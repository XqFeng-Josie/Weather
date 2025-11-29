# -*- coding: utf-8 -*-
"""
Stage-1 Direct Forecast (RAE only, integrated to your repo)

历史序列 -> ExternalRepEncoder(DINOv2, 冻结 backbone，仅训练 in_adapter 可选)
        -> tokens -> RAEDiffusionWeather.decode(tokens, Hp, Wp, H, W) -> 未来场

注意：本脚本从不直接实例化 RAEDecoder；统一通过 RAEDiffusionWeather 获得 decoder 行为。
"""

import argparse
import json
import math
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.data_loader import WeatherDataLoader
from weatherdiff.vae.rae.latent_rae_weather import (
    ExternalRepEncoder as RAEExternalRepEncoder,
    RAEDiffusionWeather,
    RAEConfig,
)

# ----------------------------
# utils
# ----------------------------
def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

def ensure_img_norm_attrs(enc: nn.Module):
    # 若未注册 mean/std buffer，补 None，避免 AttributeError
    if not hasattr(enc, "img_mean"): enc.img_mean = None
    if not hasattr(enc, "img_std"):  enc.img_std = None
    return enc

# ----------------------------
# ExternalRepEncoder (DINOv2) 构建：签名自适配 + 统一 in_chans
# ----------------------------
def build_history_encoder_dino(
    token_dim: int,
    patch_size: int,
    in_chans: int,                         # << 统一使用 in_chans
    encoder_ckpt: str = "",
    target_hw: Tuple[int, int] = (518, 518),
    freeze_backbone: bool = True,
) -> nn.Module:
    # 1) DINOv2-G/14 backbone（timm）
    backbone = timm.create_model("vit_giant_patch14_dinov2", pretrained=(encoder_ckpt == ""))
    if encoder_ckpt:
        sd = torch.load(encoder_ckpt, map_location="cpu")
        backbone.load_state_dict(sd, strict=False)

    # 2) ExternalRepEncoder 签名自适配（只传它支持的参数，避免 unexpected keyword）
    sig = inspect.signature(RAEExternalRepEncoder.__init__)
    params = sig.parameters

    kwargs = {}
    if "backbone"   in params: kwargs["backbone"] = backbone
    if "token_dim"  in params: kwargs["token_dim"] = token_dim
    if "patch_size" in params: kwargs["patch_size"] = patch_size
    if "in_chans"   in params: kwargs["in_chans"] = in_chans
    if "remove_cls" in params: kwargs["remove_cls"] = True
    if "target_hw"  in params: kwargs["target_hw"] = target_hw
    if "freeze"     in params: kwargs["freeze"] = freeze_backbone
    if "freeze_backbone" in params: kwargs["freeze_backbone"] = freeze_backbone
    if "mean" in params: kwargs["mean"] = None
    if "std"  in params: kwargs["std"] = None

    enc = RAEExternalRepEncoder(**kwargs)

    # 3) 若构造参数里没有冻结开关，事后强制冻结 backbone（双保险）
    if hasattr(enc, "backbone"):
        # 若传 freeze=True 则权重本就不需要梯度。这一步保证即使某实现在构造器里没处理冻结，也被强制冻结。
        if freeze_backbone:
            set_requires_grad(enc.backbone, False)
            enc.backbone.eval()

    # 4) 兜底：确保存在 img_mean/img_std
    ensure_img_norm_attrs(enc)
    return enc

# ----------------------------
# 通过 RAEDiffusionWeather 拿到解码行为（统一用 rae.decode(...)）
# ----------------------------
def build_rae_model(cfg: RAEConfig, device: torch.device) -> RAEDiffusionWeather:
    rae = RAEDiffusionWeather(cfg).to(device)
    # 只训练 decoder.*
    for name, p in rae.named_parameters():
        p.requires_grad = name.startswith("decoder.")
    return rae

# ----------------------------
# 直推模型：历史编码 -> rae.decode -> 未来
# ----------------------------
@dataclass
class DirectCfg:
    input_length: int
    output_length: int
    in_channels_history: int
    out_channels_future: int
    token_dim: int
    patch_size: int

class Stage1RAEPredictor(nn.Module):
    def __init__(self, hist_encoder: nn.Module, rae: RAEDiffusionWeather, cfg: DirectCfg):
        super().__init__()
        self.hist_encoder = hist_encoder
        self.rae = rae
        self.cfg = cfg

    def forward(self, x_hist: torch.Tensor) -> torch.Tensor:
        """
        x_hist: (B, T_in, C_in, H, W)
        return: (B, T_out, C_out, H, W)
        """
        B, T, C, H, W = x_hist.shape
        assert T == self.cfg.input_length and C == self.cfg.in_channels_history, "Input shape mismatch"
        xin = x_hist.view(B, T * C, H, W)         # (B, T_in*C_in, H, W)
        z, Hp, Wp = self.hist_encoder(xin)        # (B, N, D), N = Hp*Wp
        y_pred = self.rae.decode(z, Hp, Wp, H, W) # 由你工程实现：返回 (B, T_out, C_out, H, W)
        # 若某实现返回 (B*T_out, C_out, H, W)，这里也兼容一下
        if y_pred.dim() == 4 and y_pred.shape[0] == B * self.cfg.output_length:
            y_pred = y_pred.view(B, self.cfg.output_length, self.cfg.out_channels_future, H, W)
        return y_pred

# ----------------------------
# 训练 / 验证 / 测试
# ----------------------------
def train_one_epoch(model, loader, opt, device, grad_clip=1.0):
    model.train()
    loss_fn = nn.MSELoss()
    total = 0.0
    for xb, yb in tqdm(loader, desc="Train", leave=False):
        xb = xb.to(device); yb = yb.to(device)
        yp = model(xb)
        loss = loss_fn(yp, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        total += float(loss.item())
    return total / max(1, len(loader))

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    loss_fn = nn.MSELoss()
    total = 0.0
    first_batch = None
    for xb, yb in tqdm(loader, desc="Val", leave=False):
        xb = xb.to(device); yb = yb.to(device)
        yp = model(xb)
        loss = loss_fn(yp, yb)
        total += float(loss.item())
        if first_batch is None:
            first_batch = (yb.detach().cpu(), yp.detach().cpu())
    return total / max(1, len(loader)), first_batch

@torch.no_grad()
def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()
    l2_sum = 0.0; l1_sum = 0.0; n_elem = 0
    per_t_l2 = None; per_t_cnt = 0
    for xb, yb in tqdm(loader, desc="Test", leave=False):
        xb = xb.to(device); yb = yb.to(device)
        yp = model(xb)
        diff = yp - yb
        l2_sum += diff.pow(2).sum().item()
        l1_sum += diff.abs().sum().item()
        n_elem += diff.numel()
        B, T, C, H, W = diff.shape
        if per_t_l2 is None: per_t_l2 = [0.0 for _ in range(T)]
        for t in range(T):
            per_t_l2[t] += diff[:, t].pow(2).sum().item()
        per_t_cnt += (B * C * H * W)
    mse = l2_sum / n_elem
    rmse = float(math.sqrt(mse))
    mae = l1_sum / n_elem
    metrics = {"mse": float(mse), "rmse": rmse, "mae": mae}
    if per_t_l2 is not None and per_t_cnt > 0:
        for t in range(len(per_t_l2)):
            metrics[f"rmse_step_{t+1}"] = float(math.sqrt(per_t_l2[t] / per_t_cnt))
    return metrics

@torch.no_grad()
def save_spatial_example(out_dir: Path, y_true: torch.Tensor, y_pred: torch.Tensor, var_name: str, epoch: int):
    safe = var_name.replace("/", "_").replace(" ", "_")
    B, T, C, H, W = y_true.shape
    b0, c0 = 0, 0
    Tplot = min(4, T)
    fig = plt.figure(figsize=(18, 8))
    for t in range(Tplot):
        ax = plt.subplot(2, Tplot, t + 1)
        im = ax.imshow(y_true[b0, t, c0].cpu().numpy(), cmap="RdBu_r")
        ax.set_title(f"GT t+{t+1}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = plt.subplot(2, Tplot, Tplot + t + 1)
        im = ax.imshow(y_pred[b0, t, c0].cpu().numpy(), cmap="RdBu_r")
        ax.set_title(f"Pred t+{t+1}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    p = out_dir / f"spatial_examples_stage1_ep{epoch:04d}.png"
    fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig)
    latest = out_dir / "spatial_examples_stage1.png"
    try:
        if latest.exists(): latest.unlink()
        p.replace(latest)
    except Exception:
        pass

# ----------------------------
# shards 保存 / 恢复：decoder + encoder.in_adapter
# ----------------------------
def _cast_fp16_cpu(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        out[k] = (v.detach().to("cpu").half() if torch.is_floating_point(v) else v.detach().to("cpu"))
    return out

def save_stage1_shards(rae: RAEDiffusionWeather, hist_encoder: nn.Module, history: Dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # 只保存 decoder 权重 + encoder.in_adapter
    torch.save(_cast_fp16_cpu(rae.decoder.state_dict()), out_dir / "decoder_fp16.pt")
    enc_sd = hist_encoder.state_dict()
    in_ad = {k: v for k, v in enc_sd.items() if k.startswith("in_adapter.")}
    if len(in_ad) > 0:
        torch.save(_cast_fp16_cpu(in_ad), out_dir / "encoder_in_adapter_fp16.pt")
    with open(out_dir / "stage1_meta.json", "w") as f:
        json.dump({"history": history}, f, indent=2)
    print(f"✓ Saved shards to {out_dir}/decoder_fp16.pt"
          f"{', encoder_in_adapter_fp16.pt' if len(in_ad)>0 else ''}")

def load_stage1_shards(rae: RAEDiffusionWeather, hist_encoder: nn.Module, src: str, device: torch.device):
    base = Path(src)
    dec_path = base / "decoder_fp16.pt" if base.is_dir() else base.parent / "decoder_fp16.pt"
    in_ad_path = base / "encoder_in_adapter_fp16.pt" if base.is_dir() else base.parent / "encoder_in_adapter_fp16.pt"

    loaded = False
    if dec_path.exists():
        sd = torch.load(dec_path, map_location=device)
        sd = {k: (v.float() if torch.is_floating_point(v) else v) for k, v in sd.items()}
        miss = rae.decoder.load_state_dict(sd, strict=False)
        print(f"✓ Loaded decoder from {dec_path}  missing={len(miss.missing_keys)} unexpected={len(miss.unexpected_keys)}")
        loaded = True
    if in_ad_path.exists():
        sd = torch.load(in_ad_path, map_location=device)
        sd = {k: (v.float() if torch.is_floating_point(v) else v) for k, v in sd.items()}
        cur = hist_encoder.state_dict(); cur.update(sd)
        hist_encoder.load_state_dict(cur, strict=False)
        print(f"✓ Loaded encoder.in_adapter from {in_ad_path}")
        loaded = True

    # 兼容 legacy best_rae_sd.pth
    if (not loaded) and base.exists() and base.is_file():
        ckpt = torch.load(base, map_location=device)
        full = ckpt.get("model", ckpt)
        dec_sub = {k.split("decoder.", 1)[1]: v for k, v in full.items() if k.startswith("decoder.")}
        if dec_sub:
            miss = rae.decoder.load_state_dict(dec_sub, strict=False)
            print(f"✓ Loaded decoder (legacy) from {base}  missing={len(miss.missing_keys)} unexpected={len(miss.unexpected_keys)}")
            loaded = True
        in_ad_sub = {k.split("encoder.", 1)[1]: v for k, v in full.items() if k.startswith("encoder.in_adapter")}
        if in_ad_sub:
            cur = hist_encoder.state_dict(); cur.update(in_ad_sub)
            hist_encoder.load_state_dict(cur, strict=False)
            print(f"✓ Loaded encoder.in_adapter (legacy) from {base}")
            loaded = True

    if not loaded:
        print(f"[WARN] nothing loaded from {src}")

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Stage-1 Direct Forecast (RAE only)")
    # Data
    ap.add_argument("--data-path", type=str,
                    default="gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr")
    ap.add_argument("--variables", type=str, default="2m_temperature")
    ap.add_argument("--time-slice", type=str, default="2015-01-01:2019-12-31")
    ap.add_argument("--input-length", type=int, default=12)
    ap.add_argument("--output-length", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--val-ratio", type=float, default=0.15)

    # Encoder / tokens
    ap.add_argument("--encoder-ckpt", type=str, default="")
    ap.add_argument("--token-dim", type=int, default=1536)
    ap.add_argument("--patch-size", type=int, default=14)
    ap.add_argument("--freeze-backbone", type=lambda x: str(x).lower()=="true", default=True)
    ap.add_argument("--train-in-adapter", type=lambda x: str(x).lower()=="true", default=True)

    # Train
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--early-stop", type=int, default=20)
    ap.add_argument("--viz-every", type=int, default=1)

    # IO
    ap.add_argument("--output-dir", type=str, default="./outputs_stage1_direct_integrated")
    ap.add_argument("--resume", type=str, default="", help="dir(含 shards) 或 legacy best_rae_sd.pth")
    return ap.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("Stage-1 Direct Forecast (RAE only, integrated)")
    print("="*80)
    print(f"Data: {args.data_path}")
    print(f"Time slice: {args.time_slice}")
    print(f"Variables: {args.variables}")
    print(f"Token dim={args.token_dim}, patch={args.patch_size}, freeze_backbone={args.freeze_backbone}")

    # ---------- 数据 ----------
    variables = [v.strip() for v in args.variables.split(",")]
    loader = WeatherDataLoader(data_path=args.data_path, variables=variables)
    s, e = args.time_slice.split(":")
    loader.load_data(time_slice=slice(s, e))
    features = loader.prepare_features(normalize=True)

    X, y = loader.create_sequences(
        features, input_length=args.input_length, output_length=args.output_length, format="spatial"
    )
    splits = loader.split_data(X, y, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val,   y_val   = splits["X_val"],   splits["y_val"]
    X_test,  y_test  = splits["X_test"],  splits["y_test"]

    print(f"\nShapes:")
    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_val  : {X_val.shape}    y_val  : {y_val.shape}")
    print(f"  X_test : {X_test.shape}   y_test : {y_test.shape}")

    inC_hist = X.shape[2]   # 通常 1
    outC_fut = y.shape[2]   # 通常 1

    # ---------- 历史编码器 ----------
    hist_encoder = build_history_encoder_dino(
        token_dim=args.token_dim,
        patch_size=args.patch_size,
        in_chans=args.input_length * inC_hist,   # 统一 in_chans
        encoder_ckpt=args.encoder_ckpt,
        target_hw=(518, 518),
        freeze_backbone=args.freeze_backbone
    ).to(device)

    # 只训练 in_adapter（若存在）
    has_in_adapter = any(n.startswith("in_adapter.") for n, _ in hist_encoder.named_parameters())
    for n, p in hist_encoder.named_parameters():
        if has_in_adapter and n.startswith("in_adapter."):
            p.requires_grad = bool(args.train_in_adapter)
        else:
            p.requires_grad = False

    # ---------- RAE 模型（仅训练 decoder.*） ----------
    rae_cfg = RAEConfig(
        input_length=args.input_length,
        output_length=args.output_length,
        in_channels_history=inC_hist,
        out_channels_future=outC_fut,
        token_dim=args.token_dim,
        patch_size=args.patch_size
    )
    rae = build_rae_model(rae_cfg, device)

    # ---------- 直推组合 ----------
    direct_cfg = DirectCfg(
        input_length=args.input_length,
        output_length=args.output_length,
        in_channels_history=inC_hist,
        out_channels_future=outC_fut,
        token_dim=args.token_dim,
        patch_size=args.patch_size
    )
    model = Stage1RAEPredictor(hist_encoder=hist_encoder, rae=rae, cfg=direct_cfg).to(device)

    # ---------- DataLoaders ----------
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()),
        batch_size=args.batch_size, shuffle=False, num_workers=max(1, args.num_workers//2), pin_memory=True
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()),
        batch_size=args.batch_size, shuffle=False, num_workers=max(1, args.num_workers//2), pin_memory=True
    )

    # ---------- Resume（可选） ----------
    if args.resume:
        print(f"\n[Resume] Loading Stage-1 shards from: {args.resume}")
        load_stage1_shards(rae, hist_encoder, args.resume, device)

    # ---------- 训练 ----------
    params = [p for p in list(rae.decoder.parameters()) + list(hist_encoder.parameters()) if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.epochs * len(train_loader)), eta_min=args.lr * 0.05
    )
    history = {"train": [], "val": []}
    best = float("inf"); patience = 0

    print("\nTraining ...")
    for ep in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, opt, device, grad_clip=args.grad_clip)
        va, first = validate(model, val_loader, device)
        history["train"].append(tr); history["val"].append(va)
        print(f"Epoch {ep:03d}/{args.epochs}  train={tr:.6f}  val={va:.6f}")

        if first is not None and (ep % args.viz_every == 0):
            yb, yp = first
            save_spatial_example(out_dir, yb, yp, variables[0] if variables else "var", ep)

        if va < best - 1e-7:
            best = va; patience = 0
            save_stage1_shards(rae, hist_encoder, history, out_dir)
        else:
            patience += 1
            if patience >= args.early_stop:
                print(f"Early stopping at epoch {ep}.")
                break
        sched.step()

    # 载入最佳（以 shards 方式）
    if (out_dir / "decoder_fp16.pt").exists():
        load_stage1_shards(rae, hist_encoder, str(out_dir), device)

    # ---------- 测试 ----------
    print("\nEvaluating on TEST ...")
    metrics = evaluate(model, test_loader, device)
    print("\nTest Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    with open(out_dir / "metrics_stage1_direct.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✓ Results saved to {out_dir}")
    print(f"✓ Shards: {out_dir}/decoder_fp16.pt (+ encoder_in_adapter_fp16.pt if present)")
    print(f"✓ Viz: {out_dir}/spatial_examples_stage1.png")


if __name__ == "__main__":
    main()
