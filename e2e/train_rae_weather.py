# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.data_loader import WeatherDataLoader
from weatherdiff.vae.rae.latent_rae_weather import (
    RAEConfig, RAEDiffusionWeather, TrainCfg, RAEDiffusionTrainer, ExternalRepEncoder
)

def parse_args():
    
    ap = argparse.ArgumentParser("Train RAE (pretrained) Latent Diffusion for Weather")
    # Data
    ap.add_argument("--data-path", type=str,
        default="gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr")
    ap.add_argument("--variables", type=str, default="2m_temperature")
    ap.add_argument("--time-slice", type=str, default="2020-01-01:2020-12-31")
    ap.add_argument("--input-length", type=int, default=12)
    ap.add_argument("--output-length", type=int, default=4)

    # Tokens / RAE / DiT
    ap.add_argument("--token-dim", type=int, default=1536)    # ViT-XL/14 ⇒ 1536
    ap.add_argument("--patch-size", type=int, default=14)     # DINOv2/14
    ap.add_argument("--rae-depth", type=int, default=4)
    ap.add_argument("--rae-noise-tau", type=float, default=0.8)
    ap.add_argument("--dit-depth", type=int, default=8)
    ap.add_argument("--dit-heads", type=int, default=8)

    # Diffusion
    ap.add_argument("--num-diffusion-steps", type=int, default=1000)
    ap.add_argument("--beta-schedule", type=str, default="squaredcos_cap_v2",
                    choices=["linear", "cosine", "squaredcos_cap_v2"])
    ap.add_argument("--num-inference-steps", type=int, default=50)
    ap.add_argument("--schedule-shift-base-dim", type=int, default=4096)

    # Pretrained paths (填你的路径)
    ap.add_argument("--decoder-ckpt", type=str, default="./models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt")
    ap.add_argument("--dit-ckpt", type=str, default="")  # 可填 ./models/DiTs/.../stage2_model.pt（可能部分加载）

    # Train
    ap.add_argument("--epochs-rae", type=int, default=20)
    ap.add_argument("--epochs-dit", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--early-stop", type=int, default=20)
    ap.add_argument("--stage", type=str, default="both", choices=["rae", "dit", "both"])
    ap.add_argument("--output-dir", type=str, default="./outputs_rae_pretrained")

    # External encoder backend
    ap.add_argument("--encoder-backend", type=str, default="none",
                    choices=["none","dino-timm","mae-timm"])
    ap.add_argument("--encoder-ckpt", type=str, default="")  # 可选：你本地 DINO/MAE ckpt
    ap.add_argument("--resume", type=str, default="", help="path to a previous checkpoint (.pth or shards)")

    ap.add_argument("--dit-variant", type=str, default="dh", choices=["simple","dh"],
                help="选择 SimpleDiT 或 Dual-Head DiT")
    ap.add_argument("--dh-loss-eps-weight", type=float, default=1.0)
    ap.add_argument("--dh-loss-v-weight", type=float, default=1.0)
    ap.add_argument("--dh-pred", type=str, default="hybrid", choices=["eps","v","hybrid"],
                    help="采样时使用的参数化")
    ap.add_argument("--dh-pred-mix", type=float, default=0.5, help="hybrid 的加权（0~1）")
    return ap.parse_args()


def build_external_encoder(backend: str, ckpt: str, token_dim: int, patch_size: int, in_chans: int):
    if backend == "none":
        return None
    import timm, torch
    if backend == "dino-timm":
        name = "vit_giant_patch14_dinov2"  # D=1536, p=14
        backbone = timm.create_model(name, pretrained=(ckpt == ""))
        if ckpt:
            sd = torch.load(ckpt, map_location="cpu")
            backbone.load_state_dict(sd, strict=False)

        # 可选：Imagenet 统计（若你有 ./models/stats/.../stat.pt）
        mean = std = None
        # 例如：stat = torch.load("./models/stats/dinov2/wReg_base/imagenet1k/stat.pt", map_location="cpu")
        # mean = torch.tensor(stat["mean"])  # (3,)
        # std  = torch.tensor(stat["std"])   # (3,)

        return ExternalRepEncoder(
            backbone=backbone,
            token_dim=token_dim,
            patch_size=patch_size,
            in_chans=in_chans,
            remove_cls=True,
            freeze=True,
            target_hw=(518, 518),
            mean=mean,
            std=std,
        )
    # 其它 backend 可按需扩展
    return None



def main():
    args = parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Weather RAE Latent Diffusion (Pretrained)")
    print("=" * 80)

    # 1) 数据
    variables = [v.strip() for v in args.variables.split(",")]
    loader = WeatherDataLoader(data_path=args.data_path, variables=variables)
    start, end = args.time_slice.split(":")
    ds = loader.load_data(time_slice=slice(start, end))
    feats = loader.prepare_features(normalize=True)

    X, y = loader.create_sequences(
        feats,
        input_length=args.input_length,
        output_length=args.output_length,
        format="spatial",
    )
    print(f"\nX: {X.shape}  (N, T_in, C_in, H, W)")
    print(f"y: {y.shape}  (N, T_out, C_out, H, W)")

    splits = loader.split_data(X, y, train_ratio=0.7, val_ratio=0.15)
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 2) 模型（载入你的预训练）
    in_channels_history = X.shape[2]
    out_channels_future = y.shape[2]
    in_chans_future = args.output_length * out_channels_future
    
    rae_cfg = RAEConfig(
        input_length=args.input_length,
        output_length=args.output_length,
        in_channels_history=in_channels_history,
        out_channels_future=out_channels_future,
        token_dim=args.token_dim,
        patch_size=args.patch_size,
        rae_depth=args.rae_depth,
        rae_noise_tau=args.rae_noisy_tau if hasattr(args, "rae_noisy_tau") else args.rae_noise_tau,
        dit_depth=args.dit_depth,
        dit_heads=args.dit_heads,
        num_diffusion_steps=args.num_diffusion_steps,
        beta_schedule=args.beta_schedule,
        schedule_shift_base_dim=args.schedule_shift_base_dim,
            # ==== 新增：DH ====
        dit_variant=args.dit_variant,
        dh_loss_eps_weight=args.dh_loss_eps_weight,
        dh_loss_v_weight=args.dh_loss_v_weight,
        dh_pred=args.dh_pred,
        dh_pred_mix=args.dh_pred_mix,
    )

    def enc_builder():
        return build_external_encoder(args.encoder_backend, args.encoder_ckpt, args.token_dim, args.patch_size, in_chans_future)

    model = RAEDiffusionWeather.from_pretrained(
        rae_cfg,
        encoder_builder=enc_builder,
        decoder_ckpt=args.decoder_ckpt if args.decoder_ckpt else None,
        dit_ckpt=args.dit_ckpt if args.dit_ckpt else None
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable params: {n_params:,}")

    train_cfg = TrainCfg(
        device=device, lr=args.lr,
        epochs_rae=args.epochs_rae, epochs_dit=args.epochs_dit,
        batch_size=args.batch_size, early_stop=args.early_stop,
        num_infer_steps_val=args.num_inference_steps,
        checkpoint=str(out_dir / "best_rae_sd.pth"),
    )
    trainer = RAEDiffusionTrainer(model, train_cfg)
    if args.resume:
        print(f"\n[Resume] Loading checkpoint from: {args.resume}")
        trainer.load(args.resume)
    # 3) 训练
    history = trainer.fit(train_loader, val_loader, stage=args.stage)

    # 4) 测试/评估
    print("\nEvaluating on test ...")
    preds = []
    with torch.no_grad():
        for Xb, _ in tqdm(test_loader):
            Xb = Xb.to(device)
            yp = model.sample(Xb, num_inference_steps=args.num_inference_steps, use_dim_shift=True)
            preds.append(yp.cpu().numpy())
    y_pred = np.concatenate(preds, axis=0)
    np.save(out_dir / "y_test_pred_rae.npy", y_pred)
    np.save(out_dir / "y_test_rae.npy", y_test)

    mse = np.mean((y_test - y_pred) ** 2)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_test - y_pred)))
    rmse_per_step = {
    f"rmse_step_{i+1}": float(np.sqrt(np.mean((y_test[:, i] - y_pred[:, i]) ** 2)))
    for i in range(args.output_length)
}
    metrics = {"mse": float(mse), "rmse": rmse, "mae": mae, **rmse_per_step}
    with open(out_dir / "metrics_rae.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nTest Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # 5) 可视化（简单）
    if len(history["train_loss"]) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(history["train_loss"], label="Train")
        plt.plot(history["val_loss"], label="Val")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("RAE+DiT Training")
        plt.grid(True, alpha=0.3); plt.legend()
        plt.savefig(out_dir / "training_history_rae.png", dpi=160, bbox_inches="tight"); plt.close()

    steps = list(range(1, args.output_length + 1))
    vals = [metrics[f"rmse_step_{i}"] for i in steps]
    plt.figure(figsize=(8, 5))
    plt.plot(steps, vals, "o-", linewidth=2)
    plt.xlabel("Lead Time Step"); plt.ylabel("RMSE"); plt.title("RMSE vs Lead Time")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / "rmse_vs_leadtime_rae.png", dpi=160, bbox_inches="tight"); plt.close()

    # 空间例子
    sample_idx, var_idx = 0, 0
    plt.figure(figsize=(18, 8)); Tplot = min(4, args.output_length)
    for t in range(Tplot):
        plt.subplot(2, Tplot, t + 1); plt.imshow(y_test[sample_idx, t, var_idx], cmap="RdBu_r"); plt.title(f"True t+{t+1}"); plt.colorbar(fraction=0.046, pad=0.04)
        plt.subplot(2, Tplot, Tplot + t + 1); plt.imshow(y_pred[sample_idx, t, var_idx], cmap="RdBu_r"); plt.title(f"Pred t+{t+1}"); plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.savefig(out_dir / "spatial_examples_rae.png", dpi=160, bbox_inches="tight"); plt.close()

    print(f"\n✓ Results saved to {out_dir}")
    print(f"✓ Checkpoint: {out_dir/'best_rae_sd.pth'}")


if __name__ == "__main__":
    main()
