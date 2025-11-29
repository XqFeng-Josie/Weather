# src/models/sd/latent_sd_weather.py
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from diffusers import UNet2DConditionModel, DDPMScheduler


# -----------------------------
# 1) 轻量 2D VAE（通道数任意）
# -----------------------------
class SimpleVAE2D(nn.Module):
    """
    用于将未来多通道、多时间步的天气场压缩到潜空间（Latent）。
    编码下采样次数由 hidden_dims 长度决定；下采样因子 factor = 2**len(hidden_dims)。
    """
    def __init__(
        self,
        in_channels: int,
        latent_channels: int = 4,
        hidden_dims: Tuple[int, ...] = (64, 128, 256, 256),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.hidden_dims = hidden_dims
        self.down_factor = 2 ** len(hidden_dims)

        # Encoder
        enc_layers = []
        c = in_channels
        for h in hidden_dims:
            enc_layers += [
                nn.Conv2d(c, h, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(num_groups=min(32, h), num_channels=h),
                nn.SiLU(),
            ]
            c = h
        self.encoder = nn.Sequential(*enc_layers)
        self.conv_mu = nn.Conv2d(c, latent_channels, 3, padding=1)
        self.conv_logvar = nn.Conv2d(c, latent_channels, 3, padding=1)

        # Decoder
        self.proj = nn.Conv2d(latent_channels, c, 3, padding=1)
        dec_layers = []
        ch = c
        for h in reversed(hidden_dims):
            dec_layers += [
                nn.ConvTranspose2d(ch, h, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(num_groups=min(32, h), num_channels=h),
                nn.SiLU(),
            ]
            ch = h
        dec_layers += [nn.Conv2d(ch, in_channels, kernel_size=3, padding=1)]
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.conv_mu(h)
        logvar = self.conv_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.proj(z)
        x_rec = self.decoder(h)
        return x_rec

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_rec = self.decode(z)
        return x_rec, mu, logvar

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # 标准 VAE KL：E[ -log q(z|x) + log p(z) ]
        return 0.5 * torch.mean(torch.sum(mu.pow(2) + torch.exp(logvar) - 1.0 - logvar, dim=(1, 2, 3)))


# --------------------------------------------
# 2) 历史条件编码器：3D-CNN + PatchEmbed → Tokens
# --------------------------------------------
class HistoryConditionEncoder(nn.Module):
    """
    将历史序列 (B, T_in, C_in, H, W) 编码为跨注意力 tokens (B, N, D)
    - 先用 3D-CNN 聚合时间维到 (B, Cc, H, W)
    - 再以 patch_size 切块并线性映射到 cross_attn_dim
    """
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        patch_size: int = 8,
        cross_attn_dim: int = 512,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.cross_attn_dim = cross_attn_dim

        self.time_encoder = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=(3, 3, 3), padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv3d(base_channels, base_channels, kernel_size=(3, 3, 3), padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d((1, None, None)),  # (B, Cc, 1, H, W)
        )
        # patch 投影
        self.proj = None  # 动态在 forward 里构建（取决于 H、W 与 patch_size）

    def forward(self, x_hist: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        x_hist: (B, T_in, C_in, H, W)
        returns:
            tokens: (B, N_patches, cross_attn_dim)
            Hp, Wp: patch 网格尺寸（用于可视化/对齐）
        """
        # to (B, C_in, T_in, H, W)
        x = x_hist.transpose(1, 2)
        feat = self.time_encoder(x).squeeze(2)  # (B, Cc, H, W)

        B, Cc, H, W = feat.shape
        p = self.patch_size
        Hp, Wp = H // p, W // p

        feat = feat[:, :, :Hp * p, :Wp * p]  # 裁到整除 patch
        # unfold 成 (B, Cc, Hp, Wp, p, p)
        feat = feat.unfold(2, p, p).unfold(3, p, p)  # (B, Cc, Hp, Wp, p, p)
        # (B, Hp*Wp, Cc*p*p)
        feat = feat.contiguous().view(B, Cc, Hp * Wp, p * p).permute(0, 2, 1, 3).contiguous()
        feat = feat.view(B, Hp * Wp, Cc * p * p)

        if self.proj is None:
            self.proj = nn.Linear(feat.shape[-1], self.cross_attn_dim).to(feat.device)

        tokens = self.proj(feat)  # (B, N, D)
        return tokens, Hp, Wp


# ---------------------------------------------------------
# 3) Stable Diffusion 风格 Latent Weather 模型（LDM 核心）
# ---------------------------------------------------------
@dataclass
class LDMConfig:
    input_length: int = 12
    output_length: int = 4
    in_channels_history: int = 1
    out_channels_future: int = 1
    base_channels: int = 64

    # VAE 配置
    vae_latent_channels: int = 4
    vae_hidden_dims: Tuple[int, ...] = (64, 128, 256, 256)
    vae_kl_weight: float = 1e-6  # KL 权重（重建阶段）

    # 条件与 U-Net
    patch_size: int = 8
    cross_attn_dim: int = 512
    unet_block_channels: Tuple[int, ...] = (64, 128, 256)  # UNet width

    # 扩散配置
    num_diffusion_steps: int = 1000
    beta_schedule: str = "squaredcos_cap_v2"  # diffusers 默认稳定的 cosine
    scale_factor: float = 0.18215  # 与 SD 一致的 latent 缩放


class WeatherLDM(nn.Module):
    """
    - VAE 编码/解码未来场（把 T_out*C_out 拼到通道维）
    - UNet2DConditionModel 在潜空间做去噪，条件来自历史序列 tokens
    - DDPMScheduler 负责噪声添加与采样
    """
    def __init__(self, cfg: LDMConfig):
        super().__init__()
        self.cfg = cfg
        self.future_total_c = cfg.output_length * cfg.out_channels_future

        # VAE
        self.vae = SimpleVAE2D(
            in_channels=self.future_total_c,
            latent_channels=cfg.vae_latent_channels,
            hidden_dims=cfg.vae_hidden_dims,
        )

        # 条件编码器
        self.cond_encoder = HistoryConditionEncoder(
            in_channels=cfg.in_channels_history,
            base_channels=cfg.base_channels,
            patch_size=cfg.patch_size,
            cross_attn_dim=cfg.cross_attn_dim,
        )

        # UNet（跨注意力）
        self.unet = UNet2DConditionModel(
            in_channels=cfg.vae_latent_channels,
            out_channels=cfg.vae_latent_channels,
            layers_per_block=2,
            block_out_channels=cfg.unet_block_channels,  # e.g. (64, 128, 256)
            down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
            cross_attention_dim=cfg.cross_attn_dim,
            sample_size=None,  # 支持任意尺寸
        )

        # Scheduler
        self.scheduler = DDPMScheduler(
            num_train_timesteps=cfg.num_diffusion_steps,
            beta_schedule=cfg.beta_schedule,
        )

        self.mse = nn.MSELoss()

    # --------- VAE 编码/解码 ----------
    def _stack_future(self, y: torch.Tensor) -> torch.Tensor:
        # (B, T_out, C_out, H, W) -> (B, T_out*C_out, H, W)
        B = y.shape[0]
        return y.reshape(B, self.future_total_c, y.shape[-2], y.shape[-1])

    def encode_future_to_latents(
        self, y: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        y_stack = self._stack_future(y)
        x_rec, mu, logvar = self.vae(y_stack)
        z = self.vae.reparameterize(mu, logvar) * self.cfg.scale_factor
        return z, (x_rec, mu, logvar)

    def decode_latents_to_future(self, z: torch.Tensor, H: int, W: int) -> torch.Tensor:
        z = z / self.cfg.scale_factor
        y_rec = self.vae.decode(z)
        y_rec = y_rec[:, :, :H, :W]
        B = y_rec.shape[0]
        return y_rec.reshape(B, self.cfg.output_length, self.cfg.out_channels_future, H, W)

    # --------- 前向（两种模式） ----------
    def forward_vae(self, y: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """仅 VAE 重建损失（用于预训练）"""
        y_stack = self._stack_future(y)
        x_rec, mu, logvar = self.vae(y_stack)
        rec = self.mse(x_rec, y_stack)
        kl = self.vae.kl_loss(mu, logvar)
        loss = rec + self.cfg.vae_kl_weight * kl
        return loss, {"rec": float(rec.item()), "kl": float(kl.item())}

    def forward_ldm(self, x_hist: torch.Tensor, y_future: torch.Tensor) -> torch.Tensor:
        """
        Stable Diffusion 风格噪声回归
        """
        cond_tokens, _, _ = self.cond_encoder(x_hist)                   # (B, N, D)
        z, _ = self.encode_future_to_latents(y_future)                  # (B, zc, h, w)
        noise = torch.randn_like(z)
        bsz = z.shape[0]
        t = torch.randint(
            0, self.scheduler.num_train_timesteps, (bsz,), device=z.device, dtype=torch.long
        )
        z_noisy = self.scheduler.add_noise(z, noise, t)
        noise_pred = self.unet(z_noisy, t, encoder_hidden_states=cond_tokens).sample
        return self.mse(noise_pred, noise)

    # --------- 采样 ----------
    @torch.no_grad()
    def sample(self, x_hist: torch.Tensor, num_inference_steps: int = 50) -> torch.Tensor:
        """
        x_hist: (B, T_in, C_in, H, W)
        return: y_pred (B, T_out, C_out, H, W)
        """
        self.unet.eval()
        self.vae.eval()

        cond_tokens, _, _ = self.cond_encoder(x_hist)
        B, _, _, H, W = x_hist.shape

        # 估计潜空间分辨率
        df = self.vae.down_factor
        h_lat = math.ceil(H / df)
        w_lat = math.ceil(W / df)

        z = torch.randn(B, self.cfg.vae_latent_channels, h_lat, w_lat, device=x_hist.device)
        self.scheduler.set_timesteps(num_inference_steps, device=x_hist.device)

        for t in self.scheduler.timesteps:
            noise_pred = self.unet(z, t, encoder_hidden_states=cond_tokens).sample
            step = self.scheduler.step(noise_pred, t, z)
            z = step.prev_sample

        y_pred = self.decode_latents_to_future(z, H, W)
        return y_pred

    @torch.no_grad()
    def generate_ensemble(
        self, x_hist: torch.Tensor, num_members: int = 10, num_inference_steps: int = 50
    ) -> torch.Tensor:
        outs = []
        for _ in range(num_members):
            outs.append(self.sample(x_hist, num_inference_steps))
        return torch.stack(outs, dim=0)  # (M, B, T_out, C_out, H, W)


# -----------------------------
# 4) 训练器（VAE + LDM 两阶段）
# -----------------------------
@dataclass
class SDTrainConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    lr: float = 5e-5
    weight_decay: float = 1e-5
    epochs_vae: int = 20
    epochs_ldm: int = 150
    batch_size: int = 8
    num_workers: int = 4
    early_stop: int = 20
    use_amp: bool = True
    grad_clip: float = 1.0
    num_inference_steps_val: int = 50
    use_ema: bool = False  # 简化：默认关闭
    checkpoint_path: str = "outputs/best_sd.pth"


class SDTrainer:
    def __init__(self, model: WeatherLDM, cfg: SDTrainConfig):
        self.model = model.to(cfg.device)
        self.cfg = cfg

        # 两阶段分别建优化器（可共享）
        self.opt = torch.optim.AdamW(
            list(self.model.parameters()),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.999),
        )
        self.scaler = GradScaler(enabled=cfg.use_amp)

        self.history = {"train_loss": [], "val_loss": []}

    def _run_epoch_vae(self, loader) -> float:
        self.model.train()
        total = 0.0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.cfg.device)
            y_batch = y_batch.to(self.cfg.device)

            self.opt.zero_grad(set_to_none=True)
            with autocast(enabled=self.cfg.use_amp):
                loss, _ = self.model.forward_vae(y_batch)

            self.scaler.scale(loss).backward()
            if self.cfg.grad_clip is not None:
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()

            total += float(loss.item())

        return total / len(loader)

    def _run_epoch_ldm(self, loader) -> float:
        self.model.train()
        total = 0.0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.cfg.device)
            y_batch = y_batch.to(self.cfg.device)

            self.opt.zero_grad(set_to_none=True)
            with autocast(enabled=self.cfg.use_amp):
                loss = self.model.forward_ldm(X_batch, y_batch)

            self.scaler.scale(loss).backward()
            if self.cfg.grad_clip is not None:
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()

            total += float(loss.item())

        return total / len(loader)

    @torch.no_grad()
    def _validate_fast(self, loader) -> float:
        """
        快速验证（不完整采样，只看噪声回归目标）。
        """
        self.model.eval()
        total = 0.0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.cfg.device)
            y_batch = y_batch.to(self.cfg.device)
            loss = self.model.forward_ldm(X_batch, y_batch)
            total += float(loss.item())
        return total / len(loader)

    @torch.no_grad()
    def _validate_full(self, loader) -> float:
        """
        完整验证：真实采样 → 计算 MSE。
        """
        self.model.eval()
        mse = nn.MSELoss()
        total = 0.0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.cfg.device)
            y_batch = y_batch.to(self.cfg.device)

            y_pred = self.model.sample(X_batch, num_inference_steps=self.cfg.num_inference_steps_val)
            loss = mse(y_pred, y_batch)
            total += float(loss.item())

        return total / len(loader)

    def fit(
        self,
        train_loader,
        val_loader,
        stage: str = "both",  # "vae" | "ldm" | "both"
    ) -> Dict[str, list]:
        best_val = float("inf")
        patience = 0

        # -------- Stage 1: 预训练 VAE --------
        if stage in ("vae", "both"):
            print("\n[Stage 1/2] Pretraining VAE ...")
            for ep in range(self.cfg.epochs_vae):
                tr = self._run_epoch_vae(train_loader)
                va = self._validate_fast(val_loader)  # 这里仍用 fast 验证
                print(f"  VAE Epoch {ep+1}/{self.cfg.epochs_vae}  train={tr:.4f}  val={va:.4f}")

        # 冻结或部分冻结 VAE（常见做法：冻结 VAE）
        for p in self.model.vae.parameters():
            p.requires_grad = False

        # -------- Stage 2: 训练 LDM --------
        if stage in ("ldm", "both"):
            print("\n[Stage 2/2] Training Latent Diffusion ...")
            for ep in range(self.cfg.epochs_ldm):
                tr = self._run_epoch_ldm(train_loader)

                # 前 50 个 epoch 快速验证，之后每 10 个 epoch 完整采样验证
                if ep >= 50 and ((ep + 1) % 10 == 0 or ep == self.cfg.epochs_ldm - 1):
                    va = self._validate_full(val_loader)
                else:
                    va = self._validate_fast(val_loader)

                self.history["train_loss"].append(tr)
                self.history["val_loss"].append(va)
                print(f"  LDM Epoch {ep+1}/{self.cfg.epochs_ldm}  train={tr:.4f}  val={va:.4f}")

                # early stop & checkpoint
                if va < best_val:
                    best_val = va
                    patience = 0
                    self.save(self.cfg.checkpoint_path)
                else:
                    patience += 1
                    if patience >= self.cfg.early_stop:
                        print(f"Early stopping at epoch {ep+1}.")
                        break

        # 加载最优
        self.load(self.cfg.checkpoint_path)
        return self.history

    def save(self, path: str):
        ckpt = {
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "cfg": self.cfg.__dict__,
            "ldm_cfg": self.model.cfg.__dict__,
            "history": self.history,
        }
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(ckpt, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.cfg.device)
        self.model.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["opt"])
        self.history = ckpt.get("history", self.history)
