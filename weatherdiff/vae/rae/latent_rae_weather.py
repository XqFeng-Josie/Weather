# -*- coding: utf-8 -*-
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
import shutil
import json
import matplotlib.pyplot as plt
# -------------------------------
# 工具：时间步嵌入
# -------------------------------
def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    if timesteps.dtype != torch.float:
        timesteps = timesteps.float()
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float, device=timesteps.device) / half)
    args = timesteps[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


# -------------------------------
# 简化 Patch 编码器（无预训练时兜底）
# -------------------------------
class FrozenRepEncoder(nn.Module):
    def __init__(self, in_channels: int, token_dim: int, patch_size: int = 8, freeze: bool = True):
        super().__init__()
        self.patch = patch_size
        self.proj = nn.Conv2d(in_channels, token_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(token_dim)
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        B, C, H, W = x.shape
        Hp, Wp = H // self.patch, W // self.patch
        x = x[:, :, :Hp * self.patch, :Wp * self.patch]
        feat = self.proj(x)                                # (B, D, Hp, Wp)
        feat = feat.permute(0, 2, 3, 1).reshape(B, Hp * Wp, -1)
        tokens = self.norm(feat)                           # (B, N, D)
        return tokens, Hp, Wp


# -------------------------------
# 外部预训练编码器包装（DINOv2/MAE/自定义）
# 需要：forward(x)->(tokens, Hp, Wp)
# -------------------------------
class ExternalRepEncoder(nn.Module):
    """
    使用 DINOv2（timm）作为冻结编码器：
    - 把 (B, C_all, H, W) 先经 1x1 conv 适配到 (B, 3, H, W)
    - 双线性上采样到 DINO 预训练的分辨率 (518, 518)
    - 走 backbone.forward_features，拿到 patch tokens
    """
    def __init__(
        self,
        backbone: nn.Module,
        token_dim: int,
        patch_size: int,
        in_chans: int,                   # << 新增：未来场堆叠的通道数 = T_out * C_out
        remove_cls: bool = True,
        freeze: bool = True,
        target_hw: tuple = (518, 518),   # DINOv2-G/14 预训练分辨率
        mean: Optional[torch.Tensor] = None,  # 可选：Imagenet mean，用于正规化
        std: Optional[torch.Tensor] = None,   # 可选：Imagenet std
    ):
        super().__init__()
        self.backbone = backbone
        self.token_dim = token_dim
        self.patch_size = patch_size
        self.remove_cls = remove_cls
        self.img_hw = target_hw          # (H, W) = (518, 518)

        # 1x1 conv 把 (C_all) -> 3，作为输入适配器（可训练）
        self.in_adapter = nn.Conv2d(in_chans, 3, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.in_adapter.weight, nonlinearity='linear')

        # 输入正规化（可选；若提供 Imagenet 统计就用，没有就跳过）
        if mean is not None and std is not None:
            self.register_buffer("img_mean", mean.view(1, 3, 1, 1), persistent=False)
            self.register_buffer("img_std", std.view(1, 3, 1, 1), persistent=False)
        else:
            self.img_mean = None
            self.img_std = None

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # token-wise 的后归一化（去掉 affine）
        self.post_norm = nn.LayerNorm(token_dim, elementwise_affine=False)

    @torch.no_grad()
    def _backbone_tokens(self, x_3chw: torch.Tensor) -> torch.Tensor:
        fn = getattr(self.backbone, "forward_features", self.backbone)
        out = fn(x_3chw)   # 期望含有 patch tokens
        if isinstance(out, dict) and "x_norm_patchtokens" in out:
            tok = out["x_norm_patchtokens"]
        elif isinstance(out, dict) and "last_hidden_state" in out:
            tok = out["last_hidden_state"]
        elif isinstance(out, (tuple, list)):
            tok = out[0]
        else:
            tok = out
        return tok

    def forward(self, x: torch.Tensor):
        """
        x: (B, C_all, H, W)  —— C_all = T_out * C_out
        return: tokens (B, N, D), Hp, Wp   —— N = 37*37, D = token_dim
        """
        B, C, H, W = x.shape

        # (1) 通道适配到 3ch
        x = self.in_adapter(x)  # (B, 3, H, W)

        # (2) 上采样到 518x518
        x = F.interpolate(x, size=self.img_hw, mode='bilinear', align_corners=False)

        # (3) （可选）Imagenet 归一化
        if (self.img_mean is not None) and (self.img_std is not None):
            x = (x - self.img_mean) / (self.img_std + 1e-6)

        # (4) 过 DINOv2 backbone，取 patch tokens
        tok = self._backbone_tokens(x)   # (B, N (+cls), D)

        # 去 cls
        if self.remove_cls and tok.shape[1] >= 1:
            tok = tok[:, 1:, :]

        # token-wise LN
        tok = self.post_norm(tok)

        # 这里 DINOv2-G/14@518 => grid 37x37
        Hp = self.img_hw[0] // self.patch_size   # 518 // 14 = 37
        Wp = self.img_hw[1] // self.patch_size   # 518 // 14 = 37
        tok = tok[:, : Hp * Wp, :].contiguous()  # 保守起见裁到 37*37

        return tok, Hp, Wp

# -------------------------------
# ViT 风格解码器，把 tokens 还原为图像网格
# -------------------------------
class ViTDecoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        x = self.ln1(x); x, _ = self.attn(x, x, x, need_weights=False); x = x + h
        h = x
        x = self.ln2(x); x = self.mlp(x); x = x + h
        return x


class RAEDecoder(nn.Module):
    def __init__(self, token_dim: int, out_channels: int, patch_size: int = 8, depth: int = 4, num_heads: int = 8):
        super().__init__()
        self.patch = patch_size
        self.token_dim = token_dim
        self.blocks = nn.ModuleList([ViTDecoderBlock(token_dim, num_heads=num_heads) for _ in range(depth)])
        self.head = nn.Linear(token_dim, out_channels * patch_size * patch_size)

    def forward(self, tokens: torch.Tensor, Hp: int, Wp: int, H: int, W: int) -> torch.Tensor:
        B, N, D = tokens.shape
        x = tokens
        for blk in self.blocks:
            x = blk(x)
        vec = self.head(x)  # (B, N, C * p * p)
        C = self.head.out_features // (self.patch * self.patch)
        p = self.patch
        x = vec.view(B, Hp, Wp, C, p, p).permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, Hp * p, Wp * p)
        if x.shape[-2] != H or x.shape[-1] != W:
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return x


# -------------------------------
# 历史条件 tokens（3D -> 2D -> patch）
# -------------------------------
class HistoryToTokens(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 64, token_dim: int = 256, patch_size: int = 8):
        super().__init__()
        self.time_encoder = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=(3, 3, 3), padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv3d(base_channels, base_channels, kernel_size=(3, 3, 3), padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d((1, None, None)),
        )
        self.patch = patch_size
        self.proj = nn.Conv2d(base_channels, token_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, x_hist: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = x_hist.transpose(1, 2)                # (B, C_in, T_in, H, W)
        feat = self.time_encoder(x).squeeze(2)    # (B, C, H, W)
        B, C, H, W = feat.shape
        Hp, Wp = H // self.patch, W // self.patch
        feat = feat[:, :, :Hp * self.patch, :Wp * self.patch]
        tok = self.proj(feat).permute(0, 2, 3, 1).reshape(B, Hp * Wp, -1)
        tok = self.norm(tok)
        return tok, Hp, Wp


# -------------------------------
# 简化 DiT（含跨注意力）
# -------------------------------
class CrossAttnBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.ln_q = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.ln_ca = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.ln_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = x
        x = self.ln_q(x); x, _ = self.self_attn(x, x, x, need_weights=False); x = x + h
        h = x
        x = self.ln_ca(x); x, _ = self.cross_attn(x, cond, cond, need_weights=False); x = x + h
        h = x
        x = self.ln_mlp(x); x = self.mlp(x); x = x + h
        return x


class SimpleDiT(nn.Module):
    def __init__(self, token_dim: int, depth: int = 8, num_heads: int = 8):
        super().__init__()
        self.dim = token_dim
        self.time_mlp = nn.Sequential(nn.Linear(token_dim, token_dim * 4), nn.GELU(), nn.Linear(token_dim * 4, token_dim))
        self.blocks = nn.ModuleList([CrossAttnBlock(token_dim, num_heads=num_heads) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, token_dim))

    def forward(self, noisy_tokens: torch.Tensor, timesteps: torch.Tensor, cond_tokens: torch.Tensor) -> torch.Tensor:
        B, N, D = noisy_tokens.shape
        t_embed = timestep_embedding(timesteps, D)
        t_embed = self.time_mlp(t_embed).unsqueeze(1)
        x = noisy_tokens + t_embed
        for blk in self.blocks:
            x = blk(x, cond_tokens)
        return self.out(x)

    
class DiTDH(nn.Module):
    """
    Dual-Head DiT:
      - 主干与 SimpleDiT 一样（自注意 + 条件跨注意 + MLP）
      - 输出两个头：head_eps 预测 ε（噪声），head_v 预测 v（velocity）
    """
    def __init__(self, token_dim: int, depth: int = 8, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.dim = token_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim * 4), nn.GELU(), nn.Linear(token_dim * 4, token_dim)
        )
        self.blocks = nn.ModuleList([CrossAttnBlock(token_dim, num_heads=num_heads, dropout=dropout)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(token_dim)
        # 双头
        self.head_eps = nn.Linear(token_dim, token_dim)   # 预测 ε
        self.head_v   = nn.Linear(token_dim, token_dim)   # 预测 v

    def forward(self, noisy_tokens: torch.Tensor, timesteps: torch.Tensor, cond_tokens: torch.Tensor):
        B, N, D = noisy_tokens.shape
        t_embed = timestep_embedding(timesteps, D)
        t_embed = self.time_mlp(t_embed).unsqueeze(1)     # (B,1,D)
        x = noisy_tokens + t_embed
        for blk in self.blocks:
            x = blk(x, cond_tokens)
        x = self.norm(x)
        eps = self.head_eps(x)
        v   = self.head_v(x)
        return eps, v


# -------------------------------
# 调度器
# -------------------------------
class DDPMScheduler:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, beta_schedule="linear"):
        self.num_timesteps = num_timesteps
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        else:
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = torch.cos((x / num_timesteps + 0.008) / 1.008 * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clamp(betas, 1e-4, 0.999)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def to(self, device):
        for n, v in list(self.__dict__.items()):
            if isinstance(v, torch.Tensor):
                setattr(self, n, v.to(device)); 
        return self

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        s1 = self.sqrt_alphas_cumprod[t]; s2 = self.sqrt_one_minus_alphas_cumprod[t]
        while len(s1.shape) < len(x0.shape):
            s1 = s1.unsqueeze(-1); s2 = s2.unsqueeze(-1)
        return s1 * x0 + s2 * noise

    def step(self, noise_pred: torch.Tensor, t: int, x_t: torch.Tensor) -> torch.Tensor:
        alpha = self.alphas[t]; alpha_bar = self.alphas_cumprod[t]; beta = self.betas[t]
        while len(x_t.shape) > len(alpha.shape):
            alpha = alpha.unsqueeze(-1); alpha_bar = alpha_bar.unsqueeze(-1); beta = beta.unsqueeze(-1)
        mean = (1.0 / torch.sqrt(alpha)) * (x_t - (beta / torch.sqrt(1.0 - alpha_bar)) * noise_pred)
        if t > 0:
            noise = torch.randn_like(x_t); var = self.posterior_variance[t]
            while len(x_t.shape) > len(var.shape):
                var = var.unsqueeze(-1)
            return mean + torch.sqrt(var) * noise
        else:
            return mean


# -------------------------------
# 维度相关 schedule shift（RAE 论文建议）
# -------------------------------
def shift_timesteps(t: torch.Tensor, num_train_steps: int, eff_dim: int, base_dim: int = 4096) -> torch.Tensor:
    tn = t.float() / (num_train_steps - 1)
    alpha = math.sqrt(eff_dim / float(base_dim))
    tm = alpha * tn / (1.0 + (alpha - 1.0) * tn)
    t_new = torch.clamp((tm * (num_train_steps - 1)).round(), 0, num_train_steps - 1).long().to(t.device)
    return t_new


# -------------------------------
# RAE + 扩散 主模型
# -------------------------------
@dataclass
class RAEConfig:
    input_length: int = 12
    output_length: int = 4
    in_channels_history: int = 1
    out_channels_future: int = 1
    token_dim: int = 256
    patch_size: int = 8
    rae_depth: int = 4
    rae_noise_tau: float = 0.8
    dit_depth: int = 8
    dit_heads: int = 8
    num_diffusion_steps: int = 1000
    beta_schedule: str = "squaredcos_cap_v2"
    schedule_shift_base_dim: int = 4096

    # ==== 新增：DiT 变体 & 双头训练 ====
    dit_variant: str = "dh"            # "simple" 或 "dh"
    dh_loss_eps_weight: float = 1.0    # ε 头 loss 权重
    dh_loss_v_weight: float = 1.0      # v 头 loss 权重
    dh_pred: str = "hybrid"            # 采样时使用 "eps" / "v" / "hybrid"
    dh_pred_mix: float = 0.5           # "hybrid" 时，noise = mix * ε(v) + (1-mix) * ε(ε)


class RAEDiffusionWeather(nn.Module):
    def __init__(self, cfg: RAEConfig, external_encoder: nn.Module = None):
        super().__init__()
        self.cfg = cfg
        self.future_total_c = cfg.output_length * cfg.out_channels_future
        # 编码器（外部优先）
        if external_encoder is None:
            self.encoder = FrozenRepEncoder(in_channels=self.future_total_c, token_dim=cfg.token_dim, patch_size=cfg.patch_size, freeze=True)
        else:
            self.encoder = external_encoder
        # 解码器
        self.decoder = RAEDecoder(token_dim=cfg.token_dim, out_channels=self.future_total_c, patch_size=cfg.patch_size,
                                  depth=cfg.rae_depth, num_heads=cfg.dit_heads)
        # 条件
        self.hist_enc = HistoryToTokens(in_channels=cfg.in_channels_history, base_channels=64,
                                        token_dim=cfg.token_dim, patch_size=cfg.patch_size)
        # DiT
#        self.dit = SimpleDiT(token_dim=cfg.token_dim, depth=cfg.dit_depth, num_heads=cfg.dit_heads)
        if cfg.dit_variant == "dh":
            self.dit = DiTDH(token_dim=cfg.token_dim, depth=cfg.dit_depth, num_heads=cfg.dit_heads)
        else:
            self.dit = SimpleDiT(token_dim=cfg.token_dim, depth=cfg.dit_depth, num_heads=cfg.dit_heads)

        # 调度器
        self.scheduler = DDPMScheduler(num_timesteps=cfg.num_diffusion_steps, beta_schedule=cfg.beta_schedule)
        # 损失
        self.mse = nn.MSELoss(); self.l1 = nn.L1Loss()

    # ----- 帮助函数 -----
    def _stack_future(self, y: torch.Tensor) -> torch.Tensor:
        B = y.shape[0]
        return y.reshape(B, self.future_total_c, y.shape[-2], y.shape[-1])

    def encode(self, y: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        y_stack = self._stack_future(y)
        z, Hp, Wp = self.encoder(y_stack)  # (B, N, D)
        return z, Hp, Wp

    def decode(self, z: torch.Tensor, Hp: int, Wp: int, H: int, W: int) -> torch.Tensor:
        y_rec_stack = self.decoder(z, Hp, Wp, H, W)
        B = y_rec_stack.shape[0]
        return y_rec_stack.view(B, self.cfg.output_length, self.cfg.out_channels_future, H, W)
    def _future_token_grid(self, H: int, W: int) -> tuple[int, int]:
        """
        返回未来场 latent 的网格 (Hp, Wp)：
        - 若是 DINOv2 外部编码器：固定为 (518//patch, 518//patch) = (37,37)
        - 若是卷积 patchify（FrozenRepEncoder）：按输出尺寸整除
        """
        p = self.cfg.patch_size
        if hasattr(self.encoder, "img_hw"):
            return self.encoder.img_hw[0] // p, self.encoder.img_hw[1] // p
        else:
            return H // p, W // p
    # ----- 阶段一：RAE 解码器 -----
    def forward_rae(self, y: torch.Tensor, noise_aug: bool = True):
        with torch.no_grad():
            z, Hp, Wp = self.encode(y)
        if noise_aug and self.cfg.rae_noise_tau > 0:
            sigma = torch.abs(torch.randn(z.shape[0], device=z.device)) * self.cfg.rae_noise_tau
            while len(sigma.shape) < len(z.shape):
                sigma = sigma.unsqueeze(-1)
            z = z + torch.randn_like(z) * sigma
        H, W = y.shape[-2], y.shape[-1]
        y_rec = self.decode(z, Hp, Wp, H, W)
        loss = self.l1(y_rec, y)
        return loss, {"rae_l1": float(loss.item())}

    # ----- 阶段二：DiT 扩散 -----
    def forward_diffusion(self, x_hist: torch.Tensor, y_future: torch.Tensor, use_dim_shift: bool = True):
        cond_tokens, _, _ = self.hist_enc(x_hist)
        z, Hp, Wp = self.encode(y_future)              # x0 tokens
        noise = torch.randn_like(z)
        B, N, D = z.shape; T = self.cfg.num_diffusion_steps
        t = torch.randint(0, T, (B,), device=z.device, dtype=torch.long)
        if use_dim_shift:
            eff_dim = N * D
            t = shift_timesteps(t, T, eff_dim=eff_dim, base_dim=self.cfg.schedule_shift_base_dim)
    
        # α、σ 系数
        alpha = self.scheduler.sqrt_alphas_cumprod[t]                  # (B,)
        sigma = self.scheduler.sqrt_one_minus_alphas_cumprod[t]        # (B,)
        while len(alpha.shape) < len(z.shape): alpha = alpha.unsqueeze(-1)
        while len(sigma.shape) < len(z.shape): sigma = sigma.unsqueeze(-1)
    
        # q(x_t|x0)
        z_noisy = alpha * z + sigma * noise
    
        if self.cfg.dit_variant == "dh":
            eps_pred, v_pred = self.dit(z_noisy, t, cond_tokens)      # (B,N,D)
            # 目标
            v_tgt = alpha * noise - sigma * z
            loss_eps = self.mse(eps_pred, noise)
            loss_v   = self.mse(v_pred,   v_tgt)
            loss = self.cfg.dh_loss_eps_weight * loss_eps + self.cfg.dh_loss_v_weight * loss_v
            return loss
        else:
            noise_pred = self.dit(z_noisy, t, cond_tokens)
            return self.mse(noise_pred, noise)


    @torch.no_grad()
    def sample(self, x_hist: torch.Tensor, num_inference_steps: int = 50, use_dim_shift: bool = True) -> torch.Tensor:
        cond_tokens, _, _ = self.hist_enc(x_hist)
        B, H, W = x_hist.shape[0], x_hist.shape[-2], x_hist.shape[-1]
        Hp, Wp = self._future_token_grid(H, W)            # ✅ 未来 latent 网格（如 37x37）
        N, D = Hp * Wp, self.cfg.token_dim
        z = torch.randn(B, N, D, device=x_hist.device)    # x_T tokens
    
        T = self.cfg.num_diffusion_steps
        step = max(1, T // num_inference_steps)
        timesteps = list(range(T - 1, -1, -step))
    
        for t_val in timesteps:
            t = torch.full((B,), t_val, device=x_hist.device, dtype=torch.long)
            if use_dim_shift:
                eff_dim = N * D
                t = shift_timesteps(t, T, eff_dim=eff_dim, base_dim=self.cfg.schedule_shift_base_dim)
    
            # 系数
            alpha = self.scheduler.sqrt_alphas_cumprod[t]                  # (B,)
            sigma = self.scheduler.sqrt_one_minus_alphas_cumprod[t]        # (B,)
            while len(alpha.shape) < len(z.shape): alpha = alpha.unsqueeze(-1)
            while len(sigma.shape) < len(z.shape): sigma = sigma.unsqueeze(-1)
    
            if self.cfg.dit_variant == "dh":
                eps_pred, v_pred = self.dit(z, t, cond_tokens)
                # 从 v 恢复 ε： ε(v) = σ * x_t + α * v
                eps_from_v = sigma * z + alpha * v_pred
                if self.cfg.dh_pred == "eps":
                    noise_pred = eps_pred
                elif self.cfg.dh_pred == "v":
                    noise_pred = eps_from_v
                else:
                    mix = float(self.cfg.dh_pred_mix)
                    noise_pred = mix * eps_from_v + (1.0 - mix) * eps_pred
            else:
                noise_pred = self.dit(z, t, cond_tokens)
    
            z = self.scheduler.step(noise_pred, int(t[0].item()), z)
    
        y_pred = self.decode(z, Hp, Wp, H, W)
        return y_pred

    # ----- 载入你给的权重 -----
    @staticmethod
    def from_pretrained(cfg: "RAEConfig",
                        encoder_builder: Optional[Callable[[], nn.Module]] = None,
                        decoder_ckpt: Optional[str] = None,
                        dit_ckpt: Optional[str] = None) -> "RAEDiffusionWeather":
        ext_enc = encoder_builder() if encoder_builder is not None else None
        model = RAEDiffusionWeather(cfg, external_encoder=ext_enc)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 尝试载入解码器
        if decoder_ckpt:
            try:
                sd = torch.load(decoder_ckpt, map_location=device)
                if isinstance(sd, dict) and "state_dict" in sd:
                    sd = sd["state_dict"]
                # 处理 DataParallel 前缀
                sd = {k.replace("module.", ""): v for k, v in sd.items()}
                # 兼容常见命名：decoder.head.weight 等
                matched = model.decoder.load_state_dict(sd, strict=False)
                print(f"[RAE] loaded decoder: {decoder_ckpt}\n  missing={len(matched.missing_keys)} unexpected={len(matched.unexpected_keys)}")
            except Exception as e:
                print(f"[WARN] decoder load failed: {e}")

        # 尝试载入 DiT（可能结构不完全一致，容错）
        if dit_ckpt:
            try:
                sd = torch.load(dit_ckpt, map_location=device)
                if isinstance(sd, dict) and "state_dict" in sd:
                    sd = sd["state_dict"]
                sd = {k.replace("module.", ""): v for k, v in sd.items()}
                # 过滤出和我们 SimpleDiT 结构匹配的键
                cur = model.dit.state_dict()
                filt = {k: v for k, v in sd.items() if k in cur and v.shape == cur[k].shape}
                missing = [k for k in cur.keys() if k not in filt]
                model.dit.load_state_dict({**cur, **filt}, strict=False)
                print(f"[DiT] loaded (partial-ok) from: {dit_ckpt}\n  loaded={len(filt)}  skipped={len(missing)}")
            except Exception as e:
                print(f"[WARN] dit load skipped (incompatible): {e}")

        return model



# -------------------------------
# 训练器
# -------------------------------
@dataclass
class TrainCfg:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    lr: float = 5e-5
    weight_decay: float = 1e-5
    epochs_rae: int = 20
    epochs_dit: int = 150
    batch_size: int = 8
    num_workers: int = 4
    early_stop: int = 20
    grad_clip: float = 1.0
    num_infer_steps_val: int = 50
    checkpoint: str = "outputs/best_rae_sd.pth"
    viz_every: int = 1


class RAEDiffusionTrainer:
    def __init__(self, model: RAEDiffusionWeather, cfg: TrainCfg):
        self.model = model.to(cfg.device)
        if hasattr(self.model, "scheduler") and self.model.scheduler is not None:
            self.model.scheduler.to(cfg.device)
        self.cfg = cfg
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999))
        self.history = {"train_loss": [], "val_loss": []}

    def _epoch_rae(self, loader) -> float:
        self.model.train()
#        for p in self.model.encoder.parameters(): p.requires_grad = False
        for n, p in self.model.encoder.named_parameters():
            p.requires_grad = ("in_adapter" in n)     # ✅ 允许 in_adapter 学习
        for p in self.model.decoder.parameters(): p.requires_grad = True
        for p in self.model.dit.parameters(): p.requires_grad = False

        total = 0.0
        for Xb, yb in loader:
            Xb = Xb.to(self.cfg.device); yb = yb.to(self.cfg.device)
            self.opt.zero_grad(set_to_none=True)
            loss, _ = self.model.forward_rae(yb, noise_aug=True)
            loss.backward()
            if self.cfg.grad_clip: nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.opt.step()
            total += float(loss.item())
        return total / len(loader)

    def _epoch_dit(self, loader) -> float:
        self.model.train()
        for p in self.model.encoder.parameters(): p.requires_grad = False
        for p in self.model.decoder.parameters(): p.requires_grad = False
        for p in self.model.dit.parameters(): p.requires_grad = True

        total = 0.0
        for Xb, yb in loader:
            Xb = Xb.to(self.cfg.device); yb = yb.to(self.cfg.device)
            self.opt.zero_grad(set_to_none=True)
            loss = self.model.forward_diffusion(Xb, yb, use_dim_shift=True)
            loss.backward()
            if self.cfg.grad_clip: nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.opt.step()
            total += float(loss.item())
        return total / len(loader)

    @torch.no_grad()
    def _validate_fast(self, loader) -> float:
        self.model.eval()
        total = 0.0
        for Xb, yb in loader:
            Xb = Xb.to(self.cfg.device); yb = yb.to(self.cfg.device)
            total += float(self.model.forward_diffusion(Xb, yb, use_dim_shift=True).item())
        return total / len(loader)
    
    @torch.no_grad()
    def _validate_recon(self, loader) -> float:
        self.model.eval()
        total = 0.0
        l1 = torch.nn.L1Loss()
        for _, yb in loader:
            yb = yb.to(self.cfg.device)
            # 用冻结 encoder 得到 tokens，再用 decoder 重建；不要走扩散
            z, Hp, Wp = self.model.encode(yb)
            H, W = yb.shape[-2], yb.shape[-1]
            y_rec = self.model.decode(z, Hp, Wp, H, W)
            total += float(l1(y_rec, yb).item())
        return total / len(loader)

    @torch.no_grad()
    def _validate_full(self, loader) -> float:
        self.model.eval()
        mse = nn.MSELoss(); total = 0.0
        for Xb, yb in loader:
            Xb = Xb.to(self.cfg.device); yb = yb.to(self.cfg.device)
            y_pred = self.model.sample(Xb, num_inference_steps=self.cfg.num_infer_steps_val, use_dim_shift=True)
            total += float(mse(y_pred, yb).item())
        return total / len(loader)



    def _viz_epoch(self, loader, epoch: int, stage: str):
        """
        在 out_dir 下保存当期可视化：
          - spatial_examples_rae.png （最新）
          - spatial_examples_rae_epXXXX.png （归档）
        stage: "rae"（重建可视化）或 "dit"（采样可视化）
        """
        self.model.eval()
        out_dir = Path(self.cfg.checkpoint).parent
        out_dir.mkdir(parents=True, exist_ok=True)
    
        # 取验证集第一批
        try:
            Xb, yb = next(iter(loader))
        except StopIteration:
            return
    
        Xb = Xb.to(self.cfg.device)
        yb = yb.to(self.cfg.device)
    
        with torch.no_grad():
            if stage == "rae":
                # 只看 decoder 重建（不走扩散）
                z, Hp, Wp = self.model.encode(yb)
                H, W = yb.shape[-2], yb.shape[-1]
                y_pred = self.model.decode(z, Hp, Wp, H, W)
            else:
                # 走完整采样
                y_pred = self.model.sample(
                    Xb,
                    num_inference_steps=self.cfg.num_infer_steps_val,
                    use_dim_shift=True
                )
    
        # 画图（sample_idx=0, var_idx=0，最多4个 lead time）
        b0 = 0
        v0 = 0
        Tplot = min(4, yb.shape[1])
        H, W = yb.shape[-2], yb.shape[-1]
    
        fig = plt.figure(figsize=(18, 8))
        for t in range(Tplot):
            ax = plt.subplot(2, Tplot, t + 1)
            im = ax.imshow(yb[b0, t, v0].detach().cpu().numpy(), cmap="RdBu_r")
            ax.set_title(f"[GT] t+{t+1}")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
            ax = plt.subplot(2, Tplot, Tplot + t + 1)
            im = ax.imshow(y_pred[b0, t, v0].detach().cpu().numpy(), cmap="RdBu_r")
            ax.set_title(f"[Pred-{stage}] t+{t+1}")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
        plt.tight_layout()
        # 归档文件名
        fname_epoch = out_dir / f"spatial_examples_rae_ep{epoch:04d}.png"
        fig.savefig(fname_epoch, dpi=160, bbox_inches="tight")
        plt.close(fig)
    
        # 最新的固定名（覆盖）
        latest = out_dir / "spatial_examples_rae.png"
        try:
            shutil.copyfile(fname_epoch, latest)
        except Exception:
            pass

    
    def fit(self, train_loader, val_loader, stage: str = "both"):
        best = float("inf"); patience = 0

        if stage in ("both", "rae") and self.cfg.epochs_rae > 0:
            print("\n[Stage 1/2] Train RAE decoder ...")
            for ep in range(self.cfg.epochs_rae):
                tr = self._epoch_rae(train_loader)
                va = self._validate_recon(val_loader)
                #va = self._validate_fast(val_loader)
                print(f"  RAE Epoch {ep+1}/{self.cfg.epochs_rae}  train={tr:.4f}  val={va:.4f}")
                if (ep + 1) % self.cfg.viz_every == 0:
                    self._viz_epoch(val_loader, ep + 1, stage="rae")

        if stage in ("both", "dit") and self.cfg.epochs_dit > 0:
            print("\n[Stage 2/2] Train Diffusion (DiT on tokens) ...")
            for ep in range(self.cfg.epochs_dit):
                tr = self._epoch_dit(train_loader)
                if ep >= 50 and ((ep + 1) % 10 == 0 or ep == self.cfg.epochs_dit - 1):
                    va = self._validate_full(val_loader)
                else:
                    va = self._validate_fast(val_loader)
                self.history["train_loss"].append(tr); self.history["val_loss"].append(va)
                print(f"  DiT Epoch {ep+1}/{self.cfg.epochs_dit}  train={tr:.4f}  val={va:.4f}")
                if (ep + 1) % self.cfg.viz_every == 0:
                    self._viz_epoch(val_loader, ep + 1, stage="dit")
                if va < best:
                    best = va; patience = 0; self.save(self.cfg.checkpoint)
                else:
                    patience += 1
                    if patience >= self.cfg.early_stop:
                        print(f"Early stopping at epoch {ep+1}."); break

        self.load(self.cfg.checkpoint)
        return self.history

    def save(self, path: str):
        """保存为小分片：decoder_fp16.pt / dit_fp16.pt / encoder_in_adapter_fp16.pt + meta.json"""
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)

        # 1) 元信息
        meta = {"train_cfg": self.cfg.__dict__, "rae_cfg": self.model.cfg.__dict__, "history": self.history}
        with open(base.with_suffix(".json"), "w") as f:
            json.dump(meta, f, indent=2)

        def cast_fp16_cpu(sd):
            out = {}
            for k, v in sd.items():
                if torch.is_floating_point(v):
                    out[k] = v.detach().to("cpu").half()
                else:
                    out[k] = v.detach().to("cpu")
            return out

        # 2) decoder / dit
        torch.save(cast_fp16_cpu(self.model.decoder.state_dict()), base.parent / "decoder_fp16.pt")
        torch.save(cast_fp16_cpu(self.model.dit.state_dict()),     base.parent / "dit_fp16.pt")

        # 3) 额外保存 in_adapter（若存在）
        enc_sd = self.model.encoder.state_dict()
        in_adapter_sd = {k: v for k, v in enc_sd.items() if k.startswith("in_adapter.")}
        if len(in_adapter_sd) > 0:
            torch.save(cast_fp16_cpu(in_adapter_sd), base.parent / "encoder_in_adapter_fp16.pt")

        print(f"✓ Saved shards to: {base.parent}/decoder_fp16.pt, dit_fp16.pt"
              + (", encoder_in_adapter_fp16.pt" if len(in_adapter_sd) > 0 else ""))
        print(f"✓ Meta: {base.with_suffix('.json')}")

#    def load(self, path: str):
#        ckpt = torch.load(path, map_location=self.cfg.device)
#        self.model.load_state_dict(ckpt["model"])
#        self.opt.load_state_dict(ckpt["opt"])
#        self.history = ckpt.get("history", self.history)
    def load(self, path: str):
        """优先从分片恢复，其次兼容旧版单大文件 .pth（只加载 decoder/dit/encoder.in_adapter）"""
        base = Path(path); device = self.cfg.device
        meta_path = base.with_suffix(".json")
        dec_path  = base.parent / "decoder_fp16.pt"
        dit_path  = base.parent / "dit_fp16.pt"
        enc_in_ad = base.parent / "encoder_in_adapter_fp16.pt"

        loaded_any = False

        # 1) 分片
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
            self.history = meta.get("history", self.history)
            loaded_any = True

        if dec_path.exists():
            sd = torch.load(dec_path, map_location=device)
            sd = {k: (v.float() if torch.is_floating_point(v) else v) for k, v in sd.items()}
            miss = self.model.decoder.load_state_dict(sd, strict=False)
            print(f"✓ Loaded decoder from {dec_path}  missing={len(miss.missing_keys)} unexpected={len(miss.unexpected_keys)}")
            loaded_any = True

        if dit_path.exists():
            sd = torch.load(dit_path, map_location=device)
            sd = {k: (v.float() if torch.is_floating_point(v) else v) for k, v in sd.items()}
            cur = self.model.dit.state_dict()
            filt = {k: v for k, v in sd.items() if k in cur and v.shape == cur[k].shape}
            self.model.dit.load_state_dict({**cur, **filt}, strict=False)
            print(f"✓ Loaded DiT from {dit_path}  loaded={len(filt)} skipped={len(cur)-len(filt)}")
            loaded_any = True

        if enc_in_ad.exists():
            sd = torch.load(enc_in_ad, map_location=device)
            sd = {k: (v.float() if torch.is_floating_point(v) else v) for k, v in sd.items()}
            enc_cur = self.model.encoder.state_dict()
            enc_cur.update(sd)
            self.model.encoder.load_state_dict(enc_cur, strict=False)
            print(f"✓ Loaded encoder.in_adapter from {enc_in_ad}")
            loaded_any = True

        if loaded_any:
            return

        # 2) 兼容：旧版整包 .pth
        if base.exists():
            ckpt = torch.load(base, map_location=device)
            full_sd = ckpt.get("model", ckpt)

            # decoder
            dec_sub = {k.split("decoder.", 1)[1]: v for k, v in full_sd.items() if k.startswith("decoder.")}
            if dec_sub:
                miss = self.model.decoder.load_state_dict(dec_sub, strict=False)
                print(f"✓ Loaded decoder (legacy) from {base}  missing={len(miss.missing_keys)} unexpected={len(miss.unexpected_keys)}")

            # dit
            dit_sub = {k.split("dit.", 1)[1]: v for k, v in full_sd.items() if k.startswith("dit.")}
            if dit_sub:
                cur = self.model.dit.state_dict()
                filt = {k: v for k, v in dit_sub.items() if k in cur and v.shape == cur[k].shape}
                self.model.dit.load_state_dict({**cur, **filt}, strict=False)
                print(f"✓ Loaded DiT (legacy) from {base}  loaded={len(filt)} skipped={len(cur)-len(filt)}")

            # ✅ 额外：encoder.in_adapter
            enc_in_sub = {k.split("encoder.", 1)[1]: v for k, v in full_sd.items() if k.startswith("encoder.in_adapter")}
            if enc_in_sub:
                enc_cur = self.model.encoder.state_dict()
                for k, v in enc_in_sub.items():
                    enc_cur[k] = v
                self.model.encoder.load_state_dict(enc_cur, strict=False)
                print(f"✓ Loaded encoder.in_adapter (legacy) from {base}")

            # 恢复历史（可选）
            self.history = ckpt.get("history", self.history)
            return

        print(f"[WARN] Nothing loaded from {path}")