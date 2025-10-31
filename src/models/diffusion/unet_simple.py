"""
Simplified UNet for Diffusion Models

简化版本，确保稳定性
"""

import torch
import torch.nn as nn
import math


class TimeEmbedding(nn.Module):
    """时间步嵌入"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        self.register_buffer("emb", emb)

    def forward(self, timesteps):
        emb = timesteps[:, None] * self.emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class SimpleBlock(nn.Module):
    """简单的残差块"""

    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        skip = self.skip(x)
        x = nn.functional.silu(self.conv1(x))
        x = self.norm1(x)
        x = x + self.time_mlp(t_emb)[:, :, None, None]
        x = nn.functional.silu(self.conv2(x))
        x = self.norm2(x)
        return x + skip


class SimpleUNet(nn.Module):
    """
    简化的UNet - 更稳定但功能完整
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        condition_channels,
        base_channels=64,
    ):
        super().__init__()

        time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            TimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # 条件编码
        self.cond_encoder = nn.Sequential(
            nn.Conv2d(condition_channels, base_channels, 3, padding=1),
            nn.SiLU(),
        )

        # 初始层
        self.init = nn.Conv2d(in_channels + base_channels, base_channels, 3, padding=1)

        # Encoder
        self.down1 = SimpleBlock(base_channels, base_channels * 2, time_dim)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = SimpleBlock(base_channels * 2, base_channels * 4, time_dim)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = SimpleBlock(base_channels * 4, base_channels * 4, time_dim)

        # Decoder
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = SimpleBlock(
            base_channels * 2 + base_channels * 4, base_channels * 2, time_dim
        )  # up + skip

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = SimpleBlock(
            base_channels + base_channels * 2, base_channels, time_dim
        )  # up + skip

        # Output (添加归一化层确保输出稳定)
        self.out = nn.Sequential(
            nn.GroupNorm(min(8, base_channels), base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 1),
        )

    def forward(self, x, timesteps, condition):
        # Time embedding
        t_emb = self.time_mlp(timesteps)

        # Condition
        cond = self.cond_encoder(condition)

        # Initial
        x = torch.cat([x, cond], dim=1)
        x = self.init(x)
        x0 = x

        # Encoder
        x1 = self.down1(x, t_emb)
        x = self.pool1(x1)

        x2 = self.down2(x, t_emb)
        x = self.pool2(x2)

        # Bottleneck
        x = self.bottleneck(x, t_emb)

        # Decoder
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x, t_emb)

        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x, t_emb)

        # Output
        x = self.out(x)
        return x
