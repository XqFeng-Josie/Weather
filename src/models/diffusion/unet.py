"""
UNet for Diffusion Models

参考：
- UNet: https://arxiv.org/abs/1505.04597
- Improved DDPM: https://arxiv.org/abs/2102.09672
- DiT: https://arxiv.org/abs/2212.09748
"""

import torch
import torch.nn as nn
import math


class SinusoidalPositionEmbedding(nn.Module):
    """
    正弦位置编码（用于时间步嵌入）

    将离散的时间步t编码为连续的向量表示
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        """
        Args:
            timesteps: (batch,) - 时间步索引

        Returns:
            embeddings: (batch, dim) - 时间步嵌入
        """
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """
    残差块（带时间步条件）

    用于UNet的编码器和解码器
    """

    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # 时间步MLP
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))

        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)

        # 残差连接
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()

    def forward(self, x, time_emb):
        """
        Args:
            x: (batch, in_channels, H, W)
            time_emb: (batch, time_emb_dim)

        Returns:
            out: (batch, out_channels, H, W)
        """
        residual = self.residual(x)

        # 第一个卷积
        h = self.norm1(x)
        h = nn.functional.silu(h)
        h = self.conv1(h)

        # 添加时间步信息
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]

        # 第二个卷积
        h = self.norm2(h)
        h = nn.functional.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + residual


class AttentionBlock(nn.Module):
    """
    自注意力块（用于UNet bottleneck）
    """

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        """
        Args:
            x: (batch, channels, H, W)

        Returns:
            out: (batch, channels, H, W)
        """
        batch, c, h, w = x.shape
        residual = x

        x = self.norm(x)
        qkv = self.qkv(x)

        # Reshape for attention
        qkv = qkv.reshape(batch, 3, self.num_heads, c // self.num_heads, h * w)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, batch, heads, hw, c_per_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        scale = (c // self.num_heads) ** -0.5
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        out = attn @ v

        # Reshape back
        out = out.permute(0, 1, 3, 2).reshape(batch, c, h, w)
        out = self.proj(out)

        return out + residual


class UNet2D(nn.Module):
    """
    UNet去噪网络

    条件输入：
    1. 噪声数据 x_t
    2. 时间步 t
    3. 条件信息（历史天气场）
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        condition_channels,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        attention_levels=(2, 3),
        dropout=0.1,
    ):
        """
        Args:
            in_channels: 输入通道数（目标变量）
            out_channels: 输出通道数（预测变量）
            condition_channels: 条件通道数（历史数据）
            base_channels: 基础通道数
            channel_mults: 各层通道倍数
            num_res_blocks: 每层残差块数量
            attention_levels: 使用attention的层级
            dropout: Dropout比率
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # 时间步嵌入
        time_emb_dim = base_channels * 4
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # 条件编码器（压缩历史信息）
        self.condition_encoder = nn.Sequential(
            nn.Conv2d(condition_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
        )

        # 初始卷积
        self.init_conv = nn.Conv2d(
            in_channels + base_channels,  # 噪声数据 + 条件信息
            base_channels,
            3,
            padding=1,
        )

        # 编码器
        self.down_blocks = nn.ModuleList()

        channels = [base_channels]
        now_channels = base_channels

        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult

            for j in range(num_res_blocks):
                self.down_blocks.append(
                    ResidualBlock(now_channels, out_ch, time_emb_dim, dropout)
                )
                now_channels = out_ch
                channels.append(now_channels)

                # 添加attention（每个残差块后）
                if i in attention_levels:
                    self.down_blocks.append(AttentionBlock(now_channels))
                    channels.append(now_channels)

            # Downsample（除了最后一层）
            if i < len(channel_mults) - 1:
                downsample = nn.Conv2d(
                    now_channels, now_channels, 3, stride=2, padding=1
                )
                self.down_blocks.append(downsample)
                channels.append(now_channels)

        # Bottleneck
        self.mid_block1 = ResidualBlock(
            now_channels, now_channels, time_emb_dim, dropout
        )
        self.mid_attn = AttentionBlock(now_channels)
        self.mid_block2 = ResidualBlock(
            now_channels, now_channels, time_emb_dim, dropout
        )

        # 解码器
        self.up_blocks = nn.ModuleList()

        for i, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult

            for j in range(num_res_blocks + 1):
                # Skip connection from encoder
                self.up_blocks.append(
                    ResidualBlock(
                        now_channels + channels.pop(), out_ch, time_emb_dim, dropout
                    )
                )
                now_channels = out_ch

                # 添加attention
                level_idx = len(channel_mults) - 1 - i
                if level_idx in attention_levels:
                    self.up_blocks.append(AttentionBlock(now_channels))

            # Upsample（除了最后一层）
            if i < len(channel_mults) - 1:
                upsample = nn.ConvTranspose2d(
                    now_channels, now_channels, 4, stride=2, padding=1
                )
                self.up_blocks.append(upsample)

        # 输出层
        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(self, x, timesteps, condition):
        """
        Args:
            x: 噪声数据 (batch, in_channels, H, W)
            timesteps: 时间步 (batch,)
            condition: 条件信息（历史天气场） (batch, condition_channels, H, W)

        Returns:
            noise_pred: 预测的噪声 (batch, out_channels, H, W)
        """
        # 时间步嵌入
        time_emb = self.time_embedding(timesteps)

        # 编码条件信息
        cond_emb = self.condition_encoder(condition)

        # 拼接输入和条件
        h = torch.cat([x, cond_emb], dim=1)
        h = self.init_conv(h)

        # 保存skip connections
        hs = [h]

        # 编码器（包含downsample）
        for block in self.down_blocks:
            if isinstance(block, ResidualBlock):
                h = block(h, time_emb)
                hs.append(h)
            elif isinstance(block, AttentionBlock):
                h = block(h)
                hs.append(h)
            else:  # Downsample (Conv2d)
                h = block(h)
                hs.append(h)

        # Bottleneck
        h = self.mid_block1(h, time_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_emb)

        # 解码器（包含upsample）
        for block in self.up_blocks:
            if isinstance(block, ResidualBlock):
                # Skip connection
                h = torch.cat([h, hs.pop()], dim=1)
                h = block(h, time_emb)
            elif isinstance(block, AttentionBlock):
                h = block(h)
            else:  # Upsample (ConvTranspose2d)
                h = block(h)

        # 输出
        h = self.out_norm(h)
        h = nn.functional.silu(h)
        h = self.out_conv(h)

        return h
