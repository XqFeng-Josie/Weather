"""U-Net模型实现"""

import torch
import torch.nn as nn
from typing import List, Optional


class ConvBlock(nn.Module):
    """卷积块"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )
    
    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    """下采样块"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        x = self.conv(x)
        skip = x  # skip在卷积之后
        x = self.pool(x)
        return x, skip


class UpBlock(nn.Module):
    """上采样块"""
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 
                                     kernel_size=2, stride=2)
        # 拼接后的通道数 = up后的通道数 + skip通道数
        self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels)
    
    def forward(self, x, skip):
        x = self.up(x)  # in_channels -> in_channels // 2
        x = torch.cat([x, skip], dim=1)  # (in_channels // 2) + skip_channels
        x = self.conv(x)
        return x


class WeatherUNet(nn.Module):
    """
    U-Net用于天气场预测
    
    输入: 过去T_in帧，shape (B, T_in * C, H, W)
    输出: 未来T_out帧，shape (B, T_out * C, H, W)
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 base_channels: int = 64,
                 depth: int = 4):
        """
        Args:
            in_channels: 输入通道数 (T_in * C)
            out_channels: 输出通道数 (T_out * C)
            base_channels: 基础通道数
            depth: 网络深度
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        
        # 输入卷积
        self.input_conv = nn.Conv2d(in_channels, base_channels, 
                                    kernel_size=3, padding=1)
        
        # 下采样路径
        self.down_blocks = nn.ModuleList()
        channels = base_channels
        for i in range(depth):
            self.down_blocks.append(
                DownBlock(channels, channels * 2)
            )
            channels *= 2
        
        # 瓶颈层
        self.bottleneck = ConvBlock(channels, channels)
        
        # 上采样路径
        self.up_blocks = nn.ModuleList()
        # 记录skip的通道数（从下采样路径）
        skip_channels_list = []
        temp_channels = base_channels
        for i in range(depth):
            temp_channels *= 2
            skip_channels_list.append(temp_channels)
        skip_channels_list.reverse()  # 反转，因为上采样时是反向的
        
        for i in range(depth):
            skip_ch = skip_channels_list[i]
            self.up_blocks.append(
                UpBlock(channels, skip_ch, channels // 2)
            )
            channels //= 2
        
        # 输出卷积
        self.output_conv = nn.Conv2d(channels, out_channels, 
                                     kernel_size=1)
    
    def forward(self, x):
        """
        Args:
            x: shape (B, T_in * C, H, W)
        
        Returns:
            out: shape (B, T_out * C, H, W)
        """
        # 输入
        x = self.input_conv(x)
        
        # 下采样并保存skip connections
        skips = []
        for down_block in self.down_blocks:
            x, skip = down_block(x)
            skips.append(skip)
        
        # 瓶颈
        x = self.bottleneck(x)
        
        # 上采样
        for up_block, skip in zip(self.up_blocks, reversed(skips)):
            x = up_block(x, skip)
        
        # 输出
        out = self.output_conv(x)
        
        return out


class LatentUNet(nn.Module):
    """
    潜空间U-Net，用于在VAE潜空间中预测
    
    输入: 过去T_in帧的潜向量，shape (B, T_in, 4, H//8, W//8)
    输出: 未来T_out帧的潜向量，shape (B, T_out, 4, H//8, W//8)
    """
    
    def __init__(self,
                 input_length: int,
                 output_length: int,
                 latent_channels: int = 4,
                 base_channels: int = 128,
                 depth: int = 3):
        """
        Args:
            input_length: 输入序列长度 T_in
            output_length: 输出序列长度 T_out
            latent_channels: 潜向量通道数（SD VAE固定为4）
            base_channels: 基础通道数
            depth: 网络深度
        """
        super().__init__()
        
        in_channels = input_length * latent_channels
        out_channels = output_length * latent_channels
        
        self.input_length = input_length
        self.output_length = output_length
        self.latent_channels = latent_channels
        
        self.unet = WeatherUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            depth=depth
        )
    
    def forward(self, x):
        """
        Args:
            x: shape (B, T_in, 4, H//8, W//8)
        
        Returns:
            out: shape (B, T_out, 4, H//8, W//8)
        """
        B, T_in, C, H, W = x.shape
        
        # 展平时间维度到通道维度
        x = x.reshape(B, T_in * C, H, W)
        
        # U-Net预测
        out = self.unet(x)
        
        # 恢复时间维度
        out = out.reshape(B, self.output_length, self.latent_channels, H, W)
        
        return out

