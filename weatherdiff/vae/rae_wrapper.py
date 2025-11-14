"""RAE包装器，支持使用RAE的encoder和decoder"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict
from pathlib import Path

from .rae import RAE


class RAEWrapper:
    """RAE包装器，encoder固定，decoder可微调"""
    
    def __init__(self, 
                 encoder_cls: str = 'Dinov2withNorm',
                 encoder_config_path: str = 'facebook/dinov2-base',
                 encoder_input_size: int = 224,
                 encoder_params: dict = None,
                 decoder_config_path: str = 'vit_mae-base',
                 decoder_patch_size: int = 16,
                 pretrained_decoder_path: Optional[str] = None,
                 normalization_stat_path: Optional[str] = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 dtype: torch.dtype = torch.float32,
                 freeze_encoder: bool = True,
                 freeze_decoder: bool = False):
        """
        Args:
            encoder_cls: encoder类名，如 'Dinov2withNorm', 'SigLIP2wNorm', 'MAEwNorm'
            encoder_config_path: encoder配置路径（HuggingFace模型ID）
            encoder_input_size: encoder输入图像尺寸
            encoder_params: encoder参数字典
            decoder_config_path: decoder配置路径（HuggingFace模型ID）
            decoder_patch_size: decoder patch大小
            pretrained_decoder_path: 预训练decoder权重路径
            normalization_stat_path: 归一化统计量路径
            device: 设备
            dtype: 数据类型
            freeze_encoder: 是否冻结encoder（默认True）
            freeze_decoder: 是否冻结decoder（默认False，用于微调）
        """
        self.device = device
        self.dtype = dtype
        self.encoder_cls = encoder_cls
        self.encoder_config_path = encoder_config_path
        self.encoder_input_size = encoder_input_size
        self.decoder_config_path = decoder_config_path
        self.pretrained_decoder_path = pretrained_decoder_path
        
        if encoder_params is None:
            encoder_params = {}
        
        print(f"加载 RAE: encoder={encoder_cls}, decoder={decoder_config_path}")
        print(f"设备: {device}, 数据类型: {dtype}")
        
        try:
            # 创建RAE模型
            self.rae = RAE(
                encoder_cls=encoder_cls,
                encoder_config_path=encoder_config_path,
                encoder_input_size=encoder_input_size,
                encoder_params=encoder_params,
                decoder_config_path=decoder_config_path,
                decoder_patch_size=decoder_patch_size,
                pretrained_decoder_path=pretrained_decoder_path,
                normalization_stat_path=normalization_stat_path,
                reshape_to_2d=True,
                noise_tau=0.0  # 不使用noising
            ).to(device).to(dtype)
            
            # 冻结encoder
            if freeze_encoder:
                self.rae.encoder.eval()
                self.rae.encoder.requires_grad_(False)
                print("✓ Encoder已冻结")
            else:
                self.rae.encoder.train()
                self.rae.encoder.requires_grad_(True)
                print("✓ Encoder可训练")
            
            # 设置decoder
            if freeze_decoder:
                self.rae.decoder.eval()
                self.rae.decoder.requires_grad_(False)
                print("✓ Decoder已冻结")
            else:
                self.rae.decoder.train()
                self.rae.decoder.requires_grad_(True)
                print("✓ Decoder可训练（用于微调）")
            
            print("✓ RAE加载成功")
        except Exception as e:
            print(f"✗ RAE加载失败: {e}")
            raise
    
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码到潜空间
        
        Args:
            x: 输入图像，shape (B, C, H, W)，范围[-1, 1]或[0, 1]
        
        Returns:
            latent: 潜向量，shape (B, latent_dim, H_latent, W_latent)
        """
        # 确保输入在正确设备上
        x = x.to(self.device, dtype=self.dtype)
        
        # 如果输入范围是[-1, 1]，需要转换为[0, 1]（RAE期望[0, 1]范围）
        # 检查输入范围
        x_min, x_max = x.min().item(), x.max().item()
        if x_min < -0.5:  # 假设是[-1, 1]范围
            x = (x + 1.0) / 2.0  # 转换为[0, 1]
        
        # 使用RAE编码（encoder固定，不使用noising）
        self.rae.encoder.eval()
        with torch.no_grad():
            latent = self.rae.encode(x)
        
        return latent
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        从潜空间解码
        
        Args:
            latent: 潜向量，shape (B, latent_dim, H_latent, W_latent)
        
        Returns:
            x: 重建图像，shape (B, C, H, W)，范围[0, 1]（需要转换为[-1, 1]）
        """
        latent = latent.to(self.device, dtype=self.dtype)
        
        # 使用RAE解码（decoder可能可训练）
        if not self.rae.decoder.training:
            with torch.no_grad():
                x = self.rae.decode(latent)
        else:
            x = self.rae.decode(latent)
        
        # RAE输出是[0, 1]范围，转换为[-1, 1]以匹配Weather项目的数据格式
        x = x * 2.0 - 1.0
        
        return x
    
    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        重建图像（encode + decode）
        
        Args:
            x: 输入图像，shape (B, C, H, W)
        
        Returns:
            x_recon: 重建图像，shape (B, C, H, W)
        """
        latent = self.encode(x)
        x_recon = self.decode(latent)
        return x_recon
    
    def get_latent_shape(self, image_shape: tuple) -> tuple:
        """
        获取潜向量形状
        
        Args:
            image_shape: (C, H, W)
        
        Returns:
            (latent_dim, H_latent, W_latent)
        """
        # RAE encoder会将图像resize到encoder_input_size
        # 然后编码为patches，最后reshape为2D
        encoder_patch_size = self.rae.encoder_patch_size
        encoder_input_size = self.rae.encoder_input_size
        
        # 计算latent的空间维度
        h_latent = encoder_input_size // encoder_patch_size
        w_latent = encoder_input_size // encoder_patch_size
        latent_dim = self.rae.latent_dim
        
        return (latent_dim, h_latent, w_latent)
    
    def get_decoder_parameters(self):
        """获取decoder参数（用于优化器）"""
        return self.rae.decoder.parameters()
    
    def train_mode(self):
        """设置为训练模式"""
        self.rae.encoder.eval()  # encoder始终eval
        self.rae.decoder.train()  # decoder可训练
    
    def eval_mode(self):
        """设置为评估模式"""
        self.rae.encoder.eval()
        self.rae.decoder.eval()

