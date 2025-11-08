"""VAE包装器，支持使用Stable Diffusion VAE"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict
from pathlib import Path


class SDVAEWrapper:
    """Stable Diffusion VAE包装器"""
    
    def __init__(self, 
                 model_id: str = "runwayml/stable-diffusion-v1-5",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 dtype: torch.dtype = torch.float32):
        """
        Args:
            model_id: HuggingFace模型ID
            device: 设备
            dtype: 数据类型
        """
        self.device = device
        self.dtype = dtype
        self.model_id = model_id
        
        print(f"加载 Stable Diffusion VAE: {model_id}")
        print(f"设备: {device}, 数据类型: {dtype}")
        
        try:
            from diffusers import AutoencoderKL
            self.vae = AutoencoderKL.from_pretrained(
                model_id, 
                subfolder="vae",
                torch_dtype=dtype
            ).to(device)
            self.vae.eval()
            print("✓ VAE加载成功")
        except Exception as e:
            print(f"✗ VAE加载失败: {e}")
            raise
    
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码到潜空间
        
        Args:
            x: 输入图像，shape (B, C, H, W)，范围[-1, 1]
        
        Returns:
            latent: 潜向量，shape (B, 4, H//8, W//8)
        """
        x = x.to(self.device, dtype=self.dtype)
        latent_dist = self.vae.encode(x).latent_dist
        latent = latent_dist.sample() * self.vae.config.scaling_factor
        return latent
    
    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        从潜空间解码
        
        Args:
            latent: 潜向量，shape (B, 4, H//8, W//8)
        
        Returns:
            x: 重建图像，shape (B, C, H, W)，范围[-1, 1]
        """
        latent = latent.to(self.device, dtype=self.dtype)
        latent = latent / self.vae.config.scaling_factor
        x = self.vae.decode(latent).sample
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
            (4, H//8, W//8)
        """
        c, h, w = image_shape
        return (4, h // 8, w // 8)


def test_vae_reconstruction(vae_wrapper: SDVAEWrapper,
                           test_data: torch.Tensor,
                           normalizer: Optional[object] = None,
                           variable: str = 'default',
                           save_path: Optional[str] = None) -> Dict[str, float]:
    """
    测试VAE重建能力
    
    Args:
        vae_wrapper: VAE包装器
        test_data: 测试数据，shape (N, C, H, W)，已归一化到[-1, 1]
        normalizer: 归一化器（用于反归一化）
        variable: 变量名
        save_path: 结果保存路径
    
    Returns:
        评估指标字典
    """
    from ..utils.metrics import calculate_metrics, format_metrics
    
    print("\n" + "=" * 60)
    print("VAE重建测试 (E0: 直接使用SD VAE)")
    print("=" * 60)
    
    # 批量处理以避免内存溢出
    batch_size = 8
    n_samples = len(test_data)
    
    reconstructions = []
    
    print(f"处理 {n_samples} 个样本...")
    for i in range(0, n_samples, batch_size):
        batch = test_data[i:i + batch_size]
        batch_recon = vae_wrapper.reconstruct(batch)
        reconstructions.append(batch_recon.cpu())
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  已处理: {i + len(batch)}/{n_samples}")
    
    reconstructions = torch.cat(reconstructions, dim=0)
    
    # 计算指标
    print("\n计算评估指标...")
    metrics = calculate_metrics(reconstructions, test_data, ensemble=False)
    
    # 如果有normalizer，也计算原始尺度的指标
    if normalizer is not None:
        test_data_orig = normalizer.inverse_transform(
            test_data.numpy(), name=variable
        )
        recon_orig = normalizer.inverse_transform(
            reconstructions.numpy(), name=variable
        )
        
        print("\n归一化空间的指标:")
        print(format_metrics(metrics))
        
        metrics_orig = calculate_metrics(recon_orig, test_data_orig, ensemble=False)
        print("\n原始尺度的指标:")
        print(format_metrics(metrics_orig))
        
        # 保存结果
        if save_path:
            import json
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            # 转换numpy类型为Python原生类型
            def convert_to_python_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_to_python_types(v) for k, v in obj.items()}
                elif isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            results = {
                'normalized_space': convert_to_python_types(metrics),
                'original_space': convert_to_python_types(metrics_orig),
                'vae_model': vae_wrapper.model_id,
                'n_samples': int(n_samples)
            }
            
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n✓ 结果已保存到: {save_path}")
        
        return metrics_orig
    else:
        print(format_metrics(metrics))
        
        if save_path:
            import json
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            # 转换numpy类型为Python原生类型
            def convert_to_python_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_to_python_types(v) for k, v in obj.items()}
                elif isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            with open(save_path, 'w') as f:
                json.dump(convert_to_python_types(metrics), f, indent=2)
            
            print(f"\n✓ 结果已保存到: {save_path}")
        
        return metrics

