"""噪声调度器"""

import torch
import numpy as np
from typing import Optional


class DDPMScheduler:
    """DDPM噪声调度器"""
    
    def __init__(self, 
                 num_train_timesteps: int = 1000,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 beta_schedule: str = 'linear'):
        """
        Args:
            num_train_timesteps: 训练时的时间步数
            beta_start: 起始beta值
            beta_end: 结束beta值
            beta_schedule: beta调度方式 ('linear', 'scaled_linear')
        """
        self.num_train_timesteps = num_train_timesteps
        
        # 计算beta
        if beta_schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == 'scaled_linear':
            # Stable Diffusion使用的调度方式
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, 
                                       num_train_timesteps) ** 2
        else:
            raise ValueError(f"未知的beta_schedule: {beta_schedule}")
        
        # 计算alpha
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # 用于加噪和去噪
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def add_noise(self, 
                  original: torch.Tensor,
                  noise: torch.Tensor,
                  timesteps: torch.Tensor) -> torch.Tensor:
        """
        添加噪声: x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * noise
        
        Args:
            original: 原始数据 x_0
            noise: 噪声
            timesteps: 时间步
        
        Returns:
            加噪后的数据 x_t
        """
        # 确保索引在正确的设备上
        device = original.device
        sqrt_alpha_prod = self.sqrt_alphas_cumprod.to(device)[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod.to(device)[timesteps]
        
        # 调整形状以匹配batch维度
        while len(sqrt_alpha_prod.shape) < len(original.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy = sqrt_alpha_prod * original + sqrt_one_minus_alpha_prod * noise
        return noisy
    
    def step(self,
             model_output: torch.Tensor,
             timestep: int,
             sample: torch.Tensor,
             generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        去噪一步（DDPM采样）
        
        Args:
            model_output: 模型预测的噪声
            timestep: 当前时间步
            sample: 当前样本 x_t
            generator: 随机数生成器
        
        Returns:
            去噪后的样本 x_{t-1}
        """
        # 确保在正确的设备上
        device = sample.device
        
        # 获取参数
        alpha_prod_t = self.alphas_cumprod[timestep].to(device)
        alpha_prod_t_prev = self.alphas_cumprod[timestep - 1].to(device) if timestep > 0 else torch.tensor(1.0, device=device)
        beta_t = self.betas[timestep].to(device)
        
        # 预测x_0
        pred_original = (sample - torch.sqrt(1 - alpha_prod_t) * model_output) / torch.sqrt(alpha_prod_t)
        
        # 计算x_{t-1}的均值
        pred_original_coef = torch.sqrt(alpha_prod_t_prev) * beta_t / (1 - alpha_prod_t)
        current_sample_coef = torch.sqrt(1 - beta_t) * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t)
        
        pred_prev_mean = pred_original_coef * pred_original + current_sample_coef * sample
        
        # 添加噪声（除了最后一步）
        if timestep > 0:
            noise = torch.randn(sample.shape, generator=generator, 
                              device=sample.device, dtype=sample.dtype)
            variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * beta_t
            pred_prev_sample = pred_prev_mean + torch.sqrt(variance) * noise
        else:
            pred_prev_sample = pred_prev_mean
        
        return pred_prev_sample


class DDIMScheduler:
    """DDIM噪声调度器（更快的采样）"""
    
    def __init__(self,
                 num_train_timesteps: int = 1000,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 beta_schedule: str = 'linear'):
        """
        Args:
            num_train_timesteps: 训练时的时间步数
            beta_start: 起始beta值
            beta_end: 结束beta值
            beta_schedule: beta调度方式
        """
        self.num_train_timesteps = num_train_timesteps
        
        # 计算beta和alpha
        if beta_schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == 'scaled_linear':
            betas = torch.linspace(beta_start**0.5, beta_end**0.5, 
                                  num_train_timesteps) ** 2
        else:
            raise ValueError(f"未知的beta_schedule: {beta_schedule}")
        
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # 设置采样时间步
        self.timesteps = None
    
    def set_timesteps(self, num_inference_steps: int):
        """设置推理时间步"""
        # 等间隔采样
        step_ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = torch.arange(0, num_inference_steps) * step_ratio
        self.timesteps = torch.flip(self.timesteps, dims=[0])
    
    def add_noise(self, 
                  original: torch.Tensor,
                  noise: torch.Tensor,
                  timesteps: torch.Tensor) -> torch.Tensor:
        """添加噪声"""
        # 确保索引在正确的设备上
        device = original.device
        sqrt_alpha_prod = torch.sqrt(self.alphas_cumprod.to(device)[timesteps])
        sqrt_one_minus_alpha_prod = torch.sqrt(1.0 - self.alphas_cumprod.to(device)[timesteps])
        
        while len(sqrt_alpha_prod.shape) < len(original.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy = sqrt_alpha_prod * original + sqrt_one_minus_alpha_prod * noise
        return noisy
    
    def step(self,
             model_output: torch.Tensor,
             timestep: int,
             sample: torch.Tensor,
             eta: float = 0.0) -> torch.Tensor:
        """
        DDIM去噪步骤（确定性采样）
        
        Args:
            model_output: 模型预测的噪声
            timestep: 当前时间步索引
            sample: 当前样本
            eta: 随机性参数（0为完全确定性）
        """
        # 确保在正确的设备上
        device = sample.device
        
        # 获取时间步
        t = self.timesteps[timestep].item()
        prev_t = self.timesteps[timestep + 1].item() if timestep < len(self.timesteps) - 1 else 0
        
        alpha_prod_t = self.alphas_cumprod[t].to(device)
        alpha_prod_t_prev = self.alphas_cumprod[prev_t].to(device)
        
        # 预测x_0
        pred_original = (sample - torch.sqrt(1 - alpha_prod_t) * model_output) / torch.sqrt(alpha_prod_t)
        
        # 方向指向x_t
        direction = torch.sqrt(1 - alpha_prod_t_prev) * model_output
        
        # 去噪
        pred_prev_sample = torch.sqrt(alpha_prod_t_prev) * pred_original + direction
        
        return pred_prev_sample

