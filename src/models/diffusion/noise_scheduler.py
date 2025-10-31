"""
DDPM Noise Scheduler

参考：
- DDPM: https://arxiv.org/abs/2006.11239
- Improved DDPM: https://arxiv.org/abs/2102.09672
"""

import torch
import numpy as np


class DDPMScheduler:
    """
    DDPM噪声调度器

    管理扩散过程的噪声添加和去噪步骤
    """

    def __init__(
        self,
        num_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
    ):
        """
        Args:
            num_timesteps: 扩散步数（T）
            beta_start: 初始噪声方差
            beta_end: 最终噪声方差
            beta_schedule: 噪声调度策略 (linear, cosine)
        """
        self.num_timesteps = num_timesteps

        # 生成beta序列
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif beta_schedule == "cosine":
            # Improved DDPM cosine schedule
            self.betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        # 预计算常量（DDPM公式）
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]), self.alphas_cumprod[:-1]]
        )

        # 用于采样的常量
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # 用于逆向过程
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Cosine schedule from Improved DDPM
        更平滑的噪声添加过程
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def add_noise(self, x_start, noise, timesteps):
        """
        前向扩散过程：q(x_t | x_0)

        添加噪声到原始数据：x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

        Args:
            x_start: 原始数据 (batch, channels, H, W)
            noise: 高斯噪声 (batch, channels, H, W)
            timesteps: 时间步 (batch,)

        Returns:
            x_t: 加噪后的数据
        """
        # 获取对应时间步的系数
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]

        # 调整形状以支持广播
        while len(sqrt_alpha_prod.shape) < len(x_start.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # 添加噪声
        x_noisy = sqrt_alpha_prod * x_start + sqrt_one_minus_alpha_prod * noise
        return x_noisy

    def sample_prev_timestep(self, x_t, noise_pred, timestep):
        """
        逆向采样：从x_t采样x_{t-1}

        使用模型预测的噪声，逐步去噪

        Args:
            x_t: 当前噪声数据
            noise_pred: 模型预测的噪声
            timestep: 当前时间步

        Returns:
            x_{t-1}: 去噪一步后的数据
        """
        # 获取系数
        alpha = self.alphas[timestep]
        alpha_cumprod = self.alphas_cumprod[timestep]
        beta = self.betas[timestep]

        # 调整形状
        while len(alpha.shape) < len(x_t.shape):
            alpha = alpha.unsqueeze(-1)
            alpha_cumprod = alpha_cumprod.unsqueeze(-1)
            beta = beta.unsqueeze(-1)

        # 计算均值
        sqrt_recip_alpha = 1.0 / torch.sqrt(alpha)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)

        mean = sqrt_recip_alpha * (
            x_t - beta * noise_pred / sqrt_one_minus_alpha_cumprod
        )

        # 添加噪声（除了最后一步）
        if timestep > 0:
            noise = torch.randn_like(x_t)
            variance = self.posterior_variance[timestep]
            while len(variance.shape) < len(x_t.shape):
                variance = variance.unsqueeze(-1)
            x_prev = mean + torch.sqrt(variance) * noise
        else:
            x_prev = mean

        return x_prev

    def to(self, device):
        """移动到指定设备"""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(
            device
        )
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self
