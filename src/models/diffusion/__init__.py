"""
Diffusion Models for Weather Prediction

参考架构：
- DDPM (Denoising Diffusion Probabilistic Models)
- DiT (Diffusion Transformer)
- GenCast (Google DeepMind)
- Pangu-Weather (Huawei)

将天气预测视为条件生成任务：
给定历史天气场 x_{t-n:t}，生成未来天气场 x_{t+1:t+m}
"""

from .diffusion_model import DiffusionWeatherModel
from .unet import UNet2D
from .unet_simple import SimpleUNet
from .noise_scheduler import DDPMScheduler
from .diffusion_trainer import DiffusionTrainer

__all__ = [
    'DiffusionWeatherModel',
    'UNet2D',
    'SimpleUNet',
    'DDPMScheduler',
    'DiffusionTrainer',
]

