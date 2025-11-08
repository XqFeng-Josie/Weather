"""扩散模型实现"""

import torch
import torch.nn as nn
from typing import Optional
from ..unet.unet_model import ConvBlock, DownBlock, UpBlock


class TimestepEmbedding(nn.Module):
    """时间步嵌入"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: shape (B,)
        
        Returns:
            embeddings: shape (B, dim)
        """
        # 正弦位置编码
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # MLP
        emb = self.mlp(emb)
        return emb


class ConvBlockWithTime(nn.Module):
    """带时间嵌入的卷积块"""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        
        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.act = nn.SiLU()
        
        # 残差连接
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x, time_emb):
        """
        Args:
            x: shape (B, C, H, W)
            time_emb: shape (B, time_emb_dim)
        """
        residual = self.residual_conv(x)
        
        # 第一个卷积
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        
        # 添加时间嵌入
        time_emb = self.time_emb_proj(time_emb)
        x = x + time_emb[:, :, None, None]
        
        # 第二个卷积
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        
        # 残差
        return x + residual


class WeatherDiffusion(nn.Module):
    """
    扩散模型用于天气预测
    
    输入: 
        - latent: 加噪的未来帧潜向量 (B, T_out, 4, H//8, W//8)
        - timestep: 噪声时间步 (B,)
        - condition: 过去帧的潜向量 (B, T_in, 4, H//8, W//8)
    
    输出:
        - predicted_noise: 预测的噪声 (B, T_out, 4, H//8, W//8)
    """
    
    def __init__(self,
                 input_length: int,
                 output_length: int,
                 latent_channels: int = 4,
                 base_channels: int = 128,
                 depth: int = 3,
                 time_emb_dim: int = 256):
        """
        Args:
            input_length: 条件序列长度 (T_in)
            output_length: 输出序列长度 (T_out)
            latent_channels: 潜向量通道数
            base_channels: 基础通道数
            depth: 网络深度
            time_emb_dim: 时间嵌入维度
        """
        super().__init__()
        
        self.input_length = input_length
        self.output_length = output_length
        self.latent_channels = latent_channels
        
        # 时间步嵌入
        self.time_embedding = TimestepEmbedding(time_emb_dim)
        
        # 输入投影
        in_channels = (input_length + output_length) * latent_channels
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # 下采样路径
        self.down_blocks = nn.ModuleList()
        channels = base_channels
        for i in range(depth):
            self.down_blocks.append(
                ConvBlockWithTime(channels, channels * 2, time_emb_dim)
            )
            channels *= 2
        
        self.down_pools = nn.ModuleList([nn.MaxPool2d(2) for _ in range(depth)])
        
        # 瓶颈层
        self.bottleneck = ConvBlockWithTime(channels, channels, time_emb_dim)
        
        # 上采样路径
        self.up_blocks = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in range(depth):
            self.up_convs.append(
                nn.ConvTranspose2d(channels, channels // 2, 2, stride=2)
            )
            # 上采样后需要与skip连接，所以输入通道数是 (channels // 2) + channels
            self.up_blocks.append(
                ConvBlockWithTime(channels // 2 + channels, channels // 2, time_emb_dim)
            )
            channels //= 2
        
        # 输出投影
        out_channels = output_length * latent_channels
        self.output_conv = nn.Conv2d(channels, out_channels, 1)
    
    def forward(self, 
                latent: torch.Tensor,
                timestep: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: 加噪的未来帧 (B, T_out, 4, H//8, W//8)
            timestep: 时间步 (B,)
            condition: 条件（过去帧） (B, T_in, 4, H//8, W//8)
        
        Returns:
            noise: 预测的噪声 (B, T_out, 4, H//8, W//8)
        """
        B = latent.shape[0]
        
        # 时间嵌入
        t_emb = self.time_embedding(timestep)
        
        # 展平时间维度
        latent_flat = latent.reshape(B, -1, latent.shape[-2], latent.shape[-1])
        condition_flat = condition.reshape(B, -1, condition.shape[-2], condition.shape[-1])
        
        # 拼接条件
        x = torch.cat([latent_flat, condition_flat], dim=1)
        
        # 输入
        x = self.input_conv(x)
        
        # 下采样
        skips = []
        for down_block, pool in zip(self.down_blocks, self.down_pools):
            x = down_block(x, t_emb)
            skips.append(x)
            x = pool(x)
        
        # 瓶颈
        x = self.bottleneck(x, t_emb)
        
        # 上采样
        for up_conv, up_block, skip in zip(self.up_convs, self.up_blocks, reversed(skips)):
            x = up_conv(x)
            x = torch.cat([x, skip], dim=1)
            x = up_block(x, t_emb)
        
        # 输出
        noise = self.output_conv(x)
        
        # 恢复时间维度
        noise = noise.reshape(B, self.output_length, self.latent_channels, 
                            noise.shape[-2], noise.shape[-1])
        
        return noise


class DiffusionTrainer:
    """扩散模型训练器"""
    
    def __init__(self,
                 model: WeatherDiffusion,
                 vae_wrapper,
                 scheduler,
                 device: str = 'cuda',
                 lr: float = 1e-4):
        """
        Args:
            model: 扩散模型
            vae_wrapper: VAE包装器
            scheduler: 噪声调度器
            device: 设备
            lr: 学习率
        """
        self.model = model.to(device)
        self.vae = vae_wrapper
        self.scheduler = scheduler
        self.device = device
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
    
    def train_step(self, inputs, targets):
        """训练一步"""
        B, T_in, C, H, W = inputs.shape
        T_out = targets.shape[1]
        
        # 编码到潜空间
        with torch.no_grad():
            inputs_flat = inputs.reshape(B * T_in, C, H, W).to(self.device)
            condition = self.vae.encode(inputs_flat)
            condition = condition.reshape(B, T_in, 4, H // 8, W // 8)
            
            targets_flat = targets.reshape(B * T_out, C, H, W).to(self.device)
            latent_target = self.vae.encode(targets_flat)
            latent_target = latent_target.reshape(B, T_out, 4, H // 8, W // 8)
        
        # 采样噪声和时间步
        noise = torch.randn_like(latent_target)
        timesteps = torch.randint(
            0, self.scheduler.num_train_timesteps, (B,),
            device=self.device
        ).long()
        
        # 添加噪声
        noisy_latent = self.scheduler.add_noise(latent_target, noise, timesteps)
        
        # 预测噪声
        self.optimizer.zero_grad()
        predicted_noise = self.model(noisy_latent, timesteps, condition)
        loss = self.criterion(predicted_noise, noise)
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def sample(self, 
               condition: torch.Tensor,
               num_inference_steps: int = 50,
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        采样生成未来帧
        
        Args:
            condition: 条件（过去帧） (B, T_in, C, H, W)
            num_inference_steps: 推理步数
            generator: 随机数生成器
        
        Returns:
            predictions: 预测的未来帧 (B, T_out, C, H, W)
        """
        self.model.eval()
        
        B, T_in, C, H, W = condition.shape
        
        # 编码条件
        condition_flat = condition.reshape(B * T_in, C, H, W).to(self.device)
        latent_condition = self.vae.encode(condition_flat)
        latent_condition = latent_condition.reshape(B, T_in, 4, H // 8, W // 8)
        
        # 从随机噪声开始
        latent_shape = (B, self.model.output_length, 4, H // 8, W // 8)
        latent = torch.randn(latent_shape, generator=generator,
                           device=self.device, dtype=latent_condition.dtype)
        
        # 逐步去噪
        from tqdm import tqdm
        for t in tqdm(range(num_inference_steps - 1, -1, -1), desc='采样'):
            timestep = torch.full((B,), t, device=self.device, dtype=torch.long)
            
            # 预测噪声
            predicted_noise = self.model(latent, timestep, latent_condition)
            
            # 去噪
            latent = self.scheduler.step(predicted_noise, t, latent, generator)
        
        # 解码
        latent_flat = latent.reshape(B * self.model.output_length, 4, H // 8, W // 8)
        predictions_flat = self.vae.decode(latent_flat)
        predictions = predictions_flat.reshape(B, self.model.output_length, C, H, W)
        
        return predictions

