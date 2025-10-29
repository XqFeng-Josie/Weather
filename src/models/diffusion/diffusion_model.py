"""
Diffusion Weather Prediction Model

将天气预测视为条件生成任务：
- 输入：历史天气场 x_{t-n:t}
- 输出：未来天气场 x_{t+1:t+m}
- 方法：通过扩散模型学习天气场的分布
"""

import torch
import torch.nn as nn
from .unet_simple import SimpleUNet
from .noise_scheduler import DDPMScheduler


class DiffusionWeatherModel(nn.Module):
    """
    Diffusion天气预测模型
    
    核心思想：
    1. 训练阶段：学习去噪函数，从加噪的未来天气场预测噪声
    2. 推理阶段：从随机噪声开始，逐步去噪生成未来天气场
    
    优势：
    - 捕获天气的不确定性（生成多个可能的未来）
    - 更好的空间一致性
    - 适合极端天气事件预测
    """
    
    def __init__(
        self,
        input_channels,
        output_channels,
        input_length=12,
        output_length=4,
        base_channels=64,
        num_timesteps=1000,
        beta_schedule="cosine",
        dropout=0.1,
    ):
        """
        Args:
            input_channels: 输入通道数（历史变量数）
            output_channels: 输出通道数（预测变量数）
            input_length: 输入时间步长
            output_length: 输出时间步长
            base_channels: UNet基础通道数
            num_timesteps: 扩散步数
            beta_schedule: 噪声调度策略
            dropout: Dropout比率
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.input_length = input_length
        self.output_length = output_length
        self.num_timesteps = num_timesteps
        
        # 条件编码器：将历史序列压缩为条件表示
        # (batch, input_length, input_channels, H, W) -> (batch, condition_channels, H, W)
        condition_channels = base_channels
        self.condition_encoder = nn.Sequential(
            # 先在时间维度上做3D卷积
            nn.Conv3d(input_channels, base_channels // 2, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(8, base_channels // 2),
            nn.SiLU(),
            nn.Conv3d(base_channels // 2, base_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            # 在时间维度上平均池化
            nn.AdaptiveAvgPool3d((1, None, None)),  # (batch, base_channels, 1, H, W)
        )
        
        # UNet去噪网络（使用简化版本）
        self.unet = SimpleUNet(
            in_channels=output_channels,
            out_channels=output_channels,
            condition_channels=condition_channels,
            base_channels=base_channels,
        )
        
        # 噪声调度器
        self.scheduler = DDPMScheduler(
            num_timesteps=num_timesteps,
            beta_schedule=beta_schedule,
        )
    
    def encode_condition(self, x_history):
        """
        编码历史信息为条件表示
        
        Args:
            x_history: (batch, input_length, input_channels, H, W)
        
        Returns:
            condition: (batch, condition_channels, H, W)
        """
        # Transpose to (batch, channels, time, H, W) for 3D conv
        x = x_history.transpose(1, 2)
        
        # Encode
        cond = self.condition_encoder(x)
        
        # Remove time dimension: (batch, channels, 1, H, W) -> (batch, channels, H, W)
        cond = cond.squeeze(2)
        
        return cond
    
    def forward(self, x_history, x_future=None, num_inference_steps=50):
        """
        前向传播
        
        训练模式（x_future不为None）：
            返回预测的噪声和目标噪声，用于计算loss
        
        推理模式（x_future为None）：
            从随机噪声开始，逐步去噪生成预测
        
        Args:
            x_history: 历史数据 (batch, input_length, input_channels, H, W)
            x_future: 未来数据 (batch, output_length, output_channels, H, W) [训练时]
            num_inference_steps: 推理步数 [推理时]
        
        Returns:
            训练模式: (noise_pred, noise_target, x_noisy)
            推理模式: (predictions,) - (batch, output_length, output_channels, H, W)
        """
        if self.training and x_future is not None:
            return self._forward_train(x_history, x_future)
        else:
            return self._forward_inference(x_history, num_inference_steps)
    
    def _forward_train(self, x_history, x_future):
        """
        训练前向传播
        
        对于每个未来时间步，独立训练去噪模型
        """
        batch_size = x_history.shape[0]
        device = x_history.device
        
        # 编码条件
        condition = self.encode_condition(x_history)
        
        # 随机选择一个输出时间步
        # 注意：这里我们简化为每次只预测一个时间步
        # 实际应用中可以扩展为自回归或并行预测多步
        time_idx = torch.randint(0, self.output_length, (batch_size,), device=device)
        
        # 提取对应的目标数据
        x_target = torch.stack([x_future[i, time_idx[i]] for i in range(batch_size)])
        # x_target: (batch, output_channels, H, W)
        
        # 随机选择扩散时间步
        diffusion_timesteps = torch.randint(
            0, self.num_timesteps, (batch_size,), device=device, dtype=torch.long
        )
        
        # 生成噪声
        noise = torch.randn_like(x_target)
        
        # 添加噪声
        x_noisy = self.scheduler.add_noise(x_target, noise, diffusion_timesteps)
        
        # 预测噪声
        noise_pred = self.unet(x_noisy, diffusion_timesteps, condition)
        
        return noise_pred, noise, x_noisy
    
    def _forward_inference(self, x_history, num_inference_steps=50):
        """
        推理前向传播
        
        逐步去噪生成预测
        """
        batch_size = x_history.shape[0]
        device = x_history.device
        H, W = x_history.shape[-2:]
        
        # 编码条件
        condition = self.encode_condition(x_history)
        
        # 采样步长（可以比训练步数少，加快推理）
        step_size = self.num_timesteps // num_inference_steps
        timesteps = list(range(self.num_timesteps - 1, 0, -step_size))
        
        # 生成所有输出时间步的预测
        predictions = []
        
        for t_out in range(self.output_length):
            # 从随机噪声开始
            x_t = torch.randn(
                batch_size, self.output_channels, H, W,
                device=device
            )
            
            # 逐步去噪
            for t in timesteps:
                t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
                
                # 预测噪声
                with torch.no_grad():
                    noise_pred = self.unet(x_t, t_batch, condition)
                
                # 去噪一步
                x_t = self.scheduler.sample_prev_timestep(x_t, noise_pred, t)
            
            predictions.append(x_t)
        
        # Stack: (batch, output_length, output_channels, H, W)
        predictions = torch.stack(predictions, dim=1)
        
        return predictions
    
    def generate_ensemble(self, x_history, num_samples=10, num_inference_steps=50):
        """
        生成集合预测（利用diffusion的随机性）
        
        Args:
            x_history: 历史数据
            num_samples: 集合成员数
            num_inference_steps: 每个成员的推理步数
        
        Returns:
            ensemble: (num_samples, batch, output_length, output_channels, H, W)
        """
        self.eval()
        ensemble = []
        
        for _ in range(num_samples):
            pred = self._forward_inference(x_history, num_inference_steps)
            ensemble.append(pred)
        
        return torch.stack(ensemble, dim=0)

