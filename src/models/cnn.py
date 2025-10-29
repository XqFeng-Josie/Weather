"""
CNN模型 - 处理空间结构的天气场预测
适用于网格数据，能够学习空间模式
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    简单CNN模型用于天气场预测
    
    输入: (batch, input_length, channels, H, W)
    输出: (batch, output_length, channels, H, W)
    
    设计思路：
    1. 时间维度展平，将多个时间步当作多通道处理
    2. CNN提取空间特征
    3. 预测未来多个时间步
    """
    
    def __init__(
        self,
        input_channels,
        input_length=12,
        output_length=4,
        hidden_channels=64,
    ):
        """
        Args:
            input_channels: 变量数（如温度、气压等）
            input_length: 输入时间步数
            output_length: 预测时间步数
            hidden_channels: 隐藏层通道数
        """
        super().__init__()
        self.input_channels = input_channels
        self.input_length = input_length
        self.output_length = output_length
        
        # 输入是 input_length * input_channels 个通道
        in_ch = input_length * input_channels
        out_ch = output_length * input_channels
        
        self.encoder = nn.Sequential(
            # 64x32 -> 64x32
            nn.Conv2d(in_ch, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            
            # 64x32 -> 64x32
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(),
            
            # 64x32 -> 32x16
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(),
            
            # 32x16 -> 32x16
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            # 32x16 -> 64x32
            nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(),
            
            # 64x32 -> 64x32
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            
            # 64x32 -> 64x32, 输出通道
            nn.Conv2d(hidden_channels, out_ch, kernel_size=3, padding=1),
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, input_length, channels, H, W)
        Returns:
            (batch, output_length, channels, H, W)
        """
        batch_size = x.shape[0]
        
        # 将时间和通道维度合并: (batch, input_length * channels, H, W)
        x = x.view(batch_size, -1, x.shape[-2], x.shape[-1])
        
        # CNN编码解码
        features = self.encoder(x)
        out = self.decoder(features)
        
        # 重塑回时间维度: (batch, output_length, channels, H, W)
        out = out.view(batch_size, self.output_length, self.input_channels, out.shape[-2], out.shape[-1])
        
        return out


class ResidualBlock(nn.Module):
    """残差块用于更深的CNN"""
    
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class DeepCNN(nn.Module):
    """
    更深的CNN模型，带残差连接
    适用于更复杂的空间模式学习
    """
    
    def __init__(
        self,
        input_channels,
        input_length=12,
        output_length=4,
        hidden_channels=64,
        n_residual_blocks=3,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.input_length = input_length
        self.output_length = output_length
        
        in_ch = input_length * input_channels
        out_ch = output_length * input_channels
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_ch, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
        )
        
        # 残差块
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_channels) for _ in range(n_residual_blocks)]
        )
        
        # 输出投影
        self.output_proj = nn.Conv2d(hidden_channels, out_ch, kernel_size=3, padding=1)
    
    def forward(self, x):
        """
        Args:
            x: (batch, input_length, channels, H, W)
        Returns:
            (batch, output_length, channels, H, W)
        """
        batch_size = x.shape[0]
        
        # 合并时间维度
        x = x.view(batch_size, -1, x.shape[-2], x.shape[-1])
        
        # 通过网络
        x = self.input_proj(x)
        x = self.residual_blocks(x)
        out = self.output_proj(x)
        
        # 重塑
        out = out.view(batch_size, self.output_length, self.input_channels, out.shape[-2], out.shape[-1])
        
        return out


if __name__ == "__main__":
    # 测试
    print("Testing SimpleCNN...")
    model = SimpleCNN(
        input_channels=1,
        input_length=12,
        output_length=4,
        hidden_channels=32,
    )
    
    # 模拟数据: (batch=8, time=12, channels=1, H=32, W=64)
    x = torch.randn(8, 12, 1, 32, 64)
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nTesting DeepCNN...")
    model2 = DeepCNN(
        input_channels=1,
        input_length=12,
        output_length=4,
        hidden_channels=32,
        n_residual_blocks=3,
    )
    
    y2 = model2(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y2.shape}")
    print(f"Parameters: {sum(p.numel() for p in model2.parameters()):,}")

