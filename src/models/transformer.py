"""
Transformer模型 - 使用注意力机制处理序列
适用于长序列建模，但不包含空间归纳偏置
"""

import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    标准Transformer序列预测模型

    将空间维度展平，使用注意力机制建模时间依赖
    优化：减少参数以防止单变量预测时过拟合退化
    """

    def __init__(
        self,
        input_size,
        d_model=128,  # 减小默认值：256->128
        nhead=4,  # 减小默认值：8->4
        num_layers=3,  # 减小默认值：4->3
        output_length=4,
        dropout=0.2,  # 增加dropout：0.1->0.2
    ):
        """
        Args:
            input_size: 输入特征维度
            d_model: Transformer嵌入维度
            nhead: 注意力头数
            num_layers: Transformer层数
            output_length: 预测时间步数
            dropout: Dropout率
        """
        super().__init__()
        self.output_length = output_length
        self.d_model = d_model
        self.input_size = input_size

        # Input embedding
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output projection with dropout
        self.output_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, input_size * output_length),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, input_length, input_size)
        Returns:
            (batch, output_length, input_size)
        """
        batch_size = x.shape[0]

        # Project to d_model
        x = self.input_projection(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer
        x = self.transformer(x)  # (batch, seq_len, d_model)

        # Take last hidden state
        x = x[:, -1, :]  # (batch, d_model)

        # Project to output
        out = self.output_projection(x)  # (batch, input_size * output_length)
        out = out.view(batch_size, self.output_length, self.input_size)

        return out


class TransformerSeq2Seq(nn.Module):
    """
    Transformer Seq2Seq模型

    使用编码器-解码器架构，更适合序列到序列任务
    """

    def __init__(
        self,
        input_size,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        output_length=4,
        dropout=0.1,
    ):
        super().__init__()
        self.output_length = output_length
        self.d_model = d_model
        self.input_size = input_size

        # Embeddings
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )

        # Output
        self.output_projection = nn.Linear(d_model, input_size)

    def forward(self, x):
        """
        Args:
            x: (batch, input_length, input_size)
        Returns:
            (batch, output_length, input_size)
        """
        batch_size = x.shape[0]
        device = x.device

        # Encode input
        src = self.input_projection(x)
        src = self.pos_encoder(src)

        # Create decoder input (learnable query tokens)
        tgt = torch.zeros(batch_size, self.output_length, self.d_model, device=device)
        tgt = self.pos_encoder(tgt)

        # Transformer
        out = self.transformer(src, tgt)  # (batch, output_length, d_model)

        # Project to output
        out = self.output_projection(out)  # (batch, output_length, input_size)

        return out


class SpatialTransformer(nn.Module):
    """
    空间Transformer - 尝试处理空间信息

    将每个空间位置当作一个token，使用Transformer建模
    注意：这对于大的空间网格可能计算量很大
    """

    def __init__(
        self,
        input_channels,
        d_model=128,
        nhead=4,
        num_layers=3,
        spatial_size=(32, 64),
        input_length=12,
        output_length=4,
        dropout=0.1,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.d_model = d_model
        self.spatial_size = spatial_size
        self.input_length = input_length
        self.output_length = output_length
        self.n_spatial_tokens = spatial_size[0] * spatial_size[1]

        # 输入投影: 将每个时间步的空间patch映射到token
        self.input_projection = nn.Linear(input_channels * input_length, d_model)

        # 可学习的空间位置编码
        self.spatial_pos_embedding = nn.Parameter(
            torch.randn(1, self.n_spatial_tokens, d_model)
        )

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 输出投影
        self.output_projection = nn.Linear(d_model, input_channels * output_length)

    def forward(self, x):
        """
        Args:
            x: (batch, input_length, channels, H, W)
        Returns:
            (batch, output_length, channels, H, W)
        """
        batch_size = x.shape[0]
        H, W = self.spatial_size

        # 将时间维度合并，将空间展平为token
        # (batch, input_length, channels, H, W) -> (batch, H*W, input_length*channels)
        x = x.permute(0, 3, 4, 1, 2)  # (batch, H, W, input_length, channels)
        x = x.reshape(batch_size, H * W, -1)  # (batch, H*W, input_length*channels)

        # 投影到token空间
        x = self.input_projection(x)  # (batch, H*W, d_model)

        # 添加空间位置编码
        x = x + self.spatial_pos_embedding

        # Transformer处理
        x = self.transformer(x)  # (batch, H*W, d_model)

        # 输出投影
        x = self.output_projection(x)  # (batch, H*W, channels*output_length)

        # 重塑回空间形状
        x = x.view(batch_size, H, W, self.output_length, self.input_channels)
        x = x.permute(0, 3, 4, 1, 2)  # (batch, output_length, channels, H, W)

        return x


if __name__ == "__main__":
    # 测试标准Transformer
    print("Testing Standard Transformer...")
    model = TransformerModel(
        input_size=100,
        d_model=256,
        nhead=8,
        num_layers=4,
        output_length=4,
    )

    x = torch.randn(8, 12, 100)
    y = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nTesting Transformer Seq2Seq...")
    model2 = TransformerSeq2Seq(
        input_size=100,
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        output_length=4,
    )

    y2 = model2(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y2.shape}")
    print(f"Parameters: {sum(p.numel() for p in model2.parameters()):,}")

    print("\nTesting Spatial Transformer...")
    model3 = SpatialTransformer(
        input_channels=1,
        d_model=128,
        nhead=4,
        num_layers=3,
        spatial_size=(32, 64),
        input_length=12,
        output_length=4,
    )

    x_spatial = torch.randn(4, 12, 1, 32, 64)
    y3 = model3(x_spatial)
    print(f"Input shape: {x_spatial.shape}")
    print(f"Output shape: {y3.shape}")
    print(f"Parameters: {sum(p.numel() for p in model3.parameters()):,}")
