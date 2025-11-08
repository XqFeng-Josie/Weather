"""
轻量级时空Transformer - 专为天气预测设计

设计理念:
1. 空间建模: Patch-based attention (类似ViT)
2. 时间建模: Temporal attention across time steps
3. 轻量化: 参数量与ConvLSTM相当
4. 高效训练: 使用Pre-LN和更好的初始化

架构:
  Input (B, T_in, C, H, W)
    ↓
  Patch Embedding → (B, T_in, N_patches, D)
    ↓
  Positional Encoding (time + space)
    ↓
  Spatial-Temporal Encoder
    ↓
  Decoder (predict future frames)
    ↓
  Output (B, T_out, C, H, W)
"""

import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    """
    将空间网格分割为patches并嵌入
    
    例如: (32, 64) 网格 → (8, 8) patches，每个patch 4x8
    """
    
    def __init__(self, img_size=(32, 64), patch_size=(4, 8), in_channels=1, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches_h = img_size[0] // patch_size[0]
        self.n_patches_w = img_size[1] // patch_size[1]
        self.n_patches = self.n_patches_h * self.n_patches_w
        
        # 使用卷积实现patch embedding（更高效）
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            patches: (B, N_patches, D)
        """
        x = self.proj(x)  # (B, D, n_patches_h, n_patches_w)
        x = x.flatten(2)  # (B, D, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, D)
        return x


class PositionalEncoding(nn.Module):
    """
    时空位置编码
    
    - 时间位置: 正弦编码
    - 空间位置: 可学习编码
    """
    
    def __init__(self, d_model, max_time_len=20, n_spatial_patches=64):
        super().__init__()
        
        # 时间位置编码（正弦编码，更好的外推能力）
        pe_time = torch.zeros(max_time_len, d_model)
        position = torch.arange(0, max_time_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe_time[:, 0::2] = torch.sin(position * div_term)
        pe_time[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe_time', pe_time)
        
        # 空间位置编码（可学习，因为地球网格不规则）
        self.pe_spatial = nn.Parameter(torch.randn(1, n_spatial_patches, d_model) * 0.02)
        
    def forward(self, x, time_indices):
        """
        Args:
            x: (B, T, N, D)
            time_indices: (T,) - 时间步索引
        Returns:
            x: (B, T, N, D) with positional encoding added
        """
        B, T, N, D = x.shape
        
        # 添加时间编码
        time_pe = self.pe_time[time_indices].unsqueeze(0).unsqueeze(2)  # (1, T, 1, D)
        x = x + time_pe
        
        # 添加空间编码
        x = x + self.pe_spatial.unsqueeze(1)  # (1, 1, N, D)
        
        return x


class SpatialTemporalAttention(nn.Module):
    """
    时空注意力模块
    
    策略: Factorized Attention
    1. 先做spatial attention (within each time step)
    2. 再做temporal attention (across time steps)
    
    优势: 计算复杂度 O(T*N^2 + N*T^2) vs O((T*N)^2)
    """
    
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        
        # Spatial attention (patches within same time step)
        self.spatial_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.spatial_norm = nn.LayerNorm(d_model)
        
        # Temporal attention (same patch across time)
        self.temporal_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.temporal_norm = nn.LayerNorm(d_model)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Args:
            x: (B, T, N, D)
        Returns:
            x: (B, T, N, D)
        """
        B, T, N, D = x.shape
        
        # 1. Spatial Attention (within each time step)
        x_spatial = x.reshape(B * T, N, D)
        attn_out, _ = self.spatial_attn(x_spatial, x_spatial, x_spatial)
        x = x + attn_out.reshape(B, T, N, D)
        x = self.spatial_norm(x)
        
        # 2. Temporal Attention (across time steps for each patch)
        x_temporal = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        attn_out, _ = self.temporal_attn(x_temporal, x_temporal, x_temporal)
        x = x + attn_out.reshape(B, N, T, D).permute(0, 2, 1, 3)
        x = self.temporal_norm(x)
        
        # 3. FFN
        x = x + self.ffn(x)
        x = self.ffn_norm(x)
        
        return x


class WeatherTransformer(nn.Module):
    """
    时空Transformer天气预测模型
    
    特点:
    - 轻量级设计 (~2M parameters)
    - Factorized spatial-temporal attention
    - 支持多步预测
    - 适配64x32网格
    """
    
    def __init__(
        self,
        img_size=(32, 64),
        patch_size=(4, 8),
        input_channels=1,
        output_channels=1,
        input_length=12,
        output_length=4,
        d_model=128,
        n_heads=4,
        n_layers=4,
        dropout=0.1,
    ):
        """
        Args:
            img_size: (H, W) 输入图像大小
            patch_size: (pH, pW) patch大小
            input_channels: 输入通道数
            output_channels: 输出通道数
            input_length: 输入时间步数
            output_length: 输出时间步数
            d_model: 模型维度
            n_heads: 注意力头数
            n_layers: Encoder层数
            dropout: Dropout率
        """
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.input_length = input_length
        self.output_length = output_length
        self.d_model = d_model
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, input_channels, d_model
        )
        n_patches = self.patch_embed.n_patches
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            d_model, 
            max_time_len=input_length + output_length,
            n_spatial_patches=n_patches
        )
        
        # Encoder (process input sequence)
        self.encoder_layers = nn.ModuleList([
            SpatialTemporalAttention(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # Learnable output time queries
        self.output_queries = nn.Parameter(
            torch.randn(1, output_length, n_patches, d_model) * 0.02
        )
        
        # Decoder (generate output sequence)
        self.decoder_layers = nn.ModuleList([
            SpatialTemporalAttention(d_model, n_heads, dropout)
            for _ in range(n_layers // 2)  # 更浅的decoder
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, patch_size[0] * patch_size[1] * output_channels)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """更好的初始化策略"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Args:
            x: (B, T_in, C, H, W)
        Returns:
            out: (B, T_out, C, H, W)
        """
        B, T_in, C, H, W = x.shape
        
        # 1. Patch embedding for each time step
        patches = []
        for t in range(T_in):
            patch_t = self.patch_embed(x[:, t])  # (B, N, D)
            patches.append(patch_t)
        x_patches = torch.stack(patches, dim=1)  # (B, T_in, N, D)
        
        # 2. Add positional encoding
        time_indices = torch.arange(T_in, device=x.device)
        x_patches = self.pos_encoding(x_patches, time_indices)
        
        # 3. Encoder (process input)
        for layer in self.encoder_layers:
            x_patches = layer(x_patches)
        
        # 4. Prepare decoder input (learnable queries + context)
        output_queries = self.output_queries.expand(B, -1, -1, -1)  # (B, T_out, N, D)
        
        # Add positional encoding for output time steps
        output_time_indices = torch.arange(
            T_in, T_in + self.output_length, device=x.device
        )
        output_queries = self.pos_encoding(output_queries, output_time_indices)
        
        # 5. Decoder (generate predictions)
        # Concatenate encoder output and decoder queries
        decoder_input = torch.cat([x_patches, output_queries], dim=1)  # (B, T_in+T_out, N, D)
        
        for layer in self.decoder_layers:
            decoder_input = layer(decoder_input)
        
        # Extract only the output part
        output_features = decoder_input[:, T_in:, :, :]  # (B, T_out, N, D)
        
        # 6. Project back to spatial domain
        output_features = self.output_proj(output_features)  # (B, T_out, N, pH*pW*C)
        
        # 7. Reshape to spatial format
        B, T_out, N, _ = output_features.shape
        pH, pW = self.patch_size
        n_patches_h = self.patch_embed.n_patches_h
        n_patches_w = self.patch_embed.n_patches_w
        
        # (B, T_out, N, pH*pW*C) → (B, T_out, n_patches_h, n_patches_w, pH, pW, C)
        output_features = output_features.reshape(
            B, T_out, n_patches_h, n_patches_w, pH, pW, self.output_channels
        )
        
        # Rearrange to (B, T_out, C, H, W)
        output = output_features.permute(0, 1, 6, 2, 4, 3, 5)  # (B, T_out, C, n_patches_h, pH, n_patches_w, pW)
        output = output.reshape(B, T_out, self.output_channels, H, W)
        
        return output


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    """测试模型"""
    print("=" * 80)
    print("Weather Transformer - Architecture Test")
    print("=" * 80)
    
    # 创建模型
    model = WeatherTransformer(
        img_size=(32, 64),
        patch_size=(4, 8),
        input_channels=1,
        output_channels=1,
        input_length=12,
        output_length=4,
        d_model=128,
        n_heads=4,
        n_layers=4,
        dropout=0.1,
    )
    
    print(f"\n模型参数量: {count_parameters(model):,}")
    
    # 测试前向传播
    batch_size = 4
    x = torch.randn(batch_size, 12, 1, 32, 64)
    
    print(f"\n输入形状: {x.shape}")
    
    model.eval()
    with torch.no_grad():
        y = model(x)
    
    print(f"输出形状: {y.shape}")
    
    # 测试梯度
    model.train()
    y = model(x)
    loss = y.mean()
    loss.backward()
    
    print("\n✓ 前向传播和反向传播测试通过")
    
    # 对比其他模型
    print("\n" + "=" * 80)
    print("模型对比")
    print("=" * 80)
    
    from convlstm import ConvLSTM
    
    convlstm = ConvLSTM(
        input_channels=1,
        hidden_channels=64,
        num_layers=2,
        output_length=4,
    )
    
    print(f"\nConvLSTM参数量: {count_parameters(convlstm):,}")
    print(f"WeatherTransformer参数量: {count_parameters(model):,}")
    print(f"参数比例: {count_parameters(model) / count_parameters(convlstm):.2f}x")
    
    print("\n" + "=" * 80)

