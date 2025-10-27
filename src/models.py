"""
天气预测模型集合
从简单到复杂：LR -> LSTM -> CNN -> Transformer
"""
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import numpy as np


# ============================================================================
# 1. Linear Regression Baseline
# ============================================================================
class LinearRegressionModel:
    """线性回归baseline"""
    
    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def fit(self, X, y):
        """
        Args:
            X: (n_samples, input_length, n_features)
            y: (n_samples, output_length, n_features)
        """
        # 展平为2D: (n_samples, input_length * n_features)
        X_flat = X.reshape(X.shape[0], -1)
        y_flat = y.reshape(y.shape[0], -1)
        
        # 标准化
        X_scaled = self.scaler_X.fit_transform(X_flat)
        y_scaled = self.scaler_y.fit_transform(y_flat)
        
        # 训练
        print(f"Training Linear Regression on {X_scaled.shape}")
        self.model.fit(X_scaled, y_scaled)
        
        return self
    
    def predict(self, X):
        """预测"""
        X_flat = X.reshape(X.shape[0], -1)
        X_scaled = self.scaler_X.transform(X_flat)
        y_pred_scaled = self.model.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        # 还原形状
        return y_pred.reshape(X.shape[0], -1, X.shape[2])
    
    def score(self, X, y):
        """R2 score"""
        X_flat = X.reshape(X.shape[0], -1)
        y_flat = y.reshape(y.shape[0], -1)
        X_scaled = self.scaler_X.transform(X_flat)
        y_scaled = self.scaler_y.transform(y_flat)
        return self.model.score(X_scaled, y_scaled)


# ============================================================================
# 2. LSTM Model
# ============================================================================
class LSTMModel(nn.Module):
    """LSTM时间序列预测模型"""
    
    def __init__(
        self,
        input_size,
        hidden_size=128,
        num_layers=2,
        output_length=4,
        dropout=0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_length = output_length
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        
        self.fc = nn.Linear(hidden_size, input_size * output_length)
        
    def forward(self, x):
        """
        Args:
            x: (batch, input_length, input_size)
        Returns:
            (batch, output_length, input_size)
        """
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        
        # 取最后一个时间步
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # 全连接预测未来
        out = self.fc(last_hidden)  # (batch, input_size * output_length)
        
        # 重塑为 (batch, output_length, input_size)
        out = out.view(-1, self.output_length, x.shape[2])
        
        return out


# ============================================================================
# 3. CNN-LSTM Hybrid
# ============================================================================
class CNNLSTMModel(nn.Module):
    """CNN提取空间特征 + LSTM提取时间特征"""
    
    def __init__(
        self,
        input_channels,
        hidden_size=128,
        num_lstm_layers=2,
        output_length=4,
    ):
        super().__init__()
        self.output_length = output_length
        
        # CNN for spatial features (假设输入是2D空间数据)
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # 固定输出大小
        
        # LSTM for temporal features
        self.lstm = nn.LSTM(
            input_size=128 * 4 * 4,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, input_channels * 4 * 4 * output_length)
        
    def forward(self, x):
        """
        Args:
            x: (batch, time, channels, H, W)
        Returns:
            (batch, output_length, channels, H, W)
        """
        batch_size, time_steps = x.shape[0], x.shape[1]
        
        # CNN处理每个时间步
        cnn_features = []
        for t in range(time_steps):
            xt = x[:, t]  # (batch, channels, H, W)
            xt = torch.relu(self.conv1(xt))
            xt = torch.relu(self.conv2(xt))
            xt = self.pool(xt)  # (batch, 128, 4, 4)
            xt = xt.view(batch_size, -1)  # (batch, 128*4*4)
            cnn_features.append(xt)
        
        # Stack为时间序列
        cnn_features = torch.stack(cnn_features, dim=1)  # (batch, time, 2048)
        
        # LSTM
        lstm_out, _ = self.lstm(cnn_features)
        last_hidden = lstm_out[:, -1, :]
        
        # 预测
        out = self.fc(last_hidden)
        out = out.view(batch_size, self.output_length, -1, 4, 4)
        
        return out


# ============================================================================
# 4. Transformer Model
# ============================================================================
class TransformerModel(nn.Module):
    """Transformer序列预测模型"""
    
    def __init__(
        self,
        input_size,
        d_model=256,
        nhead=8,
        num_layers=4,
        output_length=4,
        dropout=0.1,
    ):
        super().__init__()
        self.output_length = output_length
        self.d_model = d_model
        
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
        
        # Output projection
        self.output_projection = nn.Linear(d_model, input_size * output_length)
        
    def forward(self, x):
        """
        Args:
            x: (batch, input_length, input_size)
        Returns:
            (batch, output_length, input_size)
        """
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
        out = out.view(-1, self.output_length, x.shape[1] // self.output_length)
        
        return out


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
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================================
# 5. U-Net for spatial prediction
# ============================================================================
class UNet2D(nn.Module):
    """U-Net用于空间天气场预测"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        
        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)
        
        # Decoder
        self.dec3 = self.conv_block(512 + 256, 256)
        self.dec2 = self.conv_block(256 + 128, 128)
        self.dec1 = self.conv_block(128 + 64, 64)
        
        # Output
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e3))
        
        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.upsample(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1))
        
        return self.out(d1)


# ============================================================================
# Helper functions
# ============================================================================
def get_model(model_name: str, **kwargs):
    """模型工厂函数"""
    models = {
        'lr': LinearRegressionModel,
        'lstm': LSTMModel,
        'cnn_lstm': CNNLSTMModel,
        'transformer': TransformerModel,
        'unet': UNet2D,
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name.lower()](**kwargs)


def count_parameters(model):
    """统计模型参数量"""
    if isinstance(model, LinearRegressionModel):
        return model.model.coef_.size
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # 测试模型
    batch_size = 32
    input_length = 12
    output_length = 4
    n_features = 10
    
    # 测试LSTM
    print("Testing LSTM...")
    lstm = LSTMModel(input_size=n_features, output_length=output_length)
    x = torch.randn(batch_size, input_length, n_features)
    y_pred = lstm(x)
    print(f"  Input: {x.shape}, Output: {y_pred.shape}")
    print(f"  Parameters: {count_parameters(lstm):,}")
    
    # 测试Transformer
    print("\nTesting Transformer...")
    transformer = TransformerModel(input_size=n_features, output_length=output_length)
    y_pred = transformer(x)
    print(f"  Input: {x.shape}, Output: {y_pred.shape}")
    print(f"  Parameters: {count_parameters(transformer):,}")

