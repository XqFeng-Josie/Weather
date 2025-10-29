"""
LSTM模型 - 处理时间序列（不考虑空间结构）
适用于展平后的特征或单点预测
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    标准LSTM时间序列预测模型
    
    将空间维度展平，仅建模时间依赖关系
    适用于：
    1. 单点预测
    2. 特征已经提取/降维的情况
    """
    
    def __init__(
        self,
        input_size,
        hidden_size=128,
        num_layers=2,
        output_length=4,
        dropout=0.2,
    ):
        """
        Args:
            input_size: 输入特征维度
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            output_length: 预测时间步数
            dropout: Dropout率
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_length = output_length
        self.input_size = input_size
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        
        # 输出层：从LSTM隐藏状态映射到预测
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, input_size * output_length)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, input_length, input_size)
        Returns:
            (batch, output_length, input_size)
        """
        batch_size = x.shape[0]
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        
        # 取最后一个时间步
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # 全连接预测未来
        out = self.fc(last_hidden)  # (batch, input_size * output_length)
        
        # 重塑为 (batch, output_length, input_size)
        out = out.view(batch_size, self.output_length, self.input_size)
        
        return out


class BidirectionalLSTM(nn.Module):
    """
    双向LSTM模型
    
    可以同时利用过去和未来的信息（仅用于编码器）
    """
    
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
        self.input_size = input_size
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True,
        )
        
        # 双向LSTM输出是 2 * hidden_size
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size * output_length)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, input_length, input_size)
        Returns:
            (batch, output_length, input_size)
        """
        batch_size = x.shape[0]
        
        # 双向LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, 2 * hidden_size)
        
        # 取最后一个时间步
        last_hidden = lstm_out[:, -1, :]
        
        # 预测
        out = self.fc(last_hidden)
        out = out.view(batch_size, self.output_length, self.input_size)
        
        return out


class LSTMSeq2Seq(nn.Module):
    """
    LSTM Seq2Seq模型
    
    编码器-解码器架构，更适合序列到序列的预测
    """
    
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
        self.input_size = input_size
        
        # 编码器
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        
        # 解码器
        self.decoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        
        # 输出层
        self.fc = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        """
        Args:
            x: (batch, input_length, input_size)
        Returns:
            (batch, output_length, input_size)
        """
        batch_size = x.shape[0]
        
        # 编码
        _, (hidden, cell) = self.encoder(x)
        
        # 解码 - 自回归
        decoder_input = x[:, -1:, :]  # 使用最后一个输入作为解码器初始输入
        outputs = []
        
        for t in range(self.output_length):
            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            out = self.fc(decoder_output)  # (batch, 1, input_size)
            outputs.append(out)
            decoder_input = out  # 使用预测作为下一步输入
        
        # 拼接
        outputs = torch.cat(outputs, dim=1)  # (batch, output_length, input_size)
        
        return outputs


if __name__ == "__main__":
    # 测试
    print("Testing Standard LSTM...")
    model = LSTMModel(
        input_size=100,
        hidden_size=128,
        num_layers=2,
        output_length=4,
    )
    
    x = torch.randn(8, 12, 100)
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nTesting Bidirectional LSTM...")
    model2 = BidirectionalLSTM(
        input_size=100,
        hidden_size=128,
        num_layers=2,
        output_length=4,
    )
    
    y2 = model2(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y2.shape}")
    print(f"Parameters: {sum(p.numel() for p in model2.parameters()):,}")
    
    print("\nTesting LSTM Seq2Seq...")
    model3 = LSTMSeq2Seq(
        input_size=100,
        hidden_size=128,
        num_layers=2,
        output_length=4,
    )
    
    y3 = model3(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y3.shape}")
    print(f"Parameters: {sum(p.numel() for p in model3.parameters()):,}")

