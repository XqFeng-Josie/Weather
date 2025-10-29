"""
ConvLSTM模型 - 结合CNN和LSTM处理时空数据
最适合处理具有时间序列和空间结构的天气数据
"""

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    ConvLSTM单元
    
    将LSTM的全连接操作替换为卷积操作，保留空间结构
    """
    
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        
        # 输入门、遗忘门、细胞门、输出门的卷积
        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,  # i, f, g, o
            kernel_size=kernel_size,
            padding=padding,
        )
    
    def forward(self, x, hidden_state):
        """
        Args:
            x: (batch, input_channels, H, W)
            hidden_state: tuple of (h, c)
                h: (batch, hidden_channels, H, W)
                c: (batch, hidden_channels, H, W)
        Returns:
            h_next, c_next
        """
        h, c = hidden_state
        
        # 拼接输入和隐藏状态
        combined = torch.cat([x, h], dim=1)
        
        # 卷积计算门
        gates = self.conv(combined)
        
        # 分割为4个门
        i, f, g, o = torch.split(gates, self.hidden_channels, dim=1)
        
        # 应用激活函数
        i = torch.sigmoid(i)  # 输入门
        f = torch.sigmoid(f)  # 遗忘门
        g = torch.tanh(g)     # 候选值
        o = torch.sigmoid(o)  # 输出门
        
        # 更新细胞状态和隐藏状态
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size, height, width, device='cpu'):
        """初始化隐藏状态"""
        return (
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device),
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device),
        )


class ConvLSTM(nn.Module):
    """
    多层ConvLSTM
    """
    
    def __init__(
        self,
        input_channels,
        hidden_channels,
        kernel_size=3,
        num_layers=1,
        batch_first=True,
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        
        # 创建多层ConvLSTM
        cell_list = []
        for i in range(num_layers):
            cur_input_channels = input_channels if i == 0 else hidden_channels
            cell_list.append(
                ConvLSTMCell(
                    input_channels=cur_input_channels,
                    hidden_channels=hidden_channels,
                    kernel_size=kernel_size,
                )
            )
        self.cell_list = nn.ModuleList(cell_list)
    
    def forward(self, x, hidden_state=None):
        """
        Args:
            x: (batch, time, channels, H, W) if batch_first
            hidden_state: list of tuples for each layer
        Returns:
            layer_output_list: list of outputs for each layer
            last_state_list: list of last (h, c) for each layer
        """
        if not self.batch_first:
            # (time, batch, channels, H, W) -> (batch, time, channels, H, W)
            x = x.permute(1, 0, 2, 3, 4)
        
        batch_size, seq_len, _, height, width = x.size()
        device = x.device
        
        # 初始化隐藏状态
        if hidden_state is None:
            hidden_state = []
            for cell in self.cell_list:
                hidden_state.append(cell.init_hidden(batch_size, height, width, device))
        
        # 前向传播
        layer_output_list = []
        last_state_list = []
        
        cur_layer_input = x
        
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](cur_layer_input[:, t, :, :, :], (h, c))
                output_inner.append(h)
            
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            last_state_list.append((h, c))
        
        return layer_output_list, last_state_list


class ConvLSTMModel(nn.Module):
    """
    基于ConvLSTM的天气预测模型
    
    输入: (batch, input_length, channels, H, W)
    输出: (batch, output_length, channels, H, W)
    """
    
    def __init__(
        self,
        input_channels,
        hidden_channels=64,
        kernel_size=3,
        num_layers=2,
        output_length=4,
    ):
        """
        Args:
            input_channels: 输入变量数
            hidden_channels: ConvLSTM隐藏通道数
            kernel_size: 卷积核大小
            num_layers: ConvLSTM层数
            output_length: 预测时间步数
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.output_length = output_length
        
        # ConvLSTM编码器
        self.convlstm = ConvLSTM(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            num_layers=num_layers,
            batch_first=True,
        )
        
        # 输出投影
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels // 2, input_channels * output_length, kernel_size=1),
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, input_length, channels, H, W)
        Returns:
            (batch, output_length, channels, H, W)
        """
        batch_size = x.shape[0]
        height, width = x.shape[-2:]
        
        # ConvLSTM编码
        layer_output_list, last_state_list = self.convlstm(x)
        
        # 取最后一层的最后一个时间步
        last_output = layer_output_list[-1][:, -1, :, :, :]  # (batch, hidden_channels, H, W)
        
        # 预测未来
        out = self.output_conv(last_output)  # (batch, channels * output_length, H, W)
        
        # 重塑为时间序列: (batch, output_length, channels, H, W)
        out = out.view(batch_size, self.output_length, self.input_channels, height, width)
        
        return out


class ConvLSTMSeq2Seq(nn.Module):
    """
    ConvLSTM Seq2Seq模型 - 更强大的版本
    
    使用编码器-解码器架构
    """
    
    def __init__(
        self,
        input_channels,
        hidden_channels=64,
        kernel_size=3,
        num_layers=2,
        output_length=4,
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.output_length = output_length
        
        # 编码器
        self.encoder = ConvLSTM(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            num_layers=num_layers,
            batch_first=True,
        )
        
        # 解码器
        self.decoder_cell = ConvLSTMCell(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
        )
        
        # 输出投影
        self.output_conv = nn.Conv2d(hidden_channels, input_channels, kernel_size=1)
    
    def forward(self, x):
        """
        Args:
            x: (batch, input_length, channels, H, W)
        Returns:
            (batch, output_length, channels, H, W)
        """
        batch_size = x.shape[0]
        height, width = x.shape[-2:]
        
        # 编码
        _, last_state_list = self.encoder(x)
        h, c = last_state_list[-1]
        
        # 解码 - 自回归预测
        outputs = []
        decoder_input = x[:, -1, :, :, :]  # 使用最后一个输入作为解码器初始输入
        
        for t in range(self.output_length):
            h, c = self.decoder_cell(decoder_input, (h, c))
            out = self.output_conv(h)
            outputs.append(out)
            decoder_input = out  # 使用预测作为下一步输入
        
        # 堆叠输出
        outputs = torch.stack(outputs, dim=1)  # (batch, output_length, channels, H, W)
        
        return outputs


if __name__ == "__main__":
    # 测试ConvLSTM
    print("Testing ConvLSTM...")
    model = ConvLSTMModel(
        input_channels=1,
        hidden_channels=32,
        num_layers=2,
        output_length=4,
    )
    
    # 模拟数据
    x = torch.randn(4, 12, 1, 32, 64)
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nTesting ConvLSTM Seq2Seq...")
    model2 = ConvLSTMSeq2Seq(
        input_channels=1,
        hidden_channels=32,
        num_layers=2,
        output_length=4,
    )
    
    y2 = model2(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y2.shape}")
    print(f"Parameters: {sum(p.numel() for p in model2.parameters()):,}")

