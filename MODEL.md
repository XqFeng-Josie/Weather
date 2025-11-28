# 模型架构详解

本文档详细介绍项目中所有模型的架构、输入输出、工作原理及其与天气预测任务的关系。

## 📋 目录

1. [任务概述](#任务概述)
2. [传统深度学习模型](#传统深度学习模型)
   - [Linear Regression](#linear-regression)
   - [LSTM](#lstm)
   - [Transformer (序列建模)](#transformer-序列建模)
   - [CNN](#cnn)
   - [ConvLSTM](#convlstm)
   - [Weather Transformer](#weather-transformer)
3. [WeatherDiff 模块](#weatherdiff-模块)
   - [VAE (Stable Diffusion VAE)](#stable-diffusion-vae)
   - [RAE (Representation Autoencoder)](#rae-representation-autoencoder)
   - [Pixel U-Net](#pixel-u-net)
   - [Latent U-Net](#latent-u-net)
   - [Diffusion Model](#diffusion-model)
4. [模型对比与选择](#模型对比与选择)

---

## 任务概述

### 预测任务定义

**目标**: 基于过去12个时间步（3天，每6小时一个时间步）的天气数据，预测未来4个时间步（1天）的天气状况。

**数据格式**:
- **输入**: `(batch, input_length=12, channels, H=32, W=64)`
  - `input_length=12`: 过去3天的数据（12个6小时间隔）
  - `channels`: 气象变量数（如温度、气压等）
  - `H=32, W=64`: 空间分辨率（纬度×经度，等角网格）
  
- **输出**: `(batch, output_length=4, channels, H=32, W=64)`
  - `output_length=4`: 未来1天的预测（4个6小时间隔）

**数据归一化**:
- **传统模型**: Z-score归一化（均值0，标准差1）
- **WeatherDiff模块**: MinMax归一化到[-1, 1]（适配Stable Diffusion VAE）

### 核心挑战

1. **时空依赖**: 天气系统具有复杂的时空相关性
2. **多尺度特征**: 从局部对流到全球环流模式
3. **不确定性**: 天气系统具有内在的混沌特性
4. **计算效率**: 全球64×32网格，需要高效处理

---

## 传统深度学习模型

### Linear Regression

#### 模型结构

```python
# 单变量版本 (lr)
X: (batch, 12, features) → 展平 → (batch, 12*features)
y: (batch, 4, features) → 展平 → (batch, 4*features)
Ridge(alpha=1.0).fit(X_flat, y_flat)

# 多变量版本 (lr_multi)
# 每个变量独立训练一个Ridge模型
for var in variables:
    model[var] = Ridge(alpha=10.0)
```

#### 输入输出

- **输入**: `(batch, 12, features)` - 展平后为 `(batch, 12*features)`
- **输出**: `(batch, 4, features)` - 展平后为 `(batch, 4*features)`
- **参数**: L2正则化系数 `alpha`

#### 工作原理

- **线性映射**: 直接学习从输入到输出的线性变换
- **无时序建模**: 将时间序列展平，丢失时间顺序信息
- **无空间建模**: 将空间网格展平，丢失空间结构

#### 与任务的关系

- **优势**: 
  - 训练极快，适合快速基线测试
  - 参数量少，不易过拟合
- **局限**: 
  - 无法建模非线性时空依赖
  - 预测精度低，仅作为baseline

#### 适用场景

- ✅ 快速验证数据流
- ✅ 单变量简单预测
- ❌ 不适合实际应用

#### 扩展：Multi-Output Linear Regression

`src/models/linear_regression.py` 中还实现了 `MultiOutputLinearRegression`：

**模型结构**:
```python
# 每个变量独立训练一个Ridge模型
for var in variables:
    X_flat = (n_samples, input_length * total_features)  # 所有变量的特征
    y_var = (n_samples, output_length * grid_points_per_var)  # 仅该变量的网格点
    model[var] = Ridge(alpha=10.0).fit(X_scaled, y_scaled)
```

**特点**:
- **每个变量一个模型**: 例如2m_temperature一个模型，geopotential一个模型
- **共享输入特征**: 所有变量使用相同的输入特征（包含所有变量的历史数据）
- **独立预测**: 每个模型只预测对应变量的所有网格点
- **正则化**: 使用更大的alpha（10.0 vs 1.0）防止过拟合

**与标准LR的区别**:
- 标准LR: 所有变量共享一个模型，预测所有变量的所有网格点
- 多输出LR: 每个变量独立模型，但输入包含所有变量的信息

**适用场景**:
- ✅ 多变量预测（每个变量有独立的物理特性）
- ✅ 需要变量特定正则化的场景

---

### LSTM

#### 模型结构

```
输入: (batch, 12, features)
  ↓
LSTM层 (hidden_size=128, num_layers=2)
  ↓
取最后时间步的隐藏状态: (batch, hidden_size)
  ↓
全连接层: (batch, hidden_size) → (batch, 4*features)
  ↓
重塑: (batch, 4, features)
```

#### 输入输出

- **输入**: `(batch, input_length=12, input_size)`
  - `input_size`: 展平后的特征数（`channels * H * W`）
- **输出**: `(batch, output_length=4, input_size)`
- **关键参数**:
  - `hidden_size`: LSTM隐藏层维度（默认128）
  - `num_layers`: LSTM层数（默认2）
  - `dropout`: Dropout率（默认0.2）

#### 工作原理

1. **时间建模**: LSTM通过门控机制（遗忘门、输入门、输出门）建模时间依赖
2. **记忆机制**: 细胞状态（cell state）存储长期记忆
3. **序列处理**: 逐时间步处理，保留时间顺序信息
4. **空间信息丢失**: 输入时已将空间维度展平，无法利用空间结构

#### 与任务的关系

- **优势**:
  - 有效建模时间依赖关系
  - 适合单点时间序列预测
- **局限**:
  - 丢失空间结构（展平操作）
  - 无法捕获空间相关性（如相邻网格的相似性）
  - 参数量大（全连接层）

#### 适用场景

- ✅ 单点预测（如单个气象站）
- ✅ 特征已提取/降维的情况
- ❌ 不适合需要空间信息的网格预测

#### 扩展：其他LSTM变体

`src/models/lstm.py` 中还实现了其他LSTM变体：

1. **BidirectionalLSTM**:
   - 双向LSTM，可以同时利用过去和未来的信息
   - 输出维度为 `2 * hidden_size`
   - 适合编码器场景，但预测任务中无法使用未来信息

2. **LSTMSeq2Seq**:
   - 编码器-解码器架构
   - 编码器处理输入序列，解码器自回归生成输出
   - 更适合序列到序列的预测任务

---

### Transformer (序列建模)

#### 模型结构

标准Transformer模型（`TransformerModel`）：

```
输入: (batch, 12, features)
  ↓
输入投影: Linear(features, d_model=128)
  ↓
位置编码: PositionalEncoding (正弦编码)
  ↓
Transformer编码器 (3层):
  MultiheadAttention (4 heads) → LayerNorm → FFN → LayerNorm
  ↓
取最后时间步: (batch, d_model)
  ↓
输出投影:
  LayerNorm → Dropout → Linear(d_model, d_model//2) → ReLU → 
  Dropout → Linear(d_model//2, features*4)
  ↓
重塑: (batch, 4, features)
```

#### 输入输出

- **输入**: `(batch, input_length=12, input_size)`
  - `input_size`: 展平后的特征数（`channels * H * W`）
- **输出**: `(batch, output_length=4, input_size)`
- **关键参数**:
  - `d_model`: Transformer嵌入维度（默认128，已优化减小）
  - `nhead`: 注意力头数（默认4，已优化减小）
  - `num_layers`: Transformer层数（默认3，已优化减小）
  - `dropout`: Dropout率（默认0.2，已优化增加）

#### 工作原理

1. **注意力机制**: 使用多头自注意力捕获序列内的依赖关系
2. **位置编码**: 正弦位置编码提供时间顺序信息
3. **序列建模**: Transformer编码器处理整个输入序列
4. **空间信息丢失**: 输入时已将空间维度展平，无法利用空间结构
5. **优化设计**: 针对单变量预测场景，减小了参数量以防止过拟合

#### 与任务的关系

- **优势**:
  - 有效建模长距离时间依赖（注意力机制）
  - 并行计算，训练效率高
  - 参数量已优化，适合单变量预测
- **局限**:
  - ❌ **丢失空间结构**（展平操作）
  - ❌ 无法捕获空间相关性
  - ❌ 计算复杂度O(n²)，序列长度受限

#### 适用场景

- ✅ 单点时间序列预测
- ✅ 特征已提取/降维的情况
- ✅ 需要捕获长距离时间依赖的任务
- ❌ 不适合需要空间信息的网格预测

#### 扩展：其他Transformer变体

`src/models/transformer.py` 中还实现了其他Transformer变体：

1. **TransformerSeq2Seq**:
   - 编码器-解码器架构
   - 编码器处理输入序列，解码器使用可学习查询token生成输出
   - 更适合序列到序列的预测任务
   - 参数：`d_model=256, nhead=8, num_encoder_layers=4, num_decoder_layers=4`

2. **SpatialTransformer**:
   - 将每个空间位置当作一个token
   - 使用Transformer建模空间位置之间的关系
   - 注意：对于大的空间网格（如32×64=2048个token）计算量很大
   - 当前实现中可能未使用

---

### CNN

#### 模型结构

```
输入: (batch, 12, channels, H, W)
  ↓
展平时间维度: (batch, 12*channels, H, W)
  ↓
编码器 (Encoder):
  Conv2d(12*channels, 64, k=3, p=1) → BN → ReLU
  Conv2d(64, 128, k=3, p=1) → BN → ReLU
  Conv2d(128, 128, k=3, s=2, p=1) → BN → ReLU  # 下采样: 64×32 → 32×16
  Conv2d(128, 256, k=3, p=1) → BN → ReLU
  ↓
解码器 (Decoder):
  ConvTranspose2d(256, 128, k=4, s=2, p=1) → BN → ReLU  # 上采样: 32×16 → 64×32
  Conv2d(128, 64, k=3, p=1) → BN → ReLU
  Conv2d(64, 4*channels, k=3, p=1)  # 输出
  ↓
重塑: (batch, 4, channels, H, W)
```

#### 输入输出

- **输入**: `(batch, input_length=12, channels, H=32, W=64)`
- **输出**: `(batch, output_length=4, channels, H=32, W=64)`
- **关键参数**:
  - `hidden_channels`: 隐藏层通道数（默认64）
  - `input_channels`: 输入变量数

#### 工作原理

1. **空间特征提取**: 卷积操作捕获局部空间模式（如温度梯度、气压系统）
2. **多尺度表示**: 通过下采样和上采样学习不同尺度的特征
3. **时间信息处理**: 将多个时间步作为多通道输入，但**不显式建模时间依赖**
4. **空间结构保留**: 保持空间维度，利用相邻网格的相关性

#### 与任务的关系

- **优势**:
  - 有效提取空间特征（如锋面、气旋）
  - 计算效率高（卷积操作）
  - 参数量相对较少
- **局限**:
  - **无时序建模**: 将时间步当作通道，无法学习时间演化规律
  - 无法捕获长期时间依赖

#### 适用场景

- ✅ 空间模式预测（如静态天气图）
- ✅ 计算资源有限的情况
- ❌ 不适合需要时间依赖的预测任务

#### 扩展：DeepCNN

`src/models/cnn.py` 中还实现了 `DeepCNN`：

- **残差连接**: 使用ResidualBlock构建更深的网络
- **结构**: 输入投影 → 多个残差块 → 输出投影
- **优势**: 更深的网络可以学习更复杂的空间模式
- **参数**: `n_residual_blocks` 控制残差块数量（默认3）

---

### ConvLSTM

#### 模型结构

```
输入: (batch, 12, channels, H, W)
  ↓
ConvLSTM编码器 (多层):
  ConvLSTMCell: 卷积LSTM单元
    - 输入门 (i): 控制新信息流入
    - 遗忘门 (f): 控制旧信息遗忘
    - 细胞门 (g): 候选值
    - 输出门 (o): 控制输出
    - 所有操作都是卷积而非全连接
  ↓
取最后一层的最后时间步: (batch, hidden_channels, H, W)
  ↓
输出投影:
  Conv2d(hidden_channels, hidden_channels//2)
  Conv2d(hidden_channels//2, 4*channels)
  ↓
重塑: (batch, 4, channels, H, W)
```

#### ConvLSTM单元详解

```python
# ConvLSTMCell 核心操作
combined = concat([input, hidden_state])  # (B, C_in+C_hid, H, W)
gates = Conv2d(combined) → (B, 4*C_hid, H, W)  # i, f, g, o
i, f, g, o = split(gates)

cell_next = f * cell + i * g  # 更新细胞状态
hidden_next = o * tanh(cell_next)  # 更新隐藏状态
```

#### 输入输出

- **输入**: `(batch, input_length=12, input_channels, H=32, W=64)`
- **输出**: `(batch, output_length=4, input_channels, H=32, W=64)`
- **关键参数**:
  - `hidden_channels`: ConvLSTM隐藏通道数（默认64）
  - `num_layers`: ConvLSTM层数（默认2）
  - `kernel_size`: 卷积核大小（默认3）

#### 工作原理

1. **时空联合建模**: 
   - **时间维度**: LSTM的门控机制建模时间依赖
   - **空间维度**: 卷积操作保留空间结构
   
2. **记忆机制**: 
   - 细胞状态存储长期时空记忆
   - 隐藏状态编码当前时空特征

3. **多尺度特征**: 
   - 多层ConvLSTM学习不同抽象层次的特征
   - 底层捕获局部模式，高层捕获全局模式

4. **空间相关性**: 
   - 卷积操作自动捕获相邻网格的相关性
   - 无需手动设计空间依赖

#### 与任务的关系

- **优势**:
  - ✅ **同时建模时空依赖**，最适合天气预测任务
  - ✅ 保留空间结构，充分利用网格数据
  - ✅ 记忆机制适合捕获天气系统的演化规律
  - ✅ 在确定性模型中表现最佳

- **局限**:
  - 参数量较大
  - 训练时间较长
  - 无法量化不确定性

#### 适用场景

- ✅ **通用天气预测**（推荐）
- ✅ 需要时空建模的任务
- ✅ 确定性预测场景

#### 扩展：ConvLSTM Seq2Seq

`src/models/convlstm.py` 中还实现了 `ConvLSTMSeq2Seq`：

**模型结构**:
```
输入: (batch, 12, channels, H, W)
  ↓
编码器 (ConvLSTM多层):
  处理输入序列，提取历史时空特征
  ↓
取最后一层的最后状态: (h, c)
  ↓
解码器 (自回归):
  for t in range(output_length):
    decoder_input = 最后一帧 (或上一步预测)
    h, c = ConvLSTMCell(decoder_input, (h, c))
    out = Conv2d(h) → (batch, channels, H, W)
    使用out作为下一步的decoder_input
  ↓
输出: (batch, 4, channels, H, W)
```

**特点**:
- 编码器-解码器架构，更适合序列到序列任务
- 自回归生成：使用预测作为下一步输入（自由运行模式）
- 可选教师强制：训练时可以使用真实值作为下一步输入
- 适合需要更长输出序列或希望在解码阶段注入外部条件的场景

---

### Weather Transformer

#### 模型结构

```
输入: (batch, 12, channels, H, W)
  ↓
Patch Embedding (每个时间步):
  Conv2d(C, embed_dim, kernel=patch_size, stride=patch_size)
  (32, 64) → (8, 8) patches, 每个patch 4×8
  ↓
位置编码:
  - 时间位置编码: 正弦编码（外推能力强）
  - 空间位置编码: 可学习参数
  ↓
时空注意力编码器 (Encoder):
  Spatial Attention: 同一时间步内，patches之间的注意力
  Temporal Attention: 同一patch，跨时间步的注意力
  Factorized设计: O(T*N² + N*T²) vs O((T*N)²)
  ↓
可学习输出查询 (Learnable Queries):
  (batch, 4, 8*8, embed_dim)  # 未来4个时间步的查询
  ↓
解码器 (Decoder):
  浅层Transformer解码器
  ↓
输出投影:
  Linear(embed_dim, patch_size² * channels)
  ↓
重塑: (batch, 4, channels, H, W)
```

#### 输入输出

- **输入**: `(batch, input_length=12, input_channels, H=32, W=64)`
- **输出**: `(batch, output_length=4, input_channels, H=32, W=64)`
- **关键参数**:
  - `d_model`: 模型维度（默认128）
  - `n_heads`: 注意力头数（默认4）
  - `n_layers`: Encoder层数（默认4）
  - `patch_size`: Patch大小（默认(4, 8)）

#### 工作原理

1. **Patch-based处理**:
   - 将空间网格分割为patches（类似ViT）
   - 每个patch作为一个token
   - 减少计算复杂度

2. **Factorized时空注意力**:
   - **Spatial Attention**: 同一时间步内，patches之间的注意力
     - 捕获空间相关性（如相邻区域的相似性）
   - **Temporal Attention**: 同一patch，跨时间步的注意力
     - 捕获时间演化规律
   - **优势**: 计算复杂度从O((T*N)²)降低到O(T*N² + N*T²)

3. **位置编码**:
   - 时间位置编码：正弦编码（更好的外推能力）
   - 空间位置编码：可学习参数（适配不规则地球网格）

4. **轻量化设计**:
   - 参数量约1.6M（与ConvLSTM相当）
   - 使用残差后LayerNorm (Post-LN) 与更好的初始化

#### 与任务的关系

- **优势**:
  - ✅ 捕获长距离时空依赖（注意力机制）
  - ✅ 轻量级设计，参数量少
  - ✅ 适合捕获大尺度天气系统（如全球环流）

- **局限**:
  - Patch分割可能丢失细节
  - 训练需要更多数据
  - 计算复杂度仍较高

#### 适用场景

- ✅ 需要捕获长距离依赖的任务
- ✅ 大尺度天气系统预测
- ✅ 计算资源充足的情况

---

## WeatherDiff 模块

WeatherDiff是基于Stable Diffusion架构的天气预测模块，将气象网格数据视为图像，利用预训练VAE和U-Net架构进行时空预测。

### 核心思想

1. **图像化处理**: 将天气场视为图像（每个时间步是一帧）
2. **VAE压缩**: 使用预训练VAE将高维图像压缩到低维潜空间
3. **潜空间预测**: 在潜空间中预测，降低计算复杂度
4. **概率建模**: Diffusion模型支持不确定性量化

---

### VAE (Stable Diffusion VAE)

#### 模型结构

```
编码器 (Encoder):
  输入图像: (B, C, H, W)  # 范围[-1, 1]
    ↓
  卷积下采样: H, W → H//8, W//8
    ↓
  潜向量: (B, 4, H//8, W//8)  # 压缩比 8×8 = 64倍

解码器 (Decoder):
  潜向量: (B, 4, H//8, W//8)
    ↓
  卷积上采样: H//8, W//8 → H, W
    ↓
  重建图像: (B, C, H, W)  # 范围[-1, 1]
```

#### 输入输出

- **编码**:
  - **输入**: `(batch, channels, H, W)` - 范围[-1, 1]
  - **输出**: `(batch, 4, H//8, W//8)` - 潜向量
  
- **解码**:
  - **输入**: `(batch, 4, H//8, W//8)` - 潜向量
  - **输出**: `(batch, channels, H, W)` - 范围[-1, 1]

#### 训练策略

当前实现仅支持加载Stable Diffusion预训练VAE权重（默认从HuggingFace获取，可通过
`--vae-pretrained-path` 指定自定义权重）。可分别控制encoder/decoder是否参与训练：

- `--freeze-encoder`: 冻结encoder，仅训练decoder+U-Net
- `--freeze-decoder`: 冻结decoder，仅训练encoder+U-Net
- 两者都加：完全冻结VAE，只训练U-Net
- 不加：encoder/decoder与U-Net一同微调

#### 工作原理

1. **预训练模型**: 使用Stable Diffusion的预训练VAE（在自然图像上训练）
2. **压缩表示**: 将512×512图像压缩到64×64潜空间（压缩比64倍）
3. **语义保持**: 潜空间保留图像的语义信息，适合生成任务
4. **重建误差**: 对于天气数据，重建RMSE约5-10K（温度单位）
5. **可训练性**: 支持冻结VAE（仅推理）或训练VAE（微调）

#### 与任务的关系

- **优势**:
  - ✅ 大幅降低计算复杂度（64倍压缩）
  - ✅ 预训练模型，无需从头训练
  - ✅ 潜空间更适合生成任务
  - ✅ 通过冻结开关灵活控制VAE微调范围

- **局限**:
  - ❌ 重建误差较大（5-10K），可能丢失细节
  - ❌ 预训练在自然图像上，可能不适合天气数据（可通过微调改善）
  - ❌ 需要数据归一化到[-1, 1]

#### 适用场景

- ✅ 大尺寸图像预测（512×512及以上）
- ✅ 需要降低显存和计算量的场景
- ✅ 需要针对天气数据微调VAE encoder/decoder的场景
- ❌ 不适合需要高精度重建的任务

---

### RAE (Representation Autoencoder)

#### 模型结构

```
编码器 (Encoder):
  输入图像: (B, C, H, W)  # 范围[0, 1]，自动从[-1, 1]转换
    ↓
  Resize到encoder_input_size (如256×256)
    ↓
  Vision Transformer (DINOv2/SigLIP2/MAE)
    ↓
  潜向量: (B, latent_dim, H_latent, W_latent)
    # latent_dim取决于encoder（如768 for DINOv2-base）
    # H_latent, W_latent = encoder_input_size // patch_size

解码器 (Decoder):
  潜向量: (B, latent_dim, H_latent, W_latent)
    ↓
  Vision Transformer Decoder (MAE-based)
    ↓
  重建图像: (B, C, H, W)  # 范围[0, 1]，自动转换为[-1, 1]
```

#### 输入输出

- **编码**:
  - **输入**: `(batch, channels, H, W)` - 范围[-1, 1]（自动转换为[0, 1]）
  - **输出**: `(batch, latent_dim, H_latent, W_latent)` - 潜向量
    - `latent_dim`: 取决于encoder（DINOv2-base: 768, SigLIP2-base: 768）
    - `H_latent, W_latent`: 取决于encoder输入尺寸和patch大小
  
- **解码**:
  - **输入**: `(batch, latent_dim, H_latent, W_latent)` - 潜向量
  - **输出**: `(batch, channels, H, W)` - 范围[-1, 1]（自动从[0, 1]转换）

#### 支持的Encoder类型

1. **DINOv2** (`Dinov2withNorm`):
   - 配置路径: `facebook/dinov2-base`
   - 输入尺寸: 224×224（默认）
   - Latent维度: 768
   - Latent空间: 14×14 (224/16=14)

2. **SigLIP2** (`SigLIP2wNorm`) ⭐ **推荐**:
   - 配置路径: `google/siglip2-base-patch16-256`
   - 输入尺寸: 256×256（默认）
   - Latent维度: 768
   - Latent空间: 16×16 (256/16=16)
   - **优势**: 更大的输入尺寸，更好的空间分辨率

3. **MAE** (`MAEwNorm`):
   - 配置路径: `facebook/vit-mae-base`
   - 输入尺寸: 224×224
   - Latent维度: 768
   - Latent空间: 14×14

#### 工作原理

1. **Encoder固定**: Encoder参数固定，不参与训练（`freeze_encoder=True`）
2. **Decoder可微调**: Decoder参数可训练（`freeze_decoder=False`），支持微调
3. **自动范围转换**: 
   - 输入: [-1, 1] → [0, 1]（RAE内部使用）
   - 输出: [0, 1] → [-1, 1]（匹配Weather项目）
4. **灵活配置**: 支持多种encoder类型，可根据需求选择

#### 与任务的关系

- **优势**:
  - ✅ **Decoder可微调**: 相比SD VAE，RAE的decoder可以针对天气数据微调
  - ✅ **多种encoder选择**: 支持DINOv2、SigLIP2、MAE等预训练encoder
  - ✅ **更大的latent维度**: 768维（vs SD VAE的4维），可能保留更多信息
  - ✅ **自动范围转换**: 无需手动处理数据范围

- **局限**:
  - ❌ Encoder需要resize输入图像（可能丢失细节）
  - ❌ Latent维度更大，可能增加计算量
  - ❌ 需要额外的decoder训练

#### 适用场景

- ✅ 需要可微调decoder的场景
- ✅ 希望利用不同预训练encoder的特征
- ✅ 大尺寸图像预测（512×512及以上）
- ✅ 显存充足的情况（latent维度较大）

#### 与SD VAE的区别

| 特性 | SD VAE | RAE |
|------|--------|-----|
| Latent channels | 4 | 取决于encoder（如768） |
| Latent shape | (4, H//8, W//8) | (latent_dim, H_latent, W_latent) |
| Encoder | 固定 | 固定（可配置） |
| Decoder | 固定 | **可微调** ⭐ |
| 输入范围 | [-1, 1] | [-1, 1]（内部转换） |
| 输出范围 | [-1, 1] | [-1, 1]（内部转换） |
| 输入resize | 无 | 有（到encoder_input_size） |

#### 使用示例

```python
from weatherdiff.vae import RAEWrapper

vae_wrapper = RAEWrapper(
    encoder_cls='SigLIP2wNorm',
    encoder_config_path='google/siglip2-base-patch16-256',
    encoder_input_size=256,
    decoder_config_path='vit_mae-base',
    decoder_patch_size=16,
    device='cuda',
    freeze_encoder=True,
    freeze_decoder=False  # decoder可微调
)

# 编码
latent = vae_wrapper.encode(images)  # (B, 768, 16, 16)

# 解码
reconstructed = vae_wrapper.decode(latent)  # (B, C, H, W)
```

---

### Pixel U-Net

#### 模型结构

```
输入: (batch, 12, channels, H, W)
  ↓
展平时间维度: (batch, 12*channels, H, W)
  ↓
U-Net:
  下采样路径 (Encoder):
    ConvBlock → MaxPool → ConvBlock → MaxPool → ...
    保留skip connections
  ↓
  瓶颈层 (Bottleneck):
    ConvBlock
  ↓
  上采样路径 (Decoder):
    UpSample → Concat(skip) → ConvBlock → ...
  ↓
输出: (batch, 4*channels, H, W)
  ↓
重塑: (batch, 4, channels, H, W)
```

#### U-Net详细结构

```python
# 卷积块（使用GroupNorm和SiLU激活）
ConvBlock:
  Conv2d → GroupNorm(8) → SiLU → Conv2d → GroupNorm(8) → SiLU

# 下采样块
DownBlock:
  ConvBlock(in_ch, out_ch) → MaxPool2d(2)
  保存skip connection（用于上采样时拼接）

# 上采样块
UpBlock:
  ConvTranspose2d(in_ch, in_ch//2, stride=2)
  Concat([up, skip])  # 拼接skip connection
  ConvBlock(in_ch//2+skip_ch, out_ch)

# 整体结构
输入: (B, T_in*C, H, W)
  → 输入卷积: (B, base_channels, H, W)
  → 下采样路径（depth层）: 逐步下采样，保存skip
  → 瓶颈层: ConvBlock
  → 上采样路径（depth层）: 逐步上采样，使用skip
  → 输出卷积: (B, T_out*C, H, W)
```

#### 输入输出

- **输入**: `(batch, input_length=12, channels, H, W)`
- **输出**: `(batch, output_length=4, channels, H, W)`
- **关键参数**:
  - `base_channels`: 基础通道数（默认64）
  - `depth`: U-Net深度（默认4）

#### 工作原理

1. **图像到图像预测**: 直接在像素空间进行预测
2. **多尺度特征**: U-Net的下采样-上采样结构捕获多尺度特征
3. **Skip连接**: 保留细节信息，避免信息丢失
4. **时间信息**: 将多个时间步作为多通道输入

#### 与任务的关系

- **优势**:
  - ✅ 训练快速（直接在像素空间）
  - ✅ 结果确定（无随机性）
  - ✅ 适合图像预测任务

- **局限**:
  - ❌ 显存需求大（大尺寸图像）
  - ❌ 无时序建模（时间步作为通道）
  - ❌ 无法量化不确定性

#### 适用场景

- ✅ 小尺寸图像（64×32）
- ✅ 快速原型验证
- ❌ 不适合大尺寸图像（512×512）

---

### Latent U-Net

#### 模型结构

**使用SD VAE**:
```
输入图像: (batch, 12, channels, H, W)
  ↓
SD VAE编码 (分批处理，控制显存):
  (batch*12, channels, H, W) → (batch*12, 4, H//8, W//8)
  ↓
重塑: (batch, 12, 4, H//8, W//8)
  ↓
展平时间维度: (batch, 12*4, H//8, W//8)
  ↓
Latent U-Net (在潜空间):
  输入卷积 → 下采样路径 → 瓶颈 → 上采样路径 → 输出卷积
  (使用与Pixel U-Net相同的U-Net结构，但输入输出通道不同)
  ↓
输出: (batch, 4*4, H//8, W//8)
  ↓
重塑: (batch, 4, 4, H//8, W//8)
  ↓
SD VAE解码 (分批处理):
  (batch*4, 4, H//8, W//8) → (batch*4, channels, H, W)
  ↓
重塑: (batch, 4, channels, H, W)
```

**使用RAE**:
```
输入图像: (batch, 12, channels, H, W)
  ↓
RAE编码 (分批处理):
  (batch*12, channels, H, W) 
    → Resize到encoder_input_size (如256×256)
    → Vision Transformer Encoder (DINOv2/SigLIP2/MAE)
    → (batch*12, latent_dim, H_latent, W_latent)
    # latent_dim取决于encoder（如768 for DINOv2-base）
    # H_latent, W_latent = encoder_input_size // patch_size
  ↓
重塑: (batch, 12, latent_dim, H_latent, W_latent)
  ↓
展平时间维度: (batch, 12*latent_dim, H_latent, W_latent)
  ↓
Latent U-Net (在潜空间):
  输入卷积 → 下采样路径 → 瓶颈 → 上采样路径 → 输出卷积
  (与SD VAE版本结构相同，但latent_channels=latent_dim)
  ↓
输出: (batch, 4*latent_dim, H_latent, W_latent)
  ↓
重塑: (batch, 4, latent_dim, H_latent, W_latent)
  ↓
RAE解码 (分批处理，decoder可微调):
  (batch*4, latent_dim, H_latent, W_latent)
    → Vision Transformer Decoder (MAE-based，可训练)
    → (batch*4, channels, H, W)
  ↓
重塑: (batch, 4, channels, H, W)
```

#### 输入输出

- **输入**: `(batch, input_length=12, channels, H, W)` - 像素空间
- **输出**: `(batch, output_length=4, channels, H, W)` - 像素空间
- **中间表示（SD VAE）**: 潜空间 `(batch, T, 4, H//8, W//8)`
- **中间表示（RAE）**: 潜空间 `(batch, T, latent_dim, H_latent, W_latent)`
- **关键参数**:
  - `vae_type`: VAE类型，'sd' 或 'rae'
  - `base_channels`: U-Net基础通道数（默认128）
  - `depth`: U-Net深度（默认3）
  - `vae_batch_size`: VAE编码/解码批次大小（控制显存）
  - **SD VAE特定参数**:
    - `vae_model_id`: SD VAE模型ID（默认'stable-diffusion-v1-5'）
    - `vae_pretrained_path`: 可选，预训练权重路径（覆盖默认SD权重）
    - `freeze_encoder`: 是否冻结encoder
    - `freeze_decoder`: 是否冻结decoder
  - **RAE特定参数**:
    - `rae_encoder_cls`: Encoder类型（'Dinov2withNorm', 'SigLIP2wNorm', 'MAEwNorm'）
    - `rae_encoder_config_path`: Encoder配置路径
    - `rae_encoder_input_size`: Encoder输入尺寸（默认256）
    - `rae_decoder_config_path`: Decoder配置路径
    - `rae_decoder_patch_size`: Decoder patch大小（默认16）
    - `freeze_encoder`: 冻结encoder（默认True）
    - `freeze_decoder`: 冻结decoder（默认False）

#### 工作原理

1. **VAE编码**: 
   - **SD VAE**: 将输入图像编码到潜空间（压缩64倍，4通道）
     - 仅支持加载预训练权重，可自定义路径
     - 通过 `freeze_encoder` / `freeze_decoder` 控制哪些部分参与训练
   - **RAE**: 将输入图像resize后编码到潜空间（latent_dim通道，如768）
2. **潜空间预测**: U-Net在潜空间中预测未来帧
3. **VAE解码**: 
   - **SD VAE**: 将预测的潜向量解码回像素空间
     - 默认解码器可训练，可通过 `freeze_decoder` 关闭微调
   - **RAE**: 将预测的潜向量解码回像素空间（可微调decoder）

4. **优势**:
   - **SD VAE**: 显存需求低（潜空间比像素空间小64倍），预训练模型稳定，支持VAE微调
   - **RAE**: Decoder可微调，可能获得更好的重建质量，支持多种encoder选择

#### 与任务的关系

- **优势**:
  - ✅ **显存需求低**（512×512 → 64×64 for SD VAE）
  - ✅ 训练更稳定
  - ✅ 生成结果更平滑
  - ✅ 适合大尺寸图像
  - ✅ **SD VAE**: encoder/decoder均可单独微调
  - ✅ **RAE**: Decoder可微调，可能获得更好的重建质量

- **局限**:
  - ❌ VAE重建误差（5-10K for SD VAE，可通过微调改善）
  - ❌ 需要VAE编码/解码步骤（增加计算时间）
  - ❌ 无法量化不确定性
  - ❌ **RAE**: Encoder需要resize输入，latent维度更大

#### 适用场景

- ✅ **大尺寸图像预测**（推荐）
- ✅ 显存有限的情况（SD VAE）
- ✅ 需要平滑预测结果的任务
- ✅ **需要可微调decoder**（RAE）

---

### Diffusion Model

#### 模型结构

```
训练阶段:
  条件（过去帧）: (batch, 12, channels, H, W)
    ↓ VAE编码
  条件潜向量: (batch, 12, 4, H//8, W//8)
  
  目标（未来帧）: (batch, 4, channels, H, W)
    ↓ VAE编码
  目标潜向量: (batch, 4, 4, H//8, W//8)
    ↓
  添加噪声: latent_target + noise * sqrt(beta_t)
    ↓
  U-Net预测噪声: predicted_noise
    ↓
  损失: MSE(predicted_noise, true_noise)

推理阶段:
  条件潜向量: (batch, 12, 4, H//8, W//8)
  随机噪声: (batch, 4, 4, H//8, W//8)
    ↓
  逐步去噪 (T步):
    for t in [T-1, ..., 0]:
      predicted_noise = U-Net(noisy_latent, t, condition)
      latent = scheduler.step(predicted_noise, t, latent)
    ↓
  去噪后的潜向量: (batch, 4, 4, H//8, W//8)
    ↓ VAE解码
  预测图像: (batch, 4, channels, H, W)
```

#### Diffusion U-Net结构

```python
# 带时间嵌入的U-Net
输入: 
  - noisy_latent: (B, 4*4, H//8, W//8)  # 加噪的未来帧
  - condition: (B, 12*4, H//8, W//8)   # 过去帧
  - timestep: (B,)
  
拼接: (B, (12+4)*4, H//8, W//8)
  ↓
时间嵌入: TimestepEmbedding(timestep) → (B, time_emb_dim)
  ↓
U-Net (带时间嵌入):
  下采样块: ConvBlockWithTime(x, time_emb)
  瓶颈层: ConvBlockWithTime(x, time_emb)
  上采样块: ConvBlockWithTime(x, time_emb)
  ↓
输出: (B, 4*4, H//8, W//8)  # 预测的噪声
```

#### 输入输出

- **训练**:
  - **输入**: 
    - 条件: `(batch, input_length=12, channels, H, W)`
    - 目标: `(batch, output_length=4, channels, H, W)`
  - **输出**: 预测的噪声 `(batch, output_length=4, 4, H//8, W//8)`

- **推理**:
  - **输入**: 条件 `(batch, input_length=12, channels, H, W)`
  - **输出**: 预测 `(batch, output_length=4, channels, H, W)`
  - **可选**: 生成多个样本（集成预测）

#### 工作原理

1. **前向扩散过程（训练）**:
   - 逐步向目标添加噪声: `x_t = sqrt(alpha_t) * x_0 + sqrt(1-alpha_t) * noise`
   - 学习预测每一步添加的噪声

2. **反向去噪过程（推理）**:
   - 从随机噪声开始
   - 逐步去噪，每步预测并移除噪声
   - 最终得到清晰的预测结果

3. **条件生成**:
   - 将过去帧作为条件输入U-Net
   - U-Net根据条件和时间步预测噪声

4. **不确定性量化**:
   - 可以生成多个样本（不同随机种子）
   - 集成预测提供不确定性估计

#### 噪声调度器

```python
# DDPM调度器
beta_t = linear_schedule(0.0001, 0.02, T=1000)
alpha_t = 1 - beta_t
alpha_bar_t = cumprod(alpha_t)

# 添加噪声
noisy = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise

# 去噪步骤
predicted_x_0 = (x_t - sqrt(1-alpha_bar_t) * predicted_noise) / sqrt(alpha_bar_t)
x_{t-1} = sqrt(alpha_bar_{t-1}) * predicted_x_0 + sqrt(1-alpha_bar_{t-1}) * predicted_noise
```

#### 与任务的关系

- **优势**:
  - ✅ **不确定性量化**: 可以生成多个未来场景
  - ✅ **集成预测**: 多个样本的平均更准确
  - ✅ **概率建模**: 适合天气系统的混沌特性
  - ✅ **高质量生成**: Diffusion模型生成质量高

- **局限**:
  - ❌ 训练时间长（需要学习去噪过程）
  - ❌ 推理慢（需要多步去噪，如50-1000步）
  - ❌ 显存需求大（U-Net + VAE）

#### 适用场景

- ✅ **需要不确定性估计的任务**（推荐）
- ✅ 集成预测
- ✅ 概率天气预报
- ❌ 不适合快速推理场景

---

## 模型对比与选择

### 模型特性对比表

| 模型 | 时空建模 | 训练速度 | 推理速度 | 不确定性 | 显存需求 | 参数量 | 推荐场景 |
|------|---------|---------|---------|---------|---------|--------|---------|
| **Linear Regression** | ✗ | ⚡⚡⚡ | ⚡⚡⚡ | ✗ | 低 | 很少 | 快速基线 |
| **LSTM** | 时序 | ⚡⚡ | ⚡⚡ | ✗ | 中 | 中 | 单点预测 |
| **Transformer** | 时序 | ⚡⚡ | ⚡⚡ | ✗ | 中 | 中 | 单点预测（长距离依赖） |
| **CNN** | 空间 | ⚡⚡ | ⚡⚡⚡ | ✗ | 中 | 中 | 空间模式 |
| **ConvLSTM** | 时空 | ⚡ | ⚡⚡ | ✗ | 中 | 中 | **通用预测** ⭐ |
| **Weather Transformer** | 时空 | ⚡ | ⚡ | ✗ | 中 | 少 | 长距离依赖 |
| **Pixel U-Net** | 空间(通道堆叠) | ⚡⚡ | ⚡⚡ | ✗ | 高 | 中 | 小图像 |
| **Latent U-Net (SD VAE)** | 空间(潜空间堆叠) | ⚡⚡ | ⚡⚡ | ✗ | 低 | 中 | **大图像** ⭐ |
| **Latent U-Net (RAE)** | 空间(潜空间堆叠) | ⚡⚡ | ⚡⚡ | ✗ | 中 | 中 | **大图像（可微调）** ⭐ |
| **Diffusion** | 时空 | 🐢 | 🐢 | ✓ | 高 | 中 | **概率预测** ⭐ |

### 选择指南

#### 1. 快速基线测试
- **推荐**: Linear Regression
- **原因**: 训练极快，验证数据流

#### 2. 确定性预测（通用）
- **推荐**: ConvLSTM 或 Latent U-Net
- **ConvLSTM**: 适合64×32小尺寸，训练快
- **Latent U-Net (SD VAE)**: 适合512×512大尺寸，显存友好
- **Latent U-Net (RAE)**: 适合512×512大尺寸，decoder可微调

#### 3. 不确定性量化
- **推荐**: Diffusion Model
- **原因**: 唯一支持概率预测的模型

#### 4. 计算资源有限
- **CPU**: Linear Regression, LSTM
- **单GPU (8GB)**: CNN, ConvLSTM
- **单GPU (12GB+)**: Latent U-Net, Diffusion

#### 5. 大尺寸图像（512×512）
- **推荐**: Latent U-Net (SD VAE 或 RAE)
- **SD VAE**: 显存需求低，训练稳定
- **RAE**: Decoder可微调，可能获得更好的重建质量

### 模型与任务的关系总结

1. **时空建模需求**:
   - ✅ ConvLSTM, Weather Transformer, Latent U-Net, Diffusion
   - ⚠️ Transformer, LSTM (仅时序，无空间)
   - ❌ Linear Regression, CNN (无时序)

2. **不确定性需求**:
   - ✅ Diffusion Model
   - ❌ 其他所有模型（确定性）

3. **计算效率需求**:
   - ✅ Linear Regression, CNN, Pixel U-Net
   - ❌ Diffusion Model

4. **大尺寸数据**:
   - ✅ Latent U-Net (SD VAE 或 RAE压缩)
   - ❌ Pixel U-Net (显存不足)

5. **需要可微调decoder**:
   - ✅ Latent U-Net (RAE)
   - ❌ Latent U-Net (SD VAE，decoder固定)

---

## 总结

本项目提供了从简单基线到复杂概率模型的完整模型体系：

1. **传统模型**: 适合快速验证和简单任务
2. **ConvLSTM**: 确定性预测的最佳选择
3. **WeatherDiff模块**: 
   - **Latent U-Net (SD VAE)**: 大尺寸图像的确定性预测，显存友好
   - **Latent U-Net (RAE)**: 大尺寸图像的确定性预测，decoder可微调
   - **Diffusion**: 概率预测和不确定性量化

选择合适的模型需要权衡：
- 预测精度 vs 计算效率
- 确定性 vs 不确定性
- 模型复杂度 vs 数据量
- 固定decoder vs 可微调decoder

建议从ConvLSTM开始，然后根据需求选择：
- **大尺寸图像（512×512）**: Latent U-Net (SD VAE 或 RAE)
- **需要不确定性估计**: Diffusion
- **需要可微调decoder**: Latent U-Net (RAE)
