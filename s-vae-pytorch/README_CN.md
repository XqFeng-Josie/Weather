# 天气数据超球面变分自编码器 (S-VAE) 架构设计

本文档详细介绍用于天气数据的超球面变分自编码器（Hyperspherical Variational Auto-Encoder, S-VAE）的架构设计。

## 目录

- [项目概述](#项目概述)
- [架构设计总览](#架构设计总览)
- [编码器设计](#编码器设计)
- [潜在空间设计](#潜在空间设计)
- [解码器设计](#解码器设计)
- [损失函数设计](#损失函数设计)
- [关键特性](#关键特性)
- [使用示例](#使用示例)

## 项目概述

本项目实现了一个专门用于处理天气网格数据的变分自编码器（VAE），支持两种潜在空间分布：

1. **标准 VAE**：使用高斯分布（Normal Distribution）作为潜在空间先验
2. **S-VAE**：使用超球面分布（Von Mises-Fisher Distribution）作为潜在空间先验

该架构设计参考了 Stable Diffusion 的 VAE 设计理念，采用空间保持的潜在表示，即潜在空间保持原始数据的空间维度结构，只进行通道压缩。

## 架构设计总览

### 整体架构

```
输入天气场 (B, C, H, W)
    ↓
[编码器 Encoder]
    ↓
潜在空间表示 (B, latent_channels, H//8, W//8)
    ↓
[重参数化 Reparameterization]
    ↓
采样潜在向量 (B, latent_channels, H//8, W//8)
    ↓
[解码器 Decoder]
    ↓
重构天气场 (B, C, H, W)
```

### 关键设计特点

1. **空间保持的潜在表示**：潜在空间保持原始空间维度，下采样因子为 8（2^4，对应 4 层编码器）
2. **通道压缩**：通过 `latent_channels` 参数控制潜在空间的通道数（默认 4）
3. **残差连接**：编码器和解码器支持残差块，提升训练稳定性和重构质量
4. **双分布支持**：灵活支持高斯分布和超球面分布两种潜在空间先验

## 编码器设计

### 编码器结构

编码器采用卷积神经网络（CNN）架构，通过多层下采样将输入天气场压缩到潜在空间。

#### 基本结构

```python
编码器 = [
    Conv2d(输入通道, 64, kernel=3, stride=2, padding=1)  # 下采样 2x
    GroupNorm(64)
    SiLU()
    
    [可选] ResidualBlock(64, 128, stride=2)  # 下采样 2x
    或
    Conv2d(64, 128, kernel=3, stride=2, padding=1)
    GroupNorm(128)
    SiLU()
    
    ... (继续到 hidden_dims[-1])
]
```

#### 残差块设计

当启用 `use_residual=True` 时，编码器使用残差块（ResidualBlock）：

```python
class ResidualBlock:
    输入 → Conv2d → GroupNorm → SiLU
         ↓
    Conv2d → GroupNorm → (输出 + 残差连接) → SiLU → 输出
```

**残差块特点**：
- 使用 `GroupNorm` 替代 `BatchNorm`，更适合小批量训练
- 使用 `SiLU` 激活函数（Swish 激活）
- 支持通道数和空间维度的投影（通过 shortcut 连接）

#### 潜在空间映射层

编码器输出后，根据分布类型使用不同的映射层：

**高斯分布（Normal）**：
```python
mu = Conv2d(hidden_dims[-1], latent_channels, kernel=3, padding=1)
logvar = Conv2d(hidden_dims[-1], latent_channels, kernel=3, padding=1)
```

**超球面分布（VMF）**：
```python
mean = Conv2d(hidden_dims[-1], latent_channels, kernel=3, padding=1)
mean = mean / (mean.norm(dim=1, keepdim=True) + 1e-8)  # 归一化到单位球面
kappa = Softplus(Conv2d(hidden_dims[-1], 1, kernel=3, padding=1)) + 1  # 浓度参数
```

### 编码器参数

- **输入**：`(B, n_channels, H, W)` - 批次大小 B，通道数 C，高度 H，宽度 W
- **输出**：
  - 高斯分布：`mu (B, latent_channels, H//8, W//8)`, `logvar (B, latent_channels, H//8, W//8)`
  - VMF 分布：`mean (B, latent_channels, H//8, W//8)`, `kappa (B, 1, H//8, W//8)`
- **下采样因子**：`2^len(hidden_dims)`（默认 2^4 = 8）

## 潜在空间设计

### 高斯分布（Standard VAE）

#### 重参数化技巧

```python
std = exp(0.5 * logvar)
eps ~ N(0, 1)  # 标准正态分布采样
z = mu + eps * std
```

#### KL 散度

```python
q(z|x) = N(mu, std^2)  # 后验分布
p(z) = N(0, 1)         # 先验分布
KL = KL(q(z|x) || p(z))
```

### 超球面分布（S-VAE）

#### Von Mises-Fisher 分布

Von Mises-Fisher (VMF) 分布是超球面上的概率分布，定义在单位球面 S^{d-1} 上。

**概率密度函数**：
```
f(x; μ, κ) = C_d(κ) * exp(κ * μ^T * x)
```

其中：
- `μ`：均值方向（单位向量）
- `κ`：浓度参数（κ > 0，越大分布越集中在 μ 方向）
- `C_d(κ)`：归一化常数

#### 空间位置独立建模

对于每个空间位置 `(h, w)`，独立建模一个 VMF 分布：

```python
# 对每个批次和空间位置
for b in range(B):
    for h, w in range(H, W):
        mean_i = mean[b, :, h, w]  # (latent_channels,)
        kappa_i = kappa[b, 0, h, w]  # 标量
        
        # 创建 VMF 分布
        q_z_i = VonMisesFisher(mean_i, kappa_i)
        
        # 先验：超球面均匀分布
        p_z_i = HypersphericalUniform(latent_channels - 1)
        
        # 采样
        z_i = q_z_i.rsample()
        
        # 计算 KL 散度
        kl_i = KL(q_z_i || p_z_i)
```

#### 关键设计决策

1. **空间位置独立**：每个空间位置独立建模，允许不同区域有不同的潜在表示
2. **单位球面约束**：均值向量归一化到单位球面，确保在超球面上
3. **浓度参数**：使用 `Softplus + 1` 确保 κ > 0

### 潜在空间维度

- **空间维度**：`(H//8, W//8)` - 保持空间结构，下采样 8 倍
- **通道维度**：`latent_channels`（默认 4）- 可配置的压缩比
- **总压缩比**：空间 8x8 = 64 倍，通道由 `n_channels` 压缩到 `latent_channels`

## 解码器设计

### 解码器结构

解码器采用转置卷积（Transposed Convolution）进行上采样，将潜在表示重构回原始空间。

#### 基本结构

```python
解码器 = [
    # 投影层：将潜在通道映射回编码器最后一层通道数
    Conv2d(latent_channels, hidden_dims[-1], kernel=3, padding=1)
    
    # 上采样层
    [可选] ResidualBlock(hidden_dims[-1], hidden_dims[-1], stride=1)
    ConvTranspose2d(hidden_dims[-1], hidden_dims[-2], kernel=4, stride=2, padding=1)
    GroupNorm(hidden_dims[-2])
    SiLU()
    
    ... (继续上采样到原始尺寸)
    
    # 输出层
    Conv2d(最后通道数, n_channels, kernel=3, padding=1)
    Tanh()  # 输出范围 [-1, 1]
]
```

#### 上采样策略

- 使用 `ConvTranspose2d` 进行 2x 上采样
- 每层上采样 2 倍，4 层共上采样 2^4 = 16 倍
- 但由于编码器下采样 8 倍，解码器需要上采样 8 倍才能恢复到原始尺寸

#### 输出归一化

解码器输出通过 `Tanh` 激活函数，将值限制在 `[-1, 1]` 范围内，这与输入数据的归一化范围一致。

## 损失函数设计

### 基础损失函数

#### 重构损失（Reconstruction Loss）

**标准 MSE 损失**：
```python
recon_loss = MSE(x_recon, x_target)
```

**高级损失函数**（`use_advanced_loss=True`）：
```python
# 感知损失：L1 + L2
perceptual_loss = L1(x_recon, x_target) + MSE(x_recon, x_target)

# 梯度损失：保持空间梯度一致性
gradient_loss = MSE(grad_x_recon, grad_x_target) + MSE(grad_y_recon, grad_y_target)

# 总重构损失
recon_loss = perceptual_weight * perceptual_loss + grad_loss_weight * gradient_loss
```

#### KL 散度损失

**高斯分布**：
```python
kl_loss = KL(N(mu, std^2) || N(0, 1))
```

**VMF 分布**：
```python
kl_loss = mean(KL(VMF(mean_i, kappa_i) || HypersphericalUniform))
```

### 总损失函数

```python
total_loss = recon_loss + kl_weight * kl_loss
```

其中：
- `kl_weight`：KL 散度权重（默认 1e-6），控制正则化强度
- 较小的 `kl_weight` 允许模型更专注于重构质量
- 较大的 `kl_weight` 鼓励潜在空间更接近先验分布

### 损失函数特性

1. **感知损失**：结合 L1 和 L2 损失，平衡细节保留和整体一致性
2. **梯度损失**：保持天气场的空间梯度信息，对天气数据特别重要
3. **KL 权重调节**：支持 KL 退火（KL Annealing），逐步增加正则化强度

## 关键特性

### 1. 空间保持的潜在表示

与传统 VAE 将空间维度展平不同，本架构保持空间结构：
- **优势**：保留空间局部性，适合处理具有空间相关性的天气数据
- **应用**：可以直接在潜在空间进行空间操作（如插值、平滑等）

### 2. 双分布支持

支持两种潜在空间分布，各有优势：

**高斯分布（Normal）**：
- 计算简单，训练稳定
- 适合大多数场景
- 潜在空间连续且易于插值

**超球面分布（VMF）**：
- 理论上有更好的表示能力
- 适合处理具有方向性的数据
- 潜在空间在单位球面上，具有更好的几何性质

### 3. 残差连接

可选的残差连接提升模型性能：
- 缓解梯度消失问题
- 加速训练收敛
- 提升重构质量

### 4. 灵活的架构配置

- **隐藏层维度**：可配置的编码器/解码器通道数
- **潜在通道数**：可调节的压缩比
- **激活函数**：使用 SiLU（Swish）激活函数
- **归一化**：使用 GroupNorm，适合小批量训练

### 5. 高级损失函数

- **感知损失**：结合多种距离度量
- **梯度损失**：保持空间结构信息
- **可配置权重**：灵活调节各项损失的贡献

## 使用示例

### 基本使用

```python
from examples.train_weather_svae_improved import ImprovedWeatherVAE

# 创建模型（高斯分布）
model = ImprovedWeatherVAE(
    spatial_shape=(64, 32),
    n_channels=1,
    latent_channels=4,
    hidden_dims=(64, 128, 256, 512),
    distribution='normal',
    use_residual=True
)

# 创建模型（超球面分布）
model_svae = ImprovedWeatherVAE(
    spatial_shape=(64, 32),
    n_channels=1,
    latent_channels=4,
    hidden_dims=(64, 128, 256, 512),
    distribution='vmf',
    use_residual=True
)
```

### 训练配置

```bash
# 标准 VAE（高斯分布）
python examples/train_weather_svae_improved.py \
    --data-path gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr \
    --variable 2m_temperature \
    --time-slice 2015-01-01:2019-12-31 \
    --distribution normal \
    --latent-channels 4 \
    --hidden-dims 64 128 256 512 \
    --use-residual \
    --use-advanced-loss \
    --lr 1e-4 \
    --kl-weight 1e-6 \
    --epochs 600 \
    --batch-size 32 \
    --output-dir outputs/svae_normal \
    --save-model

# S-VAE（超球面分布）
python examples/train_weather_svae_improved.py \
    --data-path gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr \
    --variable 2m_temperature \
    --time-slice 2015-01-01:2019-12-31 \
    --distribution vmf \
    --latent-channels 4 \
    --hidden-dims 64 128 256 512 \
    --use-residual \
    --use-advanced-loss \
    --lr 1e-4 \
    --kl-weight 1e-6 \
    --epochs 600 \
    --batch-size 32 \
    --output-dir outputs/svae_vmf \
    --save-model
```

## 架构优势总结

1. **空间结构保持**：潜在空间保持空间维度，适合处理空间相关的天气数据
2. **灵活压缩比**：通过 `latent_channels` 控制压缩程度
3. **双分布支持**：可根据需求选择高斯或超球面分布
4. **稳定训练**：残差连接和 GroupNorm 提升训练稳定性
5. **高质量重构**：高级损失函数保持细节和空间结构
6. **易于扩展**：模块化设计，易于添加新特性

## 参考文献

1. Davidson, T. R., et al. "Hyperspherical Variational Auto-Encoders." UAI 2018.
2. Kingma, D. P., & Welling, M. "Auto-Encoding Variational Bayes." ICLR 2014.
3. Rezende, D. J., et al. "Stochastic Backpropagation and Approximate Inference in Deep Generative Models." ICML 2014.

## 许可证

MIT License

