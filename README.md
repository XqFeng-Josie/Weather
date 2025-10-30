# Weather Prediction System

## 🏗️ 模型架构

### 1. Linear Regression (lr)
- **原理**：Ridge回归，对每个时间步进行独立预测
- **适用**：单变量快速baseline
- **局限**：无时序建模，多变量时退化

### 2. Multi-Output LR (lr_multi)
- **原理**：每个**变量**一个独立Ridge模型（如2m_temperature一个，geopotential一个）
- **特点**：避免变量间干扰，Ridge正则化防止病态矩阵
- **性能**：快速（并行训练），多变量基线
- **实现**：`n_variables`个Ridge模型，每个预测该变量的所有网格点

### 3. LSTM
- **原理**：循环神经网络，建模时间依赖
- **结构**：输入展平 → LSTM → 全连接 → 输出
- **适用**：单变量时间序列
- **局限**：展平丢失空间信息

### 4. Transformer
- **原理**：自注意力机制，捕获长距离依赖
- **结构**：位置编码 → Transformer编码器 → 全连接
- **问题**：易过拟合，单变量时需小模型（d_model=64, layers=2）
- **建议**：不推荐使用（即使优化后仍容易退化）

### 5. CNN
- **原理**：卷积神经网络，提取空间特征
- **结构**：Conv2D → BatchNorm → ReLU → Flatten → FC
- **适用**：空间预测，多变量
- **局限**：无时序建模

### 6. ConvLSTM
- **原理**：结合CNN和LSTM，同时建模时空依赖
- **结构**：ConvLSTM单元（卷积+LSTM门控）→ Conv2D输出
- **优势**：保留空间结构，建模时序，适合多变量
- **推荐**：最适合天气预测

### 7. Diffusion (diffusion) - 概率预测 ⭐

**核心思想**：不预测单一确定值，而是预测概率分布

#### 原理
- **前向过程**：逐步向数据添加噪声（训练时模拟）
- **反向过程**：从噪声逐步去噪恢复数据（推理时使用）
- **训练目标**：学习预测每一步添加的噪声
- **推理过程**：从随机噪声开始，迭代去噪生成预测

#### 架构
```
输入历史: (12, 1, 64, 32) → Condition Encoder
                                    ↓
随机噪声: (4, 1, 64, 32) ──→ UNet Denoiser ──→ 预测输出
                            (多次迭代去噪)
```

**UNet Denoiser**：
- Down: Conv2D + GroupNorm + SiLU (提取特征)
- Middle: ResNet blocks (处理特征)
- Up: ConvTranspose2D + Skip connections (重建空间)
- Time Embedding: 注入噪声级别信息

#### Ensemble Forecasting

**为什么需要 Ensemble？**
- Diffusion 的随机性：每次推理从不同随机噪声开始
- 自然产生多样性：无需额外设计
- 概率预测：生成多个可能的未来场景

**评估指标**：
- **CRPS** (Continuous Ranked Probability Score)：评估概率分布质量
- **Spread-Skill Ratio**：检验 ensemble 校准（理想值 ≈ 1.0）
- **Ensemble Mean RMSE**：与确定性模型对比



## 📊 数据说明

### 数据源
- **WeatherBench2**：全球ERA5再分析数据
- **分辨率**：64×32等角网格（经度×纬度）
- **时间间隔**：6小时
- **时间范围**：1959-01-01 到 2021-12-31（注意：尽管路径名为"1959-2022"，但实际只到2021年底）

### 数据结构

```python
# 原始数据 (xarray Dataset)
ds = {
    '2m_temperature': (time, lat, lon),           # (N, 32, 64)
    'geopotential': (time, level, lat, lon),      # (N, 3, 32, 64) - 选择500/700/850hPa
    ...
}

# Flat格式（lr, lstm, transformer）
X: (n_samples, input_length, n_features)
   n_features = n_variables × 32 × 64
   例：单变量 n_features=2048，双变量 n_features=4096

# Spatial格式（cnn, convlstm）
X: (n_samples, input_length, n_channels, H, W)
   n_channels = n_variables (+ levels)
   H=32, W=64
```

### 序列划分

- **输入长度** (`input_length=12`)：过去12个时间步（3天）
- **输出长度** (`output_length=4`)：未来4个时间步（1天）
- **滑动窗口**：每次移动1步

## 🔬 评估指标

### 1. 基础指标
- **RMSE**：均方根误差（主要指标）
- **MAE**：平均绝对误差
- **R²**：决定系数

### 2. 时间分辨
- **RMSE per Lead Time**：每个预测步长的RMSE
- 用于分析误差随预测时长的增长

### 3. 空间分辨
- **空间误差图**：可视化不同区域的预测误差
- 识别模型在特定区域的优劣

### 4. WeatherBench2标准
- **Bias**：系统偏差
- **ACC**：距平相关系数
- **Skill Score**：相对于基线的改进


## 🛠️ 技术细节

### 1. 数据标准化
- **StandardScaler**：每个变量独立标准化（均值0，方差1）
- **lr_multi**：每个变量的X和y分别标准化

### 2. 正则化策略
- **Ridge**：L2正则化（alpha=10.0）防止过拟合和病态矩阵
- **Dropout**：神经网络中防止过拟合
- **Early Stopping**：监控验证集，自动停止训练

### 3. 优化技巧
- **Transformer单变量**：自动使用小模型（d=64, l=2）+高dropout(0.3)+低学习率
- **lr_multi并行**：每个变量独立训练，避免2048个网格点模型
- **Batch Normalization**：CNN中稳定训练

### 4. 数据划分
- **训练集**：70%
- **验证集**：15%
- **测试集**：15%
- **时序保持**：不打乱顺序

## 📁 项目结构

```
Weather/
├── src/
│   ├── data_loader.py          # 数据加载和预处理
│   ├── trainer.py              # 训练器（支持sklearn和PyTorch）
│   ├── models/
│   │   ├── __init__.py         # 模型工厂
│   │   ├── linear_regression.py    # LR和MultiOutputLR
│   │   ├── lstm.py             # LSTM
│   │   ├── transformer.py      # Transformer
│   │   ├── cnn.py              # CNN
│   │   ├── convlstm.py         # ConvLSTM
│   │   └── diffusion/          # Diffusion模型
│   │       ├── diffusion_model.py      # 主模型
│   │       ├── diffusion_trainer.py    # 训练器（含ensemble）
│   │       ├── unet_simple.py          # UNet架构
│   │       └── noise_scheduler.py      # 噪声调度
│   └── metrics/
│       └── probabilistic.py    # 概率评估指标（CRPS等）
├── scripts/                    # 运行脚本 ⭐
│   ├── run_lr_multi.sh         # LR Multi (训练+预测+评估)
│   ├── run_lstm.sh             # LSTM (训练+预测+评估)
│   ├── run_cnn.sh              # CNN (训练+预测+评估)
│   ├── run_convlstm.sh         # ConvLSTM (训练+预测+评估)
│   ├── run_diffusion.sh        # Diffusion 基础 (训练+预测+评估)
│   └── run_diffusion_ensemble.sh   # Diffusion Ensemble (训练+预测+评估)
├── train.py                    # 训练脚本（确定性模型）
├── train_diffusion.py          # Diffusion训练脚本
├── predict.py                  # 预测脚本
├── evaluate_diffusion_ensemble.py  # Ensemble评估工具
├── evaluate_weatherbench.py    # WeatherBench2评估
├── USAGE.md                    # 使用指南 ⭐
└── README.md                   # 本文件 ⭐

outputs/                        # 训练输出
└── <exp_name>/
    ├── best_model.pth
    ├── config.json
    ├── metrics.json            # 包含确定性和概率指标
    ├── predictions_*.png       # 预测图
    ├── ensemble_*.png          # Ensemble可视化（如启用）
    └── ensemble_predictions.npy    # Ensemble数据（如启用）
```

## 🚀 快速开始

详细使用说明见 [USAGE.md](USAGE.md)

## 📚 参考资料

### 数据和基准
- [WeatherBench2](https://weatherbench2.readthedocs.io/) - 天气预测基准
- [ERA5](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5) - ECMWF再分析数据

### 模型论文
- [ConvLSTM](https://arxiv.org/abs/1506.04214) - Shi et al., 2015
- [DDPM](https://arxiv.org/abs/2006.11239) - Ho et al., 2020 (Diffusion 模型基础)
- [GenCast](https://arxiv.org/abs/2312.15796) - Price et al., 2023 (概率天气预测)