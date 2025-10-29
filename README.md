# Weather Prediction - 基于深度学习的天气预测系统

多模型对比的天气序列预测项目，支持单变量和多变量预测。

## 📋 项目目标

构建一个天气预测系统，用于：
1. **多模型对比**：对比线性回归、LSTM、Transformer、CNN、ConvLSTM等模型在天气预测任务上的表现
2. **多变量预测**：支持同时预测多个气象变量（温度、位势、风场等）
3. **时空建模**：处理具有时间序列和空间网格结构的气象数据
4. **基准评估**：使用WeatherBench2标准进行模型评估

## 🎯 核心问题

### 问题1：多变量预测退化

**现象**：标准LSTM/Transformer在多变量预测时退化为均值预测器

**原因**：
- 数据展平（32×64=2048特征）丢失空间信息
- 多变量干扰（不同气象变量混在同一特征空间）
- 过拟合（模型参数过多，数据量不足）

**解决方案**：
1. **独立建模**：`lr_multi` 为每个变量训练独立模型
2. **保留空间结构**：`cnn`, `convlstm` 使用卷积保留空间信息
3. **正则化**：增加dropout，减小模型规模

### 问题2：空间vs时间建模

**Flat模型** (lr, lstm, transformer)：
- 将空间网格展平为1D特征向量
- 适合简单时间序列，但丢失空间关系
- 多变量时容易退化

**Spatial模型** (cnn, convlstm)：
- 保留2D空间结构
- 卷积提取空间特征
- 更适合气象数据

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

#### Ensemble Forecasting（GenCast 风格）

**为什么需要 Ensemble？**
- Diffusion 的随机性：每次推理从不同随机噪声开始
- 自然产生多样性：无需额外设计
- 概率预测：生成多个可能的未来场景

**评估指标**：
- **CRPS** (Continuous Ranked Probability Score)：评估概率分布质量
- **Spread-Skill Ratio**：检验 ensemble 校准（理想值 ≈ 1.0）
- **Ensemble Mean RMSE**：与确定性模型对比

**使用方法**：
```bash
# Ensemble 评估（推荐）
python train_diffusion.py \
    --variables 2m_temperature \
    --epochs 200 \
    --enable-ensemble-eval \
    --num-ensemble-members 20
```

详细参数见 `USAGE.md`

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

## 📈 模型对比

| 模型 | 单变量 | 多变量 | 数据格式 | 参数量 | 速度 | 预测类型 | 推荐场景 |
|------|-------|-------|---------|--------|------|---------|---------|
| `lr` | ✓ | ❌ | flat | ~10K | ⚡⚡⚡ | 确定性 | 避免使用 |
| `lr_multi` | ✓ | ✓ | flat | ~10K | ⚡⚡⚡ | 确定性 | 快速baseline |
| `lstm` | ✓ | ❌ | flat | ~850K | ⚡⚡ | 确定性 | 单变量 |
| `transformer` | ⚠️ | ❌ | flat | ~500K | ⚡ | 确定性 | 不推荐 |
| `cnn` | ✓ | ✓ | spatial | ~284K | ⚡⚡ | 确定性 | 空间预测 |
| `convlstm` | ✓ | ✓ | spatial | ~117K | ⚡ | 确定性 | **首选（确定性）** |
| `diffusion` | ✓ | ✓ | spatial | ~3.7M | 🐌 | 概率 | **概率预测** |

**说明**：
- ✓ 适用  ⚠️ 需谨慎  ❌ 不推荐
- `lr_multi`：每个**变量**一个模型，快速且无数值问题
- `transformer`：即使优化后仍容易退化，不推荐
- `convlstm`：确定性预测首选，平衡性能、速度、稳定性
- `diffusion`：概率预测，提供不确定性量化，需要 ensemble 评估

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
│   │       ├── __init__.py
│   │       ├── diffusion_model.py      # 主模型
│   │       ├── diffusion_trainer.py    # 训练器（含ensemble）
│   │       ├── unet_simple.py          # UNet架构
│   │       └── noise_scheduler.py      # 噪声调度
│   └── metrics/
│       └── probabilistic.py    # 概率评估指标（CRPS等）
├── train.py                    # 训练脚本（确定性模型）
├── train_diffusion.py          # Diffusion训练脚本
├── predict.py                  # 预测脚本
├── evaluate_diffusion_ensemble.py  # Ensemble评估
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

### 确定性预测（推荐 ConvLSTM）

```bash
# 1. 安装依赖
pip install torch xarray zarr gcsfs scikit-learn matplotlib tqdm scipy

# 2. 训练 ConvLSTM
python train.py \
    --model convlstm \
    --variables 2m_temperature \
    --time-slice 2020-01-01:2020-12-31 \
    --epochs 30

# 3. 生成预测
python predict.py --model-path outputs/<exp_name>/best_model.pth

# 4. 评估
python evaluate_weatherbench.py --pred outputs/<exp_name>/predictions.npz
```

### 概率预测（Diffusion）

```bash
# 训练 Diffusion + Ensemble 评估
python train_diffusion.py \
    --variables 2m_temperature \
    --time-slice 2019-01-01:2020-12-31 \
    --epochs 200 \
    --batch-size 16 \
    --enable-ensemble-eval \
    --num-ensemble-members 20

# 查看结果（包含 CRPS, Spread-Skill Ratio）
cat outputs/<exp_name>/metrics.json
```

详细使用说明见 [USAGE.md](USAGE.md)

## 📚 参考资料

### 数据和基准
- [WeatherBench2](https://weatherbench2.readthedocs.io/) - 天气预测基准
- [ERA5](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5) - ECMWF再分析数据

### 模型论文
- [ConvLSTM](https://arxiv.org/abs/1506.04214) - Shi et al., 2015
- [DDPM](https://arxiv.org/abs/2006.11239) - Ho et al., 2020 (Diffusion 模型基础)
- [GenCast](https://arxiv.org/abs/2312.15796) - Price et al., 2023 (概率天气预测)

## 🔄 更新日志

- **2024-10-28**: 修复多变量可视化，每个变量单独显示
- **2024-10-28**: lr_multi优化为"每变量一个模型"而非"每网格点一个模型"
- **2024-10-28**: Transformer单变量自动使用小模型防止退化
- **2024-10-28**: variables参数改为逗号分割输入
- **2024-10**: 初始版本，支持6种模型，单/多变量预测

## 📝 课程项目说明

本项目为课程作业，重点在于：
1. **代码清晰**：模块化设计，每个模型独立文件
2. **实验对比**：多模型对比，分析优劣
3. **问题解决**：识别并解决多变量退化问题
4. **标准评估**：使用WeatherBench2标准

代码注重可读性和实验性，而非生产部署。
