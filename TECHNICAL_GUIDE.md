# 天气预测系统 - 技术指南

> 深入理解项目设计思路、模型原理和相关知识

## 📚 目录

1. [系统设计思路](#系统设计思路)
2. [数据和特征工程](#数据和特征工程)
3. [模型架构详解](#模型架构详解)
4. [WeatherBench2 评测标准](#weatherbench2-评测标准)
5. [训练策略和优化](#训练策略和优化)
6. [评估指标体系](#评估指标体系)
7. [最新研究进展](#最新研究进展)

---

## 系统设计思路

### 整体架构

本系统采用模块化设计，分为四个核心模块：

```
┌─────────────────────────────────────────────────────┐
│                   数据层 (src/data_loader.py)         │
│  - ERA5 数据加载和预处理                               │
│  - 特征工程和归一化                                    │
│  - 时间序列样本生成                                    │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│                   模型层 (src/models.py)              │
│  - Linear Regression (baseline)                      │
│  - LSTM (时间序列)                                    │
│  - Transformer (长序列)                               │
│  - CNN-LSTM (时空预测)                                │
│  - U-Net (空间预测)                                   │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│                  训练层 (src/trainer.py)              │
│  - 统一训练接口                                        │
│  - Early stopping                                    │
│  - 学习率调度                                         │
│  - 检查点管理                                         │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│              评估层 (evaluate_weatherbench.py)        │
│  - WeatherBench2 标准评估                             │
│  - 多维度指标计算                                      │
│  - 可视化和报告生成                                    │
└─────────────────────────────────────────────────────┘
```

### 设计原则

1. **模块化**: 每个模块职责单一，便于扩展和维护
2. **标准化**: 遵循 WeatherBench2 标准，保证评测一致性
3. **灵活性**: 支持多种模型和配置，便于实验
4. **可复现**: 固定随机种子，保存完整配置
5. **高效性**: 支持单点/空间数据切换，优化训练效率

### 数据流

```python
ERA5 原始数据 (Zarr)
    ↓ data_loader.py
标准化特征矩阵 (numpy/torch)
    ↓ create_samples()
时间序列样本 (X: [batch, seq_len, features], y: [batch, forecast_len, targets])
    ↓ DataLoader
批次数据
    ↓ model.forward()
预测结果
    ↓ inverse_transform
物理量预测
    ↓ evaluate_weatherbench.py
评估指标和可视化
```

---

## 数据和特征工程

### ERA5 数据集

**ERA5** 是 ECMWF 第五代全球大气再分析数据集，是天气预测研究的标准数据源。

**基本信息：**
- 时间范围: 1959 年至今
- 时间分辨率: 1 小时（本项目使用 6 小时）
- 空间分辨率: 0.25° × 0.25°（本项目使用 5.625° × 5.625°，即 64×32 网格）
- 变量: 100+ 个大气、陆面、海洋变量

### 核心气象变量

#### 1. 温度场 ⭐⭐⭐⭐⭐

```python
temperature_variables = [
    '2m_temperature',        # 2米温度 (K) - 最常用
    'surface_temperature',   # 地表温度 (K)
    'temperature',           # 不同气压层温度 (K)
]
```

**物理意义：**
- 2 米温度：标准观测高度，影响人类活动
- 地表温度：地面辐射和能量交换
- 多层温度：揭示大气垂直结构和稳定度

**预测难点：**
- 日变化大（昼夜温差）
- 受地形影响显著
- 极端高温/低温预测困难

#### 2. 风场 ⭐⭐⭐⭐⭐

```python
wind_variables = [
    '10m_u_component_of_wind',  # 10米U风（东西向）(m/s)
    '10m_v_component_of_wind',  # 10米V风（南北向）(m/s)
    'u_component_of_wind',      # 不同层U风 (m/s)
    'v_component_of_wind',      # 不同层V风 (m/s)
]

# 衍生特征
wind_speed = np.sqrt(u**2 + v**2)
wind_direction = np.arctan2(v, u) * 180 / np.pi
vorticity = dv/dx - du/dy      # 涡度：旋转强度
divergence = du/dx + dv/dy     # 散度：辐合/辐散
```

**物理意义：**
- 风场描述大气运动
- 决定天气系统移动方向和速度
- 涡度正值表示气旋性旋转

#### 3. 气压场 ⭐⭐⭐⭐⭐

```python
pressure_variables = [
    'mean_sea_level_pressure',  # 海平面气压 (Pa)
    'surface_pressure',         # 地表气压 (Pa)
    'geopotential',             # 位势高度 (m²/s²)
]

# 衍生特征
geopotential_height = geopotential / 9.81  # 位势高度 (m)
pressure_gradient = np.gradient(pressure)  # 气压梯度
```

**物理意义：**
- 气压决定风的强度和方向
- 位势高度 500 hPa 是最经典的天气预报指标
- 高压系统通常晴朗，低压系统多云雨

#### 4. 湿度和降水 ⭐⭐⭐⭐⭐

```python
moisture_variables = [
    'specific_humidity',           # 比湿 (kg/kg)
    'relative_humidity',           # 相对湿度 (%)
    'total_precipitation',         # 总降水 (m)
    'total_column_water_vapour',   # 整层水汽 (kg/m²)
]
```

**物理意义：**
- 比湿：单位质量空气中的水汽质量
- 相对湿度：当前水汽含量与饱和水汽的比值
- 整层水汽：降水潜力的重要指标

**预测难点：**
- 降水是最难预测的变量
- 时空分布极不均匀
- 强对流天气难以捕捉

### 特征工程策略

#### 1. 时间特征（周期性编码）

```python
# 小时周期
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)

# 月份周期
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)

# 年内周期
day_of_year_sin = np.sin(2 * np.pi * day_of_year / 365)
day_of_year_cos = np.cos(2 * np.pi * day_of_year / 365)
```

**原因：** 使用正弦/余弦编码保持周期连续性（例如 23:00 和 00:00 实际很接近）

#### 2. 空间梯度

```python
# 温度梯度
grad_T_lon = dT/dx  # 经向梯度
grad_T_lat = dT/dy  # 纬向梯度

# 气压梯度
grad_P = np.sqrt((dP/dx)**2 + (dP/dy)**2)  # 气压梯度幅度
```

**物理意义：** 梯度反映变量的空间变化率，与锋面、天气系统有关

#### 3. 滑动窗口统计

```python
windows = [6, 12, 24, 48]  # 小时

for w in windows:
    features[f'T_mean_{w}h'] = rolling_mean(T, window=w)
    features[f'T_std_{w}h'] = rolling_std(T, window=w)
    features[f'T_trend_{w}h'] = (T(t) - T(t-w)) / w
```

**作用：** 捕捉不同时间尺度的变化趋势

#### 4. 数据标准化

```python
from sklearn.preprocessing import StandardScaler

# 对每个变量分别标准化
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# 保存 scaler 用于预测时反标准化
joblib.dump(scaler, 'scaler.pkl')
```

**重要性：** 不同变量量纲差异大，标准化是模型训练的关键

---

## 模型架构详解

### 1. Linear Regression (Baseline)

**适用场景：** 快速 baseline，验证数据流

```python
class LinearRegressionModel:
    def __init__(self, input_size, output_size):
        self.model = LinearRegression()
    
    def fit(self, X, y):
        # X: (n_samples, seq_len, features) → 展平
        X_flat = X.reshape(X.shape[0], -1)
        y_flat = y.reshape(y.shape[0], -1)
        self.model.fit(X_flat, y_flat)
```

**优点：**
- 训练极快（<1 分钟）
- 可解释性强
- 稳定可靠

**缺点：**
- 无法捕捉非线性关系
- 不考虑时间依赖
- 性能有限

### 2. LSTM (推荐)

**架构设计：**

```python
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_length):
        super().__init__()
        
        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # 全连接输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, output_length)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 使用最后时间步的输出
        last_out = lstm_out[:, -1, :]
        
        # 预测未来多步
        output = self.fc(last_out)
        return output
```

**设计要点：**

1. **输入序列长度：** 默认 12 步（2 天，每步 6 小时）
   - 太短：无法捕捉天气演变
   - 太长：计算开销大，可能过拟合

2. **隐藏层大小：** 默认 128
   - 小模型：64（快速实验）
   - 中等模型：128-256（推荐）
   - 大模型：512+（需要更多数据）

3. **层数：** 默认 2-3 层
   - 单层：可能欠拟合
   - 多层：增强表达能力，但易过拟合

4. **Dropout：** 0.2-0.3
   - 防止过拟合
   - 在 LSTM 层间和全连接层使用

**适用场景：**
- 单点或区域平均预测
- 短期到中期预报（6h - 72h）
- 时间序列特征明显

**优点：**
- 有效捕捉时间依赖
- 训练效率高
- 性能稳定

**缺点：**
- 难以处理完整空间场
- 长序列梯度消失问题

### 3. Transformer

**架构设计：**

```python
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_size):
        super().__init__()
        
        # 输入嵌入
        self.embedding = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出层
        self.fc = nn.Linear(d_model, output_size)
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        # 使用最后时间步
        output = self.fc(x[:, -1, :])
        return output
```

**设计要点：**

1. **d_model（模型维度）：** 256-512
   - 影响模型容量和计算开销

2. **nhead（注意力头数）：** 4-8
   - 多头注意力捕捉不同模式
   - 必须被 d_model 整除

3. **num_layers（Transformer 层数）：** 4-6
   - 更深的网络捕捉更复杂的模式

4. **位置编码：** 关键！
   - Transformer 本身无法感知位置
   - 使用正弦/余弦位置编码

**适用场景：**
- 中长期预报（3-10 天）
- 需要捕捉长距离依赖
- 数据量充足

**优点：**
- 并行计算效率高
- 能捕捉长程依赖
- 注意力机制可解释

**缺点：**
- 需要更多数据
- 计算和内存开销大
- 超参数敏感

### 4. CNN-LSTM（未来扩展）

**设计思路：** 结合 CNN 的空间特征提取和 LSTM 的时序建模

```python
class CNNLSTMModel(nn.Module):
    def __init__(self, input_channels, hidden_channels, lstm_hidden):
        super().__init__()
        
        # CNN 提取空间特征
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # LSTM 处理时序
        self.lstm = nn.LSTM(128, lstm_hidden, num_layers=2, batch_first=True)
        
        # 解码器恢复空间分辨率
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(lstm_hidden, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
```

**适用场景：**
- 完整空间场预测
- Nowcasting（0-6 小时）
- 需要保持空间结构

### 5. U-Net（未来扩展）

**设计思路：** 编码器-解码器结构，skip connections 保持细节

**适用场景：**
- 降水、云等局地强对流预测
- 保持高空间分辨率
- 图像到图像的预测

---

## WeatherBench2 评测标准

### 什么是 WeatherBench2？

**WeatherBench 2** 是下一代数据驱动全球天气预测模型的基准测试框架，由 Google Research 和 ECMWF 联合开发。

**核心目标：**
1. 提供标准化的评测框架
2. 对比数据驱动方法和传统数值天气预报（NWP）
3. 建立公平的性能比较基准

**论文：** [WeatherBench 2 (arXiv:2308.15560)](https://arxiv.org/abs/2308.15560)

### 预测任务

**输入：** 初始时刻的大气状态  
**输出：** 未来不同 lead time 的气象变量

```
初始状态 (t=0) → 预测: t+6h, t+12h, t+24h, ..., t+10天
```

### 核心评测变量

#### 最重要的 3 个变量

| 变量 | 层级 | 重要性 | 说明 |
|------|------|--------|------|
| **geopotential** | **500 hPa** | ⭐⭐⭐⭐⭐ | 中层大气，最经典的预报指标 |
| **2m_temperature** | 地表 | ⭐⭐⭐⭐⭐ | 最贴近日常生活的预报 |
| **geopotential** | 850 hPa | ⭐⭐⭐⭐ | 低层大气，影响天气系统 |

#### 完整变量列表

**3D 变量**（有压强层 level）：
- `geopotential` - 位势高度
- `temperature` - 温度
- `u_component_of_wind` - U 风分量
- `v_component_of_wind` - V 风分量
- `specific_humidity` - 比湿

**2D 变量**（地表/单层）：
- `2m_temperature` - 2 米温度
- `10m_u_component_of_wind` - 10 米 U 风
- `10m_v_component_of_wind` - 10 米 V 风
- `10m_wind_speed` - 10 米风速
- `mean_sea_level_pressure` - 海平面气压
- `total_precipitation_24hr` - 24 小时降水

### 数据格式要求

#### Forecast 输出格式

```python
<xarray.Dataset>
Dimensions:
  - time (init_time): datetime64[ns]           # 初始化时间
  - prediction_timedelta: timedelta64[ns]      # [0h, 6h, 12h, ..., 240h]
  - latitude: float64 [-90, 90]
  - longitude: float64 [0, 360]                # ⚠️ 必须 0-360 度！
  - level: int32 [500, 700, 850] hPa

Data variables:
  - geopotential: (time, prediction_timedelta, level, longitude, latitude)
  - 2m_temperature: (time, prediction_timedelta, longitude, latitude)
```

#### 保存代码示例

```python
import xarray as xr
import pandas as pd
import numpy as np

forecast = xr.Dataset(
    {
        'geopotential': (['time', 'prediction_timedelta', 'level', 
                         'longitude', 'latitude'], predictions),
    },
    coords={
        'time': pd.date_range('2020-01-01', periods=N, freq='6H'),
        'prediction_timedelta': pd.timedelta_range('0H', '240H', freq='6H'),
        'level': [500, 700, 850],
        'latitude': np.linspace(-87.19, 87.19, 32),
        'longitude': np.linspace(0, 354.38, 64),  # 0-360 度！
    }
)

forecast.to_zarr('my_forecast.zarr')
```

### 评测指标

#### 1. RMSE (Root Mean Square Error)

```python
RMSE = sqrt(mean((prediction - truth)^2))
```

**最重要的指标！** 衡量预测误差的均方根。

**性能目标：**

| Lead Time | 500hPa Geo RMSE | 2m Temp RMSE |
|-----------|-----------------|---------------|
| 1 day     | < 50 m         | < 1.5 K       |
| 3 days    | < 150 m        | < 2.5 K       |
| 5 days    | < 300 m        | < 3.5 K       |
| 10 days   | < 600 m        | < 5.0 K       |

#### 2. ACC (Anomaly Correlation Coefficient)

```python
ACC = correlation(prediction_anomaly, truth_anomaly)
```

**黄金标准！** 数值天气预报的传统指标，范围 [-1, 1]，1 表示完美预测。

**计算方法：**
```python
pred_anomaly = prediction - climatology
truth_anomaly = truth - climatology
ACC = correlation(pred_anomaly, truth_anomaly)
```

#### 3. Bias

```python
Bias = mean(prediction - truth)
```

衡量系统性偏差：
- 正值 = 高估
- 负值 = 低估

#### 4. Skill Score

```python
SS = 1 - (RMSE_model / RMSE_baseline)
```

- SS > 0：优于 baseline
- SS < 0：差于 baseline

### 常见陷阱

#### ❌ 错误示例

```python
# 1. 经度系统错误
longitude = np.linspace(-180, 180, 64)  # ❌ WeatherBench2 使用 0-360

# 2. 变量命名错误
't2m'              # ❌ 应该是 '2m_temperature'
'geopotential_500' # ❌ 应该用 level 维度

# 3. 单位错误
precipitation_mm = 50  # ❌ 应该是 meters: 0.050
```

#### ✅ 正确示例

```python
# 1. 经度：0-360 度
longitude = np.linspace(0, 360, 64, endpoint=False)

# 2. 标准变量名
'2m_temperature'
'geopotential'

# 3. 正确单位
temperature_K = 273.15 + 20  # Kelvin
precipitation_m = 50 / 1000   # meters
```

### 标准数据集路径

```python
# ERA5 观测数据 (ground truth)
obs_path = 'gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr'

# 气候态 (用于 ACC 计算)
clim_path = 'gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_conservative.zarr'

# ECMWF HRES (强 baseline)
hres_path = 'gs://weatherbench2/datasets/hres/2016-2022-0012-64x32_equiangular_conservative.zarr'
```

---

## 训练策略和优化

### 损失函数选择

#### 1. MSE Loss（默认）

```python
criterion = nn.MSELoss()
```

**优点：**
- 数学性质好，易优化
- 对大误差惩罚重

**缺点：**
- 倾向于预测平均值
- 预测结果可能过于平滑

#### 2. MAE Loss

```python
criterion = nn.L1Loss()
```

**优点：**
- 对异常值更鲁棒
- 预测中位数

#### 3. Huber Loss（推荐）

```python
criterion = nn.SmoothL1Loss()
```

**优点：**
- 结合 MSE 和 MAE 优点
- 对异常值鲁棒，对正常值敏感

#### 4. 自定义加权损失

```python
class WeightedMSELoss(nn.Module):
    def __init__(self, temporal_weights):
        super().__init__()
        self.temporal_weights = temporal_weights  # 例如 [1.0, 0.9, 0.8, 0.7]
    
    def forward(self, pred, target):
        loss = (pred - target) ** 2
        # 对不同预测时间步加权
        loss = loss * self.temporal_weights.view(1, -1, 1)
        return loss.mean()
```

**应用：** 强调短期预测（近期更重要）

### 优化器和学习率

#### 优化器选择

```python
# Adam（最常用）
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# AdamW（推荐）
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

#### 学习率调度

```python
# ReduceLROnPlateau（根据验证集自动调整）
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',
    factor=0.5,      # 每次降低为原来的 0.5
    patience=5,      # 5 个 epoch 不改善就降低
    verbose=True
)

# CosineAnnealingLR（余弦退火）
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,       # 周期
    eta_min=1e-6     # 最小学习率
)
```

### 正则化技术

1. **Dropout：** 0.2-0.3（防止过拟合）
2. **Weight Decay：** 1e-5 到 1e-4（L2 正则化）
3. **Gradient Clipping：** max_norm=1.0（防止梯度爆炸）
4. **Early Stopping：** patience=10-20（避免过拟合）

### 训练技巧

1. **Warm-up：** 前几个 epoch 使用较小学习率
2. **Mixed Precision：** 使用 torch.cuda.amp 加速训练
3. **Gradient Accumulation：** 模拟大 batch size
4. **数据增强：** 空间翻转、旋转（适用于 CNN）

---

## 评估指标体系

### 基础指标

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MAE = mean_absolute_error(y_true, y_pred)
RMSE = np.sqrt(mean_squared_error(y_true, y_pred))
R2 = r2_score(y_true, y_pred)
```

### 天气特定指标

#### 1. ACC (Anomaly Correlation Coefficient)

```python
def calculate_acc(pred, target, climatology):
    pred_anomaly = pred - climatology
    target_anomaly = target - climatology
    
    numerator = np.sum(pred_anomaly * target_anomaly)
    denominator = np.sqrt(np.sum(pred_anomaly**2) * np.sum(target_anomaly**2))
    
    return numerator / denominator
```

#### 2. Skill Score

```python
def skill_score(rmse_model, rmse_baseline):
    return 1 - (rmse_model / rmse_baseline)
```

#### 3. 空间指标

```python
# 空间 RMSE 分布
def spatial_rmse(pred, target):
    # pred, target: (time, lat, lon)
    rmse_map = np.sqrt(np.mean((pred - target)**2, axis=0))
    return rmse_map  # (lat, lon)
```

### 分层评估

```python
# 按 lead time 分层
for lead_time in [6, 12, 24, 48, 72]:
    rmse = calculate_rmse(pred[:, lead_time], target[:, lead_time])
    print(f"Lead {lead_time}h RMSE: {rmse:.4f}")

# 按区域分层
regions = {
    'tropics': (lat > -20) & (lat < 20),
    'mid_latitudes': (lat > 30) | (lat < -30),
}
for region_name, mask in regions.items():
    rmse = calculate_rmse(pred[mask], target[mask])
    print(f"{region_name} RMSE: {rmse:.4f}")
```

---

## 最新研究进展

### 1. Pangu-Weather (华为, Nature 2023)

**特点：**
- 3D Earth-Specific Transformer
- 分层处理不同气压层

**性能：**
- 优于 ECMWF IFS 在 3-10 天预报
- 1 小时分辨率全球预报仅需 10 秒

**训练：**
- 43 年 ERA5 数据
- 4 V100 GPU，16 天

**论文：** [Nature 2023](https://www.nature.com/articles/s41586-023-06185-3)

### 2. FourCastNet (NVIDIA, 2022)

**特点：**
- Adaptive Fourier Neural Operator
- 频域处理

**性能：**
- 0.25° 分辨率
- 1 周预报 < 10 秒

**优势：**
- 极快的推理速度
- 适合实时应用

**论文：** [arXiv:2202.11214](https://arxiv.org/abs/2202.11214)

### 3. GraphCast (DeepMind, Science 2023)

**特点：**
- Graph Neural Network
- 多尺度图表示

**性能：**
- 10 天预报优于 HRES
- 90% 指标超越传统 NWP

**创新：**
- 球面网格的图表示
- 消息传递机制

**论文：** [Science 2023](https://www.science.org/doi/10.1126/science.adi2336)

### 4. ClimaX (Microsoft, 2023)

**特点：**
- Foundation model for weather
- 预训练 + 微调范式

**数据：**
- 多种气候数据集
- 跨分辨率、跨变量

**优势：**
- 泛化能力强
- 适应不同下游任务

**论文：** [arXiv:2301.10343](https://arxiv.org/abs/2301.10343)

### 技术趋势

1. **Transformer 架构**：成为主流
2. **物理约束**：融入守恒定律
3. **多模态**：结合卫星、雷达数据
4. **集成预报**：生成概率预测
5. **Foundation Models**：预训练大模型

---

## 参考资源

### 官方文档

- **ERA5**: https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5
- **WeatherBench2**: https://weatherbench2.readthedocs.io/
- **排行榜**: https://sites.research.google/weatherbench

### 代码库

- **WeatherBench**: https://github.com/pangeo-data/WeatherBench
- **WeatherBench2**: https://github.com/google-research/weatherbench2
- **Xarray**: https://xarray.dev/
- **PyTorch**: https://pytorch.org/

### 重要论文

1. WeatherBench 2: [arXiv:2308.15560](https://arxiv.org/abs/2308.15560)
2. Pangu-Weather: [Nature 2023](https://www.nature.com/articles/s41586-023-06185-3)
3. GraphCast: [Science 2023](https://www.science.org/doi/10.1126/science.adi2336)
4. FourCastNet: [arXiv:2202.11214](https://arxiv.org/abs/2202.11214)
5. ClimaX: [arXiv:2301.10343](https://arxiv.org/abs/2301.10343)

---

## 总结

### 关键要点

1. **数据质量 > 模型复杂度**：特征工程和数据预处理至关重要
2. **从简单到复杂**：先建立 baseline，逐步提升
3. **标准化评测**：遵循 WeatherBench2 标准保证可比性
4. **物理约束**：融入气象学知识提升模型可信度
5. **持续迭代**：关注最新研究，不断改进

### 下一步方向

1. **完整空间预测**：实现 CNN-LSTM 和 U-Net
2. **多变量联合**：同时预测温度、风场、降水
3. **集成预报**：生成概率预测和不确定性估计
4. **物理约束**：加入能量守恒、质量守恒等物理定律
5. **实时应用**：优化推理速度，部署到生产环境
