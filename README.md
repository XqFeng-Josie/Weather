# 天气预测系统 (Weather Prediction System)

基于 ERA5 数据的深度学习天气预测系统，支持多种模型和完整的训练评估流程。

## 🚀 快速开始

### 1. 环境安装

```bash
# 安装依赖
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 如果 PyTorch 有问题，手动安装 CPU 版本
# pip uninstall torch -y
# pip install torch --index-url https://download.pytorch.org/whl/cpu

# 验证环境
python check_environment.py
```

### 2. 第一次运行

```bash
# 测试 pipeline
python test_pipeline.py

# 一键运行完整流程（训练+预测+评估）
bash run_full_pipeline.sh lstm
```

### 3. 查看结果

```bash
# 查看训练指标
cat outputs/lstm_*/metrics.json

# 查看可视化图表
open outputs/lstm_*/training_history.png
open outputs/lstm_*/evaluation/rmse_by_leadtime.png
```

## 📁 项目结构

```
Weather/
├── src/                        # 核心代码
│   ├── data_loader.py          # ERA5 数据加载
│   ├── models.py               # 模型定义（LR, LSTM, Transformer 等）
│   └── trainer.py              # 统一训练器
├── train.py                    # 训练脚本
├── predict.py                  # 预测脚本
├── evaluate_weatherbench.py    # WeatherBench2 评估
├── run_full_pipeline.sh        # 一键运行脚本
├── test_pipeline.py            # 功能测试
├── check_environment.py        # 环境检查
├── weather_data_analysis.py    # ERA5 数据分析工具
├── weather_analysis.ipynb      # Jupyter 分析 notebook
└── outputs/                    # 实验输出目录
```

## 🎯 使用指南

### 训练模型

```bash
# 基础训练（单点数据，快速测试）
python train.py --model lstm --single-point --epochs 50

# 完整训练（全空间数据）
python train.py --model lstm --epochs 100

# 自定义超参数
python train.py \
    --model lstm \
    --hidden-size 256 \
    --num-layers 4 \
    --dropout 0.3 \
    --lr 0.001 \
    --batch-size 32 \
    --time-slice "2020-01-01:2020-12-31"
```

**支持的模型：**
- `lr` - Linear Regression（最快，baseline）
- `lstm` - LSTM（推荐，时间序列预测）
- `transformer` - Transformer（强大但慢）

### 生成预测

```bash
python predict.py \
    --model-path outputs/lstm_xxx/best_model.pth \
    --time-slice "2021-01-01:2021-12-31" \
    --output predictions.npz
```

### 评估模型

```bash
# 基础评估
python evaluate_weatherbench.py \
    --pred predictions.npz \
    --output-dir evaluation_results

# 与 baseline 对比
python evaluate_weatherbench.py \
    --pred predictions.npz \
    --compare-baseline baseline_predictions.npz
```

## 📊 数据分析

### 使用 Jupyter Notebook（推荐）

```bash
jupyter notebook weather_analysis.ipynb
```

**包含功能：**
- ✅ 数据加载和基本信息
- ✅ 统计分析（均值、标准差、分位数）
- ✅ 空间分布可视化
- ✅ 时间序列分析（含趋势）
- ✅ 季节性分析
- ✅ 区域对比分析
- ✅ 数据导出和报告生成


本系统完全兼容 WeatherBench2 评测标准：

- ✅ ERA5 数据加载
- ✅ 标准变量（geopotential, temperature, wind 等）
- ✅ 多 lead time 预测
- ✅ netCDF/Zarr 输出格式
- ✅ 标准评估指标（RMSE, ACC, Bias 等）

**主要评测变量：**
- `geopotential` (500 hPa) - 中层大气位势高度
- `2m_temperature` - 地表 2 米温度
- `10m_wind_speed` - 地表风速
- `total_precipitation` - 降水量

详见 `TECHNICAL_GUIDE.md` 了解 WeatherBench2 详细信息。


## 📧 常用参数

### train.py 参数

```bash
--model              # 模型类型: lr, lstm, transformer
--time-slice         # 训练数据时间范围: "2020-01-01:2020-12-31"
--epochs             # 训练轮数: 默认 50
--batch-size         # 批大小: 默认 32
--single-point       # 使用单点数据（快速测试）
--hidden-size        # 隐藏层大小: 默认 128
--num-layers         # 层数: 默认 2
--dropout            # Dropout 率: 默认 0.2
--lr                 # 学习率: 默认 0.001
--exp-name           # 实验名称: 默认自动生成
```

### predict.py 参数

```bash
--model-path         # 模型路径
--time-slice         # 预测时间范围
--output             # 输出文件名
--format             # 输出格式: netcdf, numpy
```

### evaluate_weatherbench.py 参数

```bash
--pred               # 预测结果文件
--output-dir         # 输出目录
--compare-baseline   # baseline 文件（可选）
```
