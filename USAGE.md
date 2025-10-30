# 使用指南

## 快速开始

### 1. 安装依赖

```
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. 数据集说明

**重要**：尽管数据集路径名为 `1959-2022-6h-64x32`，但实际数据只到 **2021-12-31**。
可用时间范围：`1959-01-01` 到 `2021-12-31`

## 常用预测变量

| 变量名 | 说明 | 维度 |
|--------|------|------|
| `2m_temperature` | 2米温度 | (time, lat, lon) |
| `geopotential` | 位势高度 | (time, level, lat, lon) |
| `10m_u_component_of_wind` | 10米U风 | (time, lat, lon) |
| `10m_v_component_of_wind` | 10米V风 | (time, lat, lon) |
| `specific_humidity` | 比湿 | (time, level, lat, lon) |

完整变量列表见 [WeatherBench2](https://weatherbench2.readthedocs.io/)

## 模型选择

### 单变量预测
- **推荐**：`lstm`, `convlstm`
- **基线**：`lr`
- **避免**：`transformer`（易退化）

### 多变量预测
- **推荐**：`convlstm`（保留空间结构）
- **快速基线**：`lr_multi`（每变量一个模型）
- **空间特征**：`cnn`
- **避免**：`lr`, `lstm`, `transformer`

## 快速运行脚本

使用 `scripts/` 目录中的脚本（每个脚本包含训练+预测+评估）：

```bash
# Linear Regression Multi
./scripts/run_lr_multi.sh

# LSTM
./scripts/run_lstm.sh

# CNN
./scripts/run_cnn.sh

# ConvLSTM（推荐）
./scripts/run_convlstm.sh

# Diffusion 基础
./scripts/run_diffusion.sh

# Diffusion + Ensemble（概率预测）
./scripts/run_diffusion_ensemble.sh
```

**修改参数**：编辑对应脚本顶部的参数配置部分

## 输出文件

训练后会在 `outputs/**` 生成：


## Diffusion 模型（概率预测）

### 基础训练

```bash
# 标准训练（确定性评估）
python train_diffusion.py \
    --variables 2m_temperature \
    --time-slice 2019-01-01:2020-12-31 \
    --epochs 200 \
    --batch-size 16 \
    --lr 5e-5

# Ensemble 评估（概率预测，推荐）⭐
python train_diffusion.py \
    --variables 2m_temperature \
    --time-slice 2019-01-01:2020-12-31 \
    --epochs 200 \
    --batch-size 16 \
    --lr 5e-5 \
    --enable-ensemble-eval \
    --num-ensemble-members 20 \
    --num-inference-steps 100
```

### train_diffusion.py 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--variables` | `2m_temperature` | 预测变量（逗号分割） |
| `--time-slice` | `2020-01-01:2020-12-31` | 训练时间范围 |
| `--epochs` | 150 | 训练轮数（Diffusion 需要更多） |
| `--batch-size` | 16 | 批次大小 |
| `--lr` | `5e-5` | 学习率（比其他模型低） |
| `--base-channels` | 64 | UNet 基础通道数 |
| `--num-diffusion-steps` | 1000 | Diffusion 步数 |
| `--beta-schedule` | `cosine` | 噪声调度（linear/cosine） |
| **Ensemble 相关** | | |
| `--enable-ensemble-eval` | False | 启用 ensemble 评估 |
| `--num-ensemble-members` | 10 | Ensemble 成员数量 |
| `--num-inference-steps` | 50 | 推理步数（影响质量和速度） |

### 输出文件

**标准输出**：
- `best_model.pth` - 模型权重
- `config.json` - 训练配置
- `metrics.json` - 评估指标（RMSE, MAE）
- `spatial_predictions_*.png` - 空间预测图
- `point_predictions.png` - 时间序列图
- `training_history.png` - 训练曲线

**Ensemble 评估额外输出**（`--enable-ensemble-eval`）：
- `ensemble_predictions.npy` - Ensemble 预测数据
- `ensemble_spread.png` - Ensemble 成员分布
- `ensemble_uncertainty_map.png` - 不确定性地图
- `crps_vs_leadtime.png` - CRPS vs RMSE 对比
- `metrics.json` 包含 `test_probabilistic` 部分（CRPS, Spread-Skill Ratio）

### 评估指标

**确定性指标**（标准输出）：
- RMSE - 均方根误差
- MAE - 平均绝对误差

**概率指标**（Ensemble 评估）：
- **CRPS** - Continuous Ranked Probability Score（越小越好）
- **Spread-Skill Ratio** - 理想值 ≈ 1.0
  - < 1.0: Under-dispersive（过度自信）
  - > 1.0: Over-dispersive（不够自信）
- **Ensemble Mean RMSE** - 与确定性模型对比

### 重新评估已训练模型

```bash
# 使用已训练模型生成 ensemble 预测
python evaluate_diffusion_ensemble.py \
    --model-path outputs/diffusion_xxx/best_model.pth \
    --num-ensemble-members 20 \
    --num-inference-steps 100
```

### Diffusion 常见问题

**Q: 为什么 RMSE 比其他模型高？**  
A: Diffusion 是概率模型，单次推理不稳定。应该使用 Ensemble 评估（`--enable-ensemble-eval`），Ensemble Mean RMSE 会更好。

**Q: 训练很慢怎么办？**  
A: Diffusion 训练确实较慢。优化方法：
- 减少 `--epochs`（最少 100）
- 减小 `--base-channels`（64 → 32）
- 减小 `--batch-size`
- 减小时间范围

**Q: 何时启用 Ensemble 评估？**  
A: 
- 最终评估时：启用（完整的概率评估）
- 快速迭代时：不启用（节省时间）

**Q: Ensemble 成员数量如何选择？**  
A: 
- 快速测试：5-10 个
- 标准评估：20 个
- 完整评估：50+ 个（GenCast 级别）