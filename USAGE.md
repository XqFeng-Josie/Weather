# 使用指南

## 快速开始

### 1. 安装依赖

```bash
pip install torch xarray zarr gcsfs scikit-learn matplotlib tqdm
```

### 2. 数据集说明

**重要**：尽管数据集路径名为 `1959-2022-6h-64x32`，但实际数据只到 **2021-12-31**。

可用时间范围：
- 训练：`1959-01-01` 到 `2021-12-31`
- 推荐训练集：`2020-01-01:2020-12-31`（1464时间步）
- 推荐测试集：`2021-01-01:2021-12-31`（1460时间步）

### 3. 基础训练

```bash
# 单变量预测（默认：2m_temperature）
python train.py --model convlstm --time-slice 2020-01-01:2020-12-31 --epochs 30

# 多变量预测（逗号分割）
python train.py --model lr_multi --variables 2m_temperature,geopotential --epochs 20

# 自定义参数
python train.py \
    --model lstm \
    --variables 2m_temperature \
    --time-slice 2020-01-01:2020-06-30 \
    --epochs 50 \
    --batch-size 16 \
    --hidden-size 128 \
    --lr 0.0005
```

### 4. 生成预测

```bash
# 使用训练好的模型预测（自动读取config中的变量列表）
python predict.py \
    --model-path outputs/convlstm_xxx/best_model.pth \
    --output predictions.npz

# 指定时间段
python predict.py \
    --model-path outputs/lstm_2020/best_model.pth \
    --time-slice 2021-01-01:2021-12-31 \
    --output predictions_2021.npz
```

### 5. 评估

```bash
# WeatherBench2标准评估
python evaluate_weatherbench.py --pred predictions.npz

# 对比baseline
python evaluate_weatherbench.py \
    --pred outputs/convlstm_xxx/predictions.npz \
    --compare-baseline outputs/lr_multi_xxx/predictions.npz
```

## 命令行参数

### train.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `lr` | 模型类型：lr, lr_multi, lstm, transformer, cnn, convlstm |
| `--variables` | `2m_temperature` | 预测变量（逗号分割） |
| `--time-slice` | `2020-01-01:2020-12-31` | 训练时间范围 |
| `--input-length` | `12` | 输入序列长度（时间步） |
| `--output-length` | `4` | 预测序列长度（时间步） |
| `--epochs` | `30` | 训练轮数 |
| `--batch-size` | `32` | 批次大小 |
| `--lr` | `0.001` | 学习率 |
| `--hidden-size` | `128` | 隐藏层大小 |
| `--num-layers` | `2` | 网络层数 |
| `--dropout` | `0.2` | Dropout比率 |
| `--early-stop` | `10` | 早停轮数 |
| `--gradient-accumulation-steps` | `1` | 梯度累积步数（减少内存） |
| `--exp-name` | 自动生成 | 实验名称 |

### predict.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-path` | 必填 | 模型文件路径 |
| `--config-path` | 自动推断 | 配置文件路径 |
| `--data-path` | WeatherBench2 | 数据路径 |
| `--time-slice` | `2021-01-01:2021-12-31` | 预测时间范围 |
| `--output` | `predictions.npz` | 输出文件 |

### evaluate_weatherbench.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pred` | 必填 | 预测文件（.npz或.nc） |
| `--compare-baseline` | 无 | 对比baseline文件 |
| `--output-dir` | `./eval_results` | 输出目录 |

## 常用变量

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

## 批量训练

使用提供的脚本：

```bash
# 编辑 run_examples.sh 设置变量
VARIABLES="2m_temperature,geopotential"

# 运行所有模型
bash run_examples.sh

# 完整流程（训练+预测+评估）
bash run_full_pipeline.sh
```

## 输出文件

训练后会在 `outputs/<exp_name>/` 生成：

```
outputs/
└── lr_multi_20251028_165014/
    ├── best_model.pth                      # 模型文件
    ├── config.json                         # 配置
    ├── metrics.json                        # 指标
    ├── predictions_2m_temperature.png      # 每个变量的预测图
    ├── predictions_geopotential.png
    ├── predictions_all_variables.png       # 所有变量汇总
    ├── rmse_vs_leadtime.png               # RMSE vs 时间步
    ├── y_test.npy                         # 测试集真实值
    └── y_test_pred.npy                    # 测试集预测值
```

## 常见问题

### 1. 内存不足（ConvLSTM OOM）

**ConvLSTM专用优化**（自动应用）：
- hidden_channels自动限制≤64（其他模型不受影响）

**推荐配置**：
```bash
# 低内存GPU (<8GB)
python train.py --model convlstm --batch-size 8 --gradient-accumulation-steps 4

# 中等GPU (8-16GB)
python train.py --model convlstm --batch-size 16 --gradient-accumulation-steps 2

# 高端GPU (>16GB)
python train.py --model convlstm --batch-size 32
```

**其他方法**：
- 减小 `--batch-size`（最直接）
- 增加 `--gradient-accumulation-steps`（模拟大batch）
- 减小 `--input-length`
- 使用单变量而非多变量
- 减小 `--time-slice` 时间范围

**详细说明**：见 [CONVLSTM_OPTIMIZATION.md](CONVLSTM_OPTIMIZATION.md)

### 2. 训练过慢
- 使用 `lr_multi` 作为快速baseline
- 减小模型：`--hidden-size 64 --num-layers 1`
- 减少 `--epochs`

### 3. 模型退化（接近均值预测）
- **Transformer**：自动使用更小的模型（单变量时）
- **LSTM**：增加dropout `--dropout 0.3`
- **多变量**：改用 `convlstm` 或 `lr_multi`

### 4. 预测时间不匹配
- `predict.py` 会自动从 `config.json` 读取变量列表
- 确保 `--time-slice` 在可用范围内（1959-01-01 到 2021-12-31）
- ⚠️ 数据集名称虽为"1959-2022"，但实际只到2021年底

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

## 高级用法

### 自定义数据路径

```bash
# 使用本地数据
python train.py \
    --data-path /path/to/local/era5.zarr \
    --model convlstm
```

### 恢复训练

训练会自动保存最佳模型，可以手动加载继续训练：

```python
# 在代码中加载
trainer = WeatherTrainer.load_checkpoint('outputs/xxx/best_model.pth')
```
