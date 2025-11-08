# 使用指南

## 环境配置

### 1. 创建虚拟环境

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 2. 安装依赖

```bash
# 基础依赖（传统模型：CNN, LSTM, ConvLSTM等）
pip install -r requirements.txt

# WeatherDiff额外依赖（如果使用WeatherDiff模块）
pip install -r requirements_weatherdiff.txt
```

### 3. 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import xarray; print('xarray OK')"
python -c "from diffusers import AutoencoderKL; print('diffusers OK')"  # 如果安装了WeatherDiff依赖
```

## 数据准备

### 数据说明

- **路径**: `gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr`
- **时间范围**: 1959-01-01 到 2021-12-31
- **注意**: 路径名虽为"1959-2022"，实际数据只到2021年底

### 常用时间切片

```bash
# 快速测试（2个月）
--time-slice 2020-01-01:2020-02-28

# 标准训练（1年）
--time-slice 2020-01-01:2020-12-31

# 完整训练（3年）
--time-slice 2018-01-01:2020-12-31
```

## 模型训练

### 传统深度学习模型

#### ConvLSTM（推荐）⭐

```bash
# 使用脚本（推荐）
./scripts/run_convlstm.sh

# 或手动运行
python train.py \
    --model convlstm \
    --variables 2m_temperature \
    --time-slice 2020-01-01:2020-12-31 \
    --epochs 50 \
    --batch-size 16
```

**主要参数**:
- `--model`: 模型类型 (lr, lstm, cnn, convlstm)
- `--variables`: 预测变量（逗号分隔）
- `--time-slice`: 训练时间范围
- `--epochs`: 训练轮数
- `--batch-size`: 批次大小

#### Weather Transformer

```bash
./scripts/run_weather_transformer.sh

# 或手动运行
python train_weather_transformer.py \
    --variables 2m_temperature \
    --time-slice 2020-01-01:2020-12-31 \
    --d-model 128 \
    --n-layers 2 \
    --epochs 50
```

**主要参数**:
- `--d-model`: 模型维度
- `--n-layers`: Transformer层数
- `--n-heads`: 注意力头数
- `--patch-size`: Patch大小

### WeatherDiff 模块

WeatherDiff是基于Stable Diffusion架构的天气预测模块，包含三个主要模型：

#### 1. Pixel U-Net

直接在像素空间进行图像到图像预测。

```bash
./scripts/run_pixel_unet.sh

# 或手动运行
python train_pixel_unet.py \
    --data-path <data.zarr> \
    --variable 2m_temperature \
    --time-slice 2020-01-01:2020-12-31 \
    --input-length 12 \
    --output-length 4 \
    --batch-size 16 \
    --epochs 50
```

#### 2. Latent U-Net（推荐）⭐

在VAE潜空间中预测，显存需求低，训练更稳定。

```bash
# 完整流程（含预处理，推荐）
./scripts/run_latent_unet_full.sh

# 或分步运行
# Step 1: 预处理数据（大数据集必需）
python preprocess_data_for_latent_unet.py \
    --data-path <data.zarr> \
    --variable 2m_temperature \
    --time-slice 2015-01-01:2019-12-31 \
    --target-size 512,512 \
    --output-dir data/preprocessed/my_data

# Step 2: 训练模型
python train_latent_unet.py \
    --preprocessed-data-dir data/preprocessed/my_data \
    --variable 2m_temperature \
    --input-length 12 \
    --output-length 4 \
    --batch-size 16 \
    --epochs 50 \
    --vae-batch-size 4
```

**关键参数**:
- `--preprocessed-data-dir`: 预处理数据目录（推荐，避免内存溢出）
- `--target-size`: 图像尺寸，必须是8的倍数（如512,512）
- `--vae-batch-size`: VAE编码批次大小（控制显存）

#### 3. Diffusion Model

扩散模型，支持概率预测和不确定性量化。

```bash
./scripts/run_diffusion.sh

# 或手动运行
python train_diffusion.py \
    --preprocessed-data-dir data/preprocessed/my_data \
    --variable 2m_temperature \
    --input-length 12 \
    --output-length 4 \
    --batch-size 8 \
    --vae-batch-size 4 \
    --epochs 20 \
    --num-train-timesteps 1000
```

**关键参数**:
- `--num-train-timesteps`: 扩散步数（默认1000）
- `--beta-schedule`: Beta调度（linear/scaled_linear）
- `--vae-batch-size`: VAE批次大小（避免显存溢出）

## 模型预测

### 传统模型预测

```bash
# 预测指定模型
python predict.py \
    --model-dir outputs/<model_name> \
    --data-path <data.zarr> \
    --time-slice 2021-01-01:2021-12-31
```

### WeatherDiff模块预测

```bash
# U-Net预测（像素或潜空间）
python predict_unet.py \
    --mode latent \
    --model-dir outputs/latent_unet \
    --vae-batch-size 4

# Diffusion预测
python predict_diffusion.py \
    --model-dir outputs/diffusion \
    --num-samples 1 \
    --sampling-method ddim \
    --num-inference-steps 50 \
    --vae-batch-size 4
```

**Diffusion参数**:
- `--num-samples`: 生成样本数（>1表示集成预测）
- `--sampling-method`: 采样方法（ddpm/ddim，ddim更快）
- `--num-inference-steps`: 推理步数（越多质量越好但越慢）

## 输出文件

训练后的输出目录结构：

```
outputs/<model_name>/
├── best_model.pt              # 最佳模型
├── config.json                # 配置
├── prediction_metrics.json    # 指标
├── training_history.json      # 训练历史
├── normalizer_stats.pkl       # 归一化参数（WeatherDiff）
├── predictions_data/          # 预测数据
│   ├── y_test.npy            # 真值
│   ├── y_test_pred.npy       # 预测值
│   └── ...
└── *.png                      # 可视化图片
```

## 常见问题

### 1. 内存不足

**问题**: 训练时内存溢出

**解决方案**:
- 减小 `--batch-size`
- 使用预处理数据（`preprocess_data_for_latent_unet.py`）
- 使用Latent U-Net而非Pixel U-Net
- 缩短时间范围

### 2. GPU显存不足

**问题**: CUDA out of memory

**解决方案**:
- 减小 `--batch-size`
- 减小 `--vae-batch-size`（WeatherDiff模块）
- 使用更小的图像尺寸
- 使用梯度累积

### 3. 训练速度慢

**问题**: 训练非常慢

**解决方案**:
- 使用预处理数据（WeatherDiff模块）
- 增加 `--num-workers`（数据加载线程）
- 使用DDIM采样（Diffusion推理时）
- 减少 `--num-inference-steps`

### 4. VAE重建误差大

**问题**: Latent U-Net效果不好

**解决方案**:
- 运行 `test_vae_reconstruction.py` 检查VAE重建质量
- 确保数据归一化到[-1, 1]（WeatherDiff使用minmax归一化）
- 使用3通道数据
- 图像尺寸必须是8的倍数

### 5. 预测结果不准

**问题**: 模型预测精度低

**解决方案**:
- 增加训练数据量（至少1年）
- 调整模型大小（`--base-channels`, `--depth`）
- 尝试不同的学习率
- 检查数据归一化方法
- 尝试更复杂的模型（LSTM → ConvLSTM → Latent U-Net）

## 超参数调优

### ConvLSTM

```bash
--hidden-dim 64        # 隐藏层维度（默认64）
--num-layers 2         # ConvLSTM层数（默认2）
--kernel-size 3        # 卷积核大小（默认3）
--lr 1e-4             # 学习率（默认1e-4）
```

### Latent U-Net

```bash
--base-channels 128    # 基础通道数（默认128）
--depth 3             # U-Net深度（默认3）
--lr 1e-4             # 学习率（默认1e-4）
--weight-decay 1e-5   # 权重衰减（默认1e-5）
--vae-batch-size 4    # VAE批次大小（默认4）
```

### Diffusion

```bash
--base-channels 128    # 基础通道数（默认128）
--depth 3             # U-Net深度（默认3）
--lr 1e-4             # 学习率（默认1e-4）
--num-train-timesteps 1000  # 扩散步数（默认1000）
--beta-schedule linear      # Beta调度（linear/scaled_linear）
--vae-batch-size 4          # VAE批次大小（默认4）
```

## 脚本说明

所有脚本位于 `scripts/` 目录，可以直接运行或编辑参数：

| 脚本 | 说明 | 模块 |
|------|------|------|
| `run_convlstm.sh` | ConvLSTM训练+预测 | 传统模型 |
| `run_weather_transformer.sh` | Transformer训练+预测 | 传统模型 |
| `run_pixel_unet.sh` | Pixel U-Net训练+预测 | WeatherDiff |
| `run_latent_unet_full.sh` | 完整流程（自动预处理）⭐ | WeatherDiff |
| `run_diffusion.sh` | Diffusion训练+预测 ⭐ | WeatherDiff |

**编辑脚本参数**: 直接打开脚本文件，修改顶部的参数配置即可。

## 模型选择建议

### 单变量预测
- **快速测试**: Linear Regression
- **标准预测**: ConvLSTM
- **高质量**: Latent U-Net

### 多变量预测
- **快速测试**: Multi-Output LR
- **标准预测**: ConvLSTM
- **高质量**: Latent U-Net

### 不确定性量化
- **唯一选择**: Diffusion Model（集成预测）

### 计算资源有限
- **CPU**: Linear Regression, LSTM
- **单GPU (8GB)**: CNN, ConvLSTM
- **单GPU (12GB+)**: Latent U-Net, Diffusion
- **多GPU**: 任意模型

## 进阶用法

### WeatherDiff数据预处理（大数据集推荐）

```bash
# 预处理数据（一次性）
python preprocess_data_for_latent_unet.py \
    --data-path <data.zarr> \
    --variable 2m_temperature \
    --time-slice 2015-01-01:2019-12-31 \
    --target-size 512,512 \
    --output-dir data/preprocessed/temp_5y \
    --chunk-size 100

# 在多个模型间共享预处理数据
# Latent U-Net
python train_latent_unet.py --preprocessed-data-dir data/preprocessed/temp_5y

# Diffusion（共享同一预处理数据）
python train_diffusion.py --preprocessed-data-dir data/preprocessed/temp_5y
```

### 模型对比

```bash
# 训练多个模型
python train.py --model lstm --exp-name lstm_baseline
python train.py --model convlstm --exp-name convlstm_v1
python train_latent_unet.py --output-dir outputs/latent_unet

# 对比结果
python compare_models.py \
    --model-dirs outputs/lstm_baseline outputs/convlstm_v1 outputs/latent_unet
```

### VAE重建测试（WeatherDiff）

在训练WeatherDiff模块前，建议先测试VAE重建质量：

```bash
python test_vae_reconstruction.py \
    --data-path <data.zarr> \
    --variable 2m_temperature \
    --time-slice 2020-01-01:2020-12-31 \
    --n-test-samples 100 \
    --output-dir outputs/vae_reconstruction
```

**验收标准**:
- RMSE < 5K 且相关系数 > 0.9：VAE适用 ✓
- RMSE > 10K：VAE可能不适合该数据

---

更多模型细节和原理请参考 [README.md](README.md)
