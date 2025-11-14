# 重建测试 (Reconstruction Testing)

本目录包含 VAE 和 RAE 两种编码器的重建测试工具，用于评估它们对天气数据的重建能力。

## 📋 目录结构

```
reconstruction/
├── README.md                          # 本文件
├── test_vae_reconstruction.py        # VAE重建测试脚本
├── test_rae_reconstruction.sh        # RAE重建测试脚本
├── prepare_weather_images.py         # 准备天气图像（VAE和RAE共用）
├── compare_reconstructions.py        # 统一对比脚本（推荐）⭐
└── outputs/                          # 输出目录（自动创建）
    ├── vae_reconstruction/           # VAE测试结果
    └── rae_reconstruction/           # RAE测试结果
```

## 🎯 快速开始

### 1. 准备天气图像（VAE 和 RAE 共用）

```bash
cd reconstruction
python prepare_weather_images.py \
    --data-path gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr \
    --variable 2m_temperature \
    --time-slice 2020-01-01:2020-01-31 \
    --target-size 256 256 \
    --output-dir weather_images \
    --n-samples 100
```

这会生成：
- `weather_images/`: 图像文件
- `weather_images/normalization_stats.json`: 归一化参数（用于后续反归一化）

### 2. 运行 VAE 重建测试

```bash
# 从 Weather 项目根目录运行
cd /path/to/Weather
python reconstruction/test_vae_reconstruction.py \
    --data-path reconstruction/weather_images \
    --n-test-samples 100 \
    --save-separate
```

### 3. 运行 RAE 重建测试

```bash
cd reconstruction
bash test_rae_reconstruction.sh
```

### 4. 统一对比分析

```bash
python compare_reconstructions.py \
    --original-dir weather_images \
    --reconstructed-dirs \
        outputs/vae_reconstruction/reconstructed \
        recon_samples_DINOv2-B/RAE-pretrained-bs4-fp32 \
    --labels VAE RAE-DINOv2-B \
    --output comparison_all.png \
    --metrics-output metrics_all.json \
    --denormalize
```

## 📖 详细说明

### VAE 重建测试

VAE 测试从图像目录加载数据，与 RAE 保持一致。

**参数说明：**
- `--data-path`: 图像目录路径（必需）
- `--n-test-samples`: 测试样本数量
- `--save-separate`: 分别保存原图和重建图到子文件夹

**输出：**
- `outputs/vae_reconstruction/vae_reconstruction_results.json`: 评估指标
- `outputs/vae_reconstruction/original/`: 原图目录
- `outputs/vae_reconstruction/reconstructed/`: 重建图目录

### RAE 重建测试

RAE 测试脚本会调用 RAE 项目进行重建。确保 RAE 项目在 `../RAE` 目录或设置 `RAE_DIR` 环境变量。

### 对比分析

`compare_reconstructions.py` 支持：
- 单个或多个重建结果对比
- 自动计算评估指标（RMSE, MAE, PSNR, SSIM 等）
- 生成对比可视化图像和指标表格
- 自动使用保存的归一化参数进行反归一化

## 🔧 环境要求

### Python 依赖

```bash
# Weather 项目依赖
pip install -r requirements.txt
pip install -r requirements_weatherdiff.txt

# VAE 测试额外依赖
pip install diffusers transformers accelerate

# 可视化（可选）
pip install cartopy

# 数据访问
pip install gcsfs
```

### RAE 项目

确保 RAE 项目已安装并配置好环境。

## 📊 归一化说明

### 数据流程

1. **准备图像** (`prepare_weather_images.py`):
   - 从 zarr 加载原始数据（物理单位，如 K）
   - 插值到目标尺寸（如 256×256）
   - 全局归一化到 [0, 255] 并保存为 PNG
   - **保存归一化参数**到 `normalization_stats.json`

2. **VAE 测试** (`test_vae_reconstruction.py`):
   - 加载图像 [0, 255]
   - 转换为 [-1, 1]（VAE 输入范围）
   - VAE 重建
   - 使用保存的归一化参数反归一化到物理单位

3. **对比分析** (`compare_reconstructions.py`):
   - 自动加载归一化参数
   - 计算归一化空间和物理单位的指标

### 归一化参数文件

`normalization_stats.json` 包含：
- `method`: 归一化方法（'minmax' 或 'zscore'）
- `variable`: 变量名
- `original_min/max`: 原始数据范围（minmax 方法）
- `original_mean/std`: 原始数据统计量（zscore 方法）

## 🎯 验收标准

- **RMSE < 10K 且 相关系数 > 0.9**: 重建质量良好 ✅
- **RMSE < 15K**: 重建质量一般，建议微调
- **RMSE > 15K**: 重建质量差，建议训练自定义编码器
