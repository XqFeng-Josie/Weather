# Weather Prediction System

基于深度学习的气象预测系统，支持多种模型架构和预测方式。

## 🎯 项目目标

本项目旨在使用深度学习技术进行全球天气预测，主要解决以下问题：

- **短期天气预测**：预测未来1天（4个时间步，6小时间隔）的天气状况
- **多变量预测**：支持温度、位势高度、风速等多个气象变量
- **不确定性量化**：通过概率预测方法量化预测的不确定性
- **全球覆盖**：基于ERA5全球再分析数据，分辨率64×32网格

## 📊 数据说明

### 数据源

- **来源**: WeatherBench2 - ERA5再分析数据
- **路径**: `gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr`
- **分辨率**: 64×32等角网格（经度×纬度）
- **经度范围**:[0.00, 354.38]
- **维度范围**:[-87.19, 87.19]
- **时间间隔**: 6小时
- **时间点**: 92044
- **时间范围**: 1959-01-01 到 2021-12-31

### 主要变量

| 变量名 | 说明 | 维度 |
|--------|------|------|
| `2m_temperature` | 2米温度 | (time, lat, lon) |
| `geopotential` | 位势高度 | (time, level, lat, lon) |
| `10m_u_component_of_wind` | 10米U风 | (time, lat, lon) |
| `10m_v_component_of_wind` | 10米V风 | (time, lat, lon) |
| `specific_humidity` | 比湿 | (time, level, lat, lon) |

### 数据格式

```python
# 输入序列
X: (n_samples, input_length, features)
   input_length = 12  # 过去12个时间步（3天）

# 输出序列  
Y: (n_samples, output_length, features)
   output_length = 4  # 未来4个时间步（1天）
```

## 🏗️ 模型架构

### 1. 传统深度学习模型

#### Linear Regression (lr)
- **原理**: Ridge回归，单变量快速基线
- **特点**: 无时序建模，训练快速
- **适用**: 快速基线测试

#### Multi-Output LR (lr_multi)
- **原理**: 每个变量独立的Ridge模型
- **特点**: 避免变量间干扰，支持并行训练
- **适用**: 多变量基线

#### LSTM
- **原理**: 循环神经网络，建模时间依赖
- **结构**: 输入展平 → LSTM → 全连接
- **局限**: 丢失空间信息
- **适用**: 单变量时间序列

#### CNN ⭐
- **原理**: 卷积神经网络，提取空间特征
- **结构**: Conv2D → BatchNorm → ReLU → FC
- **优势**: 训练快速，性能最优（RMSE=1.20 K）
- **局限**: 无时序建模
- **推荐**: 快速部署的首选模型

#### ConvLSTM
- **原理**: 结合CNN和LSTM，同时建模时空依赖
- **结构**: ConvLSTM单元 → Conv2D输出
- **优势**: 保留空间结构，建模时序
- **表现**: RMSE=1.24 K，性能优秀

#### Weather Transformer
- **原理**: Factorized时空注意力机制
- **结构**: Patch Embedding → Spatial + Temporal Attention
- **特点**: 轻量级设计，约1.6M参数
- **适用**: 捕获长距离时空依赖

### 2. WeatherDiff 模块 ⭐

基于Stable Diffusion架构的天气预测模块，将气象网格数据视为图像，利用预训练VAE和U-Net架构进行时空预测。

**模块组成**：
```
weatherdiff/
├── vae/          # VAE功能（SD VAE + RAE）
│   ├── vae_wrapper.py    # SD VAE包装器
│   ├── rae_wrapper.py    # RAE包装器
│   └── rae/              # RAE核心模块
│       ├── encoders/     # Encoder（DINOv2, SigLIP2, MAE）
│       └── decoders/     # Decoder（可微调）
├── unet/         # U-Net模型（像素和潜空间）
├── diffusion/    # 扩散模型
└── utils/        # 工具函数
```

#### Pixel U-Net ⭐
- **原理**: 图像到图像的确定性预测
- **结构**: U-Net架构，直接在像素空间预测
- **输入**: 过去N帧图像 → 未来M帧图像
- **特点**: 训练快速，结果确定，性能优异（RMSE=1.25 K）
- **推荐**: WeatherDiff模块中表现最佳

#### Latent U-Net ⭐
- **原理**: 在VAE潜空间中预测
- **VAE选项**: 
  - **SD VAE**: Stable Diffusion预训练VAE（默认）
    - **权重**: 默认从HuggingFace加载，可用 `--vae-pretrained-path` 指定自定义权重
    - **可训练性**: 通过 `--freeze-encoder/--freeze-decoder` 分别控制encoder/decoder是否参与训练
    - **训练脚本**: `train_vae.py`
    - **预测脚本**: `predict_vae.py`
  - **RAE**: Representation Autoencoder（可选，支持多种encoder）
    - **Encoder**: 固定（默认冻结），可选 DINOv2 / SigLIP2 / MAE
    - **Decoder**: 可微调，支持加载预训练权重
    - **训练脚本**: `train_rae.py`
    - **预测脚本**: `predict_rae.py`
- **优势**: 
  - 显存需求低（512×512 → 64×64潜空间 for SD VAE）
  - 训练更稳定
  - 生成结果更平滑
  - 支持针对encoder/decoder的细粒度微调配置
  - **RAE**: Decoder可微调，可能获得更好的重建质量
- **推荐**: 在大尺寸数据上训练时使用

#### Diffusion Model ⭐
- **原理**: 扩散模型，概率预测 (SD-VAE+U-Net+diffusion)
- **训练**: 学习如何“去噪”未来潜向量，从噪声恢复出真实未来。
- **推理**: 从噪声逐步去噪生成预测
- **特点**:
  - 支持生成多个未来场景
  - 量化预测不确定性
  - 适合集成预测
- **推荐**: 需要不确定性估计时使用

### 3. 模型对比

| 模型 | 时空建模 | 训练速度 | 推理速度 | 不确定性 | 推荐场景 | 性能排名 |
|------|---------|---------|---------|---------|---------|---------|
| CNN | 空间 | ⚡⚡ | ⚡⚡⚡ | ✗ | 快速部署 ⭐ | #1 |
| ConvLSTM | 时空 | ⚡ | ⚡⚡ | ✗ | 通用预测 ⭐ | #2 |
| Pixel U-Net | 时空 | ⚡⚡ | ⚡⚡ | ✗ | WeatherDiff最佳 ⭐ | #3 |
| Weather Transformer | 时空 | ⚡ | ⚡ | ✗ | 长距离依赖 | #4 |
| Latent U-Net | 时空 | ⚡⚡ | ⚡⚡ | ✗ | 大尺寸图像 ⭐ | #5 |
| LSTM | 时序 | ⚡⚡ | ⚡⚡ | ✗ | 单变量时序 | #6 |
| Linear Regression | ✗ | ⚡⚡⚡ | ⚡⚡⚡ | ✗ | 快速基线 | #7 |
| Transformer | 时空 | ⚡ | ⚡ | ✗ | 长距离依赖 | #8 |
| Diffusion | 时空 | 🐢 | 🐢 | ✓ | 概率预测 ⭐ | - |

## 📈 评估指标

### VAE重建指标

- **相关系数**: 空间模式相似度
- **SSIM**: 结构相似性指数（图像质量）
- **PSNR**: 峰值信噪比（图像质量）

### 确定性指标
- **RMSE** (Root Mean Square Error): 均方根误差，主要指标
- **MAE** (Mean Absolute Error): 平均绝对误差

### 概率指标（Diffusion模型）

- **CRPS** (Continuous Ranked Probability Score): 概率分布质量
- **Spread-Skill Ratio**: 集成校准（理想值 ≈ 1.0）
  - < 1.0: 过度自信
  - > 1.0: 不够自信
- **Ensemble Mean RMSE**: 集成平均误差

### 时空分辨指标

- **RMSE vs Lead Time**: 误差随预测步长变化
- **空间误差图**: 不同区域的预测精度
- **时间序列图**: 预测值与真值的时间序列对比

## 📁 项目结构

```
Weather/
├── src/                       # 传统深度学习模型
│   ├── data_loader.py         # 数据加载
│   ├── trainer.py             # 训练器
│   ├── visualization.py       # 可视化
│   └── models/                # 模型实现
│       ├── linear_regression.py
│       ├── lstm.py
│       ├── cnn.py
│       ├── convlstm.py
│       ├── transformer.py
│       └── weather_transformer.py
│
├── weatherdiff/               # WeatherDiff模块 ⭐
│   ├── vae/                   # VAE功能（SD VAE + RAE）
│   │   ├── vae_wrapper.py     # SD VAE包装器
│   │   ├── rae_wrapper.py     # RAE包装器
│   │   └── rae/               # RAE核心模块
│   ├── unet/                  # U-Net模型（像素/潜空间）
│   ├── diffusion/             # 扩散模型
│   └── utils/                 # 工具函数
│
├── scripts/                   # 运行脚本 ⭐
│   ├── run_convlstm.sh
│   ├── run_weather_transformer.sh
│   ├── run_pixel_unet.sh
│   ├── run_vae_latent_unet.sh  # VAE (SD) 独立训练脚本 ⭐
│   ├── run_rae_latent_unet.sh  # RAE 独立训练脚本 ⭐
│   └── run_diffusion.sh
│
├── train.py                   # 传统模型训练
├── train_weather_transformer.py
├── train_pixel_unet.py        # WeatherDiff像素空间训练脚本
├── train_vae.py               # VAE (SD) 潜空间训练脚本
├── train_rae.py               # RAE 潜空间训练脚本
├── train_diffusion.py
│
├── predict.py                 # 传统模型预测
├── predict_unet.py            # WeatherDiff统一预测脚本（支持pixel/latent模式）
├── predict_vae.py             # VAE (SD) 独立预测脚本 ⭐ 新增
├── predict_rae.py             # RAE 独立预测脚本 ⭐ 新增
├── predict_diffusion.py
│
├── preprocess_data_for_latent_unet.py  # 数据预处理
├── test_vae_reconstruction.py          # VAE重建测试
├── compare_models.py                   # 模型对比
│
├── requirements.txt           # 基础依赖
├── requirements_weatherdiff.txt  # WeatherDiff依赖
├── README.md                  # 本文件 ⭐
└── USAGE.md                   # 使用指南 ⭐
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 基础依赖（传统模型）
pip install -r requirements.txt

# WeatherDiff额外依赖（如果使用WeatherDiff模块）
pip install -r requirements_weatherdiff.txt
```

### 2. 快速训练示例

#### 2.1 VAE (SD) Latent U-Net

```bash
# 使用预训练SD VAE（推荐，快速开始）
bash scripts/run_vae_latent_unet.sh

# 或手动运行
python train_vae.py \
    --data-path gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr \
    --variable 2m_temperature \
    --time-slice 2019-01-01:2019-12-31 \
    --target-size 512,512 \
    --batch-size 16 \
    --epochs 50 \
    --output-dir outputs/vae_latent_unet \
    --freeze-decoder  # 示例：仅训练encoder

# 预测
python predict_vae.py \
    --model-dir outputs/vae_latent_unet \
    --time-slice 2020-01-01:2020-12-31
```

#### 2.2 RAE Latent U-Net

```bash
# 使用SigLIP2 encoder（推荐）
bash scripts/run_rae_latent_unet.sh

# 或手动运行
python train_rae.py \
    --data-path gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr \
    --variable 2m_temperature \
    --time-slice 2015-01-01:2019-12-31 \
    --rae-encoder-cls SigLIP2wNorm \
    --rae-encoder-config-path google/siglip2-base-patch16-256 \
    --batch-size 16 \
    --epochs 50 \
    --output-dir outputs/rae_latent_unet

# 预测
python predict_rae.py \
    --model-dir outputs/rae_latent_unet \
    --time-slice 2020-01-01:2020-12-31
```

#### 2.3 Pixel U-Net

```bash
bash scripts/run_pixel_unet.sh
```

### 3. 查看结果

训练完成后，结果保存在 `outputs/<model_name>/` 目录：

```
outputs/<model_name>/
├── best_model.pt              # 最佳模型权重
├── config.json                # 配置文件
├── prediction_metrics.json    # 评估指标
├── predictions_data/          # 预测数据
├── timeseries_*.png          # 时间序列图
├── spatial_comparison_*.png   # 空间对比图
└── rmse_vs_leadtime_*.png    # RMSE vs预测步长
```

## 🔬 实验结果

所有结果均为**物理空间**（温度单位：K）的评估指标。

### 传统深度学习模型

| 模型 | RMSE (K) | MAE (K) | RMSE Step 1 | RMSE Step 2 | RMSE Step 3 | RMSE Step 4 |
|------|----------|---------|-------------|-------------|-------------|-------------|
| CNN ⭐ | 1.2025 | 0.7530 | 0.7679 | 1.0615 | 1.3086 | 1.5347 |
| ConvLSTM | 1.2417 | 0.7582 | 0.7360 | 1.0913 | 1.3648 | 1.6039 |
| Weather Transformer | 1.3495 | 0.8630 | 0.9520 | 1.2247 | 1.4549 | 1.6618 |
| LSTM | 2.5607 | 1.7288 | 2.5232 | 2.5430 | 2.5713 | 2.6044 |
| Multi-Output LR | 2.6699 | 1.7560 | 2.2266 | 2.5490 | 2.8019 | 3.0344 |
| Transformer | 3.3667 | 2.3004 | 3.3628 | 3.3710 | 3.3673 | 3.3659 |

### WeatherDiff 模块

| 模型 | RMSE (K) | MAE (K) | RMSE Step 1 | RMSE Step 2 | RMSE Step 3 | RMSE Step 4 |
|------|----------|---------|-------------|-------------|-------------|-------------|
| Pixel U-Net ⭐ | 1.2523 | 0.7832 | 0.7753 | 1.1281 | 1.3816 | 1.5782 |
| Latent U-Net (SD-VAE, frozen) | 1.9212 | 1.4293 | 1.7892 | 1.8955 | 1.9479 | 2.0436 |
| Latent U-Net (RAE, SigLIP2) | 13.6778 | 7.7200 | 13.5848 | 13.6952 | 13.7554 | 13.6755 |

### 结果分析

**最佳模型排序（按RMSE）**：
1. **CNN** (1.20 K) - 最佳传统模型 ⭐
2. **ConvLSTM** (1.24 K) - 次优传统模型
3. **Pixel U-Net** (1.25 K) - 最佳WeatherDiff模型 ⭐
4. **Weather Transformer** (1.35 K) - 基于ViT的Transformer

**关键发现**：

1. **传统模型 vs WeatherDiff**：
   - 传统CNN模型表现最优（RMSE=1.20 K），略优于WeatherDiff的Pixel U-Net（RMSE=1.25 K）
   - 两者性能相近，说明在64×32分辨率下，简单CNN也能取得很好效果
   - ConvLSTM（RMSE=1.24 K）与Pixel U-Net性能相当

2. **WeatherDiff模块表现**：
   - Pixel U-Net表现最佳（RMSE=1.25 K），接近传统最佳模型
   - Latent U-Net（SD-VAE）次之（RMSE=1.92 K），但明显优于RAE版本
   - RAE版本表现较差（RMSE=13.68 K），可能需要进一步调优

3. **预测步长分析**：
   - 所有模型均显示误差随预测步长增加而增大（Step 1 → Step 4）
   - CNN、ConvLSTM、Pixel U-Net在短期预测（6小时）表现优异（RMSE < 0.8 K）
   - 长期预测（24小时）误差增长明显，但仍可接受（RMSE < 1.6 K for最佳模型）

4. **模型选择建议**：
   - **快速部署**：选择CNN，训练快速，性能最优
   - **平衡性能与可扩展性**：选择ConvLSTM或Pixel U-Net
   - **大尺寸图像**：选择Latent U-Net（SD-VAE），显存友好
   - **长距离依赖**：选择Weather Transformer

**说明**：
- Step 1-4 分别对应未来6、12、18、24小时的预测
- 所有指标均在物理空间计算（单位：开尔文 K）
- 数据变量：2m_temperature（2米温度）

## 🔧 使用指南

### 环境配置

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装基础依赖
pip install -r requirements.txt

# 安装WeatherDiff额外依赖
pip install -r requirements_weatherdiff.txt
```

### VAE vs RAE 实验设计

#### VAE (SD) Latent U-Net 实验设计

**预训练权重**：
- 默认直接从 HuggingFace (`stable-diffusion-v1-5/stable-diffusion-v1-5`) 拉取
- 若有自定义权重，可通过 `--vae-pretrained-path /path/to/weights.pt` 指定

**微调策略**：
- `--freeze-encoder`: 仅冻结encoder（默认不冻结）
- `--freeze-decoder`: 仅冻结decoder（默认不冻结）
- 两个都加：完全冻结VAE，只训练U-Net

**推荐配置**：
- 快速开始：全部冻结（`--freeze-encoder --freeze-decoder`）
- 进阶微调：冻结decoder，仅训练encoder（`--freeze-decoder`）
- 全量微调：不加任何freeze参数，encoder/decoder与U-Net一起训练

#### RAE Latent U-Net 实验设计

**Encoder选项**：
1. **SigLIP2wNorm** (推荐)：SigLIP-2模型，性能优秀
   - 配置：`--rae-encoder-cls SigLIP2wNorm --rae-encoder-config-path google/siglip2-base-patch16-256`
   
2. **Dinov2withNorm**：DINOv2模型，稳定可靠
   - 配置：`--rae-encoder-cls Dinov2withNorm --rae-encoder-config-path facebook/dinov2-base`
   
3. **MAEwNorm**：MAE模型，适合特定场景
   - 配置：`--rae-encoder-cls MAEwNorm --rae-encoder-config-path facebook/vit-mae-base`

**训练策略**：
- Encoder：固定（`--freeze-encoder`，默认true），不参与训练
- Decoder：可微调（默认false），支持从预训练权重fine-tuning
  - 加载预训练：`--rae-pretrained-decoder-path /path/to/decoder.pt`
  - 加载归一化统计：`--rae-normalization-stat-path /path/to/stat.pt`

**target_size说明**：
- RAE的`target_size`由decoder输出尺寸自动确定，不能手动指定
- 预处理时使用`encoder_input_size`作为初始估计
- 训练时会自动验证并调整（如果不匹配会报错）

**推荐配置**：
- 标准配置：SigLIP2 + ViT-MAE decoder（256x256）
- 高分辨率：调整`encoder_input_size`和`decoder_patch_size`（需重新预处理）

### 实验对比建议

**VAE (SD) vs RAE**：
- **显存占用**：SD VAE (512x512→64x64) vs RAE (256x256→16x16)
- **训练速度**：RAE通常更快（decoder参数更少）
- **重建质量**：RAE可能更好（可微调decoder）
- **灵活性**：SD VAE更成熟，RAE更灵活（多种encoder选择）

**参数调优建议**：
1. 先使用推荐配置快速验证
2. 根据显存调整`batch_size`和`vae_batch_size`
3. 启用混合精度训练（`--use-amp --amp-dtype bfloat16`）
4. 使用梯度累积（`--gradient-accumulation-steps 2`）减少显存

## 📚 参考文献

### 数据和基准
- [WeatherBench2](https://weatherbench2.readthedocs.io/) - 天气预测基准
- [ERA5](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5) - ECMWF再分析数据

### 模型论文
- [ConvLSTM](https://arxiv.org/abs/1506.04214) - Shi et al., 2015
- [Transformer](https://arxiv.org/abs/1706.03762) - Vaswani et al., 2017
- [U-Net](https://arxiv.org/abs/1505.04597) - Ronneberger et al., 2015
- [DDPM](https://arxiv.org/abs/2006.11239) - Ho et al., 2020
- [Stable Diffusion](https://arxiv.org/abs/2112.10752) - Rombach et al., 2022

## 📧 联系方式

如遇问题或有建议，欢迎提Issue或PR。

---

更多模型架构细节请参考 [MODEL.md](MODEL.md)
