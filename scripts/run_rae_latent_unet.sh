#!/bin/bash
set -e

# ============================================================================
# 完整的RAE潜空间U-Net训练流程（支持大规模数据）
# 
# 分两步：
# Step 1: 预处理数据（一次性，可能需要1-2小时）
# Step 2: 训练模型（使用lazy loading，内存占用小）
# ============================================================================

# ============ GPU设置 ============
export CUDA_VISIBLE_DEVICES=1

# ============ 参数配置 ============
VARIABLE="2m_temperature"
TIME_SLICE="2015-01-01:2019-12-31"  # 5年完整数据
PREDICTION_TIME_SLICE="2020-01-01:2020-12-31"
# PREDICTION_TIME_SLICE="2020-01-01:2020-01-31"

# RAE参数（优先使用SigLIP2）
VAE_TYPE="rae"
RAE_ENCODER_CLS="MAEwNorm" # Dinov2withNorm MAEwNorm default: SigLIP2wNorm
RAE_ENCODER_CONFIG_PATH="facebook/vit-mae-base" # facebook/dinov2-base facebook/vit-mae-base google/siglip2-base-patch16-256
RAE_ENCODER_INPUT_SIZE=256
RAE_DECODER_CONFIG_PATH="configs/decoder/ViTXL"
RAE_DECODER_PATCH_SIZE=16
RAE_PRETRAINED_DECODER_PATH="models/decoders/mae/base_p16/ViTXL_n08/model.pt"  # 可选，预训练decoder路径
RAE_NORMALIZATION_STAT_PATH="models/stats/mae/base_p16/ImageNet1k/stat.pt"  # 可选，归一化统计量路径
FREEZE_ENCODER=true              # 冻结encoder（默认true）
FREEZE_DECODER=false             # decoder可微调（默认false）

# 模型参数
INPUT_LENGTH=12
OUTPUT_LENGTH=4
BASE_CHANNELS=128
DEPTH=3

# 训练参数
EPOCHS=1
BATCH_SIZE=16        # 主batch size（lazy loading支持）
VAE_BATCH_SIZE=4     # VAE编码子批次（控制显存，可根据GPU调整）
LR=0.0001
EARLY_STOPPING=5

# 数据参数
NORMALIZATION="minmax"
TARGET_SIZE="256,256"  # 使用完整的256x256分辨率
DATA_PATH="gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr"

# 输出目录
PREPROCESSED_DIR="data/preprocessed/vae_pre_${TIME_SLICE//:/_}_${TARGET_SIZE//,/x}"
OUTPUT_DIR="outputs/rae_latent_unet_${VARIABLE}_${RAE_ENCODER_CLS}"

echo "========================================================================"
echo "完整RAE潜空间U-Net训练流程"
echo "========================================================================"
echo "数据: $TIME_SLICE (5年完整数据)"
echo "分辨率: $TARGET_SIZE (高分辨率)"
echo "VAE类型: $VAE_TYPE"
echo "Encoder: $RAE_ENCODER_CLS ($RAE_ENCODER_CONFIG_PATH)"
echo "Decoder: $RAE_DECODER_CONFIG_PATH (patch_size=$RAE_DECODER_PATCH_SIZE)"
echo "Encoder冻结: $FREEZE_ENCODER"
echo "Decoder冻结: $FREEZE_DECODER"
echo "预处理目录: $PREPROCESSED_DIR"
echo "模型目录: $OUTPUT_DIR"
echo ""

# ============================================================================
# Step 1: 预处理数据（如果还没有预处理）
# ============================================================================
if [ ! -d "$PREPROCESSED_DIR" ]; then
    echo "========================================================================"
    echo "Step 1: 预处理数据"
    echo "========================================================================"
    echo "这可能需要1-2小时，但只需运行一次"
    echo "预计输出大小: ~20-30 GB"
    echo ""
    
    python preprocess_data_for_latent_unet.py \
        --data-path $DATA_PATH \
        --variable $VARIABLE \
        --time-slice $TIME_SLICE \
        --target-size $TARGET_SIZE \
        --n-channels 3 \
        --normalization $NORMALIZATION \
        --output-dir $PREPROCESSED_DIR \
        --chunk-size 100
    
    echo ""
    echo "✓ 数据预处理完成！"
else
    echo "========================================================================"
    echo "跳过预处理（数据已存在）"
    echo "========================================================================"
    echo "使用现有数据: $PREPROCESSED_DIR"
    
    # 显示预处理数据信息
    if [ -f "$PREPROCESSED_DIR/metadata.json" ]; then
        echo ""
        echo "数据信息:"
        cat $PREPROCESSED_DIR/metadata.json | python -m json.tool | grep -E "variable|n_timesteps|shape|target_size"
    fi
fi

echo ""
echo "========================================================================"
echo "Step 2: 训练RAE潜空间U-Net（Lazy Loading）"
echo "========================================================================"
echo ""

# 构建训练命令
TRAIN_CMD="python train_latent_unet.py \
    --preprocessed-data-dir $PREPROCESSED_DIR \
    --vae-type $VAE_TYPE \
    --rae-encoder-cls $RAE_ENCODER_CLS \
    --rae-encoder-config-path $RAE_ENCODER_CONFIG_PATH \
    --rae-encoder-input-size $RAE_ENCODER_INPUT_SIZE \
    --rae-decoder-config-path $RAE_DECODER_CONFIG_PATH \
    --rae-decoder-patch-size $RAE_DECODER_PATCH_SIZE \
    --target-size $TARGET_SIZE \
    --input-length $INPUT_LENGTH \
    --output-length $OUTPUT_LENGTH \
    --base-channels $BASE_CHANNELS \
    --depth $DEPTH \
    --batch-size $BATCH_SIZE \
    --vae-batch-size $VAE_BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --early-stopping $EARLY_STOPPING \
    --output-dir $OUTPUT_DIR"

# 添加可选参数
if [ "$FREEZE_ENCODER" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --freeze-encoder"
fi

if [ "$FREEZE_DECODER" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --freeze-decoder"
fi

if [ -n "$RAE_PRETRAINED_DECODER_PATH" ]; then
    TRAIN_CMD="$TRAIN_CMD --rae-pretrained-decoder-path $RAE_PRETRAINED_DECODER_PATH"
fi

if [ -n "$RAE_NORMALIZATION_STAT_PATH" ]; then
    TRAIN_CMD="$TRAIN_CMD --rae-normalization-stat-path $RAE_NORMALIZATION_STAT_PATH"
fi

# 执行训练命令
eval $TRAIN_CMD

echo ""
echo "✓ 训练完成!"

# ============================================================================
# Step 3: 预测和评估
# ============================================================================
echo ""
echo "========================================================================"
echo "Step 3: 预测和评估"
echo "========================================================================"
echo ""

python predict_unet.py \
    --mode latent \
    --model-dir $OUTPUT_DIR \
    --data-path $DATA_PATH \
    --time-slice $PREDICTION_TIME_SLICE \
    --batch-size 32 \
    --vae-batch-size $VAE_BATCH_SIZE 

echo ""
echo "✓ 预测完成!"

# ============================================================================
# 总结
# ============================================================================
echo ""
echo "========================================================================"
echo "完成！"
echo "========================================================================"
echo ""
echo "输出文件:"
echo "  预处理数据: $PREPROCESSED_DIR/"
echo "    - data.npy: 预处理后的图像数据"
echo "    - metadata.json: 数据元信息"
echo "    - normalizer_stats.pkl: 归一化参数"
echo ""
echo "  训练模型: $OUTPUT_DIR/"
echo "    - best_model.pt: 最佳模型"
echo "    - config.json: 训练配置"
echo "    - training_history.json: 训练历史"
echo "    - normalizer_stats.pkl: 归一化参数（包含RAE配置）"
echo ""
echo "  预测结果: $OUTPUT_DIR/"
echo "    - prediction_metrics.json: 评估指标"
echo "    - *.png: 可视化图片"
echo ""
echo "RAE配置:"
echo "  Encoder: $RAE_ENCODER_CLS ($RAE_ENCODER_CONFIG_PATH)"
echo "  Decoder: $RAE_DECODER_CONFIG_PATH (patch_size=$RAE_DECODER_PATCH_SIZE)"
echo "  Encoder冻结: $FREEZE_ENCODER"
echo "  Decoder冻结: $FREEZE_DECODER"
echo ""
echo "内存使用情况:"
echo "  预处理阶段: 分块处理，内存占用可控"
echo "  训练阶段: Lazy loading，只加载当前batch，内存占用极小"
echo "  峰值内存: < 5 GB（相比之前的28GB）"
echo ""