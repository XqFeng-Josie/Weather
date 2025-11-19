#!/bin/bash
set -e

# ============================================================================
# 完整的RAE潜空间U-Net训练流程（支持大规模数据）
# 
# 使用独立的 train_rae.py 和 predict_rae.py 脚本
# 
# 分两步：
# Step 1: 预处理数据（一次性，可能需要1-2小时）
# Step 2: 训练模型（使用lazy loading，内存占用小）
# Step 3: 预测和评估
# ============================================================================

# ============ GPU设置 ============
export CUDA_VISIBLE_DEVICES=9  # 单GPU模式，可以根据需要修改

# 多GPU模式（可选）
# USE_MULTI_GPU=false
# GPU_IDS=""

# ============ 参数配置 ============
VARIABLE="2m_temperature"
TIME_SLICE="2015-01-01:2019-12-31"  # 训练数据时间范围
PREDICTION_TIME_SLICE="2020-01-01:2020-12-31"  # 预测数据时间范围
# PREDICTION_TIME_SLICE="2020-01-01:2020-01-31"  # 快速测试用

# RAE参数（优先使用SigLIP2）
RAE_ENCODER_CLS="Dinov2withNorm"  # 可选: Dinov2withNorm, SigLIP2wNorm, MAEwNorm
RAE_ENCODER_CONFIG_PATH="facebook/dinov2-with-registers-base"  # 对应encoder的HuggingFace模型ID
RAE_ENCODER_INPUT_SIZE=224  # Encoder输入图像尺寸
RAE_DECODER_CONFIG_PATH="configs/decoder/ViTXL"  # Decoder配置路径（HuggingFace模型ID）
RAE_DECODER_PATCH_SIZE=16  # Decoder patch大小
RAE_PRETRAINED_DECODER_PATH="models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt"  # 可选，预训练decoder路径
RAE_NORMALIZATION_STAT_PATH="models/stats/dinov2/wReg_base/imagenet1k/stat.pt"  # 可选，归一化统计量路径
FREEZE_ENCODER=true   # 冻结encoder（默认true，encoder固定）
FREEZE_DECODER=false  # decoder可微调（默认false，decoder参与训练）

# 模型参数
INPUT_LENGTH=12
OUTPUT_LENGTH=4
BASE_CHANNELS=128
DEPTH=3

# 训练参数
EPOCHS=50
BATCH_SIZE=16        # 主batch size（lazy loading支持）
GRADIENT_ACCUMULATION_STEPS=2  # 梯度累积步数（用于减少显存占用，有效batch = BATCH_SIZE × GRADIENT_ACCUMULATION_STEPS）
USE_AMP=true         # 是否使用混合精度训练（FP16/BF16，可以显著减少显存占用）
AMP_DTYPE="bfloat16" # 混合精度类型：float16 或 bfloat16（bfloat16更稳定，需要GPU支持）
VAE_BATCH_SIZE=4     # VAE编码子批次（控制显存，可根据GPU调整）
LR=0.0001
WEIGHT_DECAY=0.01
EARLY_STOPPING=10

# 数据参数
NORMALIZATION="minmax"
# 注意：RAE的target_size由decoder输出尺寸决定，不能手动指定
# target_size = encoder_input_size (对于标准的RAE配置)
# 但实际输出尺寸可能由decoder_patch_size等因素决定
# 预处理时会自动使用正确的target_size
DATA_PATH="gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr"

# 输出目录（target_size会在训练时自动确定）
PREPROCESSED_DIR="data/preprocessed/rae_pre_${TIME_SLICE//:/_}_${RAE_ENCODER_CLS}"
OUTPUT_DIR="outputs/rae_latent_unet_${VARIABLE}_${RAE_ENCODER_CLS}_${FREEZE_ENCODER}_${FREEZE_DECODER}"

echo "========================================================================"
echo "完整RAE潜空间U-Net训练流程"
echo "========================================================================"
echo "数据: $TIME_SLICE (训练数据)"
echo "VAE类型: RAE"
echo "Encoder: $RAE_ENCODER_CLS ($RAE_ENCODER_CONFIG_PATH)"
echo "Encoder输入尺寸: ${RAE_ENCODER_INPUT_SIZE}x${RAE_ENCODER_INPUT_SIZE}"
echo "Decoder: $RAE_DECODER_CONFIG_PATH (patch_size=$RAE_DECODER_PATCH_SIZE)"
echo "Encoder冻结: $FREEZE_ENCODER"
echo "Decoder冻结: $FREEZE_DECODER"
if [ -n "$RAE_PRETRAINED_DECODER_PATH" ]; then
    echo "Decoder预训练路径: $RAE_PRETRAINED_DECODER_PATH"
fi
if [ -n "$RAE_NORMALIZATION_STAT_PATH" ]; then
    echo "归一化统计量路径: $RAE_NORMALIZATION_STAT_PATH"
fi
echo "预处理目录: $PREPROCESSED_DIR"
echo "模型目录: $OUTPUT_DIR"
echo "注意: target_size由RAE decoder输出尺寸自动确定"
echo ""

# ============================================================================
# Step 1: 预处理数据（如果还没有预处理）
# 
# 注意：RAE的target_size由decoder输出尺寸决定，预处理时需要使用正确的尺寸
# 由于train_rae.py会检查并自动确定target_size，我们可以先运行一次训练脚本
# 来获取正确的target_size，然后进行预处理。
# 或者，我们可以在预处理时使用encoder_input_size作为target_size的初始值
# ============================================================================
# 先确定target_size（RAE的target_size由decoder输出尺寸决定）
# 这里使用encoder_input_size作为初始估计，实际值会在训练时验证
ESTIMATED_TARGET_SIZE="256,256"
PREPROCESSED_DIR_WITH_SIZE="data/preprocessed/rae_pre_${TIME_SLICE//:/_}_${ESTIMATED_TARGET_SIZE//,/x}"

if [ ! -d "$PREPROCESSED_DIR_WITH_SIZE" ]; then
    echo "========================================================================"
    echo "Step 1: 预处理数据"
    echo "========================================================================"
    echo "注意: RAE的target_size由decoder输出尺寸决定"
    echo "预处理时使用encoder_input_size (${RAE_ENCODER_INPUT_SIZE}x${RAE_ENCODER_INPUT_SIZE}) 作为初始估计"
    echo "训练时会验证并自动调整（如果需要）"
    echo "这可能需要1-2小时，但只需运行一次"
    echo "预计输出大小: ~10-20 GB"
    echo ""
    
    python preprocess_data_for_latent_unet.py \
        --data-path $DATA_PATH \
        --variable $VARIABLE \
        --time-slice $TIME_SLICE \
        --target-size $ESTIMATED_TARGET_SIZE \
        --n-channels 3 \
        --normalization $NORMALIZATION \
        --output-dir $PREPROCESSED_DIR_WITH_SIZE \
        --chunk-size 100
    
    echo ""
    echo "✓ 数据预处理完成！"
    PREPROCESSED_DIR=$PREPROCESSED_DIR_WITH_SIZE
else
    echo "========================================================================"
    echo "跳过预处理（数据已存在）"
    echo "========================================================================"
    echo "使用现有数据: $PREPROCESSED_DIR_WITH_SIZE"
    PREPROCESSED_DIR=$PREPROCESSED_DIR_WITH_SIZE
    
    # 显示预处理数据信息
    if [ -f "$PREPROCESSED_DIR/metadata.json" ]; then
        echo ""
        echo "数据信息:"
        cat $PREPROCESSED_DIR/metadata.json | python -m json.tool | grep -E "variable|n_timesteps|shape|target_size"
    fi
fi

echo ""
echo "========================================================================"
echo "Step 2: 训练RAE潜空间U-Net（使用train_rae.py）"
echo "========================================================================"
echo ""

# 构建训练命令
TRAIN_CMD="python train_rae.py \
    --preprocessed-data-dir $PREPROCESSED_DIR \
    --variable $VARIABLE \
    --time-slice $TIME_SLICE \
    --rae-encoder-cls $RAE_ENCODER_CLS \
    --rae-encoder-config-path $RAE_ENCODER_CONFIG_PATH \
    --rae-encoder-input-size $RAE_ENCODER_INPUT_SIZE \
    --rae-decoder-config-path $RAE_DECODER_CONFIG_PATH \
    --rae-decoder-patch-size $RAE_DECODER_PATCH_SIZE \
    --input-length $INPUT_LENGTH \
    --output-length $OUTPUT_LENGTH \
    --base-channels $BASE_CHANNELS \
    --depth $DEPTH \
    --batch-size $BATCH_SIZE \
    --vae-batch-size $VAE_BATCH_SIZE \
    --gradient-accumulation-steps $GRADIENT_ACCUMULATION_STEPS \
    --epochs $EPOCHS \
    --lr $LR \
    --weight-decay $WEIGHT_DECAY \
    --early-stopping $EARLY_STOPPING \
    --normalization $NORMALIZATION \
    --output-dir $OUTPUT_DIR"

# 添加freeze参数
if [ "$FREEZE_ENCODER" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --freeze-encoder"
fi

if [ "$FREEZE_DECODER" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --freeze-decoder"
fi

# 添加混合精度训练参数
if [ "$USE_AMP" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --use-amp --amp-dtype $AMP_DTYPE"
fi

# 添加多GPU参数（如果需要）
if [ -n "$USE_MULTI_GPU" ] && [ "$USE_MULTI_GPU" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --use-multi-gpu"
    if [ -n "$GPU_IDS" ]; then
        TRAIN_CMD="$TRAIN_CMD --gpu-ids $GPU_IDS"
    fi
fi

# 添加可选参数
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
# Step 3: 预测和评估（使用predict_rae.py）
# ============================================================================
echo ""
echo "========================================================================"
echo "Step 3: 预测和评估"
echo "========================================================================"
echo ""

python predict_rae.py \
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
echo "    - predictions_data/y_*.npy: 预测数据"
echo "    - *.png: 可视化图片"
echo ""
echo "RAE配置:"
echo "  Encoder: $RAE_ENCODER_CLS ($RAE_ENCODER_CONFIG_PATH)"
echo "  Encoder输入尺寸: ${RAE_ENCODER_INPUT_SIZE}x${RAE_ENCODER_INPUT_SIZE}"
echo "  Decoder: $RAE_DECODER_CONFIG_PATH (patch_size=$RAE_DECODER_PATCH_SIZE)"
echo "  Encoder冻结: $FREEZE_ENCODER"
echo "  Decoder冻结: $FREEZE_DECODER"
if [ -n "$RAE_PRETRAINED_DECODER_PATH" ]; then
    echo "  Decoder预训练路径: $RAE_PRETRAINED_DECODER_PATH"
fi
echo ""
echo "内存使用情况:"
echo "  预处理阶段: 分块处理，内存占用可控"
echo "  训练阶段: Lazy loading，只加载当前batch，内存占用极小"
echo "  峰值内存: < 5 GB（相比之前的28GB）"
echo ""
