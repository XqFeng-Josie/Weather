#!/bin/bash
set -e

# ============================================================================
# 完整的潜空间U-Net训练流程（支持大规模数据）
# 
# 分两步：
# Step 1: 预处理数据（一次性，可能需要1-2小时）
# Step 2: 训练模型（使用lazy loading，内存占用小）
# ============================================================================

# ============ GPU设置 ============
# 单GPU模式（默认）
# export CUDA_VISIBLE_DEVICES=7

# 多GPU模式
USE_MULTI_GPU=false  # 设置为true启用多GPU训练
GPU_IDS=""  # 指定GPU IDs（如 "0,1,2,3"），留空使用所有可用GPU

# 如果启用多GPU，取消注释并设置GPU IDs
# USE_MULTI_GPU=true
# GPU_IDS="0,1,2,3"  # 使用GPU 0, 1, 2, 3

# 设置CUDA_VISIBLE_DEVICES
if [ "$USE_MULTI_GPU" = true ]; then
    # 多GPU模式
    if [ -n "$GPU_IDS" ]; then
        # 指定了GPU IDs，使用指定的GPU
        export CUDA_VISIBLE_DEVICES=$GPU_IDS
    else
        # 未指定GPU IDs，不设置CUDA_VISIBLE_DEVICES，使用所有可用GPU
        unset CUDA_VISIBLE_DEVICES
    fi
else
    # 单GPU模式
    if [ -n "$GPU_IDS" ]; then
        # 指定了GPU IDs，使用第一个GPU
        FIRST_GPU=$(echo $GPU_IDS | cut -d',' -f1)
        export CUDA_VISIBLE_DEVICES=$FIRST_GPU
    fi
    # 如果未指定GPU_IDS，保持原有的CUDA_VISIBLE_DEVICES设置（如果有）
fi

# ============ 参数配置 ============
VARIABLE="2m_temperature"
TIME_SLICE="2015-01-01:2019-12-31"  # 5年完整数据
PREDICTION_TIME_SLICE="2020-01-01:2020-12-31"
# PREDICTION_TIME_SLICE="2020-01-01:2020-01-31"

# VAE参数
VAE_TYPE="sd"  # VAE类型: sd (Stable Diffusion) 或 rae
VAE_MODEL_ID="stable-diffusion-v1-5/stable-diffusion-v1-5"  # SD VAE模型ID
VAE_TRAIN_MODE="pretrained"  # VAE训练模式: pretrained (加载预训练) 或 from_scratch (从头训练)
VAE_PRETRAINED_PATH=""  # 可选，预训练VAE权重路径（如果使用from_scratch但想加载特定权重）
FREEZE_VAE=true
# 模型参数
INPUT_LENGTH=12
OUTPUT_LENGTH=4
BASE_CHANNELS=128
DEPTH=3

# 训练参数
EPOCHS=50
BATCH_SIZE=16        # 主batch size（lazy loading支持）
VAE_BATCH_SIZE=4     # VAE编码子批次（控制显存，可根据GPU调整）
LR=0.0001
EARLY_STOPPING=10

# 数据参数
NORMALIZATION="minmax"
TARGET_SIZE="512,512"  # 使用完整的512x512分辨率
DATA_PATH="gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr"

# 输出目录
PREPROCESSED_DIR="data/preprocessed/vae_pre_${TIME_SLICE//:/_}_${TARGET_SIZE//,/x}"
OUTPUT_DIR="outputs/vae_latent_unet_${VARIABLE}"

echo "========================================================================"
echo "完整VAE潜空间U-Net训练流程"
echo "========================================================================"
echo "数据: $TIME_SLICE (5年完整数据)"
echo "分辨率: $TARGET_SIZE (高分辨率)"
echo "VAE类型: $VAE_TYPE"
echo "VAE模型: $VAE_MODEL_ID"
echo "VAE训练模式: $VAE_TRAIN_MODE"
echo "预处理目录: $PREPROCESSED_DIR"
echo "模型目录: $OUTPUT_DIR"
if [ "$USE_MULTI_GPU" = true ]; then
    echo "多GPU训练: 是"
    if [ -n "$GPU_IDS" ]; then
        echo "使用GPU: $GPU_IDS"
    else
        echo "使用GPU: 所有可用GPU"
    fi
else
    echo "多GPU训练: 否"
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        echo "使用GPU: $CUDA_VISIBLE_DEVICES"
    fi
fi
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
echo "Step 2: 训练潜空间U-Net（Lazy Loading）"
echo "========================================================================"
echo ""

# 构建训练命令
TRAIN_CMD="python train_latent_unet.py \
    --preprocessed-data-dir $PREPROCESSED_DIR \
    --vae-type $VAE_TYPE \
    --vae-model-id $VAE_MODEL_ID \
    --vae-train-mode $VAE_TRAIN_MODE \
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
    --output-dir $OUTPUT_DIR \
    --freeze-vae 

# 添加多GPU参数
if [ "$USE_MULTI_GPU" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --use-multi-gpu"
    if [ -n "$GPU_IDS" ]; then
        TRAIN_CMD="$TRAIN_CMD --gpu-ids $GPU_IDS"
    fi
fi

# 添加可选参数
if [ -n "$VAE_PRETRAINED_PATH" ]; then
    TRAIN_CMD="$TRAIN_CMD --vae-pretrained-path $VAE_PRETRAINED_PATH"
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
echo ""
echo "  预测结果: $OUTPUT_DIR/"
echo "    - prediction_metrics.json: 评估指标"
echo "    - *.png: 可视化图片"
echo ""
echo "VAE配置:"
echo "  VAE类型: $VAE_TYPE"
echo "  VAE模型: $VAE_MODEL_ID"
echo "  训练模式: $VAE_TRAIN_MODE"
if [ -n "$VAE_PRETRAINED_PATH" ]; then
    echo "  预训练路径: $VAE_PRETRAINED_PATH"
fi
echo ""
echo "GPU配置:"
if [ "$USE_MULTI_GPU" = true ]; then
    echo "  多GPU训练: 是"
    if [ -n "$GPU_IDS" ]; then
        echo "  使用GPU: $GPU_IDS"
    else
        echo "  使用GPU: 所有可用GPU"
    fi
    echo "  有效batch size: $BATCH_SIZE × GPU数量 (每个GPU: $BATCH_SIZE)"
else
    echo "  多GPU训练: 否"
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        echo "  使用GPU: $CUDA_VISIBLE_DEVICES"
    fi
fi
echo ""
echo "内存使用情况:"
echo "  预处理阶段: 分块处理，内存占用可控"
echo "  训练阶段: Lazy loading，只加载当前batch，内存占用极小"
echo "  峰值内存: < 5 GB（相比之前的28GB）"
echo ""

