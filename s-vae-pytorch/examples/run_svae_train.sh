#!/bin/bash
# S-VAE 训练脚本

# 默认参数
DATA_PATH="gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr"
VARIABLE="2m_temperature"
TIME_SLICE="2015-01-01:2019-12-31"
OUTPUT_DIR="outputs/svae_improved"
EPOCHS=600
BATCH_SIZE=32
LATENT_CHANNELS=4
HIDDEN_DIMS="64 128 256 512"
DISTRIBUTION="normal"  # "normal" or "vmf"
USE_RESIDUAL=true
LR=1e-4
DEVICE=""
LR_SCHEDULER="plateau"
LR_SCHEDULER_PARAMS='{"mode":"min","factor":0.5,"patience":10,"min_lr":1e-6}'
KL_WEIGHT=1e-6
KL_ANNEALING=true
AUGMENT=false
USE_ADVANCED_LOSS=true
GRAD_LOSS_WEIGHT=0.1
PERCEPTUAL_WEIGHT=1.0

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --variable)
            VARIABLE="$2"
            shift 2
            ;;
        --time-slice)
            TIME_SLICE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --latent-channels)
            LATENT_CHANNELS="$2"
            shift 2
            ;;
        --hidden-dims)
            HIDDEN_DIMS="$2"
            shift 2
            ;;
        --distribution)
            DISTRIBUTION="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --lr-scheduler)
            LR_SCHEDULER="$2"
            shift 2
            ;;
        --lr-scheduler-params)
            LR_SCHEDULER_PARAMS="$2"
            shift 2
            ;;
        --kl-weight)
            KL_WEIGHT="$2"
            shift 2
            ;;
        --kl-annealing)
            KL_ANNEALING=true
            shift
            ;;
        --augment)
            AUGMENT=true
            shift
            ;;
        --use-advanced-loss)
            USE_ADVANCED_LOSS=true
            shift
            ;;
        --grad-loss-weight)
            GRAD_LOSS_WEIGHT="$2"
            shift 2
            ;;
        --perceptual-weight)
            PERCEPTUAL_WEIGHT="$2"
            shift 2
            ;;
        --levels)
            LEVELS="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# 构建命令
CMD="python examples/train_weather_svae.py \
    --data-path $DATA_PATH \
    --variable $VARIABLE \
    --time-slice $TIME_SLICE \
    --output-dir $OUTPUT_DIR \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --latent-channels $LATENT_CHANNELS \
    --hidden-dims $HIDDEN_DIMS \
    --distribution $DISTRIBUTION \
    --lr $LR \
    --kl-weight $KL_WEIGHT \
    --save-model"

# 添加可选参数
if [ ! -z "$DEVICE" ]; then
    CMD="$CMD --device $DEVICE"
fi

if [ "$USE_RESIDUAL" = true ]; then
    CMD="$CMD --use-residual"
fi

if [ ! -z "$LR_SCHEDULER" ]; then
    CMD="$CMD --lr-scheduler $LR_SCHEDULER"
    if [ ! -z "$LR_SCHEDULER_PARAMS" ]; then
        CMD="$CMD --lr-scheduler-params '$LR_SCHEDULER_PARAMS'"
    fi
fi

if [ "$KL_ANNEALING" = true ]; then
    CMD="$CMD --kl-annealing"
fi

if [ "$AUGMENT" = true ]; then
    CMD="$CMD --augment"
fi

if [ "$USE_ADVANCED_LOSS" = true ]; then
    CMD="$CMD --use-advanced-loss"
    CMD="$CMD --grad-loss-weight $GRAD_LOSS_WEIGHT"
    CMD="$CMD --perceptual-weight $PERCEPTUAL_WEIGHT"
fi

if [ ! -z "$LEVELS" ]; then
    CMD="$CMD --levels $LEVELS"
fi

if [ ! -z "$RESUME" ]; then
    CMD="$CMD --resume $RESUME"
fi

echo "=========================================="
echo "S-VAE 训练"
echo "=========================================="
echo "数据路径: $DATA_PATH"
echo "变量: $VARIABLE"
echo "时间范围: $TIME_SLICE"
echo "输出目录: $OUTPUT_DIR"
echo "训练轮数: $EPOCHS"
echo "批次大小: $BATCH_SIZE"
echo "潜在通道数: $LATENT_CHANNELS"
echo "隐藏层维度: $HIDDEN_DIMS"
echo "分布类型: $DISTRIBUTION"
echo "使用残差连接: $USE_RESIDUAL"
echo "学习率: $LR"
if [ ! -z "$DEVICE" ]; then
    echo "设备: $DEVICE"
else
    echo "设备: 自动选择"
fi
if [ ! -z "$LR_SCHEDULER" ]; then
    echo "学习率调度器: $LR_SCHEDULER"
    if [ ! -z "$LR_SCHEDULER_PARAMS" ]; then
        echo "调度器参数: $LR_SCHEDULER_PARAMS"
    fi
else
    echo "学习率调度器: 无"
fi
echo "KL散度权重: $KL_WEIGHT"
if [ "$KL_ANNEALING" = true ]; then
    echo "KL散度退火: 启用"
else
    echo "KL散度退火: 禁用"
fi
if [ "$AUGMENT" = true ]; then
    echo "数据增强: 启用"
fi
if [ "$USE_ADVANCED_LOSS" = true ]; then
    echo "改进损失函数: 启用 (梯度损失权重=$GRAD_LOSS_WEIGHT, 感知损失权重=$PERCEPTUAL_WEIGHT)"
fi
if [ ! -z "$LEVELS" ]; then
    echo "Levels: $LEVELS"
fi
if [ ! -z "$RESUME" ]; then
    echo "Resume from: $RESUME"
fi
echo "=========================================="
echo ""

# 执行命令
eval $CMD

