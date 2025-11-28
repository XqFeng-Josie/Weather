#!/bin/bash
#
# 训练和评估Weather Transformer模型
# 用法: bash scripts/run_weather_transformer.sh
#

set -e  # 遇到错误立即退出

echo "=================================="
echo "Weather Transformer Training"
echo "=================================="

# 配置
DATA_PATH="gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr"
VARIABLES="${1:-specific_humidity}" # 2m_temperature,geopotential,specific_humidity
TIME_SLICE="2015-01-01:2019-12-31"
PREDICTION_TIME_SLICE="2020-01-01:2020-12-31"
OUTPUT_DIR="outputs/weather_transformer_$VARIABLES"

# 序列参数
INPUT_LENGTH=12
OUTPUT_LENGTH=4


# 模型参数（轻量级配置）
IMG_SIZE="32 64"
PATCH_SIZE="4 8"
D_MODEL=128
N_HEADS=4
N_LAYERS=4
DROPOUT=0.1

# 训练参数
EPOCHS=200
BATCH_SIZE=32
LR=1e-4
WEIGHT_DECAY=1e-5
EARLY_STOP=15


if [ "$VARIABLES" == "geopotential" ]; then
    LEVELS="500"
elif [ "$VARIABLES" == "specific_humidity" ]; then
    LEVELS="700"
else
    LEVELS=""
fi


echo ""
echo "配置:"
echo "  数据: $DATA_PATH"
echo "  变量: $VARIABLES"
echo "  时间: $TIME_SLICE"
echo "  模型: d_model=$D_MODEL, n_heads=$N_HEADS, n_layers=$N_LAYERS"
echo "  输出: $OUTPUT_DIR"
echo ""

# 训练
echo "开始训练..."
train_cmd="python train_weather_transformer.py \
    --data-path $DATA_PATH \
    --variables $VARIABLES \
    --time-slice $TIME_SLICE \
    --input-length $INPUT_LENGTH \
    --output-length $OUTPUT_LENGTH \
    --img-size $IMG_SIZE \
    --patch-size $PATCH_SIZE \
    --d-model $D_MODEL \
    --n-heads $N_HEADS \
    --n-layers $N_LAYERS \
    --dropout $DROPOUT \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --weight-decay $WEIGHT_DECAY \
    --early-stop $EARLY_STOP \
    --output-dir $OUTPUT_DIR"
if [ -n "$LEVELS" ]; then
    train_cmd="$train_cmd --levels $LEVELS"
fi
    
eval $train_cmd


echo ""
echo "训练完成！"

# 预测和可视化
echo ""
echo "开始预测和可视化..."
predict_cmd="python predict.py \
    --model-path $OUTPUT_DIR/best_model.pth \
    --data-path $DATA_PATH \
    --time-slice $PREDICTION_TIME_SLICE \
    --visualize \
    --save-predictions \
    --output $OUTPUT_DIR/predictions.nc"

if [ -n "$LEVELS" ]; then
    predict_cmd="$predict_cmd --levels $LEVELS"
fi
    
eval $predict_cmd

echo ""
echo "=================================="
echo "完成！"
echo "=================================="
echo ""
echo "查看结果:"
echo "  - 模型: $OUTPUT_DIR/best_model.pth"
echo "  - 配置: $OUTPUT_DIR/config.json"
echo "  - 指标: $OUTPUT_DIR/metrics.json"
echo "  - 可视化: $OUTPUT_DIR/*.png"
echo ""

