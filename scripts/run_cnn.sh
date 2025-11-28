#!/bin/bash
set -e


# ============ 参数配置 ============

VARIABLES="${1:-specific_humidity}" # 2m_temperature,geopotential,specific_humidity
TIME_SLICE="2015-01-01:2019-12-31"
PREDICTION_TIME_SLICE="2020-01-01:2020-12-31"
EPOCHS=100
BATCH_SIZE=32
HIDDEN_SIZE=256
DROPOUT=0.2
LR=0.0001
INPUT_LENGTH=12
OUTPUT_LENGTH=4
OUTPUT_DIR="outputs/cnn_$VARIABLES"
if [ "$VARIABLES" == "geopotential" ]; then
    LEVELS="500"
elif [ "$VARIABLES" == "specific_humidity" ]; then
    LEVELS="700"
else
    LEVELS=""
fi

# ============ 训练 ============
echo "Training cnn..."
train_cmd="python train.py \
    --model cnn \
    --variables $VARIABLES \
    --time-slice $TIME_SLICE \
    --input-length $INPUT_LENGTH \
    --output-length $OUTPUT_LENGTH \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --hidden-size $HIDDEN_SIZE \
    --dropout $DROPOUT \
    --lr $LR \
    --output-dir $OUTPUT_DIR"
    
if [ "$LEVELS" != "" ]; then
    train_cmd="$train_cmd --levels $LEVELS"
fi

# eval $train_cmd
# 获取最新输出目录
echo "Model saved to: $OUTPUT_DIR"

# ============ 预测 + 可视化 ============
echo "Generating predictions and visualizations..."
predict_cmd="python predict.py \
    --model-path $OUTPUT_DIR/best_model.pth \
    --output $OUTPUT_DIR/predictions.nc \
    --time-slice $PREDICTION_TIME_SLICE \
    --visualize \
    --save-predictions"
    
if [ "$LEVELS" != "" ]; then
    predict_cmd="$predict_cmd --levels $LEVELS"
fi

eval $predict_cmd
# ============ 评估 ============
# echo "Evaluating with WeatherBench2..."
# python evaluate_weatherbench.py \
#     --pred $OUTPUT_DIR/predictions.nc \
#     --output-dir $OUTPUT_DIR/wb2_eval

echo "✓ Complete! Results in: $OUTPUT_DIR"
echo "  - Model: $OUTPUT_DIR/best_model.pth"
echo "  - Predictions: $OUTPUT_DIR/predictions.nc"
echo "  - Visualizations: $OUTPUT_DIR/predictions_*.png"
echo "  - Metrics: $OUTPUT_DIR/prediction_metrics.json"

