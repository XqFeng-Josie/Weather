#!/bin/bash
set -e

# ============ 参数配置 ============
VARIABLES="${1:-specific_humidity}" # 2m_temperature,geopotential,specific_humidity
# TIME_SLICE="2020-01-01:2020-12-31"
TIME_SLICE="2015-01-01:2019-12-31"
PREDICTION_TIME_SLICE="2020-01-01:2020-12-31"
EPOCHS=30
EARLY_STOP=5
INPUT_LENGTH=12
OUTPUT_LENGTH=4
OUTPUT_DIR="outputs/lr_multi_$VARIABLES"
# if variables is geopotential, then add levels 500 700 850
if [ "$VARIABLES" == "geopotential" ]; then
    LEVELS="500"
elif [ "$VARIABLES" == "specific_humidity" ]; then
    LEVELS="700"
else
    LEVELS=""
fi

# ============ 训练 ============
echo "Training LR Multi..."
train_command="python train.py \
    --model lr_multi \
    --variables $VARIABLES \
    --time-slice $TIME_SLICE \
    --input-length $INPUT_LENGTH \
    --output-length $OUTPUT_LENGTH \
    --epochs $EPOCHS \
    --output-dir $OUTPUT_DIR \
    --early-stop $EARLY_STOP"
    
if [ -n "$LEVELS" ]; then
    train_command="$train_command --levels $LEVELS"
fi
eval $train_command

echo "Model saved to: $OUTPUT_DIR"

# ============ 预测 ============
echo "Generating predictions..."
predict_command="python predict.py \
    --model-path $OUTPUT_DIR/best_model.pth \
    --output $OUTPUT_DIR/predictions.nc \
    --time-slice $PREDICTION_TIME_SLICE \
    --visualize \
    --save-predictions"
if [ -n "$LEVELS" ]; then
    predict_command="$predict_command --levels $LEVELS"
fi

eval $predict_command

# # ============ 评估 ============
# echo "Evaluating with WeatherBench2..."
# python evaluate_weatherbench.py \
#     --pred $OUTPUT_DIR/predictions.nc \
#     --output-dir $OUTPUT_DIR/wb2_eval

echo "✓ Complete! Results in: $OUTPUT_DIR"

