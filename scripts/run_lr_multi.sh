#!/bin/bash
set -e

# ============ 参数配置 ============
VARIABLES="2m_temperature"
# TIME_SLICE="2020-01-01:2020-12-31"
TIME_SLICE="2000-01-01:2016-12-31"
PREDICTION_TIME_SLICE="2017-01-01:2018-12-31"

EPOCHS=20
INPUT_LENGTH=12
OUTPUT_LENGTH=4
OUTPUT_DIR="outputs/long_lr_multi_$VARIABLES"

# ============ 训练 ============
echo "Training lr_multi..."
python train.py \
    --model lr_multi \
    --variables $VARIABLES \
    --time-slice $TIME_SLICE \
    --input-length $INPUT_LENGTH \
    --output-length $OUTPUT_LENGTH \
    --epochs $EPOCHS \
    --output-dir $OUTPUT_DIR

echo "Model saved to: $OUTPUT_DIR"

# ============ 预测 ============
echo "Generating predictions..."
python predict.py \
    --model-path $OUTPUT_DIR/best_model.pth \
    --output $OUTPUT_DIR/predictions.nc \
    --time-slice $PREDICTION_TIME_SLICE

# ============ 评估 ============
echo "Evaluating with WeatherBench2..."
python evaluate_weatherbench.py \
    --pred $OUTPUT_DIR/predictions.nc \
    --output-dir $OUTPUT_DIR/wb2_eval

echo "✓ Complete! Results in: $OUTPUT_DIR"

