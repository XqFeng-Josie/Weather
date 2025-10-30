#!/bin/bash
set -e

# ============ 参数配置 ============
VARIABLES="2m_temperature"
TIME_SLICE="2020-01-01:2020-12-31"
EPOCHS=30
BATCH_SIZE=32
HIDDEN_SIZE=128
DROPOUT=0.2
LR=0.001
INPUT_LENGTH=12
OUTPUT_LENGTH=4
OUTPUT_DIR="outputs/cnn_$VARIABLES"

# ============ 训练 ============
echo "Training cnn..."
python train.py \
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
    --output-dir $OUTPUT_DIR
# 获取最新输出目录
echo "Model saved to: $OUTPUT_DIR"

# ============ 预测 ============
echo "Generating predictions..."
python predict.py \
    --model-path $OUTPUT_DIR/best_model.pth \
    --output $OUTPUT_DIR/predictions.nc

# ============ 评估 ============
echo "Evaluating with WeatherBench2..."
python evaluate_weatherbench.py \
    --pred $OUTPUT_DIR/predictions.nc \
    --output-dir $OUTPUT_DIR/wb2_eval

echo "✓ Complete! Results in: $OUTPUT_DIR"

