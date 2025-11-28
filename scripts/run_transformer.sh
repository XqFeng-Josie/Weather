#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=6
# ============ 参数配置 ============
VARIABLES="${1:-specific_humidity}" # 2m_temperature,geopotential,specific_humidity
# TIME_SLICE="2020-01-01:2020-12-31"
TIME_SLICE="2015-01-01:2019-12-31"
PREDICTION_TIME_SLICE="2020-01-01:2020-12-31"
EPOCHS=200
BATCH_SIZE=32
HIDDEN_SIZE=256
NUM_LAYERS=2
DROPOUT=0.2
LR=0.0001
INPUT_LENGTH=12
OUTPUT_LENGTH=4
OUTPUT_DIR="outputs/transformer_$VARIABLES"


if [ "$VARIABLES" == "geopotential" ]; then
    LEVELS="500"
elif [ "$VARIABLES" == "specific_humidity" ]; then
    LEVELS="700"
else
    LEVELS=""
fi


# ============ 训练 ============
echo "Training transformer..."
train_cmd="python train.py \
    --model transformer \
    --variables $VARIABLES \
    --time-slice $TIME_SLICE \
    --input-length $INPUT_LENGTH \
    --output-length $OUTPUT_LENGTH \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --hidden-size $HIDDEN_SIZE \
    --num-layers $NUM_LAYERS \
    --dropout $DROPOUT \
    --lr $LR \
    --output-dir $OUTPUT_DIR"

if [ -n "$LEVELS" ]; then
    train_cmd="$train_cmd --levels $LEVELS"
fi
eval $train_cmd

echo "Model saved to: $OUTPUT_DIR"

# ============ 预测 ============
echo "Generating predictions..."
predict_cmd="python predict.py \
    --model-path $OUTPUT_DIR/best_model.pth \
    --output $OUTPUT_DIR/predictions.nc \
    --time-slice $PREDICTION_TIME_SLICE \
    --visualize \
    --save-predictions"

if [ -n "$LEVELS" ]; then
    predict_cmd="$predict_cmd --levels $LEVELS"
fi
    
eval $predict_cmd
# ============ 评估 ============
# echo "Evaluating with WeatherBench2..."
# python evaluate_weatherbench.py \
#     --pred $OUTPUT_DIR/predictions.nc \
#     --output-dir $OUTPUT_DIR/wb2_eval

echo "✓ Complete! Results in: $OUTPUT_DIR"

