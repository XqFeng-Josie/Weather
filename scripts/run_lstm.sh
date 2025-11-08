#!/bin/bash
set -e
# variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,total_precipitation_6hr,total_precipitation_24hr,10m_wind_speed,wind_speed

# ============ 参数配置 ============
VARIABLES="2m_temperature" # 2m_temperature,geopotential
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
OUTPUT_DIR="outputs/lstm_$VARIABLES"
# ============ 训练 ============
echo "Training lstm..."
python train.py \
    --model lstm \
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
    --output-dir $OUTPUT_DIR
# 获取最新输出目录
echo "Model saved to: $OUTPUT_DIR"

# ============ 预测 + 可视化 ============
# echo "Generating predictions and visualizations..."
python predict.py \
    --model-path $OUTPUT_DIR/best_model.pth \
    --output $OUTPUT_DIR/predictions.nc \
    --visualize \
    --save-predictions \
    --time-slice $PREDICTION_TIME_SLICE

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