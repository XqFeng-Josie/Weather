#!/bin/bash
set -e

# ============ 参数配置 ============
VARIABLES="2m_temperature"
TIME_SLICE="2019-01-01:2020-12-31"
EPOCHS=200
BATCH_SIZE=16
LR=5e-5
BASE_CHANNELS=64
BETA_SCHEDULE="cosine"
NUM_INFERENCE_STEPS=100
NUM_ENSEMBLE_MEMBERS=20
INPUT_LENGTH=12
OUTPUT_LENGTH=4
OUTPUT_DIR="outputs/diffusion_ensemble_$VARIABLES"

# ============ 训练 + Ensemble 评估 ============
echo "Training diffusion with ensemble evaluation..."
python train_diffusion.py \
    --variables $VARIABLES \
    --time-slice $TIME_SLICE \
    --input-length $INPUT_LENGTH \
    --output-length $OUTPUT_LENGTH \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --base-channels $BASE_CHANNELS \
    --beta-schedule $BETA_SCHEDULE \
    --enable-ensemble-eval \
    --num-ensemble-members $NUM_ENSEMBLE_MEMBERS \
    --num-inference-steps $NUM_INFERENCE_STEPS \
    --output-dir $OUTPUT_DIR

echo "Model saved to: $OUTPUT_DIR"

# ============ 预测 ============
echo "Generating predictions..."
python predict.py \
    --model-path $OUTPUT_DIR/best_model.pth \
    --output $OUTPUT_DIR/predictions.nc \
    --num-inference-steps $NUM_INFERENCE_STEPS

# ============ 评估 ============
echo "Evaluating with WeatherBench2..."
python evaluate_weatherbench.py \
    --pred $OUTPUT_DIR/predictions.nc \
    --output-dir $OUTPUT_DIR/wb2_eval

echo "✓ Complete! Results in: $OUTPUT_DIR"
echo "✓ Check metrics.json for CRPS and Spread-Skill Ratio"

