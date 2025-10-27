#!/bin/bash
# 完整的训练-预测-评估pipeline
# 用法: bash run_full_pipeline.sh [模型名称]

MODEL=${1:-lstm}  # 默认LSTM
YEAR=2020

echo "============================================"
echo "Weather Prediction Full Pipeline"
echo "Model: $MODEL"
echo "============================================"

# 1. 训练模型
echo ""
echo "Step 1/3: Training model..."
echo "--------------------------------------------"

python train.py \
    --model $MODEL \
    --time-slice "${YEAR}-01-01:${YEAR}-12-31" \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.001 \
    --single-point \
    --exp-name "${MODEL}_${YEAR}"

if [ $? -ne 0 ]; then
    echo "❌ Training failed!"
    exit 1
fi

# 2. 生成预测
echo ""
echo "Step 2/3: Generating predictions..."
echo "--------------------------------------------"

YEAR_TEST=2021
MODEL_PATH="outputs/${MODEL}_${YEAR}/best_model.pth"

if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model not found: $MODEL_PATH"
    exit 1
fi

python predict.py \
    --model-path "$MODEL_PATH" \
    --time-slice "${YEAR_TEST}-01-01:${YEAR_TEST}-12-31" \
    --output "outputs/${MODEL}_${YEAR}/predictions.npz" \
    --format numpy

if [ $? -ne 0 ]; then
    echo "❌ Prediction failed!"
    exit 1
fi

# 3. 评估
echo ""
echo "Step 3/3: Evaluating predictions..."
echo "--------------------------------------------"

python evaluate_weatherbench.py \
    --pred "outputs/${MODEL}_${YEAR}/predictions.npz" \
    --output-dir "outputs/${MODEL}_${YEAR}/evaluation"

if [ $? -ne 0 ]; then
    echo "❌ Evaluation failed!"
    exit 1
fi

# 完成
echo ""
echo "============================================"
echo "✓ Pipeline completed successfully!"
echo "============================================"
echo "Results:"
echo "  - Model:       outputs/${MODEL}_${YEAR}/best_model.pth"
echo "  - Predictions: outputs/${MODEL}_${YEAR}/predictions.npz"
echo "  - Evaluation:  outputs/${MODEL}_${YEAR}/evaluation/"
echo ""
echo "View results:"
echo "  cat outputs/${MODEL}_${YEAR}/metrics.json"
echo "  open outputs/${MODEL}_${YEAR}/evaluation/rmse_by_leadtime.png"

