#!/bin/bash
# Demo script to generate prediction comparison video

set -e

echo "=========================================="
echo "Prediction Video Generator Demo"
echo "=========================================="

# 设置路径
PREDICTIONS="outputs/cnn_2m_temperature/predictions.nc"
PREDICTIONS_DATA_DIR="outputs/cnn_2m_temperature/predictions_data"
OUTPUT_DIR="outputs/cnn_2m_temperature"
OUTPUT_VIDEO="${OUTPUT_DIR}/prediction_video_6h.mp4"

# 检查predictions_data文件夹是否存在（优先使用原始尺度数据）
if [ -d "$PREDICTIONS_DATA_DIR" ] && [ -f "${PREDICTIONS_DATA_DIR}/y_test_pred.npy" ]; then
    echo "Using original scale data from: $PREDICTIONS_DATA_DIR"
    USE_NPY="--use-npy"
    INPUT_INFO="$PREDICTIONS_DATA_DIR (original scale)"
elif [ -f "$PREDICTIONS" ]; then
    echo "Using normalized data from: $PREDICTIONS"
    echo "Note: Will attempt to denormalize using config.json if available"
    USE_NPY=""
    INPUT_INFO="$PREDICTIONS (will denormalize)"
else
    echo "Error: Neither predictions_data directory nor predictions.nc file found."
    echo "Please run predict.py first to generate predictions."
    exit 1
fi

echo ""
echo "Generating video for 6-hour lead time prediction..."
echo "Input: $INPUT_INFO"
echo "Output: $OUTPUT_VIDEO"
echo ""

# 生成视频（6小时预测，lead_time=0）
if [ -n "$USE_NPY" ]; then
    # 使用原始尺度的.npy文件
    python create_prediction_video.py \
        --predictions "$PREDICTIONS" \
        $USE_NPY \
        --time-slice "2020-01-01:2020-12-31" \
        --output "$OUTPUT_VIDEO" \
        --lead-time 0 \
        --fps 10 \
        --dpi 100
else
    # 使用NetCDF文件（尝试自动反归一化）
    python create_prediction_video.py \
        --predictions "$PREDICTIONS" \
        --config "${OUTPUT_DIR}/config.json" \
        --time-slice "2020-01-01:2020-12-31" \
        --output "$OUTPUT_VIDEO" \
        --lead-time 0 \
        --fps 10 \
        --dpi 100
fi

echo ""
echo "=========================================="
echo "Video generation complete!"
echo "Output: $OUTPUT_VIDEO"
echo "=========================================="

