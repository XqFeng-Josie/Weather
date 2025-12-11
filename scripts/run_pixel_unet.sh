#!/bin/bash
set -e

# ============================================================================
# 像素空间U-Net训练和预测脚本
# 
# 作用: 训练像素空间的U-Net模型，直接进行图像到图像的预测
# 优点: 不需要VAE，训练速度快，精度高
# 缺点: 内存占用较大，处理高分辨率图像时速度较慢
# ============================================================================

# ============ GPU设置 ============
# export CUDA_VISIBLE_DEVICES=6

# ============ 参数配置 ============
VARIABLES="${1:-specific_humidity}" # 2m_temperature,geopotential,specific_humidity
TIME_SLICE="2015-01-01:2019-12-31"          # 训练数据
PREDICTION_TIME_SLICE="2020-01-01:2020-12-31"  # 预测数据

# 模型参数
INPUT_LENGTH=12      # 输入12个时间步
OUTPUT_LENGTH=4      # 预测4个时间步
BASE_CHANNELS=64     # U-Net基础通道数
DEPTH=3             # U-Net深度

# 训练参数
EPOCHS=50
BATCH_SIZE=16
LR=0.0001
EARLY_STOPPING=10

if [ "$VARIABLES" == "geopotential" ]; then
    LEVELS="500"
elif [ "$VARIABLES" == "specific_humidity" ]; then
    LEVELS="700"
else
    LEVELS=""
fi


# 数据参数
NORMALIZATION="minmax"  # minmax或zscore
DATA_PATH="gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr"

# 输出目录
OUTPUT_DIR="outputs/pixel_unet_${VARIABLES}"

echo "========================================================================"
echo "像素空间U-Net训练和预测"
echo "========================================================================"
echo "变量: $VARIABLES"
echo "训练时间: $TIME_SLICE"
echo "预测时间: $PREDICTION_TIME_SLICE"
echo "输出目录: $OUTPUT_DIR"
echo ""

# ============ 训练 ============
echo "------------------------------------------------------------------------"
echo "Step 1: 训练像素空间U-Net"
echo "------------------------------------------------------------------------"

train_cmd="python train_pixel_unet.py \
    --data-path $DATA_PATH \
    --variables $VARIABLES \
    --time-slice $TIME_SLICE \
    --input-length $INPUT_LENGTH \
    --output-length $OUTPUT_LENGTH \
    --base-channels $BASE_CHANNELS \
    --depth $DEPTH \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --normalization $NORMALIZATION \
    --early-stopping $EARLY_STOPPING \
    --output-dir $OUTPUT_DIR"


if [ -n "$LEVELS" ]; then
    train_cmd="$train_cmd --levels $LEVELS"
fi
eval $train_cmd

echo ""
echo "✓ 训练完成! 模型保存在: $OUTPUT_DIR"

# ============ 预测 + 可视化 ============
echo ""
echo "------------------------------------------------------------------------"
echo "Step 2: 生成预测和可视化"
echo "------------------------------------------------------------------------"

predict_cmd="python predict_pixel_unet.py \
    --mode pixel \
    --model-dir $OUTPUT_DIR \
    --data-path $DATA_PATH \
    --time-slice $PREDICTION_TIME_SLICE \
    --output-dir $OUTPUT_DIR \
    --batch-size 32 "

if [ -n "$LEVELS" ]; then
    predict_cmd="$predict_cmd --levels $LEVELS"
fi
eval $predict_cmd

echo ""
echo "✓ 预测完成!"

# ============ 总结 ============
echo ""
echo "========================================================================"
echo "完成!"
echo "========================================================================"
echo "结果保存在: $OUTPUT_DIR"
echo ""
echo "输出文件:"
echo "  训练相关:"
echo "    - best_model.pt: 最佳模型权重"
echo "    - config.json: 模型配置"
echo "    - normalizer_stats.pkl: 归一化参数"
echo "    - training_history.json: 训练历史"
echo ""
echo "  预测相关:"
echo "    - prediction_metrics.json: 评估指标"
echo "    - y_pred_*.npy: 预测数据"
echo "    - y_true_*.npy: 真值数据"
echo ""
echo "  可视化:"
echo "    - timeseries_overall_*.png: 时间序列对比"
echo "    - spatial_comparison_*.png: 世界地图对比 ⭐"
echo "    - rmse_vs_leadtime_*.png: RMSE vs 预测步长"
echo ""
echo "查看指标:"
echo "  cat $OUTPUT_DIR/prediction_metrics.json"
echo ""

