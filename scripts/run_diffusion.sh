#!/bin/bash
set -e

# ============================================================================
# 完整的扩散模型训练流程（支持大规模数据）
# 
# 作用: 训练概率式扩散模型，生成多样化的未来预测
# 优点: 提供不确定性估计，支持集成预测
# 缺点: 训练时间长，推理慢（需要多步采样）
# 
# 分三步：
# Step 1: 预处理数据（一次性，可能需要1-2小时）
# Step 2: 训练模型（使用lazy loading，内存占用小）
# Step 3: 预测和评估
# 
# 💡 提示: Latent U-Net 和 Diffusion 可以共享预处理数据
# ============================================================================

# ============ GPU设置 ============
# export CUDA_VISIBLE_DEVICES=6

# ============ 参数配置 ============
VARIABLE="2m_temperature"
TIME_SLICE="2015-01-01:2019-12-31"          # 训练数据
PREDICTION_TIME_SLICE="2020-01-01:2020-12-31"  # 预测数据

# 模型参数
VAE_MODEL_ID="stable-diffusion-v1-5/stable-diffusion-v1-5"
INPUT_LENGTH=12      # 输入12个时间步
OUTPUT_LENGTH=4      # 预测4个时间步
BASE_CHANNELS=128    # 基础通道数
DEPTH=3              # 网络深度

# 扩散参数
NUM_TRAIN_TIMESTEPS=1000    # 训练扩散步数
BETA_SCHEDULE="linear"      # beta调度方式
NUM_INFERENCE_STEPS=100     # 推理步数（越多越好但越慢）

# 训练参数
EPOCHS=20
BATCH_SIZE=8         # 扩散模型batch size通常较小
VAE_BATCH_SIZE=4     # VAE编码子批次（控制显存）
LR=0.0001
SAVE_INTERVAL=10

# 预测参数
NUM_SAMPLES=1        # 每个输入生成的样本数（>1表示集成预测）
SAMPLING_METHOD="ddpm"  # ddpm或ddim

# 数据参数
NORMALIZATION="minmax"
TARGET_SIZE="512,512"
DATA_PATH="gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr"

# 预处理数据目录（推荐使用预处理数据以避免OOM）
PREPROCESSED_DIR="data/preprocessed/vae_pre_${TIME_SLICE//:/_}_${TARGET_SIZE//,/x}"
# 或者使用与latent_unet共享的预处理数据:
# PREPROCESSED_DIR="data/preprocessed/latent_unet_${TIME_SLICE//:/_}_${TARGET_SIZE//,/x}"

# 输出目录
OUTPUT_DIR="outputs/diffusion"

echo "========================================================================"
echo "扩散模型训练和预测"
echo "========================================================================"
echo "变量: $VARIABLE"
echo "训练时间: $TIME_SLICE"
echo "预测时间: $PREDICTION_TIME_SLICE"
echo "VAE模型: $VAE_MODEL_ID"
echo "图像尺寸: $TARGET_SIZE"
echo "扩散步数: $NUM_TRAIN_TIMESTEPS"
echo "预处理目录: $PREPROCESSED_DIR"
echo "输出目录: $OUTPUT_DIR"
echo ""

# ============================================================================
# Step 1: 预处理数据（如果还没有预处理）
# ============================================================================
if [ ! -d "$PREPROCESSED_DIR" ]; then
    echo "========================================================================"
    echo "Step 1: 预处理数据"
    echo "========================================================================"
    echo "这可能需要1-2小时，但只需运行一次"
    echo "预计输出大小: ~20-30 GB"
    echo "💡 提示: Latent U-Net 和 Diffusion 可以共享预处理数据"
    echo ""
    
    python preprocess_data_for_latent_unet.py \
        --data-path $DATA_PATH \
        --variable $VARIABLE \
        --time-slice $TIME_SLICE \
        --target-size $TARGET_SIZE \
        --n-channels 3 \
        --normalization $NORMALIZATION \
        --output-dir $PREPROCESSED_DIR \
        --chunk-size 100
    
    echo ""
    echo "✓ 数据预处理完成！"
else
    echo "========================================================================"
    echo "跳过预处理（数据已存在）"
    echo "========================================================================"
    echo "使用现有数据: $PREPROCESSED_DIR"
    
    # 显示预处理数据信息
    if [ -f "$PREPROCESSED_DIR/metadata.json" ]; then
        echo ""
        echo "数据信息:"
        cat $PREPROCESSED_DIR/metadata.json | python -m json.tool | grep -E "variable|n_timesteps|shape|target_size"
    fi
fi

echo ""

# ============================================================================
# Step 2: 训练扩散模型（Lazy Loading）
# ============================================================================
echo "========================================================================"
echo "Step 2: 训练扩散模型（Lazy Loading）"
echo "========================================================================"

# 使用预处理数据训练（推荐）
python train_diffusion.py \
    --preprocessed-data-dir $PREPROCESSED_DIR \
    --variable $VARIABLE \
    --time-slice $TIME_SLICE \
    --vae-model-id $VAE_MODEL_ID \
    --input-length $INPUT_LENGTH \
    --output-length $OUTPUT_LENGTH \
    --base-channels $BASE_CHANNELS \
    --depth $DEPTH \
    --num-train-timesteps $NUM_TRAIN_TIMESTEPS \
    --beta-schedule $BETA_SCHEDULE \
    --batch-size $BATCH_SIZE \
    --vae-batch-size $VAE_BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --normalization $NORMALIZATION \
    --save-interval $SAVE_INTERVAL \
    --output-dir $OUTPUT_DIR

echo ""
echo "✓ 训练完成! 模型保存在: $OUTPUT_DIR"

# ============================================================================
# Step 3: 预测和评估
# ============================================================================
echo ""
echo "========================================================================"
echo "Step 3: 预测和评估"
echo "========================================================================"

python predict_diffusion.py \
    --model-dir $OUTPUT_DIR \
    --data-path $DATA_PATH \
    --time-slice $PREDICTION_TIME_SLICE \
    --sampling-method $SAMPLING_METHOD \
    --num-inference-steps $NUM_INFERENCE_STEPS \
    --num-samples $NUM_SAMPLES \
    --batch-size 16 \
    --vae-batch-size $VAE_BATCH_SIZE

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
echo "提示:"
echo "  - 扩散模型提供概率预测，可以估计不确定性"
echo "  - 使用 --num-samples > 1 进行集成预测"
echo "  - 使用 --sampling-method ddim 可以更快采样"
echo "  - 预处理数据可与Latent U-Net共享（节省磁盘空间）"
echo ""
echo "集成预测示例:"
echo "  python predict_diffusion.py \\"
echo "      --model-dir $OUTPUT_DIR \\"
echo "      --num-samples 10 \\"
echo "      --sampling-method ddim \\"
echo "      --num-inference-steps 50"
echo ""
echo "💡 下次训练可以复用预处理数据:"
echo "   只需将脚本开头的 PREPROCESSED_DIR 改为现有目录"
echo "   Latent U-Net 和 Diffusion 可以共享相同的预处理数据"
echo ""


