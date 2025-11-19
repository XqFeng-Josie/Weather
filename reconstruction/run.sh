#!/bin/bash
# 完整的 VAE vs RAE 对比工作流程示例


set -e  # 遇到错误立即退出

echo "=========================================="
echo "VAE vs RAE 重建对比工作流程"
echo "=========================================="

# 获取当前脚本所在目录的上一级目录
PARENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"



# 配置
DATA_PATH="gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr"
VARIABLE="2m_temperature"
TIME_SLICE="2020-01-01:2020-01-31"
N_SAMPLES=100
TARGET_SIZE="256 256"
OUTPUT_DIR="outputs"

# 步骤 1: 准备天气图像（VAE 和 RAE 共用）
echo ""
echo "步骤 1: 准备天气图像..."
echo "----------------------------------------"
if [ ! -d "weather_images" ]; then
    python $PARENT_DIR/reconstruction/prepare_weather_images.py \
        --data-path "$DATA_PATH" \
        --variable "$VARIABLE" \
        --time-slice "$TIME_SLICE" \
        --target-size $TARGET_SIZE \
        --output-dir weather_images \
        --n-samples $N_SAMPLES
    echo "✓ 天气图像准备完成"
else
    echo "✓ 天气图像已存在，跳过"
fi

# 步骤 2: 运行 VAE 重建测试（从图像目录加载）
echo ""
echo "步骤 2: 运行 VAE 重建测试..."
echo "----------------------------------------"
cd ..  # 回到 Weather 项目根目录
python $PARENT_DIR/reconstruction/test_vae_reconstruction.py \
    --data-path $PARENT_DIR/reconstruction/weather_images \
    --n-test-samples $N_SAMPLES \
    --output-dir $PARENT_DIR/reconstruction/$OUTPUT_DIR/vae_reconstruction \
    --save-separate
echo "✓ VAE 重建测试完成"
cd $PARENT_DIR/reconstruction

# 步骤 3: 运行 RAE 重建测试（如果 RAE 项目可用）
echo ""
echo "步骤 3: 运行 RAE 重建测试..."
echo "----------------------------------------"
RAE_DIR=$PARENT_DIR/../RAE
if [ -d "$RAE_DIR" ]; then
    export RAE_DIR=$RAE_DIR
    cd $RAE_DIR
    bash $PARENT_DIR/reconstruction/test_rae_reconstruction.sh
    echo "✓ RAE 重建测试完成"
else
    echo "⚠ RAE 目录不存在，跳过 RAE 测试"
fi

# 步骤 4: 统一对比分析
echo ""
echo "步骤 4: 统一对比分析..."
echo "----------------------------------------"

# 检查 VAE 和 RAE 输出是否存在
VAE_RECON_DIR="$PARENT_DIR/reconstruction/$OUTPUT_DIR/vae_reconstruction/recon_samples/stable-diffusion-v1-5-stable-diffusion-v1-5/"
RAE_SigLIP2_RECON_DIR="$PARENT_DIR/reconstruction/$OUTPUT_DIR/rae_reconstruction/recon_samples_SigLIP2/RAE-pretrained-bs4-fp32"
RAE_DINOv2_B_RECON_DIR="$PARENT_DIR/reconstruction/$OUTPUT_DIR/rae_reconstruction/recon_samples_DINOv2-B/RAE-pretrained-bs4-fp32"
RAE_MAE_RECON_DIR="$PARENT_DIR/reconstruction/$OUTPUT_DIR/rae_reconstruction/recon_samples_MAE/RAE-pretrained-bs4-fp32"

if [ -d "$VAE_RECON_DIR" ] && [ -d "$RAE_SigLIP2_RECON_DIR" ] && [ -d "$RAE_DINOv2_B_RECON_DIR" ] && [ -d "$RAE_MAE_RECON_DIR" ]; then
    # 对比 VAE 和 RAE
    python $PARENT_DIR/reconstruction/compare_reconstructions.py \
        --original-dir $PARENT_DIR/reconstruction/weather_images \
        --reconstructed-dirs "$VAE_RECON_DIR" "$RAE_SigLIP2_RECON_DIR" "$RAE_DINOv2_B_RECON_DIR"  "$RAE_MAE_RECON_DIR" \
        --labels VAE RAE-SigLIP2 RAE-DINOv2-B RAE-MAE \
        --output "$PARENT_DIR/reconstruction/$OUTPUT_DIR/comparison_vae_vs_rae.png" \
        --metrics-output "$PARENT_DIR/reconstruction/$OUTPUT_DIR/metrics_vae_vs_rae.json" \
        --table-output "$PARENT_DIR/reconstruction/$OUTPUT_DIR/metrics_table.png" \
        --denormalize \
        --indices 0 10 20 30
    echo "✓ VAE vs RAE 对比完成"
elif [ -d "$VAE_RECON_DIR" ]; then
    # 只对比 VAE
    python $PARENT_DIR/reconstruction/compare_reconstructions.py \
        --original-dir $PARENT_DIR/reconstruction/weather_images \
        --reconstructed-dir "$VAE_RECON_DIR" \
        --output "$PARENT_DIR/reconstruction/$OUTPUT_DIR/comparison_vae.png" \
        --metrics-output "$PARENT_DIR/reconstruction/$OUTPUT_DIR/metrics_vae.json" \
        --denormalize
    echo "✓ VAE 对比完成"
else
    echo "⚠ 未找到重建结果，跳过对比"
fi

echo ""
echo "=========================================="
echo "✅ 工作流程完成！"
echo "=========================================="
echo ""
echo "结果保存在: $OUTPUT_DIR/"
echo "  - comparison_*.png: 对比可视化"
echo "  - metrics_*.json: 评估指标"
echo "  - metrics_table.png: 指标对比表"

