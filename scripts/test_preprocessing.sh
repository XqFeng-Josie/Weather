#!/bin/bash
set -e

# ============================================================================
# 测试预处理流程（小数据集）
# 用于验证预处理和lazy loading是否正常工作
# ============================================================================

export CUDA_VISIBLE_DEVICES=6

# 使用1个月数据测试
VARIABLE="2m_temperature"
TIME_SLICE="2020-01-01:2020-01-31"
TARGET_SIZE="256,256"
PREPROCESSED_DIR="data/preprocessed/test_latent_256x256"

echo "========================================================================"
echo "测试预处理流程（小数据集）"
echo "========================================================================"
echo "时间范围: $TIME_SLICE (1个月)"
echo "分辨率: $TARGET_SIZE"
echo "输出: $PREPROCESSED_DIR"
echo ""

# 预处理
python preprocess_data_for_latent_unet.py \
    --variable $VARIABLE \
    --time-slice $TIME_SLICE \
    --target-size $TARGET_SIZE \
    --n-channels 3 \
    --normalization minmax \
    --output-dir $PREPROCESSED_DIR \
    --chunk-size 50

echo ""
echo "✓ 预处理完成！"
echo ""

# 显示结果
echo "输出文件:"
ls -lh $PREPROCESSED_DIR/

echo ""
echo "元数据:"
cat $PREPROCESSED_DIR/metadata.json | python -m json.tool

echo ""
echo "测试完成！可以使用以下命令训练:"
echo "python train_latent_unet.py --preprocessed-data-dir $PREPROCESSED_DIR --epochs 5"

