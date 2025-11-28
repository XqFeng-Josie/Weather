#!/bin/bash
# RAE重建测试脚本
# 用于测试不同编码器（DINOv2-B, DINOv2-B_512, MAE, SigLIP2）的重建能力

# 注意：此脚本需要在 RAE 项目目录下运行，或确保 RAE 相关代码在路径中
# 配置路径（根据实际情况修改）
export CUDA_VISIBLE_DEVICES=5
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"


Weather_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Weather_DIR: $Weather_DIR"
source /data2/xiaoqinfeng/miniconda3/bin/activate rae
CONFIG_DIR=./configs/stage1/pretrained

# 准备天气图像（如果还没有）
if [ ! -d $Weather_DIR/reconstruction/weather_images ]; then
    echo "准备天气图像..."
    python $Weather_DIR/reconstruction/prepare_weather_images.py \
        --data-path gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr \
        --variable 2m_temperature \
        --time-slice 2020-01-01:2020-01-31 \
        --target-size 256 256 \
        --output-dir $Weather_DIR/reconstruction/weather_images \
        --n-samples 100
fi

echo "开始测试模型..."
# DINOv2-B DINOv2-B_512 MAE SigLIP2  DINOv2-B_decXL MAE_decXL
for model in  DINOv2-B MAE SigLIP2 DINOv2-B_decXL MAE_decXL; do
    echo "=========================================="
    echo "测试模型: $model"
    echo "=========================================="
    
    # 检查配置文件是否存在
    config_file="${CONFIG_DIR}/${model}.yaml"
    if [ ! -f "$config_file" ]; then
        echo "警告: 配置文件不存在: $config_file，跳过"
        continue
    fi
    mkdir -p $Weather_DIR/reconstruction/outputs/rae_reconstruction/recon_merged
    # 运行重建
    torchrun --standalone --nproc_per_node=1 \
        src/stage1_sample_ddp.py \
        --config "$config_file" \
        --data-path $Weather_DIR/reconstruction/weather_images \
        --sample-dir $Weather_DIR/reconstruction/outputs/rae_reconstruction/recon_samples_${model} \
        --per-proc-batch-size 4
    
    

    reconstructed_dir=$Weather_DIR/reconstruction/outputs/rae_reconstruction/recon_samples_${model}/RAE-pretrained-bs4-fp32
    if [ "$model" = "DINOv2-B_decXL" ]; then
        reconstructed_dir=$Weather_DIR/reconstruction/outputs/rae_reconstruction/recon_samples_${model}/RAE-best-bs4-fp32
    elif [ "$model" = "MAE_decXL" ]; then
        reconstructed_dir=$Weather_DIR/reconstruction/outputs/rae_reconstruction/recon_samples_${model}/RAE-best-bs4-fp32
    fi
    python $Weather_DIR/reconstruction/create_rae_reconstruction_comparison.py \
        --original-dir $Weather_DIR/reconstruction/weather_images/weather/ \
        --reconstructed-dir $reconstructed_dir \
        --output $Weather_DIR/reconstruction/outputs/rae_reconstruction/recon_merged/comparison_4samples_${model}.png \
        --indices 0 10 20 30
    
    echo "✓ 完成模型 $model 的测试"
done

echo ""
echo "=========================================="
echo "所有测试完成！"
echo "=========================================="
echo "结果保存在:"
echo "  - recon_samples_*/: 重建样本"
echo "  - recon_merged/: 对比图"

