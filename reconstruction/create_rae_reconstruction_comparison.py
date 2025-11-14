#!/usr/bin/env python3
"""
生成原始图和重建图的对比可视化
选择4组结果，放在一个大图中

使用方法:
    python create_reconstruction_comparison.py \
        --original-dir weather_images \
        --reconstructed-dir recon_samples_DINOv2-B/RAE-pretrained-bs4-fp32 \
        --output comparison_4samples.png \
        --indices 0 10 20 30
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import warnings
warnings.filterwarnings('ignore')


def load_image(image_path: Path) -> np.ndarray:
    """加载图片为 numpy 数组"""
    img = Image.open(image_path).convert('RGB')
    return np.array(img)


def prepare_image_for_display(image: np.ndarray,
                              colormap: Optional[str]):
    """
    假定图像为灰度图，直接应用伪彩色
    """
    if colormap is None:
        return image, None

    if image.ndim == 2:
        return image, colormap

    return image[:, :, 0], colormap


def create_comparison_grid(original_dir: Path,
                          reconstructed_dir: Path,
                          indices: list,
                          output_path: Path,
                          figsize: tuple = (16, 8),
                          colormap: Optional[str] = None):
    """
    创建4组原始vs重建的对比图
    
    Args:
        original_dir: 原始图片目录
        reconstructed_dir: 重建图片目录
        indices: 要对比的图片索引列表（4个）
        output_path: 输出路径
        figsize: 图片尺寸
    """
    # 确保有4个索引
    if len(indices) != 4:
        raise ValueError(f"需要4个索引，但提供了{len(indices)}个")
    
    # 创建2行4列的网格（每行：原始，重建）
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    
    # 获取图片文件列表
    original_files = sorted(original_dir.glob("*.png"))
    reconstructed_files = sorted(reconstructed_dir.glob("*.png"))
    
    if len(original_files) == 0:
        raise ValueError(f"在 {original_dir} 中未找到图片文件")
    if len(reconstructed_files) == 0:
        raise ValueError(f"在 {reconstructed_dir} 中未找到图片文件")
    
    print(f"原始图片目录: {original_dir} ({len(original_files)} 张)")
    print(f"重建图片目录: {reconstructed_dir} ({len(reconstructed_files)} 张)")
    
    # 创建索引到文件路径的映射
    def get_file_by_index(files, idx, prefix="sample_"):
        """根据索引获取文件，支持不同的命名格式"""
        # 尝试多种命名格式
        patterns = [
            f"{prefix}{idx:06d}.png",  # sample_000000.png
            f"{idx:06d}.png",          # 000000.png
            f"{prefix}_{idx:06d}.png", # sample_000000.png (带下划线)
        ]
        
        for pattern in patterns:
            matching = [f for f in files if f.name == pattern]
            if matching:
                return matching[0]
        
        # 如果找不到，尝试按索引直接访问
        if idx < len(files):
            return files[idx]
        
        return None
    
    # 处理每组对比
    for col, idx in enumerate(indices):
        # 获取文件路径
        orig_path = get_file_by_index(original_files, idx, prefix="sample_")
        recon_path = get_file_by_index(reconstructed_files, idx, prefix="")
        
        if orig_path is None:
            print(f"⚠ 警告: 无法找到索引 {idx} 的原始图片，跳过")
            continue
        if recon_path is None:
            print(f"⚠ 警告: 无法找到索引 {idx} 的重建图片，跳过")
            continue
        
        try:
            orig_img = load_image(orig_path)
            recon_img = load_image(recon_path)
        except Exception as e:
            print(f"⚠ 警告: 无法加载索引 {idx} 的图片: {e}")
            continue
        
        # 第一行：原始图片
        orig_display, orig_cmap = prepare_image_for_display(
            orig_img, colormap
        )
        axes[0, col].imshow(orig_display, cmap=orig_cmap)
        axes[0, col].set_title(f'Sample {idx:06d}\nOriginal', fontsize=10)
        axes[0, col].axis('off')
        
        # 第二行：重建图片
        recon_display, recon_cmap = prepare_image_for_display(
            recon_img, colormap
        )
        axes[1, col].imshow(recon_display, cmap=recon_cmap)
        axes[1, col].set_title(f'Sample {idx:06d}\nReconstructed', fontsize=10)
        axes[1, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 对比图保存到: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="生成原始图和重建图的对比可视化"
    )
    
    parser.add_argument(
        "--original-dir",
        type=str,
        required=True,
        help="原始图片目录",
    )
    parser.add_argument(
        "--reconstructed-dir",
        type=str,
        required=True,
        help="重建图片目录",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reconstruction_comparison_4samples.png",
        help="输出图片路径",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs='+',
        default=None,
        help="要对比的图片索引列表（4个），如果不指定则自动选择",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="随机选择4组图片（当未指定 --indices 时）",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[16, 8],
        help="图片尺寸 [width height]，默认 [16 8]",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="RdYlBu_r",
        help="灰度图采用的伪彩色 colormap，默认 RdYlBu_r，设为 none 关闭",
    )
    args = parser.parse_args()
    
    original_dir = Path(args.original_dir)
    reconstructed_dir = Path(args.reconstructed_dir)
    output_path = Path(args.output)
    
    # 检查目录是否存在
    if not original_dir.exists():
        raise FileNotFoundError(f"原始图片目录不存在: {original_dir}")
    if not reconstructed_dir.exists():
        raise FileNotFoundError(f"重建图片目录不存在: {reconstructed_dir}")
    
    # 确定要对比的索引
    original_files = sorted(original_dir.glob("*.png"))
    reconstructed_files = sorted(reconstructed_dir.glob("*.png"))
    
    if len(original_files) == 0:
        raise ValueError(f"在 {original_dir} 中未找到图片文件")
    if len(reconstructed_files) == 0:
        raise ValueError(f"在 {reconstructed_dir} 中未找到图片文件")
    
    max_available = min(len(original_files), len(reconstructed_files))
    
    if args.indices is not None:
        # 使用指定的索引
        indices = args.indices[:4]  # 只取前4个
        if len(indices) < 4:
            print(f"⚠ 警告: 只提供了 {len(indices)} 个索引，需要4个")
            # 补充索引
            for i in range(len(indices), 4):
                if i < max_available:
                    indices.append(i)
                else:
                    indices.append(max_available - 1)
    else:
        # 自动选择索引
        if args.random:
            # 随机选择
            indices = sorted(np.random.choice(max_available, size=min(4, max_available), replace=False).tolist())
        else:
            # 均匀分布选择
            if max_available >= 4:
                step = max_available // 4
                indices = [i * step for i in range(4)]
            else:
                indices = list(range(max_available))
    
    # 确保有4个索引
    while len(indices) < 4 and len(indices) < max_available:
        # 添加额外的索引
        next_idx = max(indices) + 1
        if next_idx < max_available:
            indices.append(next_idx)
        else:
            break
    
    # 如果还是不够，重复最后一个
    while len(indices) < 4:
        indices.append(indices[-1] if indices else 0)
    
    indices = indices[:4]  # 确保只有4个
    
    print("=" * 80)
    print("生成原始 vs 重建对比图")
    print("=" * 80)
    print(f"原始图片目录: {original_dir}")
    print(f"重建图片目录: {reconstructed_dir}")
    print(f"选择的索引: {indices}")
    print(f"输出路径: {output_path}")
    
    colormap = None if str(args.colormap).lower() in {"none", "null"} else args.colormap

    # 创建对比图
    create_comparison_grid(
        original_dir,
        reconstructed_dir,
        indices,
        output_path,
        figsize=tuple(args.figsize),
        colormap=colormap,
    )
    
    print(f"\n✅ 完成！")


if __name__ == "__main__":
    main()

