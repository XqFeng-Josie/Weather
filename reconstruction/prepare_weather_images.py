#!/usr/bin/env python3
"""
将 ERA5 天气数据（64×32）插值到目标尺寸（如 256×256）并保存为图片

使用方法:
    python prepare_weather_images.py \
        --data-path gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr \
        --variable 2m_temperature \
        --time-slice 2020-01-01:2020-01-31 \
        --target-size 256 256 \
        --output-dir weather_images \
        --n-samples 100
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
from PIL import Image
from scipy.ndimage import zoom
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings('ignore')


def prepare_weather_data(data: np.ndarray, 
                        n_channels: int = 3,
                        target_size: tuple = None) -> np.ndarray:
    """
    准备天气数据用于图像模型
    
    Args:
        data: 输入数据，shape (Time, H, W) 或 (Time, C, H, W)
        n_channels: 目标通道数（1或3）
        target_size: 目标尺寸 (H, W)，如果为None则保持原尺寸
    
    Returns:
        处理后的数据，shape (Time, C, H, W)
    """
    # 确保是4维: (Time, C, H, W)
    if data.ndim == 3:
        # (Time, H, W) -> (Time, 1, H, W)
        data = data[:, np.newaxis, :, :]
    
    # 如果需要3通道但只有1通道，复制
    if n_channels == 3 and data.shape[1] == 1:
        data = np.repeat(data, 3, axis=1)
    
    # 调整尺寸（如果需要）
    if target_size is not None:
        time, c, h, w = data.shape
        target_h, target_w = target_size
        
        zoom_factors = (1, 1, target_h / h, target_w / w)
        # data = zoom(data, zoom_factors, order=1)  # order=1 表示双线性插值
        data = zoom(data, zoom_factors, order=3)  # order=3 双三次插值

    return data


def normalize_to_image(data: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    将数据归一化到 [0, 255] 范围以便保存为图片
    
    Args:
        data: 输入数据，shape (Time, C, H, W) 或 (H, W)
        method: 归一化方法 ('minmax' 或 'zscore')
    
    Returns:
        归一化后的数据，范围 [0, 255]，dtype uint8
    """
    if method == 'minmax':
        # MinMax 归一化到 [0, 255]
        data_min = data.min()
        data_max = data.max()
        if data_max > data_min:
            normalized = (data - data_min) / (data_max - data_min) * 255.0
        else:
            normalized = np.zeros_like(data)
    elif method == 'zscore':
        # Z-score 归一化，然后映射到 [0, 255]
        mean = data.mean()
        std = data.std()
        if std > 0:
            normalized = (data - mean) / std
            # 映射到 [0, 255]，假设 ±3σ 范围
            normalized = np.clip((normalized + 3) / 6 * 255.0, 0, 255)
        else:
            normalized = np.zeros_like(data)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized.astype(np.uint8)

def save_weather_images_colormap(data: np.ndarray,
                       output_dir: Path,
                       prefix: str = 'sample',
                       start_idx: int = 0,
                       cmap_name: str = 'coolwarm'):
    """
    保存天气数据为图片（支持彩色 colormap）
    """
    import matplotlib.cm as cm

    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取 colormap
    cmap = cm.get_cmap(cmap_name)

    if data.ndim == 3:
        data = data[np.newaxis, ...]

    n_samples = len(data)
    for i in tqdm(range(n_samples), desc="保存图片"):
        img_data = data[i]  # (C, H, W)

        # 单通道 -> 彩色
        if img_data.shape[0] == 1:
            # [0,255] → [0,1]
            normalized = img_data[0] / 255.0
            colored = cmap(normalized)[:, :, :3]  # RGBA → RGB
            colored = (colored * 255).astype(np.uint8)
            img_data = np.transpose(colored, (2, 0, 1))
        else:
            # 已是RGB
            pass

        # 转换为 (H, W, C)
        img_data = np.transpose(img_data, (1, 2, 0))
        img = Image.fromarray(img_data, mode='RGB')
        output_path = output_dir / f"{prefix}_{start_idx + i:06d}.png"
        img.save(output_path)

def save_weather_images(data: np.ndarray, 
                       output_dir: Path,
                       prefix: str = 'sample',
                       start_idx: int = 0):
    """
    保存天气数据为图片
    
    Args:
        data: 数据，shape (Time, C, H, W) 或 (C, H, W)
        output_dir: 输出目录
        prefix: 文件名前缀
        start_idx: 起始索引
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 确保是4维
    if data.ndim == 3:
        data = data[np.newaxis, ...]
    
    n_samples = len(data)
    
    for i in tqdm(range(n_samples), desc="保存图片"):
        img_data = data[i]  # (C, H, W)
        
        # 如果是单通道，转换为3通道
        if img_data.shape[0] == 1:
            img_data = np.repeat(img_data, 3, axis=0)
        
        # 转换为 (H, W, C) 格式
        img_data = np.transpose(img_data, (1, 2, 0))
        
        # 保存为PNG
        img = Image.fromarray(img_data, mode='RGB')
        output_path = output_dir / f"{prefix}_{start_idx + i:06d}.png"
        img.save(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="将 ERA5 天气数据插值并保存为图片"
    )
    
    # 数据参数
    parser.add_argument(
        "--data-path",
        type=str,
        default="gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr",
        help="ERA5 数据路径（zarr格式）",
    )
    parser.add_argument(
        "--variable",
        type=str,
        default="2m_temperature",
        help="要提取的变量名（如 2m_temperature, geopotential_500, total_precipitation）",
    )
    parser.add_argument(
        "--time-slice",
        type=str,
        default=None,
        help="时间切片，格式: 2020-01-01:2020-12-31",
    )
    
    # 处理参数
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        default=[256, 256],
        help="目标图像尺寸 [height width]，默认 [256 256]",
    )
    parser.add_argument(
        "--n-channels",
        type=int,
        default=3,
        choices=[1, 3],
        help="输出通道数（1或3），默认3",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="minmax",
        choices=["minmax", "zscore"],
        help="归一化方法，默认 minmax",
    )
    
    # 输出参数
    parser.add_argument(
        "--output-dir",
        type=str,
        default="weather_images",
        help="输出目录",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="要处理的样本数量（默认处理所有数据）",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="sample",
        help="输出文件名前缀",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ERA5 天气数据 → 图片转换工具")
    print("=" * 80)
    
    # 1. 加载数据
    print(f"\n1. 加载数据: {args.data_path}")
    print(f"   变量: {args.variable}")
    
    try:
        ds = xr.open_zarr(args.data_path)
    except Exception as e:
        print(f"   ❌ 错误: 无法打开数据文件")
        print(f"   请确保已安装 gcsfs: pip install gcsfs")
        print(f"   错误详情: {e}")
        return
    
    # 检查变量是否存在
    if args.variable not in ds.data_vars:
        print(f"   ❌ 错误: 变量 '{args.variable}' 不存在")
        print(f"   可用变量: {list(ds.data_vars)[:10]}...")
        return
    
    # 时间切片
    if args.time_slice:
        start, end = args.time_slice.split(':')
        ds = ds.sel(time=slice(start, end))
        print(f"   时间范围: {start} 至 {end}")
    
    # 获取变量数据
    variable_data = ds[args.variable]
    data = variable_data.values  # (Time, H, W) 或 (Time, Lat, Lon)
    
    # ERA5 默认维度: (time, lat, lon)
    # 我们需要转换为 (time, height=lat, width=lon)
    if 'latitude' in variable_data.dims and 'longitude' in variable_data.dims:
        data = np.transpose(data, (0, 2, 1))  # 交换 (lat, lon)
        print("⚙️ 自动调整纬经度维度顺序: (time, lon, lat)")

    print(f"   原始数据 shape: {data.shape}")
    print(f"   数据范围: [{data.min():.2f}, {data.max():.2f}]")
    print(f"   数据单位: {variable_data.attrs.get('units', 'N/A')}")
    
    # 限制样本数量
    if args.n_samples is not None and args.n_samples < len(data):
        data = data[:args.n_samples]
        print(f"   限制样本数: {args.n_samples}")
    
    # 2. 插值到目标尺寸
    print(f"\n2. 插值到目标尺寸: {args.target_size}")
    data_processed = prepare_weather_data(
        data,
        n_channels=args.n_channels,
        target_size=tuple(args.target_size)
    )
    print(f"   处理后 shape: {data_processed.shape}")
    
    # 3. 归一化
    # print(f"\n3. 归一化（方法: {args.normalization}）")
    # # 对每个时间步单独归一化，保持相对关系
    # data_normalized = np.zeros_like(data_processed, dtype=np.uint8)
    # for i in range(len(data_processed)):
    #     data_normalized[i] = normalize_to_image(
    #         data_processed[i],
    #         method=args.normalization
    #     )
    # print(f"   归一化后范围: [{data_normalized.min()}, {data_normalized.max()}]")
    print(f"\n3. 全局归一化（方法: {args.normalization}）")

    if args.normalization == "minmax":
        # 全局最小值与最大值（跨所有时间和通道）
        global_min = data_processed.min()
        global_max = data_processed.max()
        print(f"   全局范围: [{global_min:.4f}, {global_max:.4f}]")

        # 归一化到 [0, 255]
        data_normalized = (data_processed - global_min) / (global_max - global_min + 1e-8)
        data_normalized = np.clip(data_normalized * 255.0, 0, 255).astype(np.uint8)
        
        # 保存全局统计量用于后续反归一化
        global_mean = None
        global_std = None

    elif args.normalization == "zscore":
        global_mean = data_processed.mean()
        global_std = data_processed.std()
        print(f"   全局均值: {global_mean:.4f}, 标准差: {global_std:.4f}")

        # Z-score 归一化再映射到 [0, 255]
        data_normalized = (data_processed - global_mean) / (global_std + 1e-8)
        data_normalized = np.clip((data_normalized + 3) / 6 * 255.0, 0, 255).astype(np.uint8)
        
        # 保存全局统计量用于后续反归一化
        global_min = None
        global_max = None

    else:
        raise ValueError(f"Unknown normalization method: {args.normalization}")

    print(f"   归一化后范围: [{data_normalized.min()}, {data_normalized.max()}]")
    
    # 4. 保存归一化参数
    print(f"\n4. 保存归一化参数")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存归一化统计信息
    norm_stats = {
        'method': args.normalization,
        'variable': args.variable,
        'original_min': float(global_min) if args.normalization == 'minmax' else None,
        'original_max': float(global_max) if args.normalization == 'minmax' else None,
        'original_mean': float(global_mean) if args.normalization == 'zscore' else None,
        'original_std': float(global_std) if args.normalization == 'zscore' else None,
    }
    
    with open(output_dir / 'normalization_stats.json', 'w') as f:
        json.dump(norm_stats, f, indent=2)
    
    print(f"   ✓ 归一化参数已保存: {output_dir / 'normalization_stats.json'}")
    
    # 5. 保存图片
    print(f"\n5. 保存图片到: {args.output_dir}")
    save_weather_images_colormap(
        data_normalized,
        output_dir,
        prefix=args.prefix,
        start_idx=0,
        cmap_name='turbo'
    )
    
    print(f"\n✅ 完成！")
    print(f"   共保存 {len(data_normalized)} 张图片")
    print(f"   输出目录: {output_dir.absolute()}")
    print(f"   图片尺寸: {args.target_size}")
    print(f"   通道数: {args.n_channels}")


if __name__ == "__main__":
    main()

