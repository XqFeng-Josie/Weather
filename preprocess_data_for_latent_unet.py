"""
数据预处理脚本 - 为潜空间U-Net准备数据

作用：
    1. 一次性将原始64x32数据resize到512x512
    2. 转换为3通道RGB格式
    3. 归一化到[-1, 1]
    4. 保存为内存映射文件（.npy），支持lazy loading
    
优点：
    - 预处理只需做一次
    - 训练时按需加载，内存占用小
    - 可以处理任意大小的数据集

使用方法：
    # 预处理2015-2019年的数据
    python preprocess_data_for_latent_unet.py \
        --time-slice 2015-01-01:2019-12-31 \
        --target-size 512,512 \
        --output-dir data/preprocessed/latent_unet_512x512
"""

import argparse
import numpy as np
import xarray as xr
from pathlib import Path
from scipy.ndimage import zoom
from tqdm import tqdm
import json
import pickle
import os


def prepare_image(data_slice, n_channels=3, target_size=(512, 512)):
    """
    准备单个时间步的图像
    
    Args:
        data_slice: (H, W) 原始数据
        n_channels: 目标通道数
        target_size: 目标尺寸
    
    Returns:
        image: (C, H, W) 处理后的图像
    """
    h_orig, w_orig = data_slice.shape
    h_target, w_target = target_size
    
    # Resize
    zoom_factors = (h_target / h_orig, w_target / w_orig)
    resized = zoom(data_slice, zoom_factors, order=1)
    
    # 转为3通道
    if n_channels == 3:
        image = np.stack([resized, resized, resized], axis=0)
    else:
        image = resized[np.newaxis, :, :]
    
    return image.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description='预处理数据')
    
    parser.add_argument('--data-path', type=str,
                       default='gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr',
                       help='原始数据路径')
    parser.add_argument('--variable', type=str, default='2m_temperature',
                       help='变量名')
    parser.add_argument('--time-slice', type=str, default='2015-01-01:2019-12-31',
                       help='时间切片')
    parser.add_argument('--target-size', type=str, default='512,512',
                       help='目标尺寸')
    parser.add_argument('--n-channels', type=int, default=3,
                       help='通道数')
    parser.add_argument('--normalization', type=str, default='minmax',
                       choices=['minmax', 'zscore'],
                       help='归一化方法')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--chunk-size', type=int, default=100,
                       help='分块处理大小（避免内存溢出）')
    
    args = parser.parse_args()
    
    target_size = tuple(map(int, args.target_size.split(',')))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # 判断output数据是不是有了，如果有了，则跳过预处理
    if os.path.exists(output_dir / 'data.npy'):
        print(f"输出数据已存在: {output_dir / 'data.npy'}")
        return
    
    print("\n" + "=" * 80)
    print("数据预处理")
    print("=" * 80)
    print(f"输入: {args.data_path}")
    print(f"变量: {args.variable}")
    print(f"时间范围: {args.time_slice}")
    print(f"目标尺寸: {target_size}")
    print(f"输出目录: {output_dir}")
    
    # ========================================================================
    # Step 1: 加载元数据
    # ========================================================================
    print("\n" + "-" * 80)
    print("Step 1: 加载元数据")
    print("-" * 80)
    
    ds = xr.open_zarr(args.data_path)
    
    if args.time_slice:
        start, end = args.time_slice.split(':')
        ds = ds.sel(time=slice(start, end))
    
    data_var = ds[args.variable]
    n_timesteps = len(data_var)
    h_orig, w_orig = data_var.shape[1], data_var.shape[2]
    
    print(f"✓ 总时间步数: {n_timesteps}")
    print(f"✓ 原始尺寸: ({h_orig}, {w_orig})")
    print(f"✓ 目标尺寸: {target_size}")
    print(f"✓ 通道数: {args.n_channels}")
    
    # 估算输出大小
    output_shape = (n_timesteps, args.n_channels, target_size[0], target_size[1])
    output_size_gb = np.prod(output_shape) * 4 / 1024**3
    print(f"\n预计输出大小: {output_size_gb:.2f} GB")
    
    # ========================================================================
    # Step 2: 创建内存映射文件
    # ========================================================================
    print("\n" + "-" * 80)
    print("Step 2: 创建内存映射文件")
    print("-" * 80)
    
    data_file = output_dir / 'data.npy'
    data_mmap = np.memmap(
        data_file,
        dtype='float32',
        mode='w+',
        shape=output_shape
    )
    
    print(f"✓ 创建文件: {data_file}")
    print(f"  形状: {output_shape}")
    
    # ========================================================================
    # Step 3: 分块处理数据
    # ========================================================================
    print("\n" + "-" * 80)
    print("Step 3: 分块处理和归一化")
    print("-" * 80)
    
    # 先扫描一遍获取全局min/max（用于归一化）
    print("\n扫描数据范围...")
    global_min = float('inf')
    global_max = float('-inf')
    
    for i in tqdm(range(0, n_timesteps, args.chunk_size), desc='扫描'):
        end_idx = min(i + args.chunk_size, n_timesteps)
        chunk = data_var.isel(time=slice(i, end_idx)).values
        global_min = min(global_min, chunk.min())
        global_max = max(global_max, chunk.max())
    
    print(f"\n✓ 数据范围: [{global_min:.2f}, {global_max:.2f}]")
    
    # 处理和归一化
    print("\n处理数据...")
    for i in tqdm(range(0, n_timesteps, args.chunk_size), desc='处理'):
        end_idx = min(i + args.chunk_size, n_timesteps)
        
        # 加载当前块
        chunk = data_var.isel(time=slice(i, end_idx)).values
        
        # 处理每个时间步
        for j, t in enumerate(range(i, end_idx)):
            # Resize和转通道
            processed = prepare_image(
                chunk[j],
                n_channels=args.n_channels,
                target_size=target_size
            )
            
            # 归一化到[-1, 1]
            if args.normalization == 'minmax':
                processed = 2 * (processed - global_min) / (global_max - global_min) - 1
            else:  # zscore
                mean = chunk.mean()
                std = chunk.std()
                processed = (processed - mean) / std
            
            # 写入内存映射
            data_mmap[t] = processed
        
        # 定期刷新到磁盘
        data_mmap.flush()
    
    print(f"✓ 数据处理完成")
    print(f"  归一化后范围: [{data_mmap[:].min():.2f}, {data_mmap[:].max():.2f}]")
    
    # ========================================================================
    # Step 4: 保存元数据
    # ========================================================================
    print("\n" + "-" * 80)
    print("Step 4: 保存元数据")
    print("-" * 80)
    
    metadata = {
        'variable': args.variable,
        'time_slice': args.time_slice,
        'n_timesteps': n_timesteps,
        'shape': output_shape,
        'target_size': target_size,
        'n_channels': args.n_channels,
        'normalization': args.normalization,
        'data_range': {
            'original_min': float(global_min),
            'original_max': float(global_max),
            'normalized_min': float(data_mmap[:].min()),
            'normalized_max': float(data_mmap[:].max())
        },
        'dtype': 'float32',
        'time_coords': ds.time.values.astype(str).tolist() if hasattr(ds, 'time') else None
    }
    
    # 保存JSON
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # 保存归一化参数（用于反归一化）
    # 使用变量名作为key，而不是'default'，这样预测时可以正确反归一化
    norm_stats = {
        'method': args.normalization,
        'stats': {
            args.variable: {  # 使用实际变量名（如'2m_temperature'）
                'min': float(global_min),
                'max': float(global_max),
                'mean': float((global_min + global_max) / 2),
                'std': float((global_max - global_min) / 2)
            }
        }
    }
    
    with open(output_dir / 'normalizer_stats.pkl', 'wb') as f:
        pickle.dump(norm_stats, f)
    
    print(f"✓ 元数据已保存: {output_dir / 'metadata.json'}")
    print(f"✓ 归一化参数已保存: {output_dir / 'normalizer_stats.pkl'}")
    
    # ========================================================================
    # 完成
    # ========================================================================
    print("\n" + "=" * 80)
    print("预处理完成！")
    print("=" * 80)
    print(f"\n输出文件:")
    print(f"  - data.npy: {output_size_gb:.2f} GB")
    print(f"  - metadata.json")
    print(f"  - normalizer_stats.pkl")
    
    print(f"\n使用方法:")
    print(f"  修改train_latent_unet.py，使用以下参数:")
    print(f"  --preprocessed-data-dir {output_dir}")
    
    print(f"\n提示:")
    print(f"  - 预处理数据支持lazy loading")
    print(f"  - 训练时内存占用将大大减少")
    print(f"  - 可以安全地训练大规模数据集")


if __name__ == '__main__':
    main()

