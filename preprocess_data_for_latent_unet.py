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


def ensure_lat_lon_order(data_array):
    """Ensure latitude dimension precedes longitude for consistent downstream usage."""
    dims = list(getattr(data_array, "dims", []))
    if not dims:
        print("  未检测到维度信息，跳过纬/经度顺序检查")
        return data_array, None

    lat_dim = next((dim for dim in dims if "lat" in dim.lower()), None)
    lon_dim = next((dim for dim in dims if "lon" in dim.lower()), None)

    if lat_dim is None or lon_dim is None:
        print(f"  未检测到纬/经度维度，dims={dims}")
        return data_array, None

    lon_first = dims.index(lon_dim) < dims.index(lat_dim)
    if lon_first:
        print(f"  检测到维度顺序 {dims} (longitude 在 latitude 前)，转置为标准顺序")
        target_dims = [dim for dim in dims if dim not in (lat_dim, lon_dim)]
        target_dims.extend([lat_dim, lon_dim])
        data_array = data_array.transpose(*target_dims)
        print(f"  转置后维度: {list(data_array.dims)}")
    else:
        print(f"  维度顺序符合标准: {dims}")

    return data_array, lon_first


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
    parser = argparse.ArgumentParser(description="预处理数据")

    parser.add_argument(
        "--data-path",
        type=str,
        default="gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr",
        help="原始数据路径",
    )
    parser.add_argument("--variable", type=str, default="2m_temperature", help="变量名")
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=None,
        help="气压层列表（用于有level维度的变量），例如：--levels 500 或 --levels 500 700 850。如果不指定，使用所有可用的levels。",
    )
    parser.add_argument(
        "--time-slice", type=str, default="2015-01-01:2019-12-31", help="时间切片"
    )
    parser.add_argument("--target-size", type=str, default="512,512", help="目标尺寸")
    parser.add_argument("--n-channels", type=int, default=3, help="通道数")
    parser.add_argument(
        "--normalization",
        type=str,
        default="minmax",
        choices=["minmax", "zscore"],
        help="归一化方法",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录")
    parser.add_argument(
        "--chunk-size", type=int, default=100, help="分块处理大小（避免内存溢出）"
    )

    args = parser.parse_args()

    target_size = tuple(map(int, args.target_size.split(",")))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # 判断output数据是不是有了，如果有了，则跳过预处理
    if os.path.exists(output_dir / "data.npy"):
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
        start, end = args.time_slice.split(":")
        ds = ds.sel(time=slice(start, end))

    data_var = ds[args.variable]

    # 处理levels维度（如果有）
    selected_levels = None
    has_level_dim = False
    if "level" in data_var.dims:
        available_levels = data_var.level.values.tolist()
        print(f"✓ 变量有level维度，可用levels: {available_levels}")

        if args.levels is not None:
            # 检查用户指定的levels是否在可用levels中
            invalid_levels = [l for l in args.levels if l not in available_levels]
            if invalid_levels:
                raise ValueError(
                    f"无效的levels: {invalid_levels}. "
                    f"可用的levels: {available_levels}"
                )
            selected_levels = args.levels
            data_var = data_var.sel(level=selected_levels)
            print(f"✓ 使用指定的levels: {selected_levels}")
        else:
            # 使用所有可用的levels
            selected_levels = available_levels
            print(f"✓ 使用所有可用的levels: {selected_levels}")

        # 重要：只有当选择了多个levels时才保持level维度
        # 如果只有一个level，将其视为单层空间数据
        if len(selected_levels) == 1:
            print(
                f"✓ 检测到单level模式 (level={selected_levels[0]})，将作为空间数据处理"
            )
            has_level_dim = False
            # data_var现在是 (Time, 1, H, W)，prepare_image会自动squeeze
        else:
            print(f"✓ 检测到多level模式 ({len(selected_levels)} levels)，将按level处理")
            has_level_dim = True

    # 统一纬经度维度顺序，确保 latitude 在 longitude 前
    spatial_lon_first = None
    if any("lat" in dim.lower() for dim in data_var.dims) and any(
        "lon" in dim.lower() for dim in data_var.dims
    ):
        data_var, spatial_lon_first = ensure_lat_lon_order(data_var)
    else:
        print(f"  未检测到纬/经度维度，dims={list(data_var.dims)}")

    # 获取数据维度
    data_shape = data_var.shape
    n_timesteps = len(data_var)

    if has_level_dim:
        # 多level模式: (Time, Level, H, W)
        if len(data_shape) == 4:
            n_levels, h_orig, w_orig = data_shape[1], data_shape[2], data_shape[3]
            # 每个level转换为n_channels，总通道数 = n_levels * n_channels
            total_channels = len(selected_levels) * args.n_channels
        else:
            raise ValueError(f"意外的数据形状: {data_shape}")
    else:
        # 单level或无level模式: (Time, H, W) 或 (Time, 1, H, W)
        # 如果是 (Time, 1, H, W)，prepare_image会squeeze到 (Time, H, W)
        if len(data_shape) == 3:
            h_orig, w_orig = data_shape[1], data_shape[2]
        elif len(data_shape) == 4 and data_shape[1] == 1:
            h_orig, w_orig = data_shape[2], data_shape[3]
        else:
            raise ValueError(f"意外的数据形状: {data_shape}")
        total_channels = args.n_channels

    print(f"✓ 总时间步数: {n_timesteps}")
    print(f"✓ 原始尺寸: ({h_orig}, {w_orig})")
    print(f"✓ 目标尺寸: {target_size}")
    if has_level_dim:
        print(f"✓ Levels数: {len(selected_levels)}")
        print(f"✓ 每个level的通道数: {args.n_channels}")
        print(f"✓ 总通道数: {total_channels}")
    else:
        print(f"✓ 通道数: {args.n_channels}")

    # 估算输出大小
    output_shape = (n_timesteps, total_channels, target_size[0], target_size[1])
    output_size_gb = np.prod(output_shape) * 4 / 1024**3
    print(f"\n预计输出大小: {output_size_gb:.2f} GB")
    print(f"  输出形状: {output_shape}")

    # ========================================================================
    # Step 2: 创建内存映射文件
    # ========================================================================
    print("\n" + "-" * 80)
    print("Step 2: 创建内存映射文件")
    print("-" * 80)

    data_file = output_dir / "data.npy"
    data_mmap = np.memmap(data_file, dtype="float32", mode="w+", shape=output_shape)

    print(f"✓ 创建文件: {data_file}")
    print(f"  形状: {output_shape}")

    # ========================================================================
    # Step 3: 分块处理数据
    # Step 3: 分块处理和归一化
    # ========================================================================
    print("\n" + "-" * 80)
    print("Step 3: 分块处理和归一化")
    print("-" * 80)

    # 计算归一化统计量
    level_stats = {}  # {level: {'min': ..., 'max': ..., 'mean': ..., 'std': ...}}

    if has_level_dim:
        # 多level模式：对每个level分别计算统计量
        print("\n扫描每个level的数据范围...")
        for level_idx, level_val in enumerate(selected_levels):
            level_min = float("inf")
            level_max = float("-inf")

            if args.normalization == "zscore":
                level_sum = 0.0
                level_sum_sq = 0.0
                level_count = 0

            for i in tqdm(
                range(0, n_timesteps, args.chunk_size), desc=f"扫描 level {level_val}"
            ):
                end_idx = min(i + args.chunk_size, n_timesteps)
                chunk = data_var.isel(time=slice(i, end_idx)).values
                # chunk shape: (Time, Level, H, W)
                level_chunk = chunk[:, level_idx, :, :]  # 提取当前level的所有时间步
                level_min = min(level_min, level_chunk.min())
                level_max = max(level_max, level_chunk.max())

                if args.normalization == "zscore":
                    level_sum += level_chunk.sum()
                    level_sum_sq += (level_chunk**2).sum()
                    level_count += level_chunk.size

            if args.normalization == "zscore":
                level_mean = level_sum / level_count
                level_std = np.sqrt(level_sum_sq / level_count - level_mean**2)
                level_stats[level_val] = {
                    "min": float(level_min),
                    "max": float(level_max),
                    "mean": float(level_mean),
                    "std": float(level_std),
                }
                print(
                    f"✓ Level {level_val}: 范围=[{level_min:.2f}, {level_max:.2f}], "
                    f"均值={level_mean:.2f}, 标准差={level_std:.2f}"
                )
            else:
                level_stats[level_val] = {
                    "min": float(level_min),
                    "max": float(level_max),
                }
                print(f"✓ Level {level_val}: 范围=[{level_min:.2f}, {level_max:.2f}]")
    else:
        # 单level或无level模式：计算全局统计量
        print("\n扫描数据范围...")
        global_min = float("inf")
        global_max = float("-inf")

        for i in tqdm(range(0, n_timesteps, args.chunk_size), desc="扫描"):
            end_idx = min(i + args.chunk_size, n_timesteps)
            chunk = data_var.isel(time=slice(i, end_idx)).values
            # 如果是 (Time, 1, H, W)，squeeze到 (Time, H, W)
            if chunk.ndim == 4 and chunk.shape[1] == 1:
                chunk = chunk.squeeze(axis=1)
            global_min = min(global_min, chunk.min())
            global_max = max(global_max, chunk.max())

        print(f"\n✓ 数据范围: [{global_min:.2f}, {global_max:.2f}]")

        if args.normalization == "zscore":
            print("\n计算统计量...")
            global_sum = 0.0
            global_sum_sq = 0.0
            global_count = 0

            for i in tqdm(range(0, n_timesteps, args.chunk_size), desc="统计"):
                end_idx = min(i + args.chunk_size, n_timesteps)
                chunk = data_var.isel(time=slice(i, end_idx)).values
                # 如果是 (Time, 1, H, W)，squeeze到 (Time, H, W)
                if chunk.ndim == 4 and chunk.shape[1] == 1:
                    chunk = chunk.squeeze(axis=1)
                global_sum += chunk.sum()
                global_sum_sq += (chunk**2).sum()
                global_count += chunk.size

            global_mean = global_sum / global_count
            global_std = np.sqrt(global_sum_sq / global_count - global_mean**2)
            level_stats["global"] = {
                "min": float(global_min),
                "max": float(global_max),
                "mean": float(global_mean),
                "std": float(global_std),
            }
            print(f"✓ 均值: {global_mean:.2f}, 标准差: {global_std:.2f}")
        else:
            level_stats["global"] = {"min": float(global_min), "max": float(global_max)}

    # 处理和归一化
    print("\\n处理数据...")
    for i in tqdm(range(0, n_timesteps, args.chunk_size), desc="处理"):
        end_idx = min(i + args.chunk_size, n_timesteps)

        # 加载当前块
        chunk = data_var.isel(time=slice(i, end_idx)).values

        # 处理每个时间步
        for j, t in enumerate(range(i, end_idx)):
            if has_level_dim:
                # 多level模式：每个level独立处理并堆叠
                # chunk[j] shape: (Level, H, W)
                processed_levels = []
                for level_idx, level_val in enumerate(selected_levels):
                    # 获取单个level的数据
                    level_data = chunk[j][level_idx]  # (H, W)

                    # Resize和转通道
                    level_processed = prepare_image(
                        level_data, n_channels=args.n_channels, target_size=target_size
                    )  # (C, H, W)

                    # 按level归一化
                    level_stat = level_stats[level_val]
                    if args.normalization == "minmax":
                        # 归一化到[-1, 1]
                        level_min = level_stat["min"]
                        level_max = level_stat["max"]
                        if level_max - level_min > 1e-8:
                            level_processed = (
                                2
                                * (level_processed - level_min)
                                / (level_max - level_min)
                                - 1
                            )
                        else:
                            # 如果min==max，保持原值或设为0
                            level_processed = np.zeros_like(level_processed)
                    else:  # zscore
                        level_mean = level_stat["mean"]
                        level_std = level_stat["std"]
                        if level_std > 1e-8:
                            level_processed = (level_processed - level_mean) / level_std
                        else:
                            # 如果std==0，设为0
                            level_processed = np.zeros_like(level_processed)

                    processed_levels.append(level_processed)

                # 将所有levels堆叠成一个多通道图像
                # processed_levels: list of (C, H, W)
                # 堆叠后: (n_levels * C, H, W)
                processed = np.concatenate(
                    processed_levels, axis=0
                )  # (total_channels, H, W)
            else:
                # 单level或无level模式：作为单层空间数据处理
                # chunk[j] shape: (H, W) 或 (1, H, W)
                data_slice = chunk[j]
                # 如果是 (1, H, W)，squeeze到 (H, W)
                if data_slice.ndim == 3 and data_slice.shape[0] == 1:
                    data_slice = data_slice.squeeze(axis=0)

                # Resize和转通道
                processed = prepare_image(
                    data_slice, n_channels=args.n_channels, target_size=target_size
                )  # (C, H, W)

                # 归一化（使用global统计量）
                global_stat = level_stats["global"]
                if args.normalization == "minmax":
                    # 归一化到[-1, 1]
                    global_min = global_stat["min"]
                    global_max = global_stat["max"]
                    if global_max - global_min > 1e-8:
                        processed = (
                            2 * (processed - global_min) / (global_max - global_min) - 1
                        )
                    else:
                        processed = np.zeros_like(processed)
                else:  # zscore
                    global_mean = global_stat["mean"]
                    global_std = global_stat["std"]
                    if global_std > 1e-8:
                        processed = (processed - global_mean) / global_std
                    else:
                        processed = np.zeros_like(processed)

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

    # 计算总体数据范围（用于元数据）
    overall_min = min([stat["min"] for stat in level_stats.values()])
    overall_max = max([stat["max"] for stat in level_stats.values()])

    metadata = {
        "variable": args.variable,
        "time_slice": args.time_slice,
        "n_timesteps": n_timesteps,
        "shape": output_shape,
        "target_size": target_size,
        "n_channels": args.n_channels,
        "normalization": args.normalization,
        "normalize_per_level": has_level_dim,  # 标记是否按level归一化
        "data_range": {
            "original_min": float(overall_min),
            "original_max": float(overall_max),
            "normalized_min": float(data_mmap[:].min()),
            "normalized_max": float(data_mmap[:].max()),
        },
        "dtype": "float32",
        "time_coords": (
            ds.time.values.astype(str).tolist() if hasattr(ds, "time") else None
        ),
        "spatial_lon_first": spatial_lon_first,
    }

    # 如果有levels，保存levels信息
    if selected_levels is not None:
        metadata["levels"] = selected_levels
        # has_level_dim已经在前面正确设置：多level为True，单level为False
        metadata["has_level_dim"] = has_level_dim
    else:
        metadata["has_level_dim"] = False

    # 保存JSON
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # 保存归一化参数（用于反归一化）
    # 如果按level归一化，保存每个level的统计量
    # 使用变量名作为key，而不是'default'，这样预测时可以正确反归一化
    norm_stats = {
        "method": args.normalization,
        "normalize_per_level": has_level_dim,  # 标记是否按level归一化
        "stats": {},
    }

    if has_level_dim:
        # 按level归一化：为每个level保存统计量
        for level_val in selected_levels:
            level_stat = level_stats[level_val]
            level_key = f"{args.variable}_level_{level_val}"
            if args.normalization == "minmax":
                norm_stats["stats"][level_key] = {
                    "min": level_stat["min"],
                    "max": level_stat["max"],
                    "mean": (level_stat["min"] + level_stat["max"]) / 2,
                    "std": (level_stat["max"] - level_stat["min"]) / 2,
                }
            else:  # zscore
                norm_stats["stats"][level_key] = {
                    "mean": level_stat["mean"],
                    "std": level_stat["std"],
                    "min": level_stat["min"],  # 保留min/max用于参考
                    "max": level_stat["max"],
                }
        # 同时保存level到统计量的映射，方便查找
        norm_stats["level_to_stats_key"] = {
            level_val: f"{args.variable}_level_{level_val}"
            for level_val in selected_levels
        }
        norm_stats["variable"] = args.variable
        norm_stats["levels"] = selected_levels
    else:
        # 全局归一化：保存全局统计量
        global_stat = level_stats["global"]
        if args.normalization == "minmax":
            norm_stats["stats"][args.variable] = {
                "min": global_stat["min"],
                "max": global_stat["max"],
                "mean": (global_stat["min"] + global_stat["max"]) / 2,
                "std": (global_stat["max"] - global_stat["min"]) / 2,
            }
        else:  # zscore
            norm_stats["stats"][args.variable] = {
                "mean": global_stat["mean"],
                "std": global_stat["std"],
                "min": global_stat["min"],  # 保留min/max用于参考
                "max": global_stat["max"],
            }

    with open(output_dir / "normalizer_stats.pkl", "wb") as f:
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
    print(f"  将该目录传给潜空间训练脚本，例如:")
    print(f"    python train_vae.py --preprocessed-data-dir {output_dir} ...")
    print(f"    # 或 train_rae.py（如果使用RAE管线）")

    print(f"\n提示:")
    print(f"  - 预处理数据支持lazy loading")
    print(f"  - 训练时内存占用将大大减少")
    print(f"  - 可以安全地训练大规模数据集")


if __name__ == "__main__":
    main()
