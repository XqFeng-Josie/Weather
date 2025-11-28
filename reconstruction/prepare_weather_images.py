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
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import numpy as np
import xarray as xr
from PIL import Image
from scipy.ndimage import zoom
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")


def _compute_chunk_stats(args_tuple):
    """
    计算数据块的统计量（用于并发计算）

    Args:
        args_tuple: (chunk_data, method)

    Returns:
        dict with stats
    """
    chunk_data, method = args_tuple
    stats = {}

    if method == "minmax":
        stats["min"] = float(chunk_data.min())
        stats["max"] = float(chunk_data.max())
    elif method == "zscore":
        stats["mean"] = float(chunk_data.mean())
        stats["sum"] = float(chunk_data.sum())
        stats["sum_sq"] = float((chunk_data**2).sum())
        stats["count"] = int(chunk_data.size)

    return stats


def _normalize_chunk(args_tuple):
    """
    归一化数据块（用于并发处理）

    Args:
        args_tuple: (chunk_data, method, stats)

    Returns:
        normalized_chunk
    """
    chunk_data, method, stats = args_tuple

    if method == "minmax":
        global_min = stats["min"]
        global_max = stats["max"]
        denom = (global_max - global_min) + 1e-8
        normalized = (chunk_data - global_min) / denom
        normalized = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    elif method == "zscore":
        global_mean = stats["mean"]
        global_std = stats["std"]
        normalized = (chunk_data - global_mean) / (global_std + 1e-8)
        normalized = np.clip((normalized + 3) / 6 * 255.0, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized


def _process_single_time_step(args_tuple):
    """
    处理单个时间步的辅助函数（用于并发处理）

    Args:
        args_tuple: (time_slice, n_channels, target_size, idx)

    Returns:
        (idx, processed_slice)
    """
    time_slice, n_channels, target_size, idx = args_tuple

    # 确保是3维: (C, H, W) 或 (H, W)
    if time_slice.ndim == 2:
        time_slice = time_slice[np.newaxis, :, :]  # (1, H, W)

    # 如果需要3通道但只有1通道，复制
    if n_channels == 3 and time_slice.shape[0] == 1:
        time_slice = np.repeat(time_slice, 3, axis=0)

    # 调整尺寸（如果需要）
    if target_size is not None:
        c, h, w = time_slice.shape
        target_h, target_w = target_size

        zoom_factors = (1, target_h / h, target_w / w)
        time_slice = zoom(time_slice, zoom_factors, order=3)  # order=3 双三次插值

    return (idx, time_slice)


def prepare_weather_data(
    data: np.ndarray,
    n_channels: int = 3,
    target_size: tuple = None,
    n_workers: int = 1,
    use_concurrent: bool = False,
) -> np.ndarray:
    """
    准备天气数据用于图像模型

    Args:
        data: 输入数据，shape (Time, H, W) 或 (Time, C, H, W)
        n_channels: 目标通道数（1或3）
        target_size: 目标尺寸 (H, W)，如果为None则保持原尺寸
        n_workers: 并发工作进程数（仅在use_concurrent=True时有效）
        use_concurrent: 是否使用并发处理（对于大数据集可能更快）

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

        # 对于大数据集，可以考虑并发处理每个时间步
        # 注意：使用ThreadPoolExecutor而不是ProcessPoolExecutor，因为scipy的zoom操作
        # 主要在C扩展中执行，GIL影响较小，且避免序列化开销
        if use_concurrent and n_workers > 1 and time > 10:
            # 准备任务
            tasks = []
            for i in range(time):
                tasks.append((data[i].copy(), n_channels, target_size, i))

            # 并发处理
            results = {}
            failed_count = 0
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(_process_single_time_step, task): task[3]
                    for task in tasks
                }

                with tqdm(
                    total=time,
                    desc="🔄 插值处理",
                    unit="帧",
                    unit_scale=False,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                ) as pbar:
                    for future in as_completed(futures):
                        try:
                            idx, processed = future.result()
                            results[idx] = processed
                        except Exception as e:
                            failed_count += 1
                            task_idx = futures[future]
                            print(f"\n⚠️  处理帧 {task_idx} 时出错: {e}")
                        pbar.update(1)
                        pbar.set_postfix({"失败": failed_count})

            if failed_count > 0:
                print(f"\n⚠️  警告: {failed_count} 帧处理失败")

            # 按顺序重组数据
            processed_data = np.stack([results[i] for i in range(time)], axis=0)
            return processed_data
        else:
            # 原始方法：一次性处理所有时间步
            zoom_factors = (1, 1, target_h / h, target_w / w)
            # data = zoom(data, zoom_factors, order=1)  # order=1 表示双线性插值
            data = zoom(data, zoom_factors, order=3)  # order=3 双三次插值

    return data


def compute_normalization_stats_concurrent(
    data: np.ndarray,
    method: str = "minmax",
    n_workers: int = 4,
    chunk_size: int = None,
) -> dict:
    """
    并发计算归一化统计量

    Args:
        data: 输入数据
        method: 归一化方法
        n_workers: 并发工作线程数
        chunk_size: 每个块的大小（None表示自动计算）

    Returns:
        统计量字典
    """
    if chunk_size is None:
        # 自动计算合适的块大小
        total_elements = data.size
        chunk_size = max(1, total_elements // (n_workers * 4))

    if method == "minmax":
        # 将数据分成块
        data_flat = data.flatten()
        n_chunks = (len(data_flat) + chunk_size - 1) // chunk_size
        chunks = [
            data_flat[i * chunk_size : (i + 1) * chunk_size] for i in range(n_chunks)
        ]

        # 并发计算每个块的最小值和最大值
        tasks = [(chunk, method) for chunk in chunks]

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            chunk_stats = list(
                tqdm(
                    executor.map(_compute_chunk_stats, tasks),
                    total=len(tasks),
                    desc="📊 计算统计量",
                    unit="块",
                    leave=False,
                )
            )

        # 合并结果
        global_min = min(stat["min"] for stat in chunk_stats)
        global_max = max(stat["max"] for stat in chunk_stats)

        return {"min": global_min, "max": global_max}

    elif method == "zscore":
        # 将数据分成块
        data_flat = data.flatten()
        n_chunks = (len(data_flat) + chunk_size - 1) // chunk_size
        chunks = [
            data_flat[i * chunk_size : (i + 1) * chunk_size] for i in range(n_chunks)
        ]

        # 并发计算每个块的统计量
        tasks = [(chunk, method) for chunk in chunks]

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            chunk_stats = list(
                tqdm(
                    executor.map(_compute_chunk_stats, tasks),
                    total=len(tasks),
                    desc="📊 计算统计量",
                    unit="块",
                    leave=False,
                )
            )

        # 合并结果（计算全局均值和标准差）
        total_sum = sum(stat["sum"] for stat in chunk_stats)
        total_sum_sq = sum(stat["sum_sq"] for stat in chunk_stats)
        total_count = sum(stat["count"] for stat in chunk_stats)

        global_mean = total_sum / total_count
        global_var = (total_sum_sq / total_count) - (global_mean**2)
        global_std = np.sqrt(max(0, global_var))

        return {"mean": global_mean, "std": global_std}

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def normalize_to_image(data: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    将数据归一化到 [0, 255] 范围以便保存为图片

    Args:
        data: 输入数据，shape (Time, C, H, W) 或 (H, W)
        method: 归一化方法 ('minmax' 或 'zscore')

    Returns:
        归一化后的数据，范围 [0, 255]，dtype uint8
    """
    if method == "minmax":
        # MinMax 归一化到 [0, 255]
        data_min = data.min()
        data_max = data.max()
        if data_max > data_min:
            normalized = (data - data_min) / (data_max - data_min) * 255.0
        else:
            normalized = np.zeros_like(data)
    elif method == "zscore":
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


def _save_single_image_colormap(args_tuple):
    """
    保存单张图片的辅助函数（用于并发处理）

    Args:
        args_tuple: (img_data, output_path, cmap, idx)

    Returns:
        (idx, success)
    """
    img_data, output_path, cmap, idx = args_tuple

    try:
        # 单通道 -> 彩色
        if img_data.shape[0] == 1:
            # [0,255] → [0,1]
            normalized = img_data[0] / 255.0
            colored = cmap(normalized)[:, :, :3]  # RGBA → RGB
            colored = (colored * 255).astype(np.uint8)
            img_data = np.transpose(colored, (2, 0, 1))

        # 转换为 (H, W, C)
        img_data = np.transpose(img_data, (1, 2, 0))
        img = Image.fromarray(img_data, mode="RGB")
        img.save(output_path)
        return (idx, True)
    except Exception as e:
        return (idx, False, str(e))


def save_weather_images_colormap(
    data: np.ndarray,
    output_dir: Path,
    prefix: str = "sample",
    start_idx: int = 0,
    cmap_name: str = "coolwarm",
    n_workers: int = 4,
):
    """
    保存天气数据为图片（支持彩色 colormap，并发版本）

    Args:
        data: 数据，shape (Time, C, H, W) 或 (C, H, W)
        output_dir: 输出目录
        prefix: 文件名前缀
        start_idx: 起始索引
        cmap_name: colormap 名称
        n_workers: 并发工作线程数
    """
    import matplotlib.cm as cm

    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取 colormap
    cmap = cm.get_cmap(cmap_name)

    if data.ndim == 3:
        data = data[np.newaxis, ...]

    n_samples = len(data)

    # 准备任务列表
    tasks = []
    for i in range(n_samples):
        img_data = data[i]  # (C, H, W)
        output_path = output_dir / f"{prefix}_{start_idx + i:06d}.png"
        tasks.append((img_data, output_path, cmap, i))

    # 并发保存
    if n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_save_single_image_colormap, task): task[3]
                for task in tasks
            }

            # 使用 tqdm 显示进度
            completed = 0
            failed = 0
            failed_indices = []
            start_time = time.time()

            with tqdm(
                total=n_samples,
                desc="💾 保存图片",
                unit="张",
                unit_scale=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            ) as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if len(result) == 2 and result[1]:
                        completed += 1
                    else:
                        failed += 1
                        failed_indices.append(result[0])
                        if len(result) > 2:
                            print(f"\n⚠️  保存图片 {result[0]} 失败: {result[2]}")
                    pbar.update(1)

                    # 更新后处理信息
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    pbar.set_postfix(
                        {"成功": completed, "失败": failed, "速度": f"{rate:.1f} 张/秒"}
                    )

            if failed > 0:
                print(
                    f"\n⚠️  警告: {failed} 张图片保存失败 (索引: {failed_indices[:10]}{'...' if len(failed_indices) > 10 else ''})"
                )
    else:
        # 单线程模式（向后兼容）
        for i in tqdm(range(n_samples), desc="保存图片", unit="张"):
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
            img = Image.fromarray(img_data, mode="RGB")
            output_path = output_dir / f"{prefix}_{start_idx + i:06d}.png"
            img.save(output_path)


def save_weather_images(
    data: np.ndarray, output_dir: Path, prefix: str = "sample", start_idx: int = 0
):
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
        img = Image.fromarray(img_data, mode="RGB")
        output_path = output_dir / f"{prefix}_{start_idx + i:06d}.png"
        img.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="将 ERA5 天气数据插值并保存为图片")

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
    parser.add_argument(
        "--norm-stats-path",
        type=str,
        default=None,
        help="可选，提供已有的 normalization_stats.json 文件以复用归一化参数",
    )
    parser.add_argument(
        "--concurrent-stats",
        action="store_true",
        help="使用并发计算归一化统计量（对于大数据集可能更快）",
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
    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="并发工作线程/进程数（用于图片保存和数据处理），默认4",
    )
    parser.add_argument(
        "--concurrent-interpolation",
        action="store_true",
        help="使用并发处理数据插值（对于大数据集可能更快）",
    )

    args = parser.parse_args()

    # 记录总开始时间
    total_start_time = time.time()

    print("=" * 80)
    print("ERA5 天气数据 → 图片转换工具")
    print("=" * 80)
    print(f"并发工作线程数: {args.n_workers}")
    print(f"并发插值: {'启用' if args.concurrent_interpolation else '禁用'}")
    print(f"并发统计量计算: {'启用' if args.concurrent_stats else '禁用'}")
    print("=" * 80)

    # 1. 加载数据
    load_start = time.time()
    print(f"\n[1/5] 📥 加载数据: {args.data_path}")
    print(f"   变量: {args.variable}")

    try:
        with tqdm(desc="   打开数据文件", leave=False) as pbar:
            ds = xr.open_zarr(args.data_path)
            pbar.update(1)
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
        start, end = args.time_slice.split(":")
        with tqdm(desc="   应用时间切片", leave=False) as pbar:
            ds = ds.sel(time=slice(start, end))
            pbar.update(1)
        print(f"   时间范围: {start} 至 {end}")

    # 获取变量数据
    with tqdm(desc="   读取变量数据", leave=False) as pbar:
        variable_data = ds[args.variable]
        data = variable_data.values  # (Time, H, W) 或 (Time, Lat, Lon)
        pbar.update(1)

    # ERA5 默认维度: (time, lat, lon)
    # 我们需要转换为 (time, height=lat, width=lon)
    if "latitude" in variable_data.dims and "longitude" in variable_data.dims:
        data = np.transpose(data, (0, 2, 1))  # 交换 (lat, lon)
        print("   ⚙️ 自动调整纬经度维度顺序: (time, lon, lat)")

    # 限制样本数量
    if args.n_samples is not None and args.n_samples < len(data):
        data = data[: args.n_samples]
        print(f"   ⚙️ 限制样本数: {args.n_samples}")

    load_time = time.time() - load_start
    print(f"   原始数据 shape: {data.shape}")
    print(f"   数据范围: [{data.min():.2f}, {data.max():.2f}]")
    print(f"   数据单位: {variable_data.attrs.get('units', 'N/A')}")
    print(f"   ✓ 加载完成 (耗时: {load_time:.2f}秒)")

    # 2. 插值到目标尺寸
    interp_start = time.time()
    print(f"\n[2/5] 🔄 插值到目标尺寸: {args.target_size}")
    if args.concurrent_interpolation:
        print(f"   使用并发处理（工作线程数: {args.n_workers}）")
    data_processed = prepare_weather_data(
        data,
        n_channels=args.n_channels,
        target_size=tuple(args.target_size),
        n_workers=args.n_workers,
        use_concurrent=args.concurrent_interpolation,
    )
    interp_time = time.time() - interp_start
    print(f"   处理后 shape: {data_processed.shape}")
    print(f"   ✓ 插值完成 (耗时: {interp_time:.2f}秒)")

    # 3. 归一化
    existing_norm_stats = None
    norm_stats_path = None
    if args.norm_stats_path:
        norm_stats_path = Path(args.norm_stats_path)
        if not norm_stats_path.exists():
            print(f"   ❌ 错误: 找不到归一化参数文件 {norm_stats_path}")
            return
        try:
            with open(norm_stats_path, "r") as f:
                existing_norm_stats = json.load(f)
            print(f"   使用外部归一化参数: {norm_stats_path}")
        except Exception as e:
            print(f"   ❌ 错误: 无法读取归一化参数文件: {e}")
            return

    normalization_method = (
        existing_norm_stats.get("method")
        if existing_norm_stats and existing_norm_stats.get("method")
        else args.normalization
    )

    if (
        existing_norm_stats
        and existing_norm_stats.get("method")
        and existing_norm_stats.get("method") != args.normalization
    ):
        print(
            f"   ⚠️ 提供的归一化方法 {existing_norm_stats.get('method')} "
            f"与命令行参数 {args.normalization} 不一致，将使用外部参数中的方法。"
        )

    norm_start = time.time()
    print(f"\n[3/5] 📊 全局归一化（方法: {normalization_method}）")

    if existing_norm_stats:
        print("   使用提供的统计量进行归一化")
    else:
        if args.concurrent_stats and args.n_workers > 1:
            print(f"   使用并发计算统计量（工作线程数: {args.n_workers}）")
            stats = compute_normalization_stats_concurrent(
                data_processed,
                method=normalization_method,
                n_workers=args.n_workers,
            )
        else:
            print("   计算统计量...", end="", flush=True)
            if normalization_method == "minmax":
                stats = {"min": data_processed.min(), "max": data_processed.max()}
            elif normalization_method == "zscore":
                stats = {
                    "mean": data_processed.mean(),
                    "std": data_processed.std(),
                }
            print(" ✓")

    if normalization_method == "minmax":
        if existing_norm_stats:
            global_min = existing_norm_stats.get("original_min")
            global_max = existing_norm_stats.get("original_max")
            if global_min is None or global_max is None:
                print("   ❌ 错误: 提供的归一化参数缺少 original_min 或 original_max")
                return
            print(
                f"   外部全局范围: [{float(global_min):.4f}, {float(global_max):.4f}]"
            )
        else:
            global_min = stats["min"]
            global_max = stats["max"]
            print(f"   全局范围: [{global_min:.4f}, {global_max:.4f}]")

        # 归一化到 [0, 255]
        print("   执行归一化...", end="", flush=True)
        denom = (global_max - global_min) + 1e-8
        data_normalized = (data_processed - global_min) / denom
        data_normalized = np.clip(data_normalized * 255.0, 0, 255).astype(np.uint8)
        print(" ✓")

        # 保存全局统计量用于后续反归一化
        global_mean = None
        global_std = None

    elif normalization_method == "zscore":
        if existing_norm_stats:
            global_mean = existing_norm_stats.get("original_mean")
            global_std = existing_norm_stats.get("original_std")
            if global_mean is None or global_std is None:
                print("   ❌ 错误: 提供的归一化参数缺少 original_mean 或 original_std")
                return
            print(
                f"   外部全局均值: {float(global_mean):.4f}, 标准差: {float(global_std):.4f}"
            )
        else:
            global_mean = stats["mean"]
            global_std = stats["std"]
            print(f"   全局均值: {global_mean:.4f}, 标准差: {global_std:.4f}")

        # Z-score 归一化再映射到 [0, 255]
        print("   执行归一化...", end="", flush=True)
        data_normalized = (data_processed - global_mean) / (global_std + 1e-8)
        data_normalized = np.clip((data_normalized + 3) / 6 * 255.0, 0, 255).astype(
            np.uint8
        )
        print(" ✓")

        # 保存全局统计量用于后续反归一化
        global_min = None
        global_max = None

    else:
        raise ValueError(f"Unknown normalization method: {normalization_method}")

    norm_time = time.time() - norm_start
    print(f"   归一化后范围: [{data_normalized.min()}, {data_normalized.max()}]")
    print(f"   ✓ 归一化完成 (耗时: {norm_time:.2f}秒)")

    # 4. 保存归一化参数
    save_stats_start = time.time()
    print(f"\n[4/5] 💾 保存归一化参数")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存归一化统计信息
    norm_stats = {
        "method": normalization_method,
        "variable": args.variable,
        "original_min": float(global_min) if normalization_method == "minmax" else None,
        "original_max": float(global_max) if normalization_method == "minmax" else None,
        "original_mean": (
            float(global_mean) if normalization_method == "zscore" else None
        ),
        "original_std": float(global_std) if normalization_method == "zscore" else None,
    }

    with open(output_dir / "normalization_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)

    save_stats_time = time.time() - save_stats_start
    print(f"   ✓ 归一化参数已保存: {output_dir / 'normalization_stats.json'}")
    print(f"   ✓ 保存完成 (耗时: {save_stats_time:.2f}秒)")

    # 5. 保存图片
    save_img_start = time.time()
    print(f"\n[5/5] 💾 保存图片到: {args.output_dir}")
    print(f"   使用并发保存（工作线程数: {args.n_workers}）")
    save_weather_images_colormap(
        data_normalized,
        output_dir,
        prefix=args.prefix,
        start_idx=0,
        cmap_name="turbo",
        n_workers=args.n_workers,
    )
    save_img_time = time.time() - save_img_start

    # 总耗时统计
    total_time = time.time() - total_start_time

    print(f"\n{'='*80}")
    print("✅ 处理完成！")
    print(f"{'='*80}")
    print(f"📊 处理统计:")
    print(f"   共保存 {len(data_normalized)} 张图片")
    print(f"   输出目录: {output_dir.absolute()}")
    print(f"   图片尺寸: {args.target_size}")
    print(f"   通道数: {args.n_channels}")
    print(f"\n⏱️  性能统计:")
    print(f"   数据加载: {load_time:.2f}秒")
    print(f"   数据插值: {interp_time:.2f}秒")
    print(f"   数据归一化: {norm_time:.2f}秒")
    print(f"   保存参数: {save_stats_time:.2f}秒")
    print(f"   保存图片: {save_img_time:.2f}秒")
    print(f"   总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    print(f"   平均速度: {len(data_normalized)/total_time:.2f} 张/秒")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
