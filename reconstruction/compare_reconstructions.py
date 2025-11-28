#!/usr/bin/env python3
"""
统一的重建对比脚本 - 支持 VAE 和 RAE 结果对比

功能：
1. 从图像目录加载原始图像和重建图像
2. 计算评估指标（RMSE, MAE, PSNR, SSIM, 相关系数等）
3. 生成对比可视化图像
4. 支持多个重建结果同时对比

使用方法:
    # 对比单个重建结果
    python compare_reconstructions.py \
        --original-dir weather_images \
        --reconstructed-dir outputs/vae_reconstruction/reconstructed \
        --output comparison_vae.png \
        --metrics-output metrics_vae.json

    # 对比多个重建结果
    python compare_reconstructions.py \
        --original-dir weather_images \
        --reconstructed-dirs outputs/vae_reconstruction/reconstructed recon_samples_DINOv2-B/RAE-pretrained-bs4-fp32 \
        --labels VAE RAE-DINOv2-B \
        --output comparison_all.png \
        --metrics-output metrics_all.json
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from weatherdiff.utils.metrics import calculate_metrics, format_metrics


def load_image(image_path: Path) -> np.ndarray:
    """加载图片为 numpy 数组，返回 (H, W, 3)"""
    img = Image.open(image_path).convert("RGB")
    return np.array(img)


def prepare_image_for_display(image: np.ndarray, colormap: Optional[str]):
    """
    如果提供了 colormap，则直接将图像以伪彩色展示（假定为灰度）
    """
    if colormap is None:
        return image, None

    if image.ndim == 2:
        return image, colormap

    # 假定图像为单通道灰度（复制到 RGB），取第一个通道应用伪彩色
    return image[:, :, 0], colormap


def load_images_from_dir(image_dir: Path, max_samples: int = None) -> np.ndarray:
    """
    从目录加载所有图像

    Returns:
        images: (N, H, W, 3) numpy array
    """
    image_files = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
    if len(image_files) == 0:
        raise ValueError(f"在 {image_dir} 中未找到图像文件")

    if max_samples is not None:
        image_files = image_files[:max_samples]

    images = []
    for img_path in image_files:
        img = load_image(img_path)
        images.append(img)

    return np.stack(images)


def denormalize_image(img_array: np.ndarray, norm_stats: dict = None) -> np.ndarray:
    """
    将图像数组从 [0, 255] 反归一化到物理单位

    Args:
        img_array: (H, W, 3) 或 (N, H, W, 3)，范围 [0, 255]
        norm_stats: 归一化统计信息字典（从 normalization_stats.json 加载）

    Returns:
        反归一化后的数据（物理单位，如 K）
    """
    # 将 [0, 255] 映射回 [0, 1]
    normalized = img_array.astype(np.float32) / 255.0

    # 取单通道（假设所有通道相同）
    if normalized.ndim == 4:
        normalized = normalized[:, :, :, 0]  # (N, H, W)
    elif normalized.ndim == 3:
        normalized = normalized[:, :, 0]  # (H, W)

    if norm_stats is None:
        # 如果没有归一化参数，假设原始范围是 [200, 320] K
        data_orig = normalized * (320 - 200) + 200
    else:
        method = norm_stats.get("method", "minmax")
        if method == "minmax":
            orig_min = norm_stats.get("original_min")
            orig_max = norm_stats.get("original_max")
            if orig_min is not None and orig_max is not None:
                # [0, 1] -> 原始范围
                data_orig = normalized * (orig_max - orig_min) + orig_min
            else:
                data_orig = normalized * (320 - 200) + 200
        else:  # zscore
            # zscore: 图像是 (z + 3) / 6 * 255，所以 z = (img/255 * 6) - 3
            orig_mean = norm_stats.get("original_mean")
            orig_std = norm_stats.get("original_std")
            if orig_mean is not None and orig_std is not None:
                z = normalized * 6.0 - 3.0
                data_orig = z * orig_std + orig_mean
            else:
                data_orig = normalized * (320 - 200) + 200

    return data_orig


def calculate_reconstruction_metrics(
    original_images: np.ndarray,
    reconstructed_images: np.ndarray,
    denormalize: bool = True,
    norm_stats: dict = None,
) -> dict:
    """
    计算重建指标

    Args:
        original_images: (N, H, W, 3) 原始图像
        reconstructed_images: (N, H, W, 3) 重建图像
        denormalize: 是否反归一化到物理单位

    Returns:
        指标字典
    """
    # 转换为 (N, H, W) 单通道
    if original_images.ndim == 4:
        orig = original_images[:, :, :, 0]  # 取第一个通道
        recon = reconstructed_images[:, :, :, 0]
    else:
        orig = original_images
        recon = reconstructed_images

    # 归一化到 [0, 1] 用于计算指标
    orig_norm = orig.astype(np.float32) / 255.0
    recon_norm = recon.astype(np.float32) / 255.0

    # 计算归一化空间的指标
    metrics_norm = calculate_metrics(recon_norm, orig_norm, ensemble=False)

    # 如果需要，计算物理单位的指标
    if denormalize:
        orig_phys = denormalize_image(orig, norm_stats)
        recon_phys = denormalize_image(recon, norm_stats)
        metrics_phys = calculate_metrics(recon_phys, orig_phys, ensemble=False)

        return {"normalized_space": metrics_norm, "original_space": metrics_phys}
    else:
        return {"normalized_space": metrics_norm}


def create_comparison_visualization(
    original_images: np.ndarray,
    reconstructed_images_list: list,
    labels: list,
    indices: list,
    output_path: Path,
    metrics_list: list = None,
    colormap: Optional[str] = None,
):
    """
    创建对比可视化

    Args:
        original_images: (N, H, W, 3) 原始图像
        reconstructed_images_list: 重建图像列表，每个元素是 (N, H, W, 3)
        labels: 重建结果的标签列表
        indices: 要显示的样本索引
        output_path: 输出路径
        metrics_list: 每个重建结果的指标列表（可选）
    """
    n_samples = len(indices)
    n_methods = len(reconstructed_images_list)

    # 创建子图：第一行是原始图像，后续行是各个重建结果
    fig, axes = plt.subplots(
        n_methods + 1, n_samples, figsize=(4 * n_samples, 4 * (n_methods + 1))
    )

    if n_samples == 1:
        axes = axes[:, np.newaxis]
    if n_methods == 0:
        axes = axes[np.newaxis, :]

    # 第一行：原始图像
    for col, idx in enumerate(indices):
        if idx < len(original_images):
            orig_display, orig_cmap = prepare_image_for_display(
                original_images[idx], colormap
            )
            axes[0, col].imshow(orig_display, cmap=orig_cmap)
            axes[0, col].set_title(f"Sample {idx:06d}\nOriginal", fontsize=10)
            axes[0, col].axis("off")

    # 后续行：各个重建结果
    for row, (recon_imgs, label) in enumerate(
        zip(reconstructed_images_list, labels), start=1
    ):
        metrics = metrics_list[row - 1] if metrics_list else None

        for col, idx in enumerate(indices):
            if idx < len(recon_imgs):
                recon_display, recon_cmap = prepare_image_for_display(
                    recon_imgs[idx], colormap
                )
                axes[row, col].imshow(recon_display, cmap=recon_cmap)

                # 添加标题和指标
                title = f"Sample {idx:06d}\n{label}"
                if metrics and "original_space" in metrics:
                    rmse = metrics["original_space"].get("rmse", 0)
                    title += f"\nRMSE: {rmse:.2f} K"
                axes[row, col].set_title(title, fontsize=10)
                axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ 对比图保存到: {output_path}")
    plt.close()


def create_metrics_comparison_table(
    metrics_list: list, labels: list, output_path: Path
):
    """
    创建指标对比表格，同时显示归一化空间和物理空间的指标

    Args:
        metrics_list: 指标列表
        labels: 标签列表
        output_path: 输出路径
    """
    # 检查是否有物理空间指标
    has_original_space = any("original_space" in m for m in metrics_list)

    if has_original_space:
        # 同时显示归一化空间和物理空间指标
        fig, ax = plt.subplots(figsize=(18, 6))
        ax.axis("tight")
        ax.axis("off")

        # 准备表格数据：每个指标有两列（norm 和 original）
        table_data = []
        headers = [
            "Method",
            "RMSE\n(norm)",
            "RMSE\n(K)",
            "MAE\n(norm)",
            "MAE\n(K)",
            "PSNR\n(norm)",
            "PSNR\n(dB)",
            "SSIM\n(norm)",
            "SSIM",
            "Correlation\n(norm)",
            "Correlation",
        ]

        for metrics, label in zip(metrics_list, labels):
            m_norm = metrics.get("normalized_space", {})
            m_orig = metrics.get("original_space", {})

            row = [
                label,
                f"{m_norm.get('rmse', 0):.4f}",
                f"{m_orig.get('rmse', 0):.4f}" if m_orig else "N/A",
                f"{m_norm.get('mae', 0):.4f}",
                f"{m_orig.get('mae', 0):.4f}" if m_orig else "N/A",
                f"{m_norm.get('psnr', 0):.2f}",
                f"{m_orig.get('psnr', 0):.2f}" if m_orig else "N/A",
                f"{m_norm.get('ssim', 0):.4f}",
                f"{m_orig.get('ssim', 0):.4f}" if m_orig else "N/A",
                f"{m_norm.get('correlation', 0):.4f}",
                f"{m_orig.get('correlation', 0):.4f}" if m_orig else "N/A",
            ]
            table_data.append(row)

        table = ax.table(
            cellText=table_data, colLabels=headers, cellLoc="center", loc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.8)

        # 设置表头样式
        for i in range(len(headers)):
            table[(0, i)].set_facecolor("#40466e")
            table[(0, i)].set_text_props(weight="bold", color="white")

        # 为归一化空间和物理空间的列设置不同的背景色以便区分
        norm_cols = [1, 3, 5, 7, 9]  # RMSE, MAE, PSNR, SSIM, Correlation 的 norm 列
        orig_cols = [2, 4, 6, 8, 10]  # 对应的 original 列

        for row_idx in range(1, len(table_data) + 1):
            for col_idx in norm_cols:
                table[(row_idx, col_idx)].set_facecolor("#f0f0f0")
            for col_idx in orig_cols:
                table[(row_idx, col_idx)].set_facecolor("#e8f4f8")

        plt.title(
            "Reconstruction Metrics Comparison (Normalized Space vs Original Space)",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
    else:
        # 只有归一化空间指标
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis("tight")
        ax.axis("off")

        table_data = []
        headers = [
            "Method",
            "RMSE (norm)",
            "MAE (norm)",
            "PSNR (norm)",
            "SSIM (norm)",
            "Correlation (norm)",
        ]

        for metrics, label in zip(metrics_list, labels):
            m = metrics.get("normalized_space", {})
            row = [
                label,
                f"{m.get('rmse', 0):.4f}",
                f"{m.get('mae', 0):.4f}",
                f"{m.get('psnr', 0):.2f}",
                f"{m.get('ssim', 0):.4f}",
                f"{m.get('correlation', 0):.4f}",
            ]
            table_data.append(row)

        table = ax.table(
            cellText=table_data, colLabels=headers, cellLoc="center", loc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # 设置表头样式
        for i in range(len(headers)):
            table[(0, i)].set_facecolor("#40466e")
            table[(0, i)].set_text_props(weight="bold", color="white")

        plt.title(
            "Reconstruction Metrics Comparison (Normalized Space)",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ 指标对比表保存到: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="统一的重建对比脚本 - 支持 VAE 和 RAE 结果对比"
    )

    parser.add_argument(
        "--original-dir",
        type=str,
        required=True,
        help="原始图像目录",
    )
    parser.add_argument(
        "--reconstructed-dir",
        type=str,
        default=None,
        help="重建图像目录（单个）",
    )
    parser.add_argument(
        "--reconstructed-dirs",
        type=str,
        nargs="+",
        default=None,
        help="重建图像目录列表（多个）",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="重建结果的标签（与 --reconstructed-dirs 对应）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reconstruction_comparison.png",
        help="输出对比图路径",
    )
    parser.add_argument(
        "--metrics-output",
        type=str,
        default=None,
        help="指标输出 JSON 文件路径",
    )
    parser.add_argument(
        "--table-output",
        type=str,
        default=None,
        help="指标对比表输出路径",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="要显示的样本索引（默认自动选择4个）",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最大样本数量（用于计算指标）",
    )
    parser.add_argument(
        "--denormalize",
        action="store_true",
        help="是否反归一化到物理单位计算指标",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="RdYlBu_r",
        help="灰度图使用的伪彩色 colormap，默认 RdYlBu_r，设为 none 关闭",
    )
    args = parser.parse_args()

    # 确定重建目录列表
    if args.reconstructed_dirs:
        recon_dirs = [Path(d) for d in args.reconstructed_dirs]
    elif args.reconstructed_dir:
        recon_dirs = [Path(args.reconstructed_dir)]
    else:
        raise ValueError("必须指定 --reconstructed-dir 或 --reconstructed-dirs")

    # 确定标签
    if args.labels:
        labels = args.labels
        if len(labels) != len(recon_dirs):
            raise ValueError(
                f"标签数量 ({len(labels)}) 与重建目录数量 ({len(recon_dirs)}) 不匹配"
            )
    else:
        labels = [f"Method_{i+1}" for i in range(len(recon_dirs))]

    original_dir = Path(args.original_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("重建对比分析")
    print("=" * 80)
    print(f"原始图像目录: {original_dir}")
    print(f"重建目录数量: {len(recon_dirs)}")
    for i, (recon_dir, label) in enumerate(zip(recon_dirs, labels)):
        print(f"  {i+1}. {label}: {recon_dir}")

    # 加载归一化参数（如果存在）
    norm_stats_file = original_dir / "normalization_stats.json"
    norm_stats = None
    if norm_stats_file.exists():
        with open(norm_stats_file, "r") as f:
            norm_stats = json.load(f)
        print(f"\n✓ 加载归一化参数: {norm_stats_file}")
    else:
        print(f"\n⚠ 未找到归一化参数文件: {norm_stats_file}")
        print("  将使用默认范围 [200, 320] K 进行反归一化")

    # 加载原始图像
    print("\n加载原始图像...")
    original_images = load_images_from_dir(original_dir, max_samples=args.max_samples)
    print(f"✓ 加载了 {len(original_images)} 张原始图像")

    # 加载重建图像
    print("\n加载重建图像...")
    reconstructed_images_list = []
    for recon_dir, label in zip(recon_dirs, labels):
        if not recon_dir.exists():
            print(f"⚠ 警告: 重建目录不存在: {recon_dir}，跳过")
            continue

        recon_imgs = load_images_from_dir(recon_dir, max_samples=args.max_samples)
        print(f"✓ {label}: 加载了 {len(recon_imgs)} 张重建图像")
        reconstructed_images_list.append(recon_imgs)

    if len(reconstructed_images_list) == 0:
        raise ValueError("没有成功加载任何重建图像")

    # 确保所有图像数量一致
    n_samples = min(
        len(original_images), min(len(imgs) for imgs in reconstructed_images_list)
    )
    original_images = original_images[:n_samples]
    reconstructed_images_list = [imgs[:n_samples] for imgs in reconstructed_images_list]

    print(f"\n使用 {n_samples} 个样本进行对比")

    # 计算指标
    print("\n计算评估指标...")
    metrics_list = []
    for recon_imgs, label in zip(reconstructed_images_list, labels):
        metrics = calculate_reconstruction_metrics(
            original_images,
            recon_imgs,
            denormalize=args.denormalize,
            norm_stats=norm_stats,
        )
        metrics_list.append(metrics)

        print(f"\n{label} 指标:")
        if "original_space" in metrics:
            print(format_metrics(metrics["original_space"]))
        else:
            print(format_metrics(metrics["normalized_space"]))

    # 保存指标
    if args.metrics_output:
        metrics_output = Path(args.metrics_output)
        metrics_output.parent.mkdir(parents=True, exist_ok=True)

        # 转换numpy类型为Python原生类型
        def convert_to_python_types(obj):
            if isinstance(obj, dict):
                return {k: convert_to_python_types(v) for k, v in obj.items()}
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        results = {
            "labels": labels,
            "n_samples": int(n_samples),
            "metrics": [convert_to_python_types(m) for m in metrics_list],
        }

        with open(metrics_output, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ 指标已保存到: {metrics_output}")

    # 确定要显示的样本索引
    if args.indices:
        indices = args.indices[:4]  # 最多4个
    else:
        # 自动选择4个均匀分布的样本
        if n_samples >= 4:
            step = n_samples // 4
            indices = [i * step for i in range(4)]
        else:
            indices = list(range(n_samples))

    colormap = None if str(args.colormap).lower() in {"none", "null"} else args.colormap

    # 创建对比可视化
    print(f"\n生成对比可视化（样本索引: {indices}）...")
    create_comparison_visualization(
        original_images,
        reconstructed_images_list,
        labels,
        indices,
        output_path,
        metrics_list,
        colormap=colormap,
    )

    # 创建指标对比表
    if args.table_output:
        print("\n生成指标对比表...")
        create_metrics_comparison_table(metrics_list, labels, Path(args.table_output))

    print("\n" + "=" * 80)
    print("✅ 对比分析完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
