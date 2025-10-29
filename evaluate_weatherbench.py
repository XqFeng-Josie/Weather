"""
WeatherBench2评估脚本
支持flat和spatial数据格式

运行示例:
python evaluate_weatherbench.py --pred predictions.nc
python evaluate_weatherbench.py --pred predictions.npz --output-dir results/
"""

import argparse
import numpy as np
import xarray as xr
from pathlib import Path
import json
import matplotlib.pyplot as plt
from typing import Dict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate predictions against WeatherBench2"
    )

    parser.add_argument(
        "--pred",
        type=str,
        required=True,
        help="Path to prediction file (netCDF or numpy)",
    )

    parser.add_argument(
        "--variables",
        type=str,
        nargs="+",
        default=None,
        help="Variables to evaluate (auto-detect if None)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results",
        help="Output directory for results",
    )

    parser.add_argument(
        "--compare-baseline",
        type=str,
        default=None,
        help="Path to baseline predictions for comparison",
    )

    return parser.parse_args()


def load_predictions(pred_path):
    """加载预测结果"""
    pred_path = Path(pred_path)

    if pred_path.suffix == ".nc":
        ds = xr.open_dataset(pred_path)
        
        # 提取pred和true变量
        pred_vars = [v for v in ds.data_vars if v.endswith('_pred')]
        
        y_pred_list = []
        y_true_list = []
        var_names = []
        
        for pv in pred_vars:
            var_name = pv.replace('_pred', '')
            tv = f"{var_name}_true"
            
            if tv in ds.data_vars:
                y_pred_list.append(ds[pv].values)
                y_true_list.append(ds[tv].values)
                var_names.append(var_name)
        
        # 确定数据格式
        if len(y_pred_list[0].shape) == 4:  # (time, lead_time, lat, lon)
            data_format = 'spatial'
            # Stack: (time, lead_time, channels, lat, lon)
            y_pred = np.stack([yp.transpose(0, 1, 2, 3) if yp.ndim == 4 else yp 
                             for yp in y_pred_list], axis=2)
            y_true = np.stack([yt.transpose(0, 1, 2, 3) if yt.ndim == 4 else yt 
                             for yt in y_true_list], axis=2)
        else:  # (time, lead_time)
            data_format = 'flat'
            y_pred = np.stack(y_pred_list, axis=2)
            y_true = np.stack(y_true_list, axis=2)
        
        return {
            "y_pred": y_pred,
            "y_true": y_true,
            "features": var_names,
            "data_format": data_format,
        }
        
    elif pred_path.suffix == ".npz":
        data = np.load(pred_path, allow_pickle=True)
        return {
            "y_pred": data["y_pred"],
            "y_true": data["y_true"],
            "features": list(data["features"]),
            "data_format": str(data.get("data_format", "flat")),
        }
    else:
        raise ValueError(f"Unsupported file format: {pred_path.suffix}")


def evaluate_by_lead_time(y_pred, y_true, data_format):
    """
    按lead time评估
    
    支持两种格式:
    - flat: (samples, lead_time, features)
    - spatial: (samples, lead_time, channels, H, W)
    """
    if data_format == 'spatial':
        # Spatial: 对H,W维度求平均后计算指标
        n_samples, n_lead_times, n_channels, H, W = y_pred.shape
        n_features = n_channels
        
        # 计算空间平均
        y_pred_mean = y_pred.reshape(n_samples, n_lead_times, n_channels, -1).mean(axis=-1)
        y_true_mean = y_true.reshape(n_samples, n_lead_times, n_channels, -1).mean(axis=-1)
        
    else:
        # Flat
        n_samples, n_lead_times, n_features = y_pred.shape
        y_pred_mean = y_pred
        y_true_mean = y_true
    
    metrics = {
        "rmse": np.zeros((n_lead_times, n_features)),
        "mae": np.zeros((n_lead_times, n_features)),
        "bias": np.zeros((n_lead_times, n_features)),
    }
    
    # 原始数据的指标（包含所有空间点）
    metrics_raw = {
        "rmse": np.zeros(n_lead_times),
        "mae": np.zeros(n_lead_times),
    }

    for t in range(n_lead_times):
        # 整体指标
        metrics_raw["rmse"][t] = np.sqrt(np.mean((y_pred[:, t] - y_true[:, t]) ** 2))
        metrics_raw["mae"][t] = np.mean(np.abs(y_pred[:, t] - y_true[:, t]))
        
        # 按变量/通道的指标
        for f in range(n_features):
            pred = y_pred_mean[:, t, f]
            true = y_true_mean[:, t, f]

            metrics["rmse"][t, f] = np.sqrt(np.mean((pred - true) ** 2))
            metrics["mae"][t, f] = np.mean(np.abs(pred - true))
            metrics["bias"][t, f] = np.mean(pred - true)
    
    metrics["rmse_raw"] = metrics_raw["rmse"]
    metrics["mae_raw"] = metrics_raw["mae"]

    return metrics


def plot_rmse_by_leadtime(metrics, variable_names, output_path, baseline_metrics=None):
    """绘制RMSE vs Lead Time"""
    n_lead_times, n_features = metrics["rmse"].shape
    lead_times = np.arange(1, n_lead_times + 1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flat

    for i in range(min(4, n_features)):
        var_name = variable_names[i] if i < len(variable_names) else f"var_{i}"
        ax = axes[i]

        # 模型RMSE
        ax.plot(
            lead_times,
            metrics["rmse"][:, i],
            "o-",
            label="Model",
            linewidth=2,
            markersize=6,
        )

        # Baseline（如果有）
        if baseline_metrics is not None and i < baseline_metrics["rmse"].shape[1]:
            ax.plot(
                lead_times,
                baseline_metrics["rmse"][:, i],
                "s--",
                label="Baseline",
                linewidth=2,
                markersize=6,
                alpha=0.7,
            )

        ax.set_xlabel("Lead Time Step")
        ax.set_ylabel("RMSE")
        ax.set_title(f"{var_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ RMSE plot saved to {output_path}")


def plot_error_distribution(y_pred, y_true, variable_names, output_path, data_format):
    """绘制误差分布"""
    errors = y_pred - y_true

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flat

    for i in range(min(4, len(variable_names))):
        var_name = variable_names[i] if i < len(variable_names) else f"var_{i}"
        ax = axes[i]

        # 提取第一个lead time的误差
        if data_format == 'spatial':
            # (samples, lead_time, channels, H, W)
            if i < errors.shape[2]:
                error_flat = errors[:, 0, i].flatten()
            else:
                continue
        else:
            # (samples, lead_time, features)
            if i < errors.shape[2]:
                error_flat = errors[:, 0, i].flatten()
            else:
                continue

        ax.hist(error_flat, bins=50, alpha=0.7, edgecolor="black")
        ax.axvline(x=0, color="r", linestyle="--", linewidth=2, label="Zero Error")
        ax.set_xlabel("Prediction Error")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{var_name} (Lead Time 1)")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Error distribution plot saved to {output_path}")


def plot_spatial_error(y_pred, y_true, variable_idx, output_path):
    """绘制空间误差图 (仅spatial格式)"""
    if len(y_pred.shape) != 5:
        return
    
    # (samples, lead_time, channels, H, W)
    sample_idx = 0
    lead_idx = 0
    
    if variable_idx >= y_pred.shape[2]:
        return
    
    error = y_pred[sample_idx, lead_idx, variable_idx] - y_true[sample_idx, lead_idx, variable_idx]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # True
    im = axes[0].imshow(y_true[sample_idx, lead_idx, variable_idx], cmap='RdBu_r')
    axes[0].set_title('Ground Truth')
    plt.colorbar(im, ax=axes[0])
    
    # Pred
    im = axes[1].imshow(y_pred[sample_idx, lead_idx, variable_idx], cmap='RdBu_r')
    axes[1].set_title('Prediction')
    plt.colorbar(im, ax=axes[1])
    
    # Error
    im = axes[2].imshow(error, cmap='seismic')
    axes[2].set_title('Error (Pred - True)')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Spatial error plot saved to {output_path}")


def main():
    args = parse_args()

    print("=" * 80)
    print("WeatherBench2 Evaluation")
    print("=" * 80)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载预测
    print("\nLoading predictions...")
    pred_data = load_predictions(args.pred)

    y_pred = pred_data["y_pred"]
    y_true = pred_data["y_true"]
    variable_names = pred_data["features"]
    data_format = pred_data["data_format"]

    print(f"Data format: {data_format}")
    print(f"Variables: {variable_names}")
    print(f"Prediction shape: {y_pred.shape}")

    # 2. 计算指标
    print("\nCalculating metrics...")
    metrics = evaluate_by_lead_time(y_pred, y_true, data_format)

    print("\nRMSE by Lead Time (spatial average):")
    for t in range(metrics["rmse"].shape[0]):
        print(f"  Lead {t+1}: ", end="")
        for f, var in enumerate(variable_names):
            if f < metrics["rmse"].shape[1]:
                print(f"{var}={metrics['rmse'][t, f]:.4f} ", end="")
        print()
    
    if "rmse_raw" in metrics:
        print("\nOverall RMSE (all points):")
        for t in range(len(metrics["rmse_raw"])):
            print(f"  Lead {t+1}: {metrics['rmse_raw'][t]:.4f}")

    # 3. 加载baseline（如果有）
    baseline_metrics = None
    if args.compare_baseline:
        print(f"\nLoading baseline from {args.compare_baseline}...")
        baseline_data = load_predictions(args.compare_baseline)
        baseline_metrics = evaluate_by_lead_time(
            baseline_data["y_pred"],
            baseline_data["y_true"],
            baseline_data["data_format"]
        )

    # 4. 可视化
    print("\nGenerating visualizations...")

    # RMSE vs Lead Time
    plot_rmse_by_leadtime(
        metrics, variable_names, output_dir / "rmse_by_leadtime.png", baseline_metrics
    )

    # 误差分布
    plot_error_distribution(
        y_pred, y_true, variable_names, output_dir / "error_distribution.png", data_format
    )
    
    # 空间误差图 (仅spatial格式)
    if data_format == 'spatial' and len(variable_names) > 0:
        plot_spatial_error(
            y_pred, y_true, 0, output_dir / "spatial_error.png"
        )

    # 5. 保存指标
    metrics_dict = {
        "data_format": data_format,
        "variables": variable_names,
        "model": {
            "rmse": metrics["rmse"].tolist(),
            "mae": metrics["mae"].tolist(),
            "bias": metrics["bias"].tolist(),
        },
    }
    
    if "rmse_raw" in metrics:
        metrics_dict["model"]["rmse_overall"] = metrics["rmse_raw"].tolist()
        metrics_dict["model"]["mae_overall"] = metrics["mae_raw"].tolist()

    if baseline_metrics is not None:
        metrics_dict["baseline"] = {
            "rmse": baseline_metrics["rmse"].tolist(),
        }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)

    print(f"\n✓ Metrics saved to {output_dir}/metrics.json")
    print(f"✓ Evaluation complete!")
    print(f"✓ Results saved to {output_dir}")


if __name__ == "__main__":
    main()
