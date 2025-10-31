"""
专门评估已训练 Diffusion 模型的 Ensemble 性能

使用方法：
    python evaluate_diffusion_ensemble.py \
        --model-path outputs/diffusion_20251029_010340/best_model.pth \
        --num-ensemble-members 20 \
        --num-inference-steps 100
"""

import argparse
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import torch

from src.data_loader import WeatherDataLoader
from src.models.diffusion import DiffusionWeatherModel, DiffusionTrainer
from src.metrics.probabilistic import (
    probabilistic_evaluation_report,
    continuous_ranked_probability_score_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Diffusion Ensemble")

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--num-ensemble-members",
        type=int,
        default=20,
        help="Number of ensemble members",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=100,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr",
        help="Path to ERA5 data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same as model)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 确定输出目录
    model_path = Path(args.model_path)
    if args.output_dir is None:
        output_dir = model_path.parent
    else:
        output_dir = Path(args.output_dir)

    print("=" * 80)
    print("Diffusion Ensemble Evaluation")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Output dir: {output_dir}")
    print(f"Ensemble members: {args.num_ensemble_members}")
    print(f"Inference steps: {args.num_inference_steps}")
    print()

    # 加载配置
    config_path = model_path.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    print("Loaded config:")
    print(f"  Variables: {config.get('variables', 'N/A')}")
    print(f"  Time slice: {config.get('time_slice', 'N/A')}")
    print(f"  Input length: {config.get('input_length', 12)}")
    print(f"  Output length: {config.get('output_length', 4)}")
    print()

    # 加载数据
    print("Loading data...")
    variables_str = config.get("variables", "2m_temperature")
    if isinstance(variables_str, str):
        variables = variables_str.split(",")
    else:
        variables = variables_str

    time_slice_str = config.get("time_slice", "2019-01-01:2020-12-31")
    start, end = time_slice_str.split(":")

    data_loader = WeatherDataLoader(
        data_path=args.data_path,
        variables=variables,
    )
    data_loader.load_data(time_slice=slice(start, end))

    X, y = data_loader.create_sequences(
        input_length=config.get("input_length", 12),
        output_length=config.get("output_length", 4),
        data_format="spatial",
    )

    # 划分数据集
    n_samples = len(X)
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)

    X_test = X[train_size + val_size :]
    y_test = y[train_size + val_size :]

    print(f"Test samples: {len(X_test)}")
    print()

    # 创建模型
    print("Creating model...")
    n_channels = config.get("n_channels", 1)

    model = DiffusionWeatherModel(
        input_channels=n_channels,
        output_channels=n_channels,
        input_length=config.get("input_length", 12),
        output_length=config.get("output_length", 4),
        base_channels=config.get("base_channels", 64),
        num_timesteps=config.get("num_diffusion_steps", 1000),
        beta_schedule=config.get("beta_schedule", "cosine"),
        dropout=config.get("dropout", 0.1),
    )

    trainer = DiffusionTrainer(
        model,
        use_ema=config.get("use_ema", True),
    )

    # 加载权重
    print(f"Loading checkpoint from {model_path}...")
    trainer.load_checkpoint(model_path)
    print("✓ Model loaded")
    print()

    # 生成 Ensemble 预测
    print("=" * 80)
    print(f"Generating {args.num_ensemble_members} ensemble members...")
    print("=" * 80)

    ensemble_predictions = trainer.predict_ensemble(
        X_test,
        num_members=args.num_ensemble_members,
        num_inference_steps=args.num_inference_steps,
    )

    print(f"Ensemble shape: {ensemble_predictions.shape}")
    print()

    # 计算概率评估指标
    print("=" * 80)
    print("Probabilistic Evaluation")
    print("=" * 80)

    probabilistic_stats = continuous_ranked_probability_score_summary(
        ensemble_predictions, y_test, variables=variables
    )

    # 打印报告
    report = probabilistic_evaluation_report(
        ensemble_predictions, y_test, variables=variables
    )
    print(report)

    # 保存结果
    results = {
        "ensemble_members": args.num_ensemble_members,
        "inference_steps": args.num_inference_steps,
        "crps_mean": probabilistic_stats["crps_mean"],
        "crps_std": probabilistic_stats["crps_std"],
        "ensemble_mean_rmse": probabilistic_stats["ensemble_mean_rmse"],
        "spread_skill_ratio": probabilistic_stats["spread_skill_ratio"],
        "crps_by_leadtime": probabilistic_stats.get("crps_by_leadtime", []),
    }

    if "crps_by_variable_names" in probabilistic_stats:
        results["crps_by_variable"] = probabilistic_stats["crps_by_variable_names"]

    results_path = output_dir / "ensemble_evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {results_path}")

    # 保存 ensemble 预测
    np.save(output_dir / "ensemble_predictions_reevaluated.npy", ensemble_predictions)
    print(
        f"✓ Ensemble predictions saved to {output_dir}/ensemble_predictions_reevaluated.npy"
    )

    # 可视化
    print("\nGenerating visualizations...")

    # Ensemble spread
    sample_idx = 0
    var_idx = 0
    time_idx = 0
    H, W = y_test.shape[3], y_test.shape[4]
    point_h, point_w = H // 2, W // 2

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 左图：Ensemble 成员
    ax = axes[0]
    for member_idx in range(min(10, args.num_ensemble_members)):
        member_pred = ensemble_predictions[member_idx, sample_idx, time_idx, var_idx]
        ax.plot(member_pred[point_h, :], alpha=0.4, label=f"Member {member_idx+1}")

    y_true_spatial = y_test[sample_idx, time_idx, var_idx]
    ax.plot(
        y_true_spatial[point_h, :], "k-", linewidth=2, label="Observation", alpha=0.8
    )
    ax.set_title(f"Ensemble Spread (Spatial Profile)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Normalized Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 右图：Ensemble mean ± spread
    ax = axes[1]
    ensemble_mean = np.mean(
        ensemble_predictions[:, sample_idx, time_idx, var_idx], axis=0
    )
    ensemble_std = np.std(
        ensemble_predictions[:, sample_idx, time_idx, var_idx], axis=0
    )

    lon_range = np.arange(ensemble_mean.shape[1])
    ax.plot(
        lon_range, ensemble_mean[point_h, :], "r-", linewidth=2, label="Ensemble Mean"
    )
    ax.fill_between(
        lon_range,
        ensemble_mean[point_h, :] - ensemble_std[point_h, :],
        ensemble_mean[point_h, :] + ensemble_std[point_h, :],
        alpha=0.3,
        color="red",
        label="±1 std (spread)",
    )
    ax.plot(
        lon_range,
        y_true_spatial[point_h, :],
        "k-",
        linewidth=2,
        label="Observation",
        alpha=0.8,
    )
    ax.set_title("Ensemble Mean ± Spread")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Normalized Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "ensemble_evaluation_spread.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"✓ Visualizations saved")
    print()
    print("=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
