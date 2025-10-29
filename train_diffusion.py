"""
Diffusion Model Training Script

独立的Diffusion模型训练脚本，与其他模型分离以保持项目整洁

参考架构：
- DDPM: https://arxiv.org/abs/2006.11239
- DiT: https://arxiv.org/abs/2212.09748
- GenCast (Google DeepMind)
- Pangu-Weather (Huawei)
"""

import argparse
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from src.data_loader import WeatherDataLoader
from src.models.diffusion import DiffusionWeatherModel, DiffusionTrainer
from src.metrics.probabilistic import (
    probabilistic_evaluation_report,
    continuous_ranked_probability_score_summary
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Diffusion weather prediction model")
    
    # Data
    parser.add_argument(
        "--data-path",
        type=str,
        default="gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr",
        help="Path to ERA5 data",
    )
    parser.add_argument(
        "--variables",
        type=str,
        default="2m_temperature",
        help="Variables to predict (comma-separated)",
    )
    parser.add_argument(
        "--time-slice",
        type=str,
        default="2020-01-01:2020-12-31",
        help="Time slice for training (start:end)",
    )
    parser.add_argument(
        "--input-length",
        type=int,
        default=12,
        help="Input sequence length",
    )
    parser.add_argument(
        "--output-length",
        type=int,
        default=4,
        help="Output sequence length",
    )
    
    # Model
    parser.add_argument(
        "--base-channels",
        type=int,
        default=64,
        help="UNet base channels (keep small to avoid OOM)",
    )
    parser.add_argument(
        "--num-diffusion-steps",
        type=int,
        default=1000,
        help="Number of diffusion timesteps",
    )
    parser.add_argument(
        "--beta-schedule",
        type=str,
        default="cosine",
        choices=["linear", "cosine"],
        help="Noise schedule",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of inference steps (fewer = faster)",
    )
    parser.add_argument(
        "--num-ensemble-members",
        type=int,
        default=10,
        help="Number of ensemble members for probabilistic evaluation (GenCast style)",
    )
    parser.add_argument(
        "--enable-ensemble-eval",
        action="store_true",
        help="Enable ensemble evaluation (CRPS, spread-skill, etc.)",
    )
    
    # Training
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs (Diffusion需要长时间训练)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (keep small!)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (降低以提高稳定性)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--early-stop", type=int, default=15, help="Early stopping patience"
    )
    parser.add_argument(
        "--use-ema",
        type=lambda x: (str(x).lower() == 'true'),
        default=True,
        help="Use exponential moving average (True/False)",
    )
    
    # Output
    parser.add_argument(
        "--output-dir", type=str, default="./outputs", help="Output directory"
    )
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 创建输出目录
    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"diffusion_{timestamp}"
    
    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Diffusion Weather Prediction Model")
    print("="*80)
    print(f"Experiment: {args.exp_name}")
    print(f"Output dir: {output_dir}")
    
    # ========================================================================
    # 1. 加载数据
    # ========================================================================
    print("\n" + "=" * 80)
    print("Loading Data")
    print("=" * 80)
    
    variables = [v.strip() for v in args.variables.split(',')]
    loader = WeatherDataLoader(data_path=args.data_path, variables=variables)
    
    start, end = args.time_slice.split(":")
    ds = loader.load_data(time_slice=slice(start, end))
    features = loader.prepare_features(normalize=True)
    
    print(f"\nVariables: {variables}")
    
    # Diffusion模型使用spatial格式
    print(f"\nUsing SPATIAL format for diffusion")
    X, y = loader.create_sequences(
        features,
        input_length=args.input_length,
        output_length=args.output_length,
        format='spatial'
    )
    
    print(f"\nData shapes:")
    print(f"  X: {X.shape} - (samples, input_length, channels, H, W)")
    print(f"  y: {y.shape} - (samples, output_length, channels, H, W)")
    
    # 划分数据
    data_splits = loader.split_data(X, y, train_ratio=0.7, val_ratio=0.15)
    X_train = data_splits["X_train"]
    y_train = data_splits["y_train"]
    X_val = data_splits["X_val"]
    y_val = data_splits["y_val"]
    X_test = data_splits["X_test"]
    y_test = data_splits["y_test"]
    
    print(f"\nData splits:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")
    
    # ========================================================================
    # 2. 创建模型
    # ========================================================================
    print("\n" + "=" * 80)
    print("Creating Diffusion Model")
    print("=" * 80)
    
    n_channels = X.shape[2]
    
    model = DiffusionWeatherModel(
        input_channels=n_channels,
        output_channels=n_channels,
        input_length=args.input_length,
        output_length=args.output_length,
        base_channels=args.base_channels,
        num_timesteps=args.num_diffusion_steps,
        beta_schedule=args.beta_schedule,
        dropout=args.dropout,
    )
    
    # 统计参数
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: Diffusion")
    print(f"Parameters: {n_params:,}")
    print(f"Diffusion steps: {args.num_diffusion_steps}")
    print(f"Beta schedule: {args.beta_schedule}")
    print(f"Using EMA: {args.use_ema}")
    
    # 保存配置
    config = vars(args)
    config['model'] = 'diffusion'  # 添加模型类型
    config['n_channels'] = int(n_channels)
    config['input_channels'] = int(n_channels)
    config['output_channels'] = int(n_channels)
    config['variables'] = variables
    config['n_params'] = int(n_params)
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # ========================================================================
    # 3. 训练
    # ========================================================================
    print("\n" + "=" * 80)
    print("Training")
    print("=" * 80)
    
    trainer = DiffusionTrainer(
        model,
        learning_rate=args.lr,
        use_ema=args.use_ema,
    )
    
    history = trainer.train_model(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping_patience=args.early_stop,
        checkpoint_path=str(output_dir / "best_model.pth"),
        num_inference_steps=args.num_inference_steps,
    )
    
    # ========================================================================
    # 4. 评估
    # ========================================================================
    print("\n" + "=" * 80)
    print("Evaluation")
    print("=" * 80)
    
    print("\nGenerating test predictions...")
    y_test_pred = trainer.predict(
        X_test,
        num_inference_steps=args.num_inference_steps,
        use_ema=args.use_ema
    )
    
    # 计算指标
    mse = np.mean((y_test - y_test_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_test_pred))
    
    # 按时间步计算RMSE
    rmse_per_step = {}
    for i in range(args.output_length):
        rmse_step = np.sqrt(np.mean((y_test[:, i] - y_test_pred[:, i]) ** 2))
        rmse_per_step[f"rmse_step_{i+1}"] = float(rmse_step)
    
    test_metrics = {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        **rmse_per_step
    }
    
    print("\nTest Metrics (Deterministic - Ensemble Mean):")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # ========================================================================
    # 4.5 Ensemble 概率评估 (GenCast 风格)
    # ========================================================================
    ensemble_predictions = None
    probabilistic_metrics = {}
    
    if args.enable_ensemble_eval:
        print("\n" + "=" * 80)
        print(f"Ensemble Probabilistic Evaluation (GenCast Style)")
        print("=" * 80)
        print(f"\nGenerating {args.num_ensemble_members} ensemble members...")
        
        # 生成 ensemble 预测
        ensemble_predictions = trainer.predict_ensemble(
            X_test,
            num_members=args.num_ensemble_members,
            num_inference_steps=args.num_inference_steps
        )
        
        print(f"Ensemble shape: {ensemble_predictions.shape}")
        print(f"  - num_members: {ensemble_predictions.shape[0]}")
        print(f"  - batch_size: {ensemble_predictions.shape[1]}")
        print(f"  - time_steps: {ensemble_predictions.shape[2]}")
        
        # 计算概率评估指标
        probabilistic_stats = continuous_ranked_probability_score_summary(
            ensemble_predictions,
            y_test,
            variables=variables
        )
        
        # 打印报告
        report = probabilistic_evaluation_report(
            ensemble_predictions,
            y_test,
            variables=variables
        )
        print(report)
        
        # 保存概率指标
        probabilistic_metrics = {
            "crps_mean": probabilistic_stats["crps_mean"],
            "crps_std": probabilistic_stats["crps_std"],
            "ensemble_mean_rmse": probabilistic_stats["ensemble_mean_rmse"],
            "spread_skill_ratio": probabilistic_stats["spread_skill_ratio"],
            "crps_by_leadtime": probabilistic_stats.get("crps_by_leadtime", []),
        }
        
        if "crps_by_variable_names" in probabilistic_stats:
            probabilistic_metrics["crps_by_variable"] = probabilistic_stats["crps_by_variable_names"]
        
        # 保存 ensemble 预测
        np.save(output_dir / "ensemble_predictions.npy", ensemble_predictions)
        print(f"\n✓ Ensemble predictions saved to {output_dir}/ensemble_predictions.npy")
    
    # 保存所有结果
    metrics = {
        "test_deterministic": test_metrics,
        "test_probabilistic": probabilistic_metrics,
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    np.save(output_dir / "y_test.npy", y_test)
    np.save(output_dir / "y_test_pred.npy", y_test_pred)
    
    # ========================================================================
    # 5. 可视化
    # ========================================================================
    print("\n" + "=" * 80)
    print("Visualization")
    print("=" * 80)
    
    import matplotlib.pyplot as plt
    
    # 1. 训练历史
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history['train_loss'], label='Train Loss', alpha=0.7)
    ax.plot(history['val_loss'], label='Val Loss', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Diffusion Model Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(output_dir / "training_history.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 2. RMSE vs Lead Time
    lead_times = list(range(1, args.output_length + 1))
    rmses = [rmse_per_step[f"rmse_step_{i}"] for i in lead_times]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(lead_times, rmses, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Lead Time Step')
    ax.set_ylabel('RMSE')
    ax.set_title('Diffusion Model - RMSE vs Lead Time')
    ax.grid(True, alpha=0.3)
    plt.savefig(output_dir / "rmse_vs_leadtime.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 3. 空间预测示例（第一个变量，第一个样本）
    sample_idx = 0
    var_idx = 0
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"Diffusion Model - Variable: {variables[var_idx]}", fontsize=16, fontweight='bold')
    
    for t in range(min(4, args.output_length)):
        # 真实值
        ax_true = axes[0, t]
        im = ax_true.imshow(y_test[sample_idx, t, var_idx], cmap='RdBu_r')
        ax_true.set_title(f"True - Lead Time {t+1}")
        plt.colorbar(im, ax=ax_true)
        
        # 预测值
        ax_pred = axes[1, t]
        im = ax_pred.imshow(y_test_pred[sample_idx, t, var_idx], cmap='RdBu_r')
        ax_pred.set_title(f"Pred - Lead Time {t+1}")
        plt.colorbar(im, ax=ax_pred)
    
    plt.tight_layout()
    safe_var_name = variables[var_idx].replace("/", "_").replace(" ", "_")
    plt.savefig(output_dir / f"spatial_predictions_{safe_var_name}.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 4. 绘制时间序列（选择一个空间点）
    H, W = y_test.shape[3], y_test.shape[4]
    point_h, point_w = H // 2, W // 2  # 中心点
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for i, ax in enumerate(axes.flat):
        if i >= min(4, args.output_length):
            break
        
        # 提取该时间步的空间点
        sample_indices = np.linspace(0, len(y_test) - 1, 50, dtype=int)
        y_true_point = y_test[sample_indices, i, var_idx, point_h, point_w]
        y_pred_point = y_test_pred[sample_indices, i, var_idx, point_h, point_w]
        
        ax.plot(y_true_point, "b-o", label="True", alpha=0.7, markersize=3)
        ax.plot(y_pred_point, "r-s", label="Pred", alpha=0.7, markersize=3)
        ax.set_title(f"Lead Time {i+1} (Point [{point_h}, {point_w}])")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Normalized Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "point_predictions.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # ========================================================================
    # 6. Ensemble 可视化 (如果启用)
    # ========================================================================
    if args.enable_ensemble_eval and ensemble_predictions is not None:
        print("\nGenerating ensemble visualizations...")
        
        # 6.1 Ensemble 成员散布图
        sample_idx = 0
        var_idx = 0
        time_idx = 0
        point_h, point_w = H // 2, W // 2
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 左图：所有 ensemble 成员的空间分布
        ax = axes[0]
        for member_idx in range(min(5, args.num_ensemble_members)):
            member_pred = ensemble_predictions[member_idx, sample_idx, time_idx, var_idx]
            ax.plot(member_pred[point_h, :], alpha=0.5, label=f"Member {member_idx+1}")
        
        y_true_spatial = y_test[sample_idx, time_idx, var_idx]
        ax.plot(y_true_spatial[point_h, :], 'k-', linewidth=2, label="Observation", alpha=0.8)
        ax.set_title(f"Ensemble Spread (Spatial Profile at lat={point_h})")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Normalized Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 右图：Ensemble mean ± spread
        ax = axes[1]
        ensemble_mean = np.mean(ensemble_predictions[:, sample_idx, time_idx, var_idx], axis=0)
        ensemble_std = np.std(ensemble_predictions[:, sample_idx, time_idx, var_idx], axis=0)
        
        lon_range = np.arange(ensemble_mean.shape[1])
        ax.plot(lon_range, ensemble_mean[point_h, :], 'r-', linewidth=2, label="Ensemble Mean")
        ax.fill_between(
            lon_range,
            ensemble_mean[point_h, :] - ensemble_std[point_h, :],
            ensemble_mean[point_h, :] + ensemble_std[point_h, :],
            alpha=0.3,
            color='red',
            label="±1 std (spread)"
        )
        ax.plot(lon_range, y_true_spatial[point_h, :], 'k-', linewidth=2, label="Observation", alpha=0.8)
        ax.set_title("Ensemble Mean ± Spread")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Normalized Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "ensemble_spread.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # 6.2 CRPS vs Lead Time
        if 'crps_by_leadtime' in probabilistic_metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            lead_times = list(range(1, len(probabilistic_metrics['crps_by_leadtime']) + 1))
            ax.plot(lead_times, probabilistic_metrics['crps_by_leadtime'], 'o-', 
                    linewidth=2, markersize=8, label='CRPS', color='purple')
            
            # 同时绘制 RMSE 作为对比
            rmses_for_plot = [rmse_per_step[f"rmse_step_{i}"] for i in lead_times]
            ax.plot(lead_times, rmses_for_plot, 's--', 
                    linewidth=2, markersize=8, label='RMSE (Deterministic)', color='orange', alpha=0.7)
            
            ax.set_xlabel('Lead Time Step')
            ax.set_ylabel('Score')
            ax.set_title('CRPS vs RMSE by Lead Time (Lower is Better)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.savefig(output_dir / "crps_vs_leadtime.png", dpi=300, bbox_inches="tight")
            plt.close()
        
        # 6.3 不确定性地图（Ensemble Spread）
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f"Ensemble Uncertainty (Spread) - Variable: {variables[var_idx]}", 
                     fontsize=16, fontweight='bold')
        
        for t in range(min(4, args.output_length)):
            ax = axes[t]
            # 计算该时间步的 spread
            spread_map = np.std(ensemble_predictions[:, sample_idx, t, var_idx], axis=0)
            im = ax.imshow(spread_map, cmap='YlOrRd')
            ax.set_title(f"Lead Time {t+1}\nSpread (std)")
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig(output_dir / "ensemble_uncertainty_map.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        print("✓ Ensemble visualizations saved")
    
    print(f"\n✓ Results saved to {output_dir}")
    print(f"✓ Model saved to {output_dir}/best_model.pth")


if __name__ == "__main__":
    main()

