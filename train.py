"""
主训练脚本
运行: python train.py --model lstm --epochs 50
"""
import argparse
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from src.data_loader import prepare_single_point_data, WeatherDataLoader
from src.models import get_model, LinearRegressionModel, LSTMModel, TransformerModel
from src.trainer import WeatherTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train weather prediction model')
    
    # Model
    parser.add_argument('--model', type=str, default='lstm',
                       choices=['lr', 'lstm', 'transformer'],
                       help='Model type')
    
    # Data
    parser.add_argument('--data-path', type=str,
                       default='gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr',
                       help='Path to ERA5 data')
    parser.add_argument('--time-slice', type=str, default='2020-01-01:2020-12-31',
                       help='Time slice for training (start:end)')
    parser.add_argument('--single-point', action='store_true',
                       help='Use single point data for quick testing')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--early-stop', type=int, default=10, help='Early stopping patience')
    
    # Model params
    parser.add_argument('--hidden-size', type=int, default=128, help='Hidden size')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='Output directory')
    parser.add_argument('--exp-name', type=str, default=None,
                       help='Experiment name')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 创建输出目录
    if args.exp_name is None:
        args.exp_name = f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Experiment: {args.exp_name}")
    print(f"Output dir: {output_dir}")
    
    # 保存配置
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # ========================================================================
    # 1. 加载数据
    # ========================================================================
    print("\n" + "="*80)
    print("Loading Data")
    print("="*80)
    
    if args.single_point:
        # 快速测试：单点数据
        print("Loading single-point data for quick testing...")
        start, end = args.time_slice.split(':')
        X, y = prepare_single_point_data(
            args.data_path,
            time_slice=slice(start, end)
        )
    else:
        # 完整空间数据
        print("Loading full spatial data...")
        loader = WeatherDataLoader(data_path=args.data_path)
        start, end = args.time_slice.split(':')
        ds = loader.load_data(time_slice=slice(start, end))
        features = loader.prepare_features(normalize=True)
        X, y = loader.create_sequences(features, input_length=12, output_length=4)
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # 划分数据集
    n_samples = len(X)
    train_end = int(n_samples * 0.7)
    val_end = int(n_samples * 0.85)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # ========================================================================
    # 2. 创建模型
    # ========================================================================
    print("\n" + "="*80)
    print("Creating Model")
    print("="*80)
    
    n_features = X.shape[2]
    output_length = y.shape[1]
    
    if args.model == 'lr':
        model = LinearRegressionModel(alpha=1.0)
    elif args.model == 'lstm':
        model = LSTMModel(
            input_size=n_features,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            output_length=output_length,
            dropout=args.dropout,
        )
    elif args.model == 'transformer':
        model = TransformerModel(
            input_size=n_features,
            d_model=args.hidden_size,
            nhead=8,
            num_layers=args.num_layers,
            output_length=output_length,
            dropout=args.dropout,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    print(f"Model: {args.model}")
    if hasattr(model, 'parameters'):
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params:,}")
    
    # ========================================================================
    # 3. 训练
    # ========================================================================
    print("\n" + "="*80)
    print("Training")
    print("="*80)
    
    trainer = WeatherTrainer(model, learning_rate=args.lr)
    
    if args.model == 'lr':
        # Linear Regression
        results = trainer.train_sklearn_model(X_train, y_train, X_val, y_val)
    else:
        # PyTorch models
        history = trainer.train_pytorch_model(
            X_train, y_train, X_val, y_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            early_stopping_patience=args.early_stop,
        )
        
        # 绘制训练曲线
        trainer.plot_history(save_path=output_dir / 'training_history.png')
    
    # ========================================================================
    # 4. 评估
    # ========================================================================
    print("\n" + "="*80)
    print("Evaluation")
    print("="*80)
    
    # 验证集
    val_metrics, y_val_pred = trainer.evaluate(X_val, y_val)
    print("\nValidation Metrics:")
    for key, val in val_metrics.items():
        print(f"  {key}: {val:.4f}")
    
    # 测试集
    test_metrics, y_test_pred = trainer.evaluate(X_test, y_test)
    print("\nTest Metrics:")
    for key, val in test_metrics.items():
        print(f"  {key}: {val:.4f}")
    
    # 保存指标
    metrics = {
        'val': val_metrics,
        'test': test_metrics,
    }
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # 保存预测结果
    np.save(output_dir / 'y_test.npy', y_test)
    np.save(output_dir / 'y_test_pred.npy', y_test_pred)
    
    # ========================================================================
    # 5. 可视化
    # ========================================================================
    print("\n" + "="*80)
    print("Visualization")
    print("="*80)
    
    import matplotlib.pyplot as plt
    
    # 绘制预测vs真实值（第一个特征）
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for i, ax in enumerate(axes.flat):
        if i >= min(4, y_test.shape[1]):
            break
        
        # 选择几个样本
        sample_idx = np.linspace(0, len(y_test)-1, 50, dtype=int)
        
        ax.plot(y_test[sample_idx, i, 0], 'b-o', label='True', alpha=0.7, markersize=3)
        ax.plot(y_test_pred[sample_idx, i, 0], 'r-s', label='Pred', alpha=0.7, markersize=3)
        ax.set_title(f'Lead Time {i+1} (Feature 0)')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Normalized Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制RMSE per lead time
    rmse_per_step = [test_metrics[f'rmse_step_{i+1}'] for i in range(y_test.shape[1])]
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(rmse_per_step)+1), rmse_per_step, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Lead Time Step')
    plt.ylabel('RMSE')
    plt.title(f'RMSE vs Lead Time ({args.model.upper()})')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'rmse_vs_leadtime.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Results saved to {output_dir}")
    print(f"✓ Model saved to {output_dir}/best_model.pth")


if __name__ == '__main__':
    main()

