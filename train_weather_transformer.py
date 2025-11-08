"""
训练时空Transformer模型

用法:
  python train_weather_transformer.py \
    --variables 2m_temperature \
    --input-length 12 \
    --output-length 4 \
    --d-model 128 \
    --n-heads 4 \
    --n-layers 4 \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-4
"""

import argparse
import json
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

from src.data_loader import WeatherDataLoader
from src.models.weather_transformer import WeatherTransformer
from src.trainer import WeatherTrainer
from src.losses import MultiVariableLoss, compute_variable_wise_metrics
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Train Weather Transformer")

    # 数据参数
    parser.add_argument(
        "--data-path",
        type=str,
        default="gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr",
        help="Path to ERA5 data",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=["2m_temperature"],
        help="Variables to predict",
    )
    parser.add_argument(
        "--time-slice",
        type=str,
        default="2000-01-01:2016-12-31",
        help="Time slice for training (format: start:end)",
    )

    # 序列参数
    parser.add_argument(
        "--input-length", type=int, default=12, help="Input sequence length"
    )
    parser.add_argument(
        "--output-length", type=int, default=4, help="Output sequence length"
    )

    # 模型参数
    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        default=[32, 64],
        help="Image size (H W)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=[4, 8],
        help="Patch size (pH pW)",
    )
    parser.add_argument(
        "--d-model", type=int, default=128, help="Model dimension"
    )
    parser.add_argument(
        "--n-heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument(
        "--n-layers", type=int, default=4, help="Number of encoder layers"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout rate"
    )

    # 训练参数
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-5, help="Weight decay"
    )
    parser.add_argument(
        "--early-stop", type=int, default=15, help="Early stopping patience"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.2, help="Validation split ratio"
    )

    # 多变量损失权重
    parser.add_argument(
        "--loss-weights",
        type=float,
        nargs="+",
        default=None,
        help="Loss weights for multi-variable (default: equal weights)",
    )

    # 输出参数
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (auto-generated if None)",
    )
    parser.add_argument(
        "--save-best-only", action="store_true", help="Only save best model"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 创建输出目录
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        vars_str = "_".join(args.variables)
        args.output_dir = f"outputs/weather_transformer_{vars_str}_{timestamp}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Weather Transformer Training")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Variables: {args.variables}")
    print(f"Time slice: {args.time_slice}")

    # 1. 加载数据
    print("\n" + "=" * 80)
    print("Loading Data")
    print("=" * 80)

    loader = WeatherDataLoader(
        data_path=args.data_path,
        variables=args.variables,
    )

    start, end = args.time_slice.split(":")
    ds = loader.load_data(time_slice=slice(start, end))

    # 准备特征（归一化）
    features = loader.prepare_features(normalize=True)

    # 创建序列（spatial格式）
    X, y = loader.create_sequences(
        features,
        input_length=args.input_length,
        output_length=args.output_length,
        format="spatial",  # Weather Transformer使用spatial格式
    )

    print(f"\nSequence shape: X={X.shape}, y={y.shape}")

    # 2. 创建模型
    print("\n" + "=" * 80)
    print("Creating Model")
    print("=" * 80)

    n_channels = len(args.variables)

    model = WeatherTransformer(
        img_size=tuple(args.img_size),
        patch_size=tuple(args.patch_size),
        input_channels=n_channels,
        output_channels=n_channels,
        input_length=args.input_length,
        output_length=args.output_length,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型: Weather Transformer")
    print(f"参数量: {n_params:,}")
    print(f"模型维度: d_model={args.d_model}, n_heads={args.n_heads}, n_layers={args.n_layers}")
    print(f"Patch设置: img_size={args.img_size}, patch_size={args.patch_size}")
    print(f"时间步: input={args.input_length}, output={args.output_length}")

    # 3. 设置损失函数
    if len(args.variables) > 1:
        # 多变量：使用加权损失
        if args.loss_weights is None:
            loss_weights = [1.0] * len(args.variables)
        else:
            loss_weights = args.loss_weights
            assert len(loss_weights) == len(args.variables), \
                f"Loss weights ({len(loss_weights)}) must match variables ({len(args.variables)})"

        criterion = MultiVariableLoss(
            variable_names=args.variables,
            weights=loss_weights,
        )
        print(f"\n使用多变量加权损失")
        print(f"权重: {dict(zip(args.variables, loss_weights))}")
    else:
        # 单变量：MSE
        criterion = nn.MSELoss()
        print(f"\n使用MSE损失")

    # 4. 创建训练器
    trainer = WeatherTrainer(
        model=model,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        criterion=criterion,
    )

    # 5. 训练模型
    print("\n" + "=" * 80)
    print("Training")
    print("=" * 80)

    # 划分训练集和验证集
    n_samples = X.shape[0]
    n_val = int(n_samples * args.val_split)
    n_train = n_samples - n_val
    
    # 随机划分
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    print(f"\n训练集: {X_train.shape[0]} samples")
    print(f"验证集: {X_val.shape[0]} samples")
    
    history = trainer.train_pytorch_model(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping_patience=args.early_stop,
        checkpoint_path=str(output_dir / "best_model.pth"),
    )

    # 6. 保存配置和历史
    config = {
        "model": "weather_transformer",
        "data_path": args.data_path,
        "variables": args.variables,
        "time_slice": args.time_slice,
        "input_length": args.input_length,
        "output_length": args.output_length,
        "img_size": args.img_size,
        "patch_size": args.patch_size,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "early_stop": args.early_stop,
        "val_split": args.val_split,
        "n_channels": n_channels,
        "n_params": n_params,
        "input_channels": n_channels,
        "output_channels": n_channels,
        # 保存归一化参数（重要！）
        "normalization": {
            "mean": {var: float(loader.mean[var]) for var in args.variables},
            "std": {var: float(loader.std[var]) for var in args.variables},
        },
    }

    if len(args.variables) > 1 and args.loss_weights is not None:
        config["loss_weights"] = args.loss_weights

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # with open(output_dir / "history.json", "w") as f:
    #     json.dump(history, f, indent=2)

    print(f"\n✓ 配置已保存: {output_dir / 'config.json'}")
    print(f"✓ 历史已保存: {output_dir / 'history.json'}")

    # 7. 评估
    print("\n" + "=" * 80)
    print("Evaluation")
    print("=" * 80)

    # 加载最佳模型
    trainer.load_checkpoint(str(output_dir / "best_model.pth"))

    # 预测
    y_pred = trainer.predict(X)

    # 计算总体指标
    from src.visualization import compute_metrics
    metrics = compute_metrics(y_pred, y)

    print("\n总体指标:")
    print(f"  MSE:  {metrics['mse']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE:  {metrics['mae']:.6f}")

    print("\n各lead time RMSE:")
    for t in range(args.output_length):
        rmse_t = metrics[f"rmse_step_{t+1}"]
        print(f"  Step {t+1}: {rmse_t:.6f}")

    # 计算每个变量的指标
    if len(args.variables) > 1:
        var_metrics = compute_variable_wise_metrics(
            y_pred, y, args.variables
        )
        print("\n各变量指标:")
        for var, m in var_metrics.items():
            print(f"\n  {var}:")
            print(f"    RMSE: {m['rmse']:.6f}")
            print(f"    MAE:  {m['mae']:.6f}")

        metrics["variables"] = var_metrics

    # 保存指标
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✓ 指标已保存: {output_dir / 'metrics.json'}")

    # 8. 提示如何进行预测
    print("\n" + "=" * 80)
    print("Next Steps")
    print("=" * 80)
    print("\n运行预测和可视化:")
    print(f"\npython predict.py \\")
    print(f"  --model-path {output_dir / 'best_model.pth'} \\")
    print(f"  --time-slice 2017-01-01:2018-12-31 \\")
    print(f"  --visualize \\")
    print(f"  --save-predictions")

    print("\n" + "=" * 80)
    print("训练完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()

