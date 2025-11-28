"""
主训练脚本 - 支持所有模型类型
运行示例:
  python train.py --model lr --epochs 50
  python train.py --model lr_multi --epochs 50
  python train.py --model cnn --epochs 50
  python train.py --model convlstm --epochs 50
  python train.py --model lstm --epochs 50
  python train.py --model transformer --epochs 50
"""

import argparse
from tkinter import N
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from src.data_loader import WeatherDataLoader
from src.models import get_model, count_parameters
from src.models.linear_regression import (
    LinearRegressionModel,
    MultiOutputLinearRegression,
)
from src.trainer import WeatherTrainer


# 模型配置：哪些模型需要空间数据
SPATIAL_MODELS = ["cnn", "convlstm"]
FLAT_MODELS = ["lr", "lr_multi", "lstm", "transformer"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train weather prediction model")

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="lstm",
        choices=["lr", "lr_multi", "cnn", "convlstm", "lstm", "transformer"],
        help="Model type",
    )

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
        help="Variables to predict (comma-separated, e.g., 2m_temperature,geopotential)",
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Pressure levels for variables with a level dimension "
            "(e.g. 500 700 850). "
            "如果指定，则只使用这些层的数据进行训练（适用于 flat 和 spatial 模型）。"
            "如果不指定，默认使用所有可用的 levels。"
            "例如：`--levels 500` 只使用 500hPa，`--levels 500 700 850` 使用多个层。"
        ),
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

    # Training
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--early-stop", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (for reducing memory usage)",
    )

    # Model params
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden size")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")

    # Output
    parser.add_argument(
        "--output-dir", type=str, default="./outputs", help="Output directory"
    )

    return parser.parse_args()


def create_model(args, data_info):
    """
    根据参数创建模型

    Args:
        args: 命令行参数
        data_info: 数据信息字典，包含特征数、通道数、空间大小等
    """
    model_name = args.model
    variables = [v.strip() for v in args.variables.split(",")]
    n_variables = len(variables)  # 变量数量

    # Linear Regression models
    if model_name == "lr":
        model = LinearRegressionModel(alpha=1.0)
    elif model_name == "lr_multi":
        model = MultiOutputLinearRegression(alpha=10.0, n_variables=n_variables)

    # Spatial models (CNN, ConvLSTM)
    elif model_name == "cnn":
        model = get_model(
            "cnn",
            input_channels=data_info["n_channels"],
            input_length=args.input_length,
            output_length=args.output_length,
            hidden_channels=args.hidden_size,
        )
    elif model_name == "convlstm":
        # convlstm_hidden = min(args.hidden_size, 64)  # 限制最大为64
        convlstm_hidden = args.hidden_size
        model = get_model(
            "convlstm",
            input_channels=data_info["n_channels"],
            hidden_channels=convlstm_hidden,
            num_layers=args.num_layers,
            output_length=args.output_length,
        )

        # if args.hidden_size > 64:
        #     print(f"⚠️  ConvLSTM memory optimization: hidden_channels {args.hidden_size} -> {convlstm_hidden}")

    # Flat models (LSTM, Transformer)
    elif model_name == "lstm":
        model = get_model(
            "lstm",
            input_size=data_info["n_features"],
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            output_length=args.output_length,
            dropout=args.dropout,
        )
    elif model_name == "transformer":
        # # 单变量时使用更小的模型防止过拟合
        # if n_variables == 1:
        #     d_model = 64  # 更小的模型
        #     nhead = 4
        #     num_layers = 2  # 减少层数
        #     dropout = 0.3  # 更高dropout
        # else:
        #     d_model = args.hidden_size
        #     nhead = 8
        #     num_layers = args.num_layers
        #     dropout = args.dropout

        model = get_model(
            "transformer",
            input_size=data_info["n_features"],
            d_model=args.hidden_size,
            nhead=4 if n_variables == 1 else 8,
            num_layers=args.num_layers,
            output_length=args.output_length,
            dropout=args.dropout,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def main():
    args = parse_args()
    # levels=""
    if args.levels is None or args.levels == '""':
        args.levels = ""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Output dir: {output_dir}")
    print("=" * 80)

    # 保存配置（稍后会添加数据信息）
    config = vars(args).copy()

    # ========================================================================
    # 1. 加载数据
    # ========================================================================
    print("\n" + "=" * 80)
    print("Loading Data")
    print("=" * 80)

    # 解析变量列表（逗号分割）
    variables = [v.strip() for v in args.variables.split(",")]

    # 如果用户指定了 levels，则传给数据加载器
    # 适用于 flat 和 spatial 模型
    levels = args.levels
    if levels is not None:
        print(f"Using specified levels: {levels}")
    else:
        print("Using all available levels (default)")

    loader = WeatherDataLoader(
        data_path=args.data_path,
        variables=variables,
        levels=levels,
    )
    start, end = args.time_slice.split(":")
    ds = loader.load_data(time_slice=slice(start, end))
    features = loader.prepare_features(normalize=True)

    print(f"\nVariables: {variables}")

    # 根据模型类型选择数据格式
    if args.model in SPATIAL_MODELS:
        data_format = "spatial"
        print(f"\nUsing SPATIAL format for {args.model}")
    else:
        data_format = "flat"
        print(f"\nUsing FLAT format for {args.model}")

    X, y = loader.create_sequences(
        features,
        input_length=args.input_length,
        output_length=args.output_length,
        format=data_format,
    )

    print(f"\nData shape: X={X.shape}, y={y.shape}")

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
    print("\n" + "=" * 80)
    print("Creating Model")
    print("=" * 80)

    # 准备数据信息
    if data_format == "spatial":
        # X shape: (n_samples, input_length, n_channels, H, W)
        data_info = {
            "n_channels": X.shape[2],
            "spatial_size": (X.shape[3], X.shape[4]),
            "n_features": None,
        }
        print(
            f"Channels: {data_info['n_channels']}, Spatial size: {data_info['spatial_size']}"
        )

        # 添加到config
        config["input_channels"] = int(data_info["n_channels"])
        config["spatial_size"] = data_info["spatial_size"]
    else:
        # X shape: (n_samples, input_length, n_features)
        data_info = {
            "n_features": X.shape[2],
            "n_channels": None,
            "spatial_size": None,
        }
        print(f"Features: {data_info['n_features']}")

        # 添加到config
        config["input_size"] = int(data_info["n_features"])

    # 保存完整配置
    config["output_length"] = args.output_length
    config["variables"] = variables  # 保存变量列表，predict时使用
    # 记录实际使用的 levels（无 level 变量时此字段可选）
    # loader.levels 会是实际使用的 levels（可能是默认值 [500, 700, 850] 或用户指定的值）
    config["levels"] = loader.levels

    # 保存归一化参数（关键！）
    # 此处需要兼容：
    # - 无 level 变量：mean/std 为标量
    # - 有 level 变量：mean/std 为 shape=(n_level,) 的数组
    norm_mean = {}
    norm_std = {}
    for k, v in loader.mean.items():
        arr = np.array(v)
        if arr.ndim == 0:
            norm_mean[k] = float(arr)
        else:
            norm_mean[k] = arr.tolist()
    for k, v in loader.std.items():
        arr = np.array(v)
        if arr.ndim == 0:
            norm_std[k] = float(arr)
        else:
            norm_std[k] = arr.tolist()

    config["normalization"] = {
        "mean": norm_mean,
        "std": norm_std,
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    model = create_model(args, data_info)

    print(f"Model: {args.model}")
    n_params = count_parameters(model)
    if n_params > 0:
        print(f"Parameters: {n_params:,}")

    # ========================================================================
    # 3. 训练
    # ========================================================================
    print("\n" + "=" * 80)
    print("Training")
    print("=" * 80)

    # 创建多变量加权损失函数（如果是多变量）
    criterion = None
    if len(variables) > 1 and not isinstance(
        model, (LinearRegressionModel, MultiOutputLinearRegression)
    ):
        from src.losses import MultiVariableLoss

        print(f"\nUsing Multi-Variable Weighted Loss for {len(variables)} variables:")
        print(f"  Variables: {variables}")
        print(f"  Format: {data_format}")

        # 计算变量范围（flat格式）
        variable_ranges = None
        if data_format == "flat":
            n_features = X.shape[2]
            features_per_var = n_features // len(variables)
            variable_ranges = [
                (i * features_per_var, (i + 1) * features_per_var)
                for i in range(len(variables))
            ]
            print(f"  Feature ranges: {variable_ranges}")

        # 使用自动平衡的加权损失
        criterion = MultiVariableLoss(
            n_variables=len(variables),
            weights=None,  # 初始均等权重
            variable_ranges=variable_ranges,
            format=data_format,
            auto_balance=True,  # 启用自动权重调整
        )
        print(f"  Auto-balancing: Enabled")

    # Transformer单变量时使用更小的学习率
    lr = (
        args.lr * 0.1
        if (args.model == "transformer" and len(variables) == 1)
        else args.lr
    )
    trainer = WeatherTrainer(
        model,
        learning_rate=lr,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        criterion=criterion,
    )

    if isinstance(model, (LinearRegressionModel, MultiOutputLinearRegression)):
        # sklearn模型
        results = trainer.train_sklearn_model(X_train, y_train, X_val, y_val)
        trainer.save_checkpoint(str(output_dir / "best_model.pth"))
    else:
        # PyTorch模型
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

        # 绘制训练曲线
        trainer.plot_history(save_path=output_dir / "training_history.png")

    # ========================================================================
    # 4. 评估
    # ========================================================================
    print("\n" + "=" * 80)
    print("Evaluation")
    print("=" * 80)

    # 验证集
    val_metrics, y_val_pred = trainer.evaluate(X_val, y_val)
    print("\nValidation Metrics (Overall):")
    for key, val in val_metrics.items():
        print(f"  {key}: {val:.4f}")

    # 测试集
    test_metrics, y_test_pred = trainer.evaluate(X_test, y_test)
    print("\nTest Metrics (Overall):")
    for key, val in test_metrics.items():
        print(f"  {key}: {val:.4f}")

    # 多变量独立评估
    if len(variables) > 1:
        from src.losses import compute_variable_wise_metrics

        print("\n" + "-" * 80)
        print("Per-Variable Metrics")
        print("-" * 80)

        val_var_metrics = compute_variable_wise_metrics(
            y_val_pred, y_val, len(variables), data_format
        )
        test_var_metrics = compute_variable_wise_metrics(
            y_test_pred, y_test, len(variables), data_format
        )

        print("\nValidation (Per Variable):")
        for var_idx, var_name in enumerate(variables):
            rmse = val_var_metrics.get(f"var_{var_idx}_rmse", 0)
            mae = val_var_metrics.get(f"var_{var_idx}_mae", 0)
            print(f"  {var_name}:")
            print(f"    RMSE: {rmse:.4f}")
            print(f"    MAE:  {mae:.4f}")

        print("\nTest (Per Variable):")
        for var_idx, var_name in enumerate(variables):
            rmse = test_var_metrics.get(f"var_{var_idx}_rmse", 0)
            mae = test_var_metrics.get(f"var_{var_idx}_mae", 0)
            print(f"  {var_name}:")
            print(f"    RMSE: {rmse:.4f}")
            print(f"    MAE:  {mae:.4f}")

        # 合并到metrics字典
        val_metrics.update(val_var_metrics)
        test_metrics.update(test_var_metrics)

    # 保存指标
    def convert_to_python_types(obj):
        """递归转换numpy类型为Python原生类型"""
        if isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_python_types(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj

    metrics = {
        "val": val_metrics,
        "test": test_metrics,
    }
    metrics = convert_to_python_types(metrics)
    # Print metrics
    for key, val in metrics.items():
        print(f"{key}:")
        for k, v in val.items():
            print(f"  {k}: {v:.4f}")
        print()

    # with open(output_dir / "metrics.json", "w") as f:
    #     json.dump(metrics, f, indent=2)

    # # 保存预测结果
    # np.save(output_dir / "y_test.npy", y_test)
    # np.save(output_dir / "y_test_pred.npy", y_test_pred)

    print(f"\n✓ Results saved to {output_dir}")
    print(f"✓ Model saved to {output_dir}/best_model.pth")
    print(f"\nTo generate visualizations, run:")
    test_time = config.get("time_slice", "2017-01-01:2018-12-31").split(":")[-1]
    next_year = str(int(test_time[:4]) + 1)
    print(
        f"  python predict.py --model-path {output_dir}/best_model.pth --time-slice {next_year}-01-01:{next_year}-12-31 --visualize --save-predictions"
    )


if __name__ == "__main__":
    main()
