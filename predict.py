"""
预测脚本 - 支持所有模型类型
运行: python predict.py --model-path outputs/convlstm_xxx/best_model.pth --output predictions.nc
"""

import argparse
import numpy as np
import xarray as xr
from pathlib import Path
import torch
from datetime import datetime
import json
import pandas as pd

from src.data_loader import WeatherDataLoader
from src.models import get_model
from src.trainer import WeatherTrainer
from src.visualization import (
    visualize_predictions_improved,
    compute_metrics,
    compute_variable_wise_metrics,
)


# 模型数据格式映射
SPATIAL_MODELS = ["cnn", "convlstm", "diffusion", "weather_transformer"]
FLAT_MODELS = ["lr", "lr_multi", "lstm", "transformer"]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate predictions")

    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to model config.json (auto-detect if None)",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr",
        help="Path to ERA5 data",
    )
    parser.add_argument(
        "--time-slice",
        type=str,
        default="2021-01-01:2021-12-31",
        help="Time slice for prediction",
    )
    parser.add_argument(
        "--output", type=str, default="predictions.nc", help="Output file"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="netcdf",
        choices=["netcdf", "numpy"],
        help="Output format",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Number of inference steps for diffusion models (default: use model's default)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization plots",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save y_pred and y_true as numpy files for later analysis",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for prediction (default: auto, 32 for diffusion, 256 for others)",
    )

    return parser.parse_args()


def load_model_and_config(model_path, config_path=None):
    """加载模型和配置"""
    model_path = Path(model_path)

    # 加载配置
    if config_path is None:
        config_path = model_path.parent / "config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    model_name = config["model"]
    print(f"Loading model: {model_name}")
    print(f"Config: {config_path}")

    # 确定数据格式
    if model_name in SPATIAL_MODELS:
        data_format = "spatial"
    else:
        data_format = "flat"

    # sklearn模型
    if model_name in ["lr", "lr_multi"]:
        import pickle

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        trainer = WeatherTrainer(model)
        return model, trainer, config, data_format

    # PyTorch模型 - 需要推断输入维度
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # 根据模型类型推断参数
    if model_name == "lstm":
        first_weight = state_dict["lstm.weight_ih_l0"]
        input_size = first_weight.shape[1]
        # 从权重推断 hidden_size
        # weight_ih_l0 形状是 [4*hidden_size, input_size] (LSTM有4个门)
        hidden_size = first_weight.shape[0] // 4

        model = get_model(
            "lstm",
            input_size=input_size,
            hidden_size=hidden_size,  # 使用推断的 hidden_size
            num_layers=config.get("num_layers"),
            output_length=config.get("output_length"),
            dropout=config.get("dropout"),
        )

        print(f"  Inferred: input_size={input_size}, hidden_size={hidden_size}")

    elif model_name == "transformer":
        first_weight = state_dict["input_projection.weight"]
        input_size = first_weight.shape[1]
        d_model = first_weight.shape[0]  # 从权重推断 d_model

        # nhead 必须能整除 d_model
        if d_model == 64:
            nhead = 4  # 单变量优化版本
        elif d_model == 128:
            nhead = 4
        else:
            nhead = 8

        model = get_model(
            "transformer",
            input_size=input_size,
            d_model=d_model,  # 使用推断的 d_model
            nhead=nhead,
            num_layers=config.get("num_layers", 4),
            output_length=config.get("output_length", 4),
            dropout=config.get("dropout", 0.1),
        )

        print(f"  Inferred: input_size={input_size}, d_model={d_model}, nhead={nhead}")

    elif model_name == "cnn":
        # 从config读取
        model = get_model(
            "cnn",
            input_channels=config.get("input_channels", 1),
            input_length=config.get("input_length", 12),
            output_length=config.get("output_length", 4),
            hidden_channels=config.get("hidden_size", 64),
        )

    elif model_name == "convlstm":
        model = get_model(
            "convlstm",
            input_channels=config.get("input_channels", 1),
            hidden_channels=config.get("hidden_size", 64),
            num_layers=config.get("num_layers", 2),
            output_length=config.get("output_length", 4),
        )

    elif model_name == "weather_transformer":
        # 加载weather_transformer模型
        from src.models.weather_transformer import WeatherTransformer

        model = WeatherTransformer(
            img_size=tuple(config.get("img_size", [32, 64])),
            patch_size=tuple(config.get("patch_size", [4, 8])),
            input_channels=config.get("input_channels", 1),
            output_channels=config.get("output_channels", 1),
            input_length=config.get("input_length", 12),
            output_length=config.get("output_length", 4),
            d_model=config.get("d_model", 128),
            n_heads=config.get("n_heads", 4),
            n_layers=config.get("n_layers", 4),
            dropout=config.get("dropout", 0.1),
        )

    elif model_name == "diffusion":
        # 加载diffusion模型
        from src.models.diffusion import DiffusionWeatherModel, DiffusionTrainer

        model = DiffusionWeatherModel(
            input_channels=config.get("input_channels", 1),
            output_channels=config.get("output_channels", 1),
            input_length=config.get("input_length", 12),
            output_length=config.get("output_length", 4),
            base_channels=config.get("base_channels", 64),
            beta_schedule=config.get("beta_schedule", "cosine"),
            num_timesteps=config.get("num_diffusion_steps", 1000),
        )

        # Diffusion使用专门的trainer
        trainer = DiffusionTrainer(model, use_ema=config.get("use_ema", True))
        trainer.load_checkpoint(model_path)

        return model, trainer, config, data_format

    else:
        raise ValueError(f"Unknown model: {model_name}")

    # 加载权重
    trainer = WeatherTrainer(model)
    trainer.load_checkpoint(model_path)

    return model, trainer, config, data_format


def generate_predictions(
    trainer,
    data_path,
    time_slice,
    data_format,
    variables,
    input_length=12,
    output_length=4,
    num_inference_steps=None,
    norm_params=None,
    batch_size=None,
):
    """生成预测"""
    print(f"\nGenerating predictions (format: {data_format})...")
    print(f"Variables: {variables}")

    data_loader = WeatherDataLoader(data_path=data_path, variables=variables)
    start, end = time_slice.split(":")
    ds = data_loader.load_data(time_slice=slice(start, end))

    # 准备特征（使用训练时的归一化参数）
    features = data_loader.prepare_features(normalize=True, norm_params=norm_params)

    # 创建序列（根据数据格式）
    X, y_true = data_loader.create_sequences(
        features, input_length, output_length, format=data_format
    )
    feature_names = data_loader.variables

    print(f"Input shape: {X.shape}")

    # 预测（支持batch预测以节省内存）
    from src.models.diffusion import DiffusionTrainer
    
    # 判断是否需要batch预测
    n_samples = X.shape[0]
    
    # 确定batch size
    if batch_size is None:
        # 自动选择batch size
        if isinstance(trainer, DiffusionTrainer):
            batch_size = 32  # Diffusion模型使用小batch
            print(f"Using batch prediction for Diffusion model (batch_size={batch_size})")
        else:
            batch_size = 256  # 其他模型可以用大batch
    else:
        print(f"Using custom batch_size={batch_size}")
    
    # 如果样本数较少，直接一次预测
    if n_samples <= batch_size:
        if isinstance(trainer, DiffusionTrainer) and num_inference_steps is not None:
            y_pred = trainer.predict(X, num_inference_steps=num_inference_steps)
        else:
            y_pred = trainer.predict(X)
    else:
        # 分batch预测
        print(f"Predicting in batches ({n_samples} samples, batch_size={batch_size})...")
        y_pred_list = []
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_batch = X[i:end_idx]
            
            if isinstance(trainer, DiffusionTrainer) and num_inference_steps is not None:
                y_pred_batch = trainer.predict(X_batch, num_inference_steps=num_inference_steps)
            else:
                y_pred_batch = trainer.predict(X_batch)
            
            y_pred_list.append(y_pred_batch)
            
            # 打印进度
            print(f"  Batch {i//batch_size + 1}/{(n_samples + batch_size - 1)//batch_size}: "
                  f"samples {i} to {end_idx-1}")
        
        # 合并所有batch的结果
        import numpy as np
        y_pred = np.concatenate(y_pred_list, axis=0)
        print(f"✓ Batch prediction completed")

    print(f"Prediction shape: {y_pred.shape}")
    
    # 获取空间坐标（如果是空间数据）
    spatial_coords = None
    if data_format == "spatial":
        H, W = data_loader.spatial_shape
        print(f"Spatial shape: H={H}, W={W}")
        
        if hasattr(ds, 'latitude') and hasattr(ds, 'longitude'):
            lat_values = ds.latitude.values
            lon_values = ds.longitude.values
            print(f"Dataset coords: lat={len(lat_values)}, lon={len(lon_values)}")
            
            # 验证坐标长度是否匹配
            if len(lat_values) == H and len(lon_values) == W:
                spatial_coords = {
                    'lat': lat_values,
                    'lon': lon_values,
                }
                print("Using dataset coordinates")
            else:
                print(f"Warning: Coordinate mismatch. Using default coordinates.")
                spatial_coords = {
                    'lat': np.linspace(-90, 90, H),
                    'lon': np.linspace(0, 360, W),
                }
        else:
            # 默认坐标
            print("Dataset has no coordinates. Using default coordinates.")
            spatial_coords = {
                'lat': np.linspace(-90, 90, H),
                'lon': np.linspace(0, 360, W),
            }

    return {
        "X": X,
        "y_true": y_true,
        "y_pred": y_pred,
        "features": feature_names,
        "data_format": data_format,
        "spatial_shape": (
            data_loader.spatial_shape if data_format == "spatial" else None
        ),
        "spatial_coords": spatial_coords,
    }


def save_predictions_netcdf(results, output_path, start_time=None):
    """保存为netCDF格式"""
    print(f"\nSaving predictions to {output_path}...")

    y_pred = results["y_pred"]
    y_true = results["y_true"]
    features = results["features"]
    data_format = results["data_format"]

    if data_format == "spatial":
        # Spatial格式: (samples, lead_time, channels, H, W)
        n_samples, n_lead_times, n_channels, H, W = y_pred.shape

        # 创建空间坐标
        lat = np.linspace(-90, 90, H)
        lon = np.linspace(0, 360, W)

        # 为每个通道创建变量
        data_vars = {}
        for c in range(n_channels):
            # 假设channel对应features
            var_name = features[c] if c < len(features) else f"var_{c}"
            data_vars[f"{var_name}_pred"] = (
                ["time", "lead_time", "lat", "lon"],
                y_pred[:, :, c, :, :],
            )
            data_vars[f"{var_name}_true"] = (
                ["time", "lead_time", "lat", "lon"],
                y_true[:, :, c, :, :],
            )

        # Lead times
        lead_times = np.arange(1, n_lead_times + 1) * 6  # hours

        if start_time is not None:
            times = pd.date_range(start_time, periods=n_samples, freq="6h")
        else:
            times = np.arange(n_samples)

        ds = xr.Dataset(
            data_vars,
            coords={
                "time": times,
                "lead_time": lead_times,
                "lat": lat,
                "lon": lon,
            },
        )

    else:
        # Flat格式: (samples, lead_time, features)
        n_samples, n_lead_times, n_features = y_pred.shape

        data_vars = {}
        for i, var_name in enumerate(features):
            data_vars[f"{var_name}_pred"] = (["time", "lead_time"], y_pred[:, :, i])
            data_vars[f"{var_name}_true"] = (["time", "lead_time"], y_true[:, :, i])

        lead_times = np.arange(1, n_lead_times + 1) * 6

        if start_time is not None:
            times = pd.date_range(start_time, periods=n_samples, freq="6h")
        else:
            times = np.arange(n_samples)

        ds = xr.Dataset(
            data_vars,
            coords={
                "time": times,
                "lead_time": lead_times,
            },
        )

    # 添加metadata
    ds.attrs["title"] = "Weather Predictions"
    ds.attrs["source"] = "Deep Learning Weather Model"
    ds.attrs["creation_date"] = datetime.now().isoformat()
    ds.attrs["data_format"] = data_format

    # 保存
    ds.to_netcdf(output_path)
    print(f"✓ Saved to {output_path}")
    print(f"  Format: {data_format}")
    print(f"  Variables: {list(features)}")
    print(f"  Time steps: {n_samples}")

    return ds


def save_predictions_numpy(results, output_path):
    """保存为numpy格式"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        y_pred=results["y_pred"],
        y_true=results["y_true"],
        features=results["features"],
        data_format=results["data_format"],
    )
    print(f"✓ Saved to {output_path}")


def main():
    args = parse_args()

    print("=" * 80)
    print("Weather Prediction")
    print("=" * 80)

    # 1. 加载模型
    model, trainer, config, data_format = load_model_and_config(
        args.model_path, args.config_path
    )

    # 2. 生成预测
    # 从config读取变量列表（如果有的话）
    variables = config.get("variables", ["2m_temperature"])

    # 设置推理步数（仅对diffusion模型有效）
    num_inference_steps = args.num_inference_steps
    if num_inference_steps is None and config.get("model") == "diffusion":
        num_inference_steps = config.get("num_inference_steps", 50)

    # 获取归一化参数（关键！）
    norm_params = config.get("normalization", None)
    if norm_params is None:
        print("\n⚠️  WARNING: No normalization parameters found in config!")
        print("   Predictions may be inaccurate. Please retrain the model.")
    
    results = generate_predictions(
        trainer,
        args.data_path,
        args.time_slice,
        data_format,
        variables,
        input_length=config.get("input_length", 12),
        output_length=config.get("output_length", 4),
        num_inference_steps=num_inference_steps,
        norm_params=norm_params,
        batch_size=args.batch_size,
    )

    # 3. 保存
    start_time = args.time_slice.split(":")[0] if ":" in args.time_slice else None

    if args.format == "netcdf":
        save_predictions_netcdf(results, args.output, start_time=start_time)
    elif args.format == "numpy":
        save_predictions_numpy(results, args.output)

    # 4. 评估和可视化
    print("\n" + "=" * 80)
    print("Evaluation")
    print("=" * 80)

    y_true = results["y_true"]
    y_pred = results["y_pred"]
    
    # 计算指标
    metrics = compute_metrics(y_pred, y_true)
    
    print("\nOverall Metrics:")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    
    print("\nPer Lead Time:")
    for t in range(y_true.shape[1]):
        rmse_t = metrics[f"rmse_step_{t+1}"]
        print(f"  Lead time {t+1}: RMSE = {rmse_t:.4f}")
    
    # 多变量独立评估
    if len(variables) > 1:
        var_metrics = compute_variable_wise_metrics(
            y_pred, y_true, len(variables), data_format
        )
        
        print("\n" + "-" * 80)
        print("Per-Variable Metrics")
        print("-" * 80)
        
        for var_idx, var_name in enumerate(variables):
            rmse = var_metrics.get(f"var_{var_idx}_rmse", 0)
            mae = var_metrics.get(f"var_{var_idx}_mae", 0)
            print(f"  {var_name}:")
            print(f"    RMSE: {rmse:.4f}")
            print(f"    MAE:  {mae:.4f}")
        
        # 合并到metrics字典
        metrics.update(var_metrics)
    
    # 保存指标到文件
    output_dir = Path(args.output).parent
    metrics_path = output_dir / "prediction_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Metrics saved to {metrics_path}")
    
    # 5. 可视化
    if args.visualize:
        print("\n" + "=" * 80)
        print("Generating Visualizations")
        print("=" * 80)
        
        model_name = config.get("model", "unknown")
        
        # 生成可视化（非重叠样本）
        visualize_predictions_improved(
            y_true, y_pred, metrics, variables, model_name, output_dir, data_format,
            norm_params=norm_params,
            spatial_coords=results.get("spatial_coords", None)
        )
        
        print(f"\n✓ Visualizations saved to {output_dir}/")
    
    # 6. 保存预测结果（可选）
    if args.save_predictions:
        pred_dir = output_dir / "predictions_data"
        pred_dir.mkdir(exist_ok=True)
        np.save(pred_dir / "y_test.npy", y_true)
        np.save(pred_dir / "y_test_pred.npy", y_pred)
        print(f"✓ Predictions saved to {pred_dir}/")

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
