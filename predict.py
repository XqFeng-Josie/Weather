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
SPATIAL_MODELS = ["cnn", "convlstm", "weather_transformer"]
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
        help="Batch size for prediction (default: auto, 256)",
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
    # 判断是否需要batch预测
    n_samples = X.shape[0]
    
    # 确定batch size
    if batch_size is None:
        batch_size = 256  # 默认batch size
    else:
        print(f"Using custom batch_size={batch_size}")
    
    # 如果样本数较少，直接一次预测
    if n_samples <= batch_size:
        y_pred = trainer.predict(X)
    else:
        # 分batch预测
        print(f"Predicting in batches ({n_samples} samples, batch_size={batch_size})...")
        y_pred_list = []
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_batch = X[i:end_idx]
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


def denormalize_predictions(y_pred, y_true, variables, data_format, norm_params):
    """
    反归一化预测结果到物理单位
    
    Args:
        y_pred: 归一化后的预测值
        y_true: 归一化后的真值
        variables: 变量列表
        data_format: 数据格式 ('flat' 或 'spatial')
        norm_params: 归一化参数 {'mean': {var: value}, 'std': {var: value}}
    
    Returns:
        y_pred_phys, y_true_phys: 物理单位的预测和真值
    """
    if norm_params is None:
        print("⚠️  警告: 没有归一化参数，无法反归一化")
        return y_pred, y_true
    
    mean_dict = norm_params.get("mean", {})
    std_dict = norm_params.get("std", {})
    
    if not mean_dict or not std_dict:
        print("⚠️  警告: 归一化参数格式不正确，无法反归一化")
        return y_pred, y_true
    
    # 复制数据以避免修改原始数组
    y_pred_phys = y_pred.copy()
    y_true_phys = y_true.copy()
    
    if data_format == "spatial":
        # Spatial格式: (samples, lead_time, channels, H, W)
        # 假设每个变量对应一个或多个通道
        # 对于单变量，所有通道使用该变量的归一化参数
        # 对于多变量，按顺序分配通道
        
        n_channels = y_pred.shape[2]
        n_variables = len(variables)
        
        if n_variables == 1:
            # 单变量：所有通道使用同一变量的归一化参数
            var_name = variables[0]
            if var_name in mean_dict and var_name in std_dict:
                mean_val = mean_dict[var_name]
                std_val = std_dict[var_name]
                # 反归一化: x * std + mean
                y_pred_phys = y_pred_phys * std_val + mean_val
                y_true_phys = y_true_phys * std_val + mean_val
                print(f"  反归一化: {var_name} (所有 {n_channels} 个通道)")
            else:
                print(f"⚠️  警告: 变量 {var_name} 的归一化参数不存在")
        else:
            # 多变量：假设每个变量对应一个通道（简化处理）
            # 如果通道数不等于变量数，使用第一个变量的参数
            if n_channels == n_variables:
                for ch_idx, var_name in enumerate(variables):
                    if var_name in mean_dict and var_name in std_dict:
                        mean_val = mean_dict[var_name]
                        std_val = std_dict[var_name]
                        y_pred_phys[:, :, ch_idx, :, :] = (
                            y_pred_phys[:, :, ch_idx, :, :] * std_val + mean_val
                        )
                        y_true_phys[:, :, ch_idx, :, :] = (
                            y_true_phys[:, :, ch_idx, :, :] * std_val + mean_val
                        )
                        print(f"  反归一化通道 {ch_idx}: {var_name}")
            else:
                # 通道数与变量数不匹配，使用第一个变量的参数
                var_name = variables[0]
                if var_name in mean_dict and var_name in std_dict:
                    mean_val = mean_dict[var_name]
                    std_val = std_dict[var_name]
                    y_pred_phys = y_pred_phys * std_val + mean_val
                    y_true_phys = y_true_phys * std_val + mean_val
                    print(f"⚠️  通道数({n_channels})与变量数({n_variables})不匹配，使用第一个变量 {var_name} 的参数")
    
    else:  # flat format
        # Flat格式: (samples, lead_time, features)
        # 特征被展平，需要知道每个特征对应的变量
        # 简化处理：假设每个变量贡献相同数量的特征，或使用第一个变量的参数
        
        n_features = y_pred.shape[2]
        n_variables = len(variables)
        
        if n_variables == 1:
            # 单变量：所有特征使用该变量的归一化参数
            var_name = variables[0]
            if var_name in mean_dict and var_name in std_dict:
                mean_val = mean_dict[var_name]
                std_val = std_dict[var_name]
                y_pred_phys = y_pred_phys * std_val + mean_val
                y_true_phys = y_true_phys * std_val + mean_val
                print(f"  反归一化: {var_name} (所有 {n_features} 个特征)")
            else:
                print(f"⚠️  警告: 变量 {var_name} 的归一化参数不存在")
        else:
            # 多变量：假设特征均匀分配给各变量
            # 简化处理：使用第一个变量的参数（因为无法准确知道每个特征对应的变量）
            var_name = variables[0]
            if var_name in mean_dict and var_name in std_dict:
                mean_val = mean_dict[var_name]
                std_val = std_dict[var_name]
                y_pred_phys = y_pred_phys * std_val + mean_val
                y_true_phys = y_true_phys * std_val + mean_val
                print(f"⚠️  多变量flat格式: 使用第一个变量 {var_name} 的参数进行反归一化")
                print(f"   (注意: 这可能导致多变量情况下某些变量的反归一化不准确)")
    
    return y_pred_phys, y_true_phys


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
    
    # 4.1 计算归一化空间的指标
    print("\n" + "-" * 80)
    print("归一化空间的指标")
    print("-" * 80)
    
    metrics_norm = compute_metrics(y_pred, y_true)
    
    print("\nOverall Metrics (Normalized Space):")
    print(f"  RMSE: {metrics_norm['rmse']:.4f}")
    print(f"  MAE:  {metrics_norm['mae']:.4f}")
    
    print("\nPer Lead Time (Normalized Space):")
    for t in range(y_true.shape[1]):
        rmse_t = metrics_norm[f"rmse_step_{t+1}"]
        print(f"  Lead time {t+1}: RMSE = {rmse_t:.4f}")
    
    # 多变量独立评估（归一化空间）
    if len(variables) > 1:
        var_metrics_norm = compute_variable_wise_metrics(
            y_pred, y_true, len(variables), data_format
        )
        
        print("\nPer-Variable Metrics (Normalized Space):")
        for var_idx, var_name in enumerate(variables):
            rmse = var_metrics_norm.get(f"var_{var_idx}_rmse", 0)
            mae = var_metrics_norm.get(f"var_{var_idx}_mae", 0)
            print(f"  {var_name}:")
            print(f"    RMSE: {rmse:.4f}")
            print(f"    MAE:  {mae:.4f}")
        
        # 合并到metrics字典
        metrics_norm.update(var_metrics_norm)
    
    # 4.2 反归一化到物理单位
    print("\n" + "-" * 80)
    print("反归一化到物理单位")
    print("-" * 80)
    
    y_pred_phys, y_true_phys = denormalize_predictions(
        y_pred, y_true, variables, data_format, norm_params
    )
    
    # 显示物理值的范围
    print(f"\n✓ 反归一化完成")
    print(f"  预测范围: [{y_pred_phys.min():.2f}, {y_pred_phys.max():.2f}]")
    print(f"  真值范围: [{y_true_phys.min():.2f}, {y_true_phys.max():.2f}]")
    
    # 4.3 计算物理空间的指标
    print("\n" + "-" * 80)
    print("物理空间的指标 (原始尺度)")
    print("-" * 80)
    
    metrics_phys = compute_metrics(y_pred_phys, y_true_phys)
    
    print("\nOverall Metrics (Physical Space):")
    print(f"  RMSE: {metrics_phys['rmse']:.4f}")
    print(f"  MAE:  {metrics_phys['mae']:.4f}")
    
    print("\nPer Lead Time (Physical Space):")
    T_out = y_true_phys.shape[1]
    rmse_per_leadtime = {}
    for t in range(T_out):
        rmse_t = metrics_phys[f"rmse_step_{t+1}"]
        rmse_per_leadtime[f"rmse_step_{t+1}"] = float(rmse_t)
        print(f"  Lead time {t+1} ({(t+1)*6}h): RMSE = {rmse_t:.4f}")
    
    # 多变量独立评估（物理空间）
    if len(variables) > 1:
        var_metrics_phys = compute_variable_wise_metrics(
            y_pred_phys, y_true_phys, len(variables), data_format
        )
        
        print("\nPer-Variable Metrics (Physical Space):")
        for var_idx, var_name in enumerate(variables):
            rmse = var_metrics_phys.get(f"var_{var_idx}_rmse", 0)
            mae = var_metrics_phys.get(f"var_{var_idx}_mae", 0)
            print(f"  {var_name}:")
            print(f"    RMSE: {rmse:.4f}")
            print(f"    MAE:  {mae:.4f}")
        
        # 合并到metrics字典
        metrics_phys.update(var_metrics_phys)
    
    # 4.4 合并所有指标
    metrics = {
        "mode": model_name,
        "normalized_space": {k: float(v) for k, v in metrics_norm.items()},
        "physical_space": {k: float(v) for k, v in metrics_phys.items()},
        "physical_space_rmse_per_leadtime": rmse_per_leadtime,
        "time_slice": args.time_slice,
        "n_samples": int(y_pred.shape[0]),
        "variables": variables,
        "data_format": data_format,
    }
    
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
        
        # 生成可视化（使用物理值，不传入norm_params表示已经是物理值）
        visualize_predictions_improved(
            y_true_phys, y_pred_phys, metrics_phys, variables, model_name, output_dir, data_format,
            norm_params=None,  # None表示数据已经是物理值
            spatial_coords=results.get("spatial_coords", None)
        )
        
        print(f"\n✓ Visualizations saved to {output_dir}/")
    
    # 6. 保存预测结果（可选）
    if args.save_predictions:
        pred_dir = output_dir / "predictions_data"
        pred_dir.mkdir(exist_ok=True)
        # 保存归一化值
        np.save(pred_dir / "y_test_norm.npy", y_true)
        np.save(pred_dir / "y_test_pred_norm.npy", y_pred)
        # 保存物理值
        np.save(pred_dir / "y_test.npy", y_true_phys)
        np.save(pred_dir / "y_test_pred.npy", y_pred_phys)
        print(f"✓ Predictions saved to {pred_dir}/")
        print(f"  - y_test_norm.npy / y_test_pred_norm.npy: 归一化值")
        print(f"  - y_test.npy / y_test_pred.npy: 物理值")

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
