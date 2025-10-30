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


# 模型数据格式映射
SPATIAL_MODELS = ['cnn', 'convlstm', 'diffusion']
FLAT_MODELS = ['lr', 'lr_multi', 'lstm', 'transformer']


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

    return parser.parse_args()


def load_model_and_config(model_path, config_path=None):
    """加载模型和配置"""
    model_path = Path(model_path)

    # 加载配置
    if config_path is None:
        config_path = model_path.parent / "config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    model_name = config['model']
    print(f"Loading model: {model_name}")
    print(f"Config: {config_path}")

    # 确定数据格式
    if model_name in SPATIAL_MODELS:
        data_format = 'spatial'
    else:
        data_format = 'flat'

    # sklearn模型
    if model_name in ['lr', 'lr_multi']:
        import pickle
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        trainer = WeatherTrainer(model)
        return model, trainer, config, data_format

    # PyTorch模型 - 需要推断输入维度
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # 根据模型类型推断参数
    if model_name == 'lstm':
        first_weight = state_dict["lstm.weight_ih_l0"]
        input_size = first_weight.shape[1]
        
        # 从权重推断 hidden_size
        # weight_ih_l0 形状是 [4*hidden_size, input_size] (LSTM有4个门)
        hidden_size = first_weight.shape[0] // 4
        
        model = get_model(
            'lstm',
            input_size=input_size,
            hidden_size=hidden_size,  # 使用推断的 hidden_size
            num_layers=config.get("num_layers", 2),
            output_length=config.get("output_length", 4),
            dropout=config.get("dropout", 0.2),
        )
        
        print(f"  Inferred: input_size={input_size}, hidden_size={hidden_size}")

    elif model_name == 'transformer':
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
            'transformer',
            input_size=input_size,
            d_model=d_model,  # 使用推断的 d_model
            nhead=nhead,
            num_layers=config.get("num_layers", 4),
            output_length=config.get("output_length", 4),
            dropout=config.get("dropout", 0.1),
        )
        
        print(f"  Inferred: input_size={input_size}, d_model={d_model}, nhead={nhead}")

    elif model_name == 'cnn':
        # 从config读取
        model = get_model(
            'cnn',
            input_channels=config.get("input_channels", 1),
            input_length=config.get("input_length", 12),
            output_length=config.get("output_length", 4),
            hidden_channels=config.get("hidden_size", 64),
        )

    elif model_name == 'convlstm':
        model = get_model(
            'convlstm',
            input_channels=config.get("input_channels", 1),
            hidden_channels=config.get("hidden_size", 64),
            num_layers=config.get("num_layers", 2),
            output_length=config.get("output_length", 4),
        )

    elif model_name == 'diffusion':
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
):
    """生成预测"""
    print(f"\nGenerating predictions (format: {data_format})...")
    print(f"Variables: {variables}")

    data_loader = WeatherDataLoader(data_path=data_path, variables=variables)
    start, end = time_slice.split(":")
    ds = data_loader.load_data(time_slice=slice(start, end))

    # 准备特征
    features = data_loader.prepare_features(normalize=True)

    # 创建序列（根据数据格式）
    X, y_true = data_loader.create_sequences(
        features, input_length, output_length, format=data_format
    )
    feature_names = data_loader.variables

    print(f"Input shape: {X.shape}")

    # 预测（Diffusion模型需要num_inference_steps）
    from src.models.diffusion import DiffusionTrainer
    if isinstance(trainer, DiffusionTrainer) and num_inference_steps is not None:
        y_pred = trainer.predict(X, num_inference_steps=num_inference_steps)
    else:
        y_pred = trainer.predict(X)

    print(f"Prediction shape: {y_pred.shape}")

    return {
        "X": X,
        "y_true": y_true,
        "y_pred": y_pred,
        "features": feature_names,
        "data_format": data_format,
        "spatial_shape": data_loader.spatial_shape if data_format == 'spatial' else None,
    }


def save_predictions_netcdf(results, output_path, start_time=None):
    """保存为netCDF格式"""
    print(f"\nSaving predictions to {output_path}...")

    y_pred = results["y_pred"]
    y_true = results["y_true"]
    features = results["features"]
    data_format = results["data_format"]

    if data_format == 'spatial':
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
            data_vars[f"{var_name}_pred"] = (["time", "lead_time", "lat", "lon"], 
                                            y_pred[:, :, c, :, :])
            data_vars[f"{var_name}_true"] = (["time", "lead_time", "lat", "lon"], 
                                            y_true[:, :, c, :, :])
        
        # Lead times
        lead_times = np.arange(1, n_lead_times + 1) * 6  # hours
        
        if start_time is not None:
            times = pd.date_range(start_time, periods=n_samples, freq='6H')
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
            times = pd.date_range(start_time, periods=n_samples, freq='6H')
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
    
    results = generate_predictions(
        trainer,
        args.data_path,
        args.time_slice,
        data_format,
        variables,
        input_length=config.get("input_length", 12),
        output_length=config.get("output_length", 4),
        num_inference_steps=num_inference_steps,
    )

    # 3. 保存
    start_time = args.time_slice.split(":")[0] if ":" in args.time_slice else None
    
    if args.format == "netcdf":
        save_predictions_netcdf(results, args.output, start_time=start_time)
    elif args.format == "numpy":
        save_predictions_numpy(results, args.output)

    # 4. 快速评估
    print("\n" + "=" * 80)
    print("Quick Evaluation")
    print("=" * 80)
    
    y_true = results["y_true"]
    y_pred = results["y_pred"]
    
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    
    # 按lead time计算
    n_lead = y_true.shape[1]
    for t in range(n_lead):
        rmse_t = np.sqrt(np.mean((y_true[:, t] - y_pred[:, t]) ** 2))
        print(f"  Lead time {t+1}: RMSE = {rmse_t:.4f}")
    
    print(f"\nOverall:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
