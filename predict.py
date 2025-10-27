"""
预测脚本 - 用于生成WeatherBench2格式的预测结果
运行: python predict.py --model-path outputs/lstm_xxx/best_model.pth --output predictions.nc
"""
import argparse
import numpy as np
import xarray as xr
from pathlib import Path
import torch
from datetime import datetime, timedelta
import json

from src.data_loader import WeatherDataLoader
from src.models import LSTMModel, TransformerModel
from src.trainer import WeatherTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Generate predictions')
    
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config-path', type=str, default=None,
                       help='Path to model config.json (auto-detect if None)')
    
    parser.add_argument('--data-path', type=str,
                       default='gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr',
                       help='Path to ERA5 data')
    parser.add_argument('--time-slice', type=str, default='2021-01-01:2021-12-31',
                       help='Time slice for prediction')
    
    parser.add_argument('--output', type=str, default='predictions.nc',
                       help='Output netCDF file')
    parser.add_argument('--format', type=str, default='netcdf',
                       choices=['netcdf', 'zarr', 'numpy'],
                       help='Output format')
    
    return parser.parse_args()


def load_model_and_config(model_path, config_path=None):
    """加载模型和配置"""
    model_path = Path(model_path)
    
    # 加载配置
    if config_path is None:
        config_path = model_path.parent / 'config.json'
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Loading model: {config['model']}")
    print(f"Config: {config_path}")
    
    # 创建模型
    if config['model'] == 'lstm':
        # 从checkpoint中获取input_size
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict']
        
        # 从第一层权重推断input_size
        first_weight = state_dict['lstm.weight_ih_l0']
        input_size = first_weight.shape[1]
        
        model = LSTMModel(
            input_size=input_size,
            hidden_size=config.get('hidden_size', 128),
            num_layers=config.get('num_layers', 2),
            output_length=4,  # 默认预测4步
            dropout=config.get('dropout', 0.2),
        )
        
    elif config['model'] == 'transformer':
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict']
        
        # 推断input_size
        first_weight = state_dict['input_projection.weight']
        input_size = first_weight.shape[1]
        
        model = TransformerModel(
            input_size=input_size,
            d_model=config.get('hidden_size', 256),
            nhead=8,
            num_layers=config.get('num_layers', 4),
            output_length=4,
            dropout=config.get('dropout', 0.1),
        )
    else:
        raise ValueError(f"Unknown model: {config['model']}")
    
    # 加载权重
    trainer = WeatherTrainer(model)
    trainer.load_checkpoint(model_path)
    
    return model, trainer, config


def generate_predictions(
    trainer,
    data_loader,
    time_slice,
    input_length=12,
    output_length=4,
):
    """生成预测"""
    print("\nGenerating predictions...")
    
    # 加载数据
    start, end = time_slice.split(':')
    ds = data_loader.load_data(time_slice=slice(start, end))
    
    # 准备特征
    features = data_loader.prepare_features(normalize=True)
    
    # 创建序列
    X, y_true = data_loader.create_sequences(features, input_length, output_length)
    
    print(f"Input shape: {X.shape}")
    
    # 预测
    y_pred = trainer.predict(X)
    
    print(f"Prediction shape: {y_pred.shape}")
    
    return {
        'X': X,
        'y_true': y_true,
        'y_pred': y_pred,
        'times': ds['time'].values[input_length:],
        'features': data_loader.selected_vars,
    }


def save_predictions_netcdf(results, output_path, data_loader):
    """保存为WeatherBench2兼容的netCDF格式"""
    print(f"\nSaving predictions to {output_path}...")
    
    y_pred = results['y_pred']
    times = results['times']
    features = results['features']
    
    # 创建时间维度（每个预测的lead time）
    n_samples, n_lead_times, n_features = y_pred.shape
    
    # 创建xarray Dataset
    data_vars = {}
    
    for i, var_name in enumerate(features):
        # 对每个变量，创建(time, lead_time)数组
        data_vars[var_name] = (
            ['time', 'lead_time'],
            y_pred[:len(times), :, i]
        )
    
    # Lead times（以小时为单位，假设6小时间隔）
    lead_times = np.array([6, 12, 18, 24])[:n_lead_times]
    
    ds = xr.Dataset(
        data_vars,
        coords={
            'time': times[:len(y_pred)],
            'lead_time': lead_times,
        }
    )
    
    # 添加metadata
    ds.attrs['title'] = 'Weather Predictions'
    ds.attrs['source'] = 'LSTM Weather Model'
    ds.attrs['creation_date'] = datetime.now().isoformat()
    
    # 保存
    ds.to_netcdf(output_path)
    print(f"✓ Saved to {output_path}")
    
    return ds


def save_predictions_numpy(results, output_path):
    """保存为numpy格式"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        output_path,
        y_pred=results['y_pred'],
        y_true=results['y_true'],
        times=results['times'],
        features=results['features'],
    )
    print(f"✓ Saved to {output_path}")


def main():
    args = parse_args()
    
    print("="*80)
    print("Weather Prediction")
    print("="*80)
    
    # 1. 加载模型
    model, trainer, config = load_model_and_config(args.model_path, args.config_path)
    
    # 2. 加载数据
    data_loader = WeatherDataLoader(data_path=args.data_path)
    
    # 3. 生成预测
    results = generate_predictions(
        trainer,
        data_loader,
        args.time_slice,
        input_length=12,
        output_length=4,
    )
    
    # 4. 保存
    if args.format == 'netcdf':
        save_predictions_netcdf(results, args.output, data_loader)
    elif args.format == 'numpy':
        save_predictions_numpy(results, args.output)
    elif args.format == 'zarr':
        # TODO: Implement zarr saving
        print("Zarr format not yet implemented, saving as netCDF instead")
        save_predictions_netcdf(results, args.output.replace('.zarr', '.nc'), data_loader)
    
    # 5. 快速评估
    from src.trainer import calculate_weatherbench_metrics
    
    metrics = calculate_weatherbench_metrics(results['y_true'], results['y_pred'])
    
    print("\nQuick Evaluation:")
    for key, val in metrics.items():
        print(f"  {key}: {val:.4f}")
    
    print("\n✓ Done!")


if __name__ == '__main__':
    main()

