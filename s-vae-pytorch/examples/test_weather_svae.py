#!/usr/bin/env python3
"""测试 S-VAE 模型对天气数据的重建能力"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import xarray as xr

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_weather_svae import WeatherGridDataset, WeatherVAE


def convert_to_python_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_python_types(item) for item in obj)
    else:
        return obj


def calculate_mse(pred: np.ndarray, target: np.ndarray) -> float:
    """计算均方误差 (MSE)"""
    return np.mean((pred - target) ** 2)


def calculate_mae(pred: np.ndarray, target: np.ndarray) -> float:
    """计算平均绝对误差 (MAE)"""
    return np.mean(np.abs(pred - target))


def calculate_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """计算均方根误差 (RMSE)"""
    return np.sqrt(np.mean((pred - target) ** 2))


def calculate_ssim(pred: np.ndarray, target: np.ndarray, 
                   k1: float = 0.01, k2: float = 0.03) -> float:
    C1 = (k1 * 2) ** 2
    C2 = (k2 * 2) ** 2
    
    mu1 = pred.mean()
    mu2 = target.mean()
    
    sigma1_sq = np.var(pred)
    sigma2_sq = np.var(target)
    sigma12 = np.cov(pred.flatten(), target.flatten())[0, 1]
    
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim


def denormalize_data(data: np.ndarray, norm_stats: dict) -> np.ndarray:
    z_score = data * 3.0
    mean = norm_stats['mean']
    std = norm_stats['std']
    physical = z_score * std + mean
    return physical


def calculate_metrics_norm_space(pred: np.ndarray, target: np.ndarray) -> dict:
    return {
        'mae': calculate_mae(pred, target),
        'rmse': calculate_rmse(pred, target),
        'mse': calculate_mse(pred, target),
        'ssim': calculate_ssim(pred, target)
    }


def calculate_metrics_physical_space(pred: np.ndarray, target: np.ndarray, 
                                     norm_stats: dict) -> dict:
    pred_phys = denormalize_data(pred, norm_stats)
    target_phys = denormalize_data(target, norm_stats)
    
    return {
        'mae': calculate_mae(pred_phys, target_phys),
        'rmse': calculate_rmse(pred_phys, target_phys),
        'mse': calculate_mse(pred_phys, target_phys),
        'ssim': calculate_ssim(pred_phys, target_phys)
    }


def test_model(model, test_loader, device, norm_stats):
    model.eval()
    
    all_preds_norm = []
    all_targets_norm = []
    
    print("\n进行重建测试...")
    with torch.no_grad():
        for x_mb in tqdm(test_loader, desc="Testing"):
            x_mb = x_mb.to(device)
            x_recon, _, _, _ = model(x_mb)
            all_preds_norm.append(x_recon.cpu().numpy())
            all_targets_norm.append(x_mb.cpu().numpy())
    
    preds_norm = np.concatenate(all_preds_norm, axis=0)  # (N, C, H, W)
    targets_norm = np.concatenate(all_targets_norm, axis=0)  # (N, C, H, W)
    
    n_samples = preds_norm.shape[0]
    metrics_norm_list = []
    metrics_phys_list = []
    
    print("\n计算指标...")
    for i in tqdm(range(n_samples), desc="Computing metrics"):
        pred_i = preds_norm[i]
        target_i = targets_norm[i]
        n_channels = pred_i.shape[0]
        metrics_norm_i = defaultdict(list)
        metrics_phys_i = defaultdict(list)
        
        for c in range(n_channels):
            pred_c = pred_i[c]
            target_c = target_i[c]
            
            m_norm = calculate_metrics_norm_space(pred_c, target_c)
            for k, v in m_norm.items():
                metrics_norm_i[k].append(v)
            
            m_phys = calculate_metrics_physical_space(pred_c, target_c, norm_stats)
            for k, v in m_phys.items():
                metrics_phys_i[k].append(v)
        
        metrics_norm_list.append({k: np.mean(v) for k, v in metrics_norm_i.items()})
        metrics_phys_list.append({k: np.mean(v) for k, v in metrics_phys_i.items()})
    
    metrics_norm_avg = {}
    metrics_phys_avg = {}
    
    for key in ['mae', 'rmse', 'mse', 'ssim']:
        metrics_norm_avg[key] = np.mean([m[key] for m in metrics_norm_list])
        metrics_phys_avg[key] = np.mean([m[key] for m in metrics_phys_list])
    
    return {
        'normalized_space': metrics_norm_avg,
        'physical_space': metrics_phys_avg,
        'per_sample_norm': metrics_norm_list,
        'per_sample_phys': metrics_phys_list
    }


def main():
    parser = argparse.ArgumentParser(description="测试S-VAE模型重建能力")
    
    # 数据参数
    parser.add_argument('--data-path', type=str, required=True,
                       help='zarr数据路径')
    parser.add_argument('--variable', type=str, default='2m_temperature',
                       help='变量名（如 2m_temperature）')
    parser.add_argument('--time-slice', type=str, required=True,
                       help='时间切片，格式: 2020-01-01:2020-12-31')
    parser.add_argument('--levels', type=int, nargs='+', default=None,
                       help='如果变量有level维度，指定要使用的levels')
    
    # 模型参数
    parser.add_argument('--model-path', type=str, required=True,
                       help='模型文件路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='设备 (cuda/cpu)')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, default='outputs/svae_test',
                       help='输出目录')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批次大小')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"\n加载模型: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    spatial_shape = config.get('spatial_shape', (32, 64))
    n_channels = config.get('n_channels', 1)
    hidden_dims = config.get('hidden_dims', [64, 128, 256, 512])
    latent_channels = config.get('latent_channels', 4)
    distribution = config.get('distribution', 'normal')
    use_residual = config.get('use_residual', True)
    norm_stats = config.get('norm_stats')
    
    if norm_stats is None:
        raise ValueError("模型配置中缺少归一化统计量 (norm_stats)")
    
    print(f"模型配置:")
    print(f"  空间维度: {spatial_shape}")
    print(f"  通道数: {n_channels}")
    print(f"  编码器/解码器通道: {hidden_dims}")
    print(f"  潜在通道数: {latent_channels}")
    print(f"  潜在分布: {distribution}")
    print(f"  归一化统计量: mean={norm_stats['mean']:.2f}, std={norm_stats['std']:.2f}")
    
    # 创建模型
    model = WeatherVAE(
        spatial_shape=spatial_shape,
        n_channels=n_channels,
        latent_channels=latent_channels,
        hidden_dims=hidden_dims,
        distribution=distribution,
        use_residual=use_residual,
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ 模型加载成功")
    
    print(f"\n加载测试数据...")
    print(f"  数据路径: {args.data_path}")
    print(f"  变量: {args.variable}")
    print(f"  时间切片: {args.time_slice}")
    
    test_dataset = WeatherGridDataset(
        args.data_path,
        variable=args.variable,
        time_slice=args.time_slice,
        normalize=True,
        norm_stats=norm_stats,
        levels=args.levels
    )
    
    print(f"  测试样本数: {len(test_dataset)}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    results = test_model(model, test_loader, device, norm_stats)
    
    print("\n" + "=" * 80)
    print("测试结果")
    print("=" * 80)
    
    print("\n归一化空间指标 ([-1, 1]):")
    print("-" * 80)
    metrics_norm = results['normalized_space']
    print(f"  MAE:  {metrics_norm['mae']:.6f}")
    print(f"  RMSE: {metrics_norm['rmse']:.6f}")
    print(f"  MSE:  {metrics_norm['mse']:.6f}")
    print(f"  SSIM: {metrics_norm['ssim']:.6f}")
    
    print("\n物理空间指标:")
    print("-" * 80)
    metrics_phys = results['physical_space']
    print(f"  MAE:  {metrics_phys['mae']:.4f}")
    print(f"  RMSE: {metrics_phys['rmse']:.4f}")
    print(f"  MSE:  {metrics_phys['mse']:.4f}")
    print(f"  SSIM: {metrics_phys['ssim']:.6f}")
    
    output_file = output_dir / "test_results.json"
    with open(output_file, 'w') as f:
        json.dump(convert_to_python_types({
            'args': vars(args),
            'config': config,
            'metrics': {
                'normalized_space': metrics_norm,
                'physical_space': metrics_phys
            },
            'n_samples': len(test_dataset)
        }), f, indent=2)
    
    print(f"\n✓ 结果已保存到: {output_file}")
    
    detailed_file = output_dir / "detailed_metrics.json"
    with open(detailed_file, 'w') as f:
        json.dump(convert_to_python_types({
            'per_sample_normalized': results['per_sample_norm'],
            'per_sample_physical': results['per_sample_phys']
        }), f, indent=2)
    
    print(f"✓ 详细指标已保存到: {detailed_file}")
    
    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()

