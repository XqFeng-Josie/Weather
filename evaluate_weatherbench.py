"""
WeatherBench2评估脚本
用于评估预测结果，并与WeatherBench2基准比较

运行示例:
python evaluate_weatherbench.py --pred predictions.nc --truth era5_test.nc
"""
import argparse
import numpy as np
import xarray as xr
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate predictions against WeatherBench2')
    
    parser.add_argument('--pred', type=str, required=True,
                       help='Path to prediction file (netCDF or numpy)')
    parser.add_argument('--truth', type=str, default=None,
                       help='Path to ground truth file (if not in pred file)')
    
    parser.add_argument('--variables', type=str, nargs='+',
                       default=['geopotential_500', 'temperature_850', '2m_temperature'],
                       help='Variables to evaluate')
    
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                       help='Output directory for results')
    
    parser.add_argument('--compare-baseline', type=str, default=None,
                       help='Path to baseline predictions for comparison')
    
    return parser.parse_args()


def load_predictions(pred_path):
    """加载预测结果"""
    pred_path = Path(pred_path)
    
    if pred_path.suffix == '.nc':
        return xr.open_dataset(pred_path)
    elif pred_path.suffix == '.npz':
        data = np.load(pred_path)
        # 转换为xarray（简化版）
        return {
            'y_pred': data['y_pred'],
            'y_true': data['y_true'],
            'features': data['features'],
        }
    else:
        raise ValueError(f"Unsupported file format: {pred_path.suffix}")


def calculate_rmse(pred, truth, dim=None):
    """计算RMSE"""
    mse = ((pred - truth) ** 2).mean(dim=dim)
    return np.sqrt(mse)


def calculate_acc(pred, truth, climatology):
    """
    计算异常相关系数 (Anomaly Correlation Coefficient)
    ACC = corr(pred - clim, truth - clim)
    """
    pred_anom = pred - climatology
    truth_anom = truth - climatology
    
    numerator = (pred_anom * truth_anom).mean()
    denominator = np.sqrt((pred_anom ** 2).mean() * (truth_anom ** 2).mean())
    
    return numerator / (denominator + 1e-8)


def calculate_mae(pred, truth, dim=None):
    """计算MAE"""
    return np.abs(pred - truth).mean(dim=dim)


def calculate_bias(pred, truth, dim=None):
    """计算Bias"""
    return (pred - truth).mean(dim=dim)


def evaluate_by_lead_time(y_pred, y_true, variable_names):
    """按lead time评估"""
    n_samples, n_lead_times, n_features = y_pred.shape
    
    metrics = {
        'rmse': np.zeros((n_lead_times, n_features)),
        'mae': np.zeros((n_lead_times, n_features)),
        'bias': np.zeros((n_lead_times, n_features)),
    }
    
    for t in range(n_lead_times):
        for f in range(n_features):
            pred = y_pred[:, t, f]
            true = y_true[:, t, f]
            
            metrics['rmse'][t, f] = np.sqrt(np.mean((pred - true) ** 2))
            metrics['mae'][t, f] = np.mean(np.abs(pred - true))
            metrics['bias'][t, f] = np.mean(pred - true)
    
    return metrics


def plot_rmse_by_leadtime(metrics, variable_names, output_path, baseline_metrics=None):
    """绘制RMSE vs Lead Time"""
    n_lead_times, n_features = metrics['rmse'].shape
    lead_times = np.arange(1, n_lead_times + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flat
    
    for i, var_name in enumerate(variable_names[:4]):
        if i >= n_features:
            break
        
        ax = axes[i]
        
        # 模型RMSE
        ax.plot(lead_times, metrics['rmse'][:, i], 'o-', 
                label='Model', linewidth=2, markersize=6)
        
        # Baseline（如果有）
        if baseline_metrics is not None:
            ax.plot(lead_times, baseline_metrics['rmse'][:, i], 's--',
                   label='Baseline', linewidth=2, markersize=6, alpha=0.7)
        
        ax.set_xlabel('Lead Time Step', fontsize=12)
        ax.set_ylabel('RMSE', fontsize=12)
        ax.set_title(f'{var_name}', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ RMSE plot saved to {output_path}")


def plot_skill_score(metrics, baseline_metrics, variable_names, output_path):
    """
    绘制Skill Score
    SS = 1 - (RMSE_model / RMSE_baseline)
    """
    rmse_model = metrics['rmse']
    rmse_baseline = baseline_metrics['rmse']
    
    skill_score = 1 - (rmse_model / (rmse_baseline + 1e-8))
    
    n_lead_times, n_features = skill_score.shape
    lead_times = np.arange(1, n_lead_times + 1)
    
    plt.figure(figsize=(12, 6))
    
    for i, var_name in enumerate(variable_names[:n_features]):
        plt.plot(lead_times, skill_score[:, i], 'o-', 
                label=var_name, linewidth=2, markersize=6)
    
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Baseline')
    plt.xlabel('Lead Time Step', fontsize=12)
    plt.ylabel('Skill Score', fontsize=12)
    plt.title('Skill Score vs Baseline (higher is better)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Skill score plot saved to {output_path}")


def plot_error_distribution(y_pred, y_true, variable_names, output_path):
    """绘制误差分布"""
    errors = y_pred - y_true
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flat
    
    for i, var_name in enumerate(variable_names[:4]):
        if i >= errors.shape[2]:
            break
        
        ax = axes[i]
        
        # 选择第一个lead time的误差
        error_flat = errors[:, 0, i].flatten()
        
        ax.hist(error_flat, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        ax.set_xlabel('Prediction Error', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{var_name} (Lead Time 1)', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Error distribution plot saved to {output_path}")


def compare_with_weatherbench_baselines():
    """
    与WeatherBench2基准比较
    这些是WeatherBench2论文中的参考值
    """
    # 示例：500hPa位势高度的RMSE（单位：m）
    # 来源：WeatherBench2论文 Table 1
    weatherbench_baselines = {
        'geopotential_500': {
            'climatology': [350, 420, 480, 530],  # 1, 3, 5, 7天
            'persistence': [180, 320, 400, 460],
            'ifs': [80, 180, 260, 320],  # ECMWF IFS (最好的NWP)
        },
        'temperature_850': {
            'climatology': [4.5, 5.2, 5.8, 6.2],
            'persistence': [2.5, 3.8, 4.5, 5.0],
            'ifs': [1.2, 2.1, 2.8, 3.3],
        },
    }
    
    return weatherbench_baselines


def main():
    args = parse_args()
    
    print("="*80)
    print("WeatherBench2 Evaluation")
    print("="*80)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载预测
    print("\nLoading predictions...")
    pred_data = load_predictions(args.pred)
    
    if isinstance(pred_data, dict):
        # numpy格式
        y_pred = pred_data['y_pred']
        y_true = pred_data['y_true']
        variable_names = list(pred_data['features'])
    else:
        # xarray格式
        variable_names = args.variables
        # TODO: 从xarray提取数据
        print("xarray format - extracting variables...")
    
    print(f"Variables: {variable_names}")
    print(f"Prediction shape: {y_pred.shape}")
    
    # 2. 计算指标
    print("\nCalculating metrics...")
    metrics = evaluate_by_lead_time(y_pred, y_true, variable_names)
    
    print("\nRMSE by Lead Time:")
    for t in range(metrics['rmse'].shape[0]):
        print(f"  Lead {t+1}: ", end="")
        for f, var in enumerate(variable_names):
            if f < metrics['rmse'].shape[1]:
                print(f"{var}={metrics['rmse'][t, f]:.4f} ", end="")
        print()
    
    # 3. 加载baseline（如果有）
    baseline_metrics = None
    if args.compare_baseline:
        print(f"\nLoading baseline from {args.compare_baseline}...")
        baseline_data = load_predictions(args.compare_baseline)
        if isinstance(baseline_data, dict):
            baseline_metrics = evaluate_by_lead_time(
                baseline_data['y_pred'],
                baseline_data['y_true'],
                variable_names
            )
    
    # 4. 可视化
    print("\nGenerating visualizations...")
    
    # RMSE vs Lead Time
    plot_rmse_by_leadtime(
        metrics, variable_names,
        output_dir / 'rmse_by_leadtime.png',
        baseline_metrics
    )
    
    # 误差分布
    plot_error_distribution(
        y_pred, y_true, variable_names,
        output_dir / 'error_distribution.png'
    )
    
    # Skill Score（如果有baseline）
    if baseline_metrics is not None:
        plot_skill_score(
            metrics, baseline_metrics, variable_names,
            output_dir / 'skill_score.png'
        )
    
    # 5. 保存指标
    metrics_dict = {
        'model': {
            'rmse': metrics['rmse'].tolist(),
            'mae': metrics['mae'].tolist(),
            'bias': metrics['bias'].tolist(),
        },
        'variables': variable_names,
    }
    
    if baseline_metrics is not None:
        metrics_dict['baseline'] = {
            'rmse': baseline_metrics['rmse'].tolist(),
        }
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    print(f"\n✓ Metrics saved to {output_dir}/metrics.json")
    
    # 6. 与WeatherBench2基准比较
    print("\n" + "="*80)
    print("Comparison with WeatherBench2 Baselines")
    print("="*80)
    
    wb2_baselines = compare_with_weatherbench_baselines()
    
    for var in variable_names:
        if var in wb2_baselines:
            print(f"\n{var}:")
            print(f"  Your model RMSE: {metrics['rmse'][0, variable_names.index(var)]:.2f}")
            print(f"  Climatology:     {wb2_baselines[var]['climatology'][0]:.2f}")
            print(f"  Persistence:     {wb2_baselines[var]['persistence'][0]:.2f}")
            print(f"  IFS (SOTA):      {wb2_baselines[var]['ifs'][0]:.2f}")
    
    print("\n✓ Evaluation complete!")
    print(f"✓ Results saved to {output_dir}")


if __name__ == '__main__':
    main()

