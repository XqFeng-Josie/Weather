"""评估指标计算模块"""

import numpy as np
import torch
from typing import Dict, Union


def calculate_mae(pred: np.ndarray, target: np.ndarray) -> float:
    """计算平均绝对误差 (MAE)"""
    return np.mean(np.abs(pred - target))


def calculate_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """计算均方根误差 (RMSE)"""
    return np.sqrt(np.mean((pred - target) ** 2))


def calculate_psnr(pred: np.ndarray, target: np.ndarray, max_val: float = 1.0) -> float:
    """
    计算峰值信噪比 (PSNR)
    
    Args:
        pred: 预测值
        target: 目标值
        max_val: 数据的最大值范围
    """
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))


def calculate_ssim(pred: np.ndarray, target: np.ndarray, 
                   window_size: int = 11, k1: float = 0.01, k2: float = 0.03) -> float:
    """
    计算结构相似性指数 (SSIM)
    
    简化版本，用于单通道图像
    """
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


def calculate_spatial_stats(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """
    计算物理一致性指标
    
    Returns:
        包含空间均值偏差、方差比等指标的字典
    """
    mean_bias = np.mean(pred) - np.mean(target)
    std_ratio = np.std(pred) / (np.std(target) + 1e-8)
    
    # 相关系数
    correlation = np.corrcoef(pred.flatten(), target.flatten())[0, 1]
    
    return {
        'mean_bias': float(mean_bias),
        'std_ratio': float(std_ratio),
        'correlation': float(correlation)
    }


def calculate_crps(ensemble_pred: np.ndarray, target: np.ndarray) -> float:
    """
    计算连续秩概率评分 (CRPS)
    
    Args:
        ensemble_pred: 集合预测，shape (n_samples, ...)
        target: 目标值，shape (...)
    """
    n_samples = ensemble_pred.shape[0]
    
    # 计算预测与观测的平均绝对差
    term1 = np.mean(np.abs(ensemble_pred - target))
    
    # 计算预测内部的平均绝对差
    term2 = 0
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            term2 += np.mean(np.abs(ensemble_pred[i] - ensemble_pred[j]))
    term2 = term2 / (n_samples * (n_samples - 1) / 2)
    
    crps = term1 - 0.5 * term2
    return float(crps)


def calculate_metrics(pred: Union[np.ndarray, torch.Tensor], 
                     target: Union[np.ndarray, torch.Tensor],
                     ensemble: bool = False) -> Dict[str, float]:
    """
    计算所有评估指标
    
    Args:
        pred: 预测值，如果是ensemble则shape为 (n_samples, ...)
        target: 目标值
        ensemble: 是否为集合预测
    """
    # 转换为numpy
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    metrics = {}
    
    if ensemble:
        # 使用集合均值计算基础指标
        pred_mean = pred.mean(axis=0)
        metrics['mae'] = calculate_mae(pred_mean, target)
        metrics['rmse'] = calculate_rmse(pred_mean, target)
        metrics['psnr'] = calculate_psnr(pred_mean, target)
        metrics['ssim'] = calculate_ssim(pred_mean, target)
        
        # 计算CRPS
        metrics['crps'] = calculate_crps(pred, target)
        
        # 空间统计（使用集合均值）
        spatial_stats = calculate_spatial_stats(pred_mean, target)
        metrics.update(spatial_stats)
        
        # 集合扩散度
        metrics['ensemble_spread'] = float(np.mean(np.std(pred, axis=0)))
    else:
        # 确定性预测的指标
        metrics['mae'] = calculate_mae(pred, target)
        metrics['rmse'] = calculate_rmse(pred, target)
        metrics['psnr'] = calculate_psnr(pred, target)
        metrics['ssim'] = calculate_ssim(pred, target)
        
        # 空间统计
        spatial_stats = calculate_spatial_stats(pred, target)
        metrics.update(spatial_stats)
    
    return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    """格式化指标输出"""
    lines = []
    lines.append("=" * 50)
    lines.append("评估指标:")
    lines.append("-" * 50)
    
    if 'mae' in metrics:
        lines.append(f"  MAE:         {metrics['mae']:.4f}")
    if 'rmse' in metrics:
        lines.append(f"  RMSE:        {metrics['rmse']:.4f}")
    if 'psnr' in metrics:
        lines.append(f"  PSNR:        {metrics['psnr']:.2f} dB")
    if 'ssim' in metrics:
        lines.append(f"  SSIM:        {metrics['ssim']:.4f}")
    if 'correlation' in metrics:
        lines.append(f"  相关系数:     {metrics['correlation']:.4f}")
    if 'mean_bias' in metrics:
        lines.append(f"  均值偏差:     {metrics['mean_bias']:.4f}")
    if 'std_ratio' in metrics:
        lines.append(f"  标准差比:     {metrics['std_ratio']:.4f}")
    if 'crps' in metrics:
        lines.append(f"  CRPS:        {metrics['crps']:.4f}")
    if 'ensemble_spread' in metrics:
        lines.append(f"  集合扩散度:   {metrics['ensemble_spread']:.4f}")
    
    lines.append("=" * 50)
    return "\n".join(lines)

