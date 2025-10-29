"""
概率预测评估指标（GenCast 风格）

主要包括：
1. CRPS (Continuous Ranked Probability Score)
2. Spread-Skill Relationship
3. Rank Histogram
4. 概率校准
"""

import numpy as np
from typing import Tuple, Optional


def crps_ensemble(
    observations: np.ndarray,
    forecasts: np.ndarray,
    axis: int = 0
) -> np.ndarray:
    """
    计算 CRPS (Continuous Ranked Probability Score) for ensemble forecasts
    
    CRPS 衡量概率预测分布与观测值的差异，越小越好。
    对于 ensemble forecasts，使用简化公式：
    
    CRPS = E[|X - y|] - 0.5 * E[|X - X'|]
    
    其中 X, X' 是独立的 ensemble 成员，y 是观测值。
    
    Args:
        observations: 观测值 (shape: [..., spatial_dims...])
        forecasts: Ensemble 预测 (shape: [num_members, ..., spatial_dims...])
        axis: Ensemble 维度（默认为 0）
        
    Returns:
        CRPS 值 (shape: [..., spatial_dims...])
    """
    # 确保 forecasts 的 ensemble 维度在第一个位置
    if axis != 0:
        forecasts = np.moveaxis(forecasts, axis, 0)
    
    num_members = forecasts.shape[0]
    
    # E[|X - y|]: 每个 ensemble 成员与观测值的平均绝对误差
    abs_error = np.abs(forecasts - observations)
    term1 = np.mean(abs_error, axis=0)
    
    # E[|X - X'|]: ensemble 成员之间的平均绝对差异
    # 计算所有成员对之间的差异
    pairwise_diff = 0.0
    count = 0
    for i in range(num_members):
        for j in range(i + 1, num_members):
            pairwise_diff += np.abs(forecasts[i] - forecasts[j])
            count += 1
    
    term2 = pairwise_diff / count if count > 0 else 0.0
    
    crps = term1 - 0.5 * term2
    
    return crps


def crps_gaussian(
    observations: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray
) -> np.ndarray:
    """
    计算 CRPS for Gaussian distribution
    
    对于高斯分布 N(μ, σ²)，CRPS 有解析解：
    CRPS = σ * [z * (2Φ(z) - 1) + 2φ(z) - 1/√π]
    
    其中 z = (y - μ) / σ, Φ 是标准正态 CDF, φ 是标准正态 PDF
    
    Args:
        observations: 观测值
        mean: 预测均值
        std: 预测标准差
        
    Returns:
        CRPS 值
    """
    from scipy import stats
    
    # 标准化
    z = (observations - mean) / (std + 1e-10)
    
    # 标准正态 CDF 和 PDF
    cdf_z = stats.norm.cdf(z)
    pdf_z = stats.norm.pdf(z)
    
    # CRPS 公式
    crps = std * (z * (2 * cdf_z - 1) + 2 * pdf_z - 1 / np.sqrt(np.pi))
    
    return crps


def ensemble_spread(forecasts: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    计算 Ensemble Spread (标准差)
    
    Args:
        forecasts: Ensemble 预测 (shape: [num_members, ...])
        axis: Ensemble 维度
        
    Returns:
        Spread (标准差)
    """
    return np.std(forecasts, axis=axis)


def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    计算 RMSE (Root Mean Square Error)
    
    Args:
        predictions: 预测值
        targets: 真实值
        
    Returns:
        RMSE
    """
    return np.sqrt(np.mean((predictions - targets) ** 2))


def spread_skill_ratio(
    forecasts: np.ndarray,
    observations: np.ndarray,
    axis: int = 0
) -> float:
    """
    计算 Spread-Skill Ratio
    
    理想情况下，ensemble spread 应该等于预测误差（skill），
    即 Spread-Skill Ratio ≈ 1.0
    
    - Ratio < 1: Under-dispersive (spread 太小，过度自信)
    - Ratio > 1: Over-dispersive (spread 太大，不够自信)
    
    Args:
        forecasts: Ensemble 预测 (shape: [num_members, ...])
        observations: 观测值
        axis: Ensemble 维度
        
    Returns:
        Spread-Skill Ratio
    """
    # Ensemble mean
    ensemble_mean = np.mean(forecasts, axis=axis)
    
    # Spread (标准差)
    spread = ensemble_spread(forecasts, axis=axis)
    mean_spread = np.mean(spread)
    
    # Skill (RMSE)
    skill = rmse(ensemble_mean, observations)
    
    # Ratio
    ratio = mean_spread / (skill + 1e-10)
    
    return ratio


def rank_histogram(
    forecasts: np.ndarray,
    observations: np.ndarray,
    axis: int = 0,
    num_bins: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 Rank Histogram (Talagrand Diagram)
    
    Rank histogram 检验 ensemble 的统计一致性：
    - 如果 ensemble 校准良好，观测值应该均匀分布在 ensemble 成员的排序中
    - U-shaped: Under-dispersive
    - Dome-shaped: Over-dispersive
    - Skewed: Biased
    
    Args:
        forecasts: Ensemble 预测 (shape: [num_members, n_samples, ...])
        observations: 观测值 (shape: [n_samples, ...])
        axis: Ensemble 维度
        num_bins: Bins 数量（默认为 num_members + 1）
        
    Returns:
        (bins, histogram)
    """
    if axis != 0:
        forecasts = np.moveaxis(forecasts, axis, 0)
    
    num_members = forecasts.shape[0]
    if num_bins is None:
        num_bins = num_members + 1
    
    # 展平空间维度
    forecasts_flat = forecasts.reshape(num_members, -1)
    observations_flat = observations.flatten()
    
    # 计算每个观测值在 ensemble 中的排序位置
    ranks = []
    for i in range(forecasts_flat.shape[1]):
        ensemble_values = forecasts_flat[:, i]
        obs_value = observations_flat[i]
        
        # 计算有多少 ensemble 成员小于观测值
        rank = np.sum(ensemble_values < obs_value)
        ranks.append(rank)
    
    ranks = np.array(ranks)
    
    # 统计直方图
    histogram, bins = np.histogram(ranks, bins=np.arange(num_bins + 1))
    
    return bins, histogram


def continuous_ranked_probability_score_summary(
    forecasts: np.ndarray,
    observations: np.ndarray,
    variables: list = None
) -> dict:
    """
    计算综合的 CRPS 统计信息
    
    Args:
        forecasts: Ensemble 预测 (shape: [num_members, batch, time, channels, H, W])
        observations: 观测值 (shape: [batch, time, channels, H, W])
        variables: 变量名列表
        
    Returns:
        统计字典
    """
    stats = {}
    
    # 全局 CRPS
    crps_values = crps_ensemble(observations, forecasts, axis=0)
    stats['crps_mean'] = float(np.mean(crps_values))
    stats['crps_std'] = float(np.std(crps_values))
    
    # 按时间步统计
    crps_by_time = []
    for t in range(observations.shape[1]):
        crps_t = crps_ensemble(observations[:, t], forecasts[:, :, t], axis=0)
        crps_by_time.append(float(np.mean(crps_t)))
    stats['crps_by_leadtime'] = crps_by_time
    
    # 按变量统计（如果有多个变量）
    if observations.shape[2] > 1:
        crps_by_var = []
        for c in range(observations.shape[2]):
            crps_c = crps_ensemble(observations[:, :, c], forecasts[:, :, :, c], axis=0)
            crps_by_var.append(float(np.mean(crps_c)))
        stats['crps_by_variable'] = crps_by_var
        
        if variables:
            stats['crps_by_variable_names'] = dict(zip(variables, crps_by_var))
    
    # Ensemble mean RMSE (用于对比)
    ensemble_mean = np.mean(forecasts, axis=0)
    stats['ensemble_mean_rmse'] = float(rmse(ensemble_mean, observations))
    
    # Spread-Skill Ratio
    stats['spread_skill_ratio'] = float(spread_skill_ratio(forecasts, observations, axis=0))
    
    return stats


def probabilistic_evaluation_report(
    forecasts: np.ndarray,
    observations: np.ndarray,
    variables: list = None
) -> str:
    """
    生成概率预测评估报告
    
    Args:
        forecasts: Ensemble 预测
        observations: 观测值
        variables: 变量名列表
        
    Returns:
        报告字符串
    """
    stats = continuous_ranked_probability_score_summary(forecasts, observations, variables)
    
    report = "\n" + "=" * 80 + "\n"
    report += "概率预测评估报告 (GenCast 风格)\n"
    report += "=" * 80 + "\n\n"
    
    report += f"Ensemble 成员数: {forecasts.shape[0]}\n"
    report += f"样本数: {forecasts.shape[1]}\n"
    report += f"预测时间步: {forecasts.shape[2]}\n\n"
    
    report += "📊 核心指标:\n"
    report += "-" * 80 + "\n"
    report += f"  CRPS (Continuous Ranked Probability Score): {stats['crps_mean']:.4f} ± {stats['crps_std']:.4f}\n"
    report += f"  Ensemble Mean RMSE: {stats['ensemble_mean_rmse']:.4f}\n"
    report += f"  Spread-Skill Ratio: {stats['spread_skill_ratio']:.4f}\n"
    
    if stats['spread_skill_ratio'] < 0.8:
        report += "    ⚠️  Under-dispersive (ensemble spread 太小，过度自信)\n"
    elif stats['spread_skill_ratio'] > 1.2:
        report += "    ⚠️  Over-dispersive (ensemble spread 太大，不够自信)\n"
    else:
        report += "    ✅ Well-calibrated (spread ≈ skill)\n"
    
    report += "\n"
    
    # 按时间步
    if 'crps_by_leadtime' in stats:
        report += "📈 CRPS by Lead Time:\n"
        report += "-" * 80 + "\n"
        for t, crps_t in enumerate(stats['crps_by_leadtime'], 1):
            report += f"  Lead Time {t}: {crps_t:.4f}\n"
        report += "\n"
    
    # 按变量
    if 'crps_by_variable_names' in stats:
        report += "🌍 CRPS by Variable:\n"
        report += "-" * 80 + "\n"
        for var_name, crps_var in stats['crps_by_variable_names'].items():
            report += f"  {var_name}: {crps_var:.4f}\n"
        report += "\n"
    
    report += "=" * 80 + "\n"
    
    return report

