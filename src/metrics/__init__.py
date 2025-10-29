"""
评估指标模块

包括：
- deterministic.py: 确定性预测指标 (RMSE, MAE, etc.)
- probabilistic.py: 概率预测指标 (CRPS, Spread-Skill, etc.)
"""

from .probabilistic import (
    crps_ensemble,
    crps_gaussian,
    ensemble_spread,
    spread_skill_ratio,
    rank_histogram,
    continuous_ranked_probability_score_summary,
    probabilistic_evaluation_report,
)

__all__ = [
    'crps_ensemble',
    'crps_gaussian',
    'ensemble_spread',
    'spread_skill_ratio',
    'rank_histogram',
    'continuous_ranked_probability_score_summary',
    'probabilistic_evaluation_report',
]

