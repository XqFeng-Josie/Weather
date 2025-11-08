"""数据归一化模块"""

import numpy as np
from typing import Dict, Tuple, Optional, Literal


class Normalizer:
    """数据归一化器"""
    
    def __init__(self, method: Literal['minmax', 'zscore'] = 'minmax'):
        """
        Args:
            method: 归一化方法，'minmax'将数据归一化到[-1, 1]，'zscore'使用z-score
        """
        self.method = method
        self.stats = {}
    
    def fit(self, data: np.ndarray, name: str = 'default'):
        """计算归一化统计量
        
        Args:
            data: 输入数据
            name: 变量名称
        """
        if self.method == 'minmax':
            self.stats[name] = {
                'min': float(data.min()),
                'max': float(data.max())
            }
        elif self.method == 'zscore':
            self.stats[name] = {
                'mean': float(data.mean()),
                'std': float(data.std())
            }
        else:
            raise ValueError(f"未知的归一化方法: {self.method}")
    
    def transform(self, data: np.ndarray, name: str = 'default') -> np.ndarray:
        """归一化数据
        
        Args:
            data: 输入数据
            name: 变量名称
        """
        if name not in self.stats:
            raise ValueError(f"变量 '{name}' 未进行fit")
        
        stats = self.stats[name]
        
        if self.method == 'minmax':
            # 归一化到[-1, 1]
            min_val = stats['min']
            max_val = stats['max']
            normalized = 2.0 * (data - min_val) / (max_val - min_val + 1e-8) - 1.0
        elif self.method == 'zscore':
            # z-score归一化
            normalized = (data - stats['mean']) / (stats['std'] + 1e-8)
        
        return normalized
    
    def inverse_transform(self, data: np.ndarray, name: str = 'default') -> np.ndarray:
        """反归一化数据
        
        Args:
            data: 归一化后的数据
            name: 变量名称
        """
        if name not in self.stats:
            raise ValueError(f"变量 '{name}' 未进行fit")
        
        stats = self.stats[name]
        
        if self.method == 'minmax':
            # 从[-1, 1]反归一化
            min_val = stats['min']
            max_val = stats['max']
            denormalized = (data + 1.0) / 2.0 * (max_val - min_val) + min_val
        elif self.method == 'zscore':
            # z-score反归一化
            denormalized = data * stats['std'] + stats['mean']
        
        return denormalized
    
    def fit_transform(self, data: np.ndarray, name: str = 'default') -> np.ndarray:
        """fit并transform"""
        self.fit(data, name)
        return self.transform(data, name)
    
    def get_stats(self) -> Dict:
        """获取统计量"""
        return self.stats
    
    def load_stats(self, stats: Dict):
        """加载统计量"""
        self.stats = stats


def normalize_data(data: np.ndarray, 
                   method: Literal['minmax', 'zscore'] = 'minmax',
                   stats: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
    """
    归一化数据（便捷函数）
    
    Args:
        data: 输入数据
        method: 归一化方法
        stats: 如果提供，使用已有统计量；否则从数据计算
    
    Returns:
        (归一化后的数据, 统计量字典)
    """
    normalizer = Normalizer(method=method)
    
    if stats is not None:
        normalizer.load_stats(stats)
        normalized = normalizer.transform(data)
    else:
        normalized = normalizer.fit_transform(data)
    
    return normalized, normalizer.get_stats()


def denormalize_data(data: np.ndarray, 
                     stats: Dict,
                     method: Literal['minmax', 'zscore'] = 'minmax') -> np.ndarray:
    """
    反归一化数据（便捷函数）
    
    Args:
        data: 归一化后的数据
        stats: 统计量字典
        method: 归一化方法
    
    Returns:
        反归一化后的数据
    """
    normalizer = Normalizer(method=method)
    normalizer.load_stats(stats)
    return normalizer.inverse_transform(data)

