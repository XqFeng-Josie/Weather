"""数据归一化模块"""

import numpy as np
from typing import Dict, Tuple, Optional, Literal


class Normalizer:
    """数据归一化器

    支持两种归一化模式：
    1. 全局归一化：对所有数据使用相同的统计量
    2. 按level归一化：每个level（气压层）使用独立的统计量
    """

    def __init__(self, method: Literal["minmax", "zscore"] = "minmax"):
        """
        Args:
            method: 归一化方法，'minmax'将数据归一化到[-1, 1]，'zscore'使用z-score
        """
        self.method = method
        self.stats = {}
        self.normalize_per_level = False  # 是否按level归一化
        self.level_to_stats_key = {}  # level到统计量key的映射
        self.variable = None  # 变量名
        self.levels = []  # level列表
        self.n_channels_per_level = None  # 每个level的通道数

    def fit(self, data: np.ndarray, name: str = "default"):
        """计算归一化统计量

        Args:
            data: 输入数据
            name: 变量名称
        """
        if self.method == "minmax":
            self.stats[name] = {"min": float(data.min()), "max": float(data.max())}
        elif self.method == "zscore":
            self.stats[name] = {"mean": float(data.mean()), "std": float(data.std())}
        else:
            raise ValueError(f"未知的归一化方法: {self.method}")

    def transform(self, data: np.ndarray, name: str = "default") -> np.ndarray:
        """归一化数据

        Args:
            data: 输入数据，形状可以是 (H, W), (C, H, W), (T, C, H, W) 等
            name: 变量名称
        """
        if self.normalize_per_level:
            # 按level归一化：需要知道数据中每个level对应的通道
            return self._transform_per_level(data, name)
        else:
            # 全局归一化
            if name not in self.stats:
                raise ValueError(f"变量 '{name}' 未进行fit")

            stats = self.stats[name]

            if self.method == "minmax":
                # 归一化到[-1, 1]
                min_val = stats["min"]
                max_val = stats["max"]
                normalized = 2.0 * (data - min_val) / (max_val - min_val + 1e-8) - 1.0
            elif self.method == "zscore":
                # z-score归一化
                normalized = (data - stats["mean"]) / (stats["std"] + 1e-8)

            return normalized

    def inverse_transform(self, data: np.ndarray, name: str = "default") -> np.ndarray:
        """反归一化数据

        Args:
            data: 归一化后的数据，形状可以是 (H, W), (C, H, W), (T, C, H, W) 等
            name: 变量名称
        """
        if self.normalize_per_level:
            # 按level反归一化
            return self._inverse_transform_per_level(data, name)
        else:
            # 全局反归一化
            if name not in self.stats:
                raise ValueError(f"变量 '{name}' 未进行fit")

            stats = self.stats[name]

            if self.method == "minmax":
                # 从[-1, 1]反归一化
                min_val = stats["min"]
                max_val = stats["max"]
                denormalized = (data + 1.0) / 2.0 * (max_val - min_val) + min_val
            elif self.method == "zscore":
                # z-score反归一化
                denormalized = data * stats["std"] + stats["mean"]

            return denormalized

    def _transform_per_level(self, data: np.ndarray, name: str) -> np.ndarray:
        """按level归一化数据

        数据格式：如果有level维度，通道顺序为 [level1_channels, level2_channels, ...]
        其中每个level有 n_channels_per_level 个通道
        """
        data = np.asarray(data)
        original_shape = data.shape

        # 确定通道维度
        if len(data.shape) == 2:
            # (H, W) - 应该是单个level的数据
            raise ValueError("按level归一化需要通道维度信息")
        elif len(data.shape) == 3:
            # (C, H, W)
            channels_dim = 0
            data = data[np.newaxis, ...]  # 添加时间维度: (1, C, H, W)
        elif len(data.shape) == 4:
            # (T, C, H, W)
            channels_dim = 1
        else:
            raise ValueError(f"不支持的数据形状: {data.shape}")

        T, C, H, W = data.shape

        # 确定每个level的通道数
        if self.n_channels_per_level is None:
            # 如果没有指定，尝试从统计量推断
            # 假设所有level的通道数相同
            if self.levels and C % len(self.levels) == 0:
                self.n_channels_per_level = C // len(self.levels)
            else:
                raise ValueError(
                    f"无法确定每个level的通道数。总通道数: {C}, Levels: {self.levels}"
                )

        normalized = np.zeros_like(data)

        # 对每个level分别归一化
        for level_idx, level_val in enumerate(self.levels):
            # 计算该level对应的通道范围
            start_channel = level_idx * self.n_channels_per_level
            end_channel = (level_idx + 1) * self.n_channels_per_level

            # 获取该level的数据
            level_data = data[:, start_channel:end_channel, :, :]

            # 获取该level的统计量
            stats_key = self.level_to_stats_key.get(level_val)
            if stats_key is None:
                stats_key = f"{self.variable}_level_{level_val}"

            if stats_key not in self.stats:
                raise ValueError(f"Level {level_val} 的统计量未找到 (key: {stats_key})")

            stats = self.stats[stats_key]

            # 归一化
            if self.method == "minmax":
                min_val = stats["min"]
                max_val = stats["max"]
                if max_val - min_val > 1e-8:
                    level_normalized = (
                        2.0 * (level_data - min_val) / (max_val - min_val) - 1.0
                    )
                else:
                    level_normalized = np.zeros_like(level_data)
            elif self.method == "zscore":
                mean_val = stats["mean"]
                std_val = stats["std"]
                if std_val > 1e-8:
                    level_normalized = (level_data - mean_val) / std_val
                else:
                    level_normalized = np.zeros_like(level_data)

            normalized[:, start_channel:end_channel, :, :] = level_normalized

        # 恢复原始形状
        if len(original_shape) == 3:
            normalized = normalized[0]  # 移除时间维度

        return normalized

    def _inverse_transform_per_level(self, data: np.ndarray, name: str) -> np.ndarray:
        """按level反归一化数据"""
        data = np.asarray(data)
        original_shape = data.shape

        # 确定通道维度
        if len(data.shape) == 2:
            raise ValueError("按level反归一化需要通道维度信息")
        elif len(data.shape) == 3:
            # (C, H, W)
            channels_dim = 0
            data = data[np.newaxis, ...]  # 添加时间维度: (1, C, H, W)
        elif len(data.shape) == 4:
            # (T, C, H, W)
            channels_dim = 1
        else:
            raise ValueError(f"不支持的数据形状: {data.shape}")

        T, C, H, W = data.shape

        # 确定每个level的通道数
        if self.n_channels_per_level is None:
            if self.levels and C % len(self.levels) == 0:
                self.n_channels_per_level = C // len(self.levels)
            else:
                raise ValueError(
                    f"无法确定每个level的通道数。总通道数: {C}, Levels: {self.levels}"
                )

        denormalized = np.zeros_like(data)

        # 对每个level分别反归一化
        for level_idx, level_val in enumerate(self.levels):
            # 计算该level对应的通道范围
            start_channel = level_idx * self.n_channels_per_level
            end_channel = (level_idx + 1) * self.n_channels_per_level

            # 获取该level的数据
            level_data = data[:, start_channel:end_channel, :, :]

            # 获取该level的统计量
            stats_key = self.level_to_stats_key.get(level_val)
            if stats_key is None:
                stats_key = f"{self.variable}_level_{level_val}"

            if stats_key not in self.stats:
                raise ValueError(f"Level {level_val} 的统计量未找到 (key: {stats_key})")

            stats = self.stats[stats_key]

            # 反归一化
            if self.method == "minmax":
                min_val = stats["min"]
                max_val = stats["max"]
                level_denormalized = (level_data + 1.0) / 2.0 * (
                    max_val - min_val
                ) + min_val
            elif self.method == "zscore":
                mean_val = stats["mean"]
                std_val = stats["std"]
                level_denormalized = level_data * std_val + mean_val

            denormalized[:, start_channel:end_channel, :, :] = level_denormalized

        # 恢复原始形状
        if len(original_shape) == 3:
            denormalized = denormalized[0]  # 移除时间维度

        return denormalized

    def fit_transform(self, data: np.ndarray, name: str = "default") -> np.ndarray:
        """fit并transform"""
        self.fit(data, name)
        return self.transform(data, name)

    def get_stats(self) -> Dict:
        """获取统计量"""
        return self.stats

    def load_stats(self, stats: Dict):
        """加载统计量

        Args:
            stats: 统计量字典，可能包含：
                - 'stats': 实际的统计量字典
                - 'normalize_per_level': 是否按level归一化
                - 'level_to_stats_key': level到统计量key的映射
                - 'variable': 变量名
                - 'levels': level列表
                - 'method': 归一化方法
        """
        if isinstance(stats, dict):
            # 检查是否是新的格式（包含normalize_per_level等信息）
            if "normalize_per_level" in stats:
                self.normalize_per_level = stats["normalize_per_level"]
                self.stats = stats.get(
                    "stats", stats
                )  # 如果stats是嵌套的，提取内部的stats

                if self.normalize_per_level:
                    self.level_to_stats_key = stats.get("level_to_stats_key", {})
                    self.variable = stats.get("variable")
                    self.levels = stats.get("levels", [])
                    # n_channels_per_level 需要从数据或元数据中推断
            else:
                # 旧格式：直接是统计量字典
                self.stats = stats
                self.normalize_per_level = False
        else:
            self.stats = stats
            self.normalize_per_level = False


def normalize_data(
    data: np.ndarray,
    method: Literal["minmax", "zscore"] = "minmax",
    stats: Optional[Dict] = None,
) -> Tuple[np.ndarray, Dict]:
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


def denormalize_data(
    data: np.ndarray, stats: Dict, method: Literal["minmax", "zscore"] = "minmax"
) -> np.ndarray:
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
