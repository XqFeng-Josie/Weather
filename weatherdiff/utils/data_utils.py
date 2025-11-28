"""数据处理工具模块"""

import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict, List
from pathlib import Path


def prepare_weather_data(
    data: np.ndarray, n_channels: int = 3, target_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    准备天气数据用于图像模型

    Args:
        data: 输入数据，shape (Time, H, W) 或 (Time, Level, H, W) 或 (Time, C, H, W)
        n_channels: 目标通道数（1或3）
        target_size: 目标尺寸 (H, W)，如果为None则保持原尺寸

    Returns:
        处理后的数据，shape (Time, C, H, W)

    Note:
        当输入数据有Level维度且Level=1时（单个气压层选择后），会先squeeze掉该维度，
        然后再转换为n_channels通道。这确保了level和channel的概念分离：
        - level: 气象数据的气压层（从原始数据选择）
        - channel: 图像模型的通道数（RGB转换后）
    """
    # 处理4维数据：可能是 (Time, Level, H, W) 或 (Time, C, H, W)
    if data.ndim == 4:
        # 如果第二维度是1，可能是单个level选择后的结果，需要squeeze
        # 这样可以统一处理为3维数据，再转换为n_channels通道
        if data.shape[1] == 1:
            # (Time, 1, H, W) -> (Time, H, W)
            data = data.squeeze(axis=1)

    # 确保是4维: (Time, C, H, W)
    if data.ndim == 3:
        # (Time, H, W) -> (Time, 1, H, W)
        data = data[:, np.newaxis, :, :]

    # 如果需要3通道但只有1通道，复制
    if n_channels == 3 and data.shape[1] == 1:
        data = np.repeat(data, 3, axis=1)

    # 调整尺寸（如果需要）
    if target_size is not None:
        from scipy.ndimage import zoom

        time, c, h, w = data.shape
        target_h, target_w = target_size

        zoom_factors = (1, 1, target_h / h, target_w / w)
        data = zoom(data, zoom_factors, order=1)

    return data


def split_dataset(
    data: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    shuffle: bool = False,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    分割数据集为训练/验证/测试集

    Args:
        data: 输入数据
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        shuffle: 是否打乱数据
        seed: 随机种子

    Returns:
        (train_data, val_data, test_data)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"

    n_samples = len(data)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    if shuffle:
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n_samples)
        data = data[indices]

    train_data = data[:n_train]
    val_data = data[n_train : n_train + n_val]
    test_data = data[n_train + n_val :]

    return train_data, val_data, test_data


class WeatherSequenceDataset(Dataset):
    """天气序列数据集"""

    def __init__(
        self,
        data: np.ndarray,
        input_length: int = 4,
        output_length: int = 1,
        stride: int = 1,
    ):
        """
        Args:
            data: 数据，shape (Time, C, H, W)
            input_length: 输入序列长度
            output_length: 输出序列长度
            stride: 滑动窗口步长
        """
        self.data = data
        self.input_length = input_length
        self.output_length = output_length
        self.stride = stride

        # 计算有效样本数
        total_length = input_length + output_length
        self.n_samples = (len(data) - total_length) // stride + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Returns:
            input_seq: shape (T_in, C, H, W)
            output_seq: shape (T_out, C, H, W)
        """
        start_idx = idx * self.stride
        end_input = start_idx + self.input_length
        end_output = end_input + self.output_length

        input_seq = self.data[start_idx:end_input]
        output_seq = self.data[end_input:end_output]

        return torch.from_numpy(input_seq).float(), torch.from_numpy(output_seq).float()


class WeatherDataModule:
    """天气数据模块，统一管理数据加载"""

    def __init__(
        self,
        data_path: str,
        variable: str = "2m_temperature",
        time_slice: Optional[str] = None,
        input_length: int = 4,
        output_length: int = 1,
        batch_size: int = 16,
        num_workers: int = 4,
        normalization: str = "minmax",
        n_channels: int = 3,
        target_size: Optional[Tuple[int, int]] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        levels: Optional[List[int]] = None,
    ):
        """
        Args:
            data_path: 数据文件路径
            variable: 变量名
            time_slice: 时间切片，格式 "2020-01-01:2020-12-31"
            input_length: 输入序列长度
            output_length: 输出序列长度
            batch_size: 批次大小
            num_workers: 数据加载线程数
            normalization: 归一化方法
            n_channels: 通道数
            target_size: 目标尺寸
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            levels: 气压层列表（用于有level维度的变量），None表示使用所有可用的levels
        """
        self.data_path = data_path
        self.variable = variable
        self.time_slice = time_slice
        self.input_length = input_length
        self.output_length = output_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalization = normalization
        self.n_channels = n_channels
        self.target_size = target_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.levels = levels

        self.normalizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self):
        """设置数据集"""
        print(f"加载数据: {self.data_path}")

        # 加载数据
        ds = xr.open_zarr(self.data_path)

        # 时间切片
        if self.time_slice:
            start, end = self.time_slice.split(":")
            ds = ds.sel(time=slice(start, end))

        # 获取变量数据
        var_data = ds[self.variable]

        # 处理levels维度（如果有）
        if "level" in var_data.dims:
            if self.levels is not None:
                # 使用指定的levels
                print(f"使用指定的levels: {self.levels}")
                var_data = var_data.sel(level=self.levels)
            else:
                # 使用所有可用的levels
                available_levels = var_data.level.values.tolist()
                print(f"使用所有可用的levels: {available_levels}")
                self.levels = available_levels  # 保存实际使用的levels

        # 确保维度顺序为标准的 (time, latitude, longitude) 或 (time, level, latitude, longitude)
        # ERA5数据可能以 (time, longitude, latitude) 的顺序存储
        if "latitude" in var_data.dims and "longitude" in var_data.dims:
            # 检查当前维度顺序
            dims = list(var_data.dims)
            lon_idx = dims.index("longitude")
            lat_idx = dims.index("latitude")

            # 如果longitude在latitude之前，需要转置
            if lon_idx < lat_idx:
                print(f"  检测到维度顺序: {dims}")
                print(f"  转置为标准顺序 (latitude在longitude之前)")
                # 构建目标维度顺序: time, [level], latitude, longitude
                if "level" in dims:
                    target_dims = ["time", "level", "latitude", "longitude"]
                else:
                    target_dims = ["time", "latitude", "longitude"]
                var_data = var_data.transpose(*target_dims)
                print(f"  转置后维度: {list(var_data.dims)}")

        data = var_data.values  # 现在是 (Time, Lat, Lon) 或 (Time, Level, Lat, Lon)

        print(f"原始数据 shape: {data.shape}")
        print(f"数据范围: [{data.min():.2f}, {data.max():.2f}]")

        # 准备为图像格式
        data = prepare_weather_data(
            data, n_channels=self.n_channels, target_size=self.target_size
        )

        print(f"处理后 shape: {data.shape}")

        # 归一化
        from .normalization import Normalizer

        self.normalizer = Normalizer(method=self.normalization)
        data = self.normalizer.fit_transform(data, name=self.variable)

        print(f"归一化后范围: [{data.min():.2f}, {data.max():.2f}]")

        # 分割数据集
        train_data, val_data, test_data = split_dataset(
            data, train_ratio=self.train_ratio, val_ratio=self.val_ratio, shuffle=False
        )

        print(f"训练集: {len(train_data)} 样本")
        print(f"验证集: {len(val_data)} 样本")
        print(f"测试集: {len(test_data)} 样本")

        # 创建Dataset
        self.train_dataset = WeatherSequenceDataset(
            train_data, self.input_length, self.output_length
        )
        self.val_dataset = WeatherSequenceDataset(
            val_data, self.input_length, self.output_length
        )
        self.test_dataset = WeatherSequenceDataset(
            test_data, self.input_length, self.output_length
        )

        print(
            f"序列数 - 训练: {len(self.train_dataset)}, "
            f"验证: {len(self.val_dataset)}, "
            f"测试: {len(self.test_dataset)}"
        )

    def train_dataloader(self):
        """训练集DataLoader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        """验证集DataLoader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        """测试集DataLoader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_sample(self, split: str = "test", idx: int = 0):
        """获取单个样本

        Args:
            split: 'train', 'val', 或 'test'
            idx: 样本索引
        """
        if split == "train":
            dataset = self.train_dataset
        elif split == "val":
            dataset = self.val_dataset
        else:
            dataset = self.test_dataset

        return dataset[idx]
