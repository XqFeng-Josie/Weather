"""
数据加载和预处理模块
负责从ERA5加载数据，创建训练样本
支持两种数据格式：
1. 展平格式 (flat): 用于LR, LSTM, Transformer
2. 空间格式 (spatial): 用于CNN, ConvLSTM
"""

import xarray as xr
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class WeatherDataLoader:
    """天气数据加载器"""

    def __init__(
        self,
        data_path: str = "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr",
        variables: List[str] = None,
        levels: List[int] = None,
    ):
        """
        Args:
            data_path: ERA5数据路径
            variables: 要加载的变量列表，默认["2m_temperature"]
            levels: 气压层列表（用于有level维度的变量）
        """
        self.data_path = data_path
        self.variables = variables if variables is not None else ["2m_temperature"]
        self.levels = levels or [500, 700, 850]
        self.ds = None
        self.spatial_shape = None

    def load_data(self, time_slice: slice = None):
        """加载数据"""
        print(f"Loading data from {self.data_path}...")
        self.ds = xr.open_zarr(self.data_path)
        
        # 显示数据集的时间范围
        time_start = str(self.ds.time.values[0])[:10]
        time_end = str(self.ds.time.values[-1])[:10]
        print(f"Dataset time range: {time_start} to {time_end}")

        if time_slice:
            print(f"Selecting time slice: {time_slice}")
            self.ds = self.ds.sel(time=time_slice)
            
            # 检查切片后是否有数据
            if len(self.ds.time) == 0:
                raise ValueError(
                    f"Time slice {time_slice} resulted in empty dataset! "
                    f"Available range: {time_start} to {time_end}. "
                    f"Note: Despite the filename '1959-2022', data only goes to 2021-12-31."
                )

        print(f"Data loaded: {self.ds.dims}")
        print(f"Time steps: {len(self.ds.time)}")
        return self.ds

    def prepare_features(self, normalize: bool = True):
        """准备特征数据"""
        features = {}

        for var in self.variables:
            data = self.ds[var]
            print(f"Variable {var} shape: {data.shape}")

            # 如果有level维度，选择指定层
            if "level" in data.dims:
                data = data.sel(level=self.levels)

            features[var] = data
        
        # 记录空间形状
        first_var = list(features.values())[0]
        if 'latitude' in first_var.dims and 'longitude' in first_var.dims:
            self.spatial_shape = (
                len(first_var.latitude),
                len(first_var.longitude)
            )
            print(f"Spatial shape: {self.spatial_shape}")
        
        # 转换为numpy并标准化
        if normalize:
            self.mean = {}
            self.std = {}

            for var, data in features.items():
                values = data.values
                self.mean[var] = np.nanmean(values)
                self.std[var] = np.nanstd(values)
                features[var] = (values - self.mean[var]) / (self.std[var] + 1e-8)
        else:
            features = {k: v.values for k, v in features.items()}
        return features

    def create_sequences(
        self,
        features: Dict[str, np.ndarray],
        input_length: int = 4,
        output_length: int = 4,
        stride: int = 1,
        format: str = 'flat',  # 'flat' or 'spatial'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建时间序列样本
        
        Args:
            features: 特征字典
            input_length: 输入时间步数
            output_length: 输出时间步数
            stride: 滑动窗口步长
            format: 数据格式
                'flat': (n_samples, time, features) - 用于LR/LSTM/Transformer
                'spatial': (n_samples, time, channels, H, W) - 用于CNN/ConvLSTM

        Returns:
            X, y in specified format
        """
        if format == 'flat':
            return self._create_sequences_flat(features, input_length, output_length, stride)
        elif format == 'spatial':
            return self._create_sequences_spatial(features, input_length, output_length, stride)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'flat' or 'spatial'")

    def _create_sequences_flat(
        self,
        features: Dict[str, np.ndarray],
        input_length: int,
        output_length: int,
        stride: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建展平格式的序列
        
        Returns:
            X: (n_samples, input_length, n_features)
            y: (n_samples, output_length, n_features)
        """
        # 将所有特征拼接并展平空间维度
        feature_list = []
        for var, data in features.items():
            print(f"{var} - shape: {data.shape}")
            if len(data.shape) == 4:  # (time, level, lat, lon)
                # 展平空间和层级维度
                n_time = data.shape[0]
                data_flat = data.reshape(n_time, -1)
            elif len(data.shape) == 3:  # (time, lat, lon)
                n_time = data.shape[0]
                data_flat = data.reshape(n_time, -1)
            elif len(data.shape) == 2:  # (time, features)
                data_flat = data
            else:  # (time,) - single feature
                data_flat = data.reshape(-1, 1)
            
            # 确保所有数据都是2D: (time, features)
            if len(data_flat.shape) == 1:
                data_flat = data_flat.reshape(-1, 1)
            print(f"  -> flat shape: {data_flat.shape}")
            feature_list.append(data_flat)

        # 拼接所有特征: (time, total_features)
        all_features = np.concatenate([f for f in feature_list], axis=1)

        print(f"\nTotal features shape: {all_features.shape}")

        # 创建滑动窗口样本
        X, y = [], []
        max_index = len(all_features) - input_length - output_length + 1
        for i in range(0, max_index, stride):
            X.append(all_features[i : i + input_length])
            y.append(all_features[i + input_length : i + input_length + output_length])

        if len(X) == 0:
            raise ValueError(
                f"Cannot create sequences: data length {len(all_features)} is too short "
                f"for input_length={input_length} + output_length={output_length}"
            )

        X = np.array(X)
        y = np.array(y)

        print(f"Created {len(X)} samples (flat format)")
        print(f"X shape: {X.shape}, y shape: {y.shape}")

        return X, y

    def _create_sequences_spatial(
        self,
        features: Dict[str, np.ndarray],
        input_length: int,
        output_length: int,
        stride: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建保留空间结构的序列
        
        Returns:
            X: (n_samples, input_length, n_channels, H, W)
            y: (n_samples, output_length, n_channels, H, W)
        """
        # 处理每个变量，保留空间结构
        channel_list = []
        for var, data in features.items():
            print(f"{var} - shape: {data.shape}")
            
            if len(data.shape) == 4:  # (time, level, lat, lon)
                # 将level维度当作通道
                n_time, n_level, H, W = data.shape
                # 重排为 (time, level, H, W)
                data_spatial = data
                print(f"  -> spatial shape: {data_spatial.shape} ({n_level} channels)")
                
                # 将每个level当作一个通道
                for level_idx in range(n_level):
                    channel_list.append(data_spatial[:, level_idx, :, :])
                    
            elif len(data.shape) == 3:  # (time, lat, lon)
                # 单通道
                print(f"  -> spatial shape: {data.shape} (1 channel)")
                channel_list.append(data)
            else:
                raise ValueError(f"Cannot handle shape {data.shape} for spatial format")
        
        # 堆叠为 (time, n_channels, H, W)
        all_channels = np.stack(channel_list, axis=1)
        n_time, n_channels, H, W = all_channels.shape
        
        print(f"\nTotal spatial shape: {all_channels.shape}")
        print(f"  {n_channels} channels, {H}x{W} spatial grid")
        
        # 创建滑动窗口样本
        X, y = [], []
        max_index = n_time - input_length - output_length + 1
        
        for i in range(0, max_index, stride):
            X.append(all_channels[i : i + input_length])
            y.append(all_channels[i + input_length : i + input_length + output_length])
        
        if len(X) == 0:
            raise ValueError(
                f"Cannot create sequences: data length {n_time} is too short "
                f"for input_length={input_length} + output_length={output_length}"
            )
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Created {len(X)} samples (spatial format)")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        
        return X, y

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Dict[str, np.ndarray]:
        """划分训练/验证/测试集（按时间顺序）"""
        n_samples = len(X)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))

        return {
            "X_train": X[:train_end],
            "y_train": y[:train_end],
            "X_val": X[train_end:val_end],
            "y_val": y[train_end:val_end],
            "X_test": X[val_end:],
            "y_test": y[val_end:],
        }

    def compute_climatology(self, variables: List[str] = None) -> Dict[str, np.ndarray]:
        """
        计算气候态（多年平均），用于WeatherBench2 ACC指标计算
        
        Returns:
            Dict with climatology mean for each variable
        """
        if self.ds is None:
            raise ValueError("Must load data first using load_data()")
        
        variables = variables or self.variables
        climatology = {}
        
        for var in variables:
            if var in self.ds:
                # 按照day of year和hour分组计算平均
                clim_data = self.ds[var].groupby('time.dayofyear').mean()
                climatology[var] = clim_data.values
                
        return climatology


if __name__ == "__main__":
    # 测试数据加载
    print("="*60)
    print("Testing WeatherDataLoader")
    print("="*60)
    
    loader = WeatherDataLoader()

    # 加载小样本测试
    ds = loader.load_data(time_slice=slice("2020-01-01", "2020-01-31"))

    # 准备特征
    features = loader.prepare_features(normalize=True)
    print(f"\nFeatures keys: {features.keys()}")
    
    # 测试展平格式
    print("\n" + "="*60)
    print("Testing FLAT format (for LR/LSTM/Transformer)")
    print("="*60)
    X_flat, y_flat = loader.create_sequences(
        features, 
        input_length=4, 
        output_length=4,
        format='flat'
    )
    print(f"X shape: {X_flat.shape}, y shape: {y_flat.shape}")

    # 测试空间格式
    print("\n" + "="*60)
    print("Testing SPATIAL format (for CNN/ConvLSTM)")
    print("="*60)
    X_spatial, y_spatial = loader.create_sequences(
        features,
        input_length=4,
        output_length=4,
        format='spatial'
    )
    print(f"X shape: {X_spatial.shape}, y shape: {y_spatial.shape}")

    # 划分数据
    print("\n" + "="*60)
    print("Testing data splits")
    print("="*60)
    data_splits = loader.split_data(X_flat, y_flat)
    print("\nFlat format splits:")
    for key, arr in data_splits.items():
        print(f"  {key}: {arr.shape}")
