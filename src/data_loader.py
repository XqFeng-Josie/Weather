"""
数据加载和预处理模块
负责从ERA5加载数据，创建训练样本
"""
import xarray as xr
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path


class WeatherDataLoader:
    """天气数据加载器"""
    
    def __init__(
        self,
        data_path: str = 'gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr',
        variables: List[str] = None,
        levels: List[int] = None,
    ):
        """
        Args:
            data_path: ERA5数据路径
            variables: 要加载的变量列表
            levels: 气压层列表
        """
        self.data_path = data_path
        self.variables = variables or ['2m_temperature', 'geopotential', '10m_u_component_of_wind', '10m_v_component_of_wind']
        self.levels = levels or [500, 700, 850]
        self.ds = None
        
    def load_data(self, time_slice: slice = None):
        """加载数据"""
        print(f"Loading data from {self.data_path}...")
        self.ds = xr.open_zarr(self.data_path)
        
        if time_slice:
            self.ds = self.ds.sel(time=time_slice)
            
        print(f"Data loaded: {self.ds.dims}")
        return self.ds
    
    def prepare_features(self, normalize: bool = True):
        """准备特征数据"""
        features = {}
        
        for var in self.variables:
            data = self.ds[var]
            
            # 如果有level维度，选择指定层
            if 'level' in data.dims:
                data = data.sel(level=self.levels)
                
            features[var] = data
            
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
        input_length: int = 4,  # 输入：过去4个时间步（24小时）
        output_length: int = 4,  # 输出：未来4个时间步（24小时）
        stride: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建时间序列样本
        
        Returns:
            X: (n_samples, input_length, n_features) 
            y: (n_samples, output_length, n_features)
        """
        # 将所有特征拼接
        feature_list = []
        for var, data in features.items():
            if len(data.shape) == 4:  # (time, level, lat, lon)
                # 展平空间和层级维度
                n_time = data.shape[0]
                data_flat = data.reshape(n_time, -1)
            elif len(data.shape) == 3:  # (time, lat, lon)
                n_time = data.shape[0]
                data_flat = data.reshape(n_time, -1)
            else:
                data_flat = data
            feature_list.append(data_flat)
        
        # 拼接所有特征: (time, total_features)
        all_features = np.concatenate([f for f in feature_list], axis=1)
        
        print(f"All features shape: {all_features.shape}")
        
        # 创建滑动窗口样本
        X, y = [], []
        for i in range(0, len(all_features) - input_length - output_length, stride):
            X.append(all_features[i:i+input_length])
            y.append(all_features[i+input_length:i+input_length+output_length])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Created {len(X)} samples")
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
            'X_train': X[:train_end],
            'y_train': y[:train_end],
            'X_val': X[train_end:val_end],
            'y_val': y[train_end:val_end],
            'X_test': X[val_end:],
            'y_test': y[val_end:],
        }


def prepare_single_point_data(
    data_path: str,
    lat_idx: int = 16,  # 中间纬度
    lon_idx: int = 32,  # 中间经度
    time_slice: slice = None,
):
    """
    准备单点预测数据（简化版，用于快速测试）
    
    Returns:
        X: (n_samples, input_length, n_features)
        y: (n_samples, output_length, n_features)
    """
    print("Preparing single-point data for quick testing...")
    
    ds = xr.open_zarr(data_path)
    if time_slice:
        ds = ds.sel(time=time_slice)
    
    # 选择单个空间点
    ds = ds.isel(latitude=lat_idx, longitude=lon_idx)
    
    # 选择关键变量
    variables = ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind']
    
    # 提取并标准化
    features = []
    for var in variables:
        data = ds[var].values
        data = (data - np.mean(data)) / (np.std(data) + 1e-8)
        features.append(data)
    
    # 拼接特征: (time, n_features)
    features = np.stack(features, axis=1)
    
    # 创建序列
    input_length = 12  # 过去72小时（12个6小时步）
    output_length = 4  # 未来24小时（4个6小时步）
    
    X, y = [], []
    for i in range(len(features) - input_length - output_length):
        X.append(features[i:i+input_length])
        y.append(features[i+input_length:i+input_length+output_length])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Single-point data prepared: X={X.shape}, y={y.shape}")
    
    return X, y


if __name__ == '__main__':
    # 测试数据加载
    loader = WeatherDataLoader()
    
    # 加载小样本测试
    ds = loader.load_data(time_slice=slice('2020-01-01', '2020-01-31'))
    
    # 准备特征
    features = loader.prepare_features(normalize=True)
    
    # 创建序列
    X, y = loader.create_sequences(features, input_length=4, output_length=4)
    
    # 划分数据
    data_splits = loader.split_data(X, y)
    
    print("\nData splits:")
    for key, arr in data_splits.items():
        print(f"  {key}: {arr.shape}")

