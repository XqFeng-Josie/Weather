"""
Lazy Loading Dataset - 按需加载预处理数据

优点：
    - 内存占用极小（只加载当前batch需要的数据）
    - 支持任意大小的数据集
    - IO优化（使用内存映射）
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from typing import Tuple, Optional


class LazyWeatherDataset(Dataset):
    """
    从预处理的内存映射文件中按需加载数据
    
    内存占用：O(1) - 只加载当前样本，不加载全部数据
    """
    
    def __init__(self,
                 preprocessed_dir: str,
                 input_length: int = 12,
                 output_length: int = 4,
                 stride: int = 1,
                 start_idx: int = 0,
                 end_idx: Optional[int] = None):
        """
        Args:
            preprocessed_dir: 预处理数据目录
            input_length: 输入序列长度
            output_length: 输出序列长度
            stride: 滑动窗口步长
            start_idx: 数据起始索引（用于train/val/test分割）
            end_idx: 数据结束索引
        """
        self.preprocessed_dir = Path(preprocessed_dir)
        self.input_length = input_length
        self.output_length = output_length
        self.stride = stride
        
        # 加载元数据
        with open(self.preprocessed_dir / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        # 打开内存映射（不加载到内存）
        self.data_mmap = np.memmap(
            self.preprocessed_dir / 'data.npy',
            dtype='float32',
            mode='r',
            shape=tuple(self.metadata['shape'])
        )
        
        # 计算有效样本范围
        total_timesteps = self.metadata['n_timesteps']
        if end_idx is None:
            end_idx = total_timesteps
        
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.valid_length = end_idx - start_idx
        
        # 计算样本数
        total_length = input_length + output_length
        self.n_samples = max(0, (self.valid_length - total_length) // stride + 1)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """
        按需加载单个样本（只加载需要的时间步，不加载全部数据）
        
        Returns:
            input_seq: (T_in, C, H, W)
            output_seq: (T_out, C, H, W)
        """
        # 计算全局索引
        global_start = self.start_idx + idx * self.stride
        input_end = global_start + self.input_length
        output_end = input_end + self.output_length
        
        # 只加载需要的切片（lazy loading的关键！）
        input_seq = np.array(self.data_mmap[global_start:input_end])
        output_seq = np.array(self.data_mmap[input_end:output_end])
        
        return torch.from_numpy(input_seq), torch.from_numpy(output_seq)
    
    def get_normalization_stats(self):
        """获取归一化统计量"""
        import pickle
        stats_file = self.preprocessed_dir / 'normalizer_stats.pkl'
        if stats_file.exists():
            with open(stats_file, 'rb') as f:
                return pickle.load(f)
        return None


class LazyWeatherDataModule:
    """
    使用预处理数据的Data Module
    
    自动处理train/val/test分割，支持lazy loading
    """
    
    def __init__(self,
                 preprocessed_dir: str,
                 input_length: int = 12,
                 output_length: int = 4,
                 batch_size: int = 16,
                 num_workers: int = 4,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15):
        """
        Args:
            preprocessed_dir: 预处理数据目录
            input_length: 输入序列长度
            output_length: 输出序列长度
            batch_size: 批次大小
            num_workers: 数据加载线程数
            train_ratio: 训练集比例
            val_ratio: 验证集比例
        """
        self.preprocessed_dir = Path(preprocessed_dir)
        self.input_length = input_length
        self.output_length = output_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        
        # 加载元数据
        with open(self.preprocessed_dir / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        self.n_timesteps = self.metadata['n_timesteps']
        self.variable = self.metadata['variable']
        
        # 计算分割点
        n_train = int(self.n_timesteps * train_ratio)
        n_val = int(self.n_timesteps * val_ratio)
        
        self.train_end = n_train
        self.val_end = n_train + n_val
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self):
        """创建datasets（不加载数据到内存）"""
        print(f"设置Lazy Loading数据集")
        print(f"  总时间步数: {self.n_timesteps}")
        print(f"  变量: {self.variable}")
        print(f"  数据形状: {self.metadata['shape']}")
        
        # 创建datasets（只创建对象，不加载数据）
        self.train_dataset = LazyWeatherDataset(
            self.preprocessed_dir,
            input_length=self.input_length,
            output_length=self.output_length,
            start_idx=0,
            end_idx=self.train_end
        )
        
        self.val_dataset = LazyWeatherDataset(
            self.preprocessed_dir,
            input_length=self.input_length,
            output_length=self.output_length,
            start_idx=self.train_end,
            end_idx=self.val_end
        )
        
        self.test_dataset = LazyWeatherDataset(
            self.preprocessed_dir,
            input_length=self.input_length,
            output_length=self.output_length,
            start_idx=self.val_end,
            end_idx=self.n_timesteps
        )
        
        print(f"  训练样本: {len(self.train_dataset)}")
        print(f"  验证样本: {len(self.val_dataset)}")
        print(f"  测试样本: {len(self.test_dataset)}")
        print(f"  ✓ Lazy loading已就绪（内存占用: ~0 MB）")
    
    def train_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def get_sample(self, split: str, idx: int):
        """获取单个样本（用于测试）"""
        if split == 'train':
            return self.train_dataset[idx]
        elif split == 'val':
            return self.val_dataset[idx]
        else:
            return self.test_dataset[idx]

