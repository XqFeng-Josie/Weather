#!/usr/bin/env python3
"""S-VAE 天气数据训练脚本"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import xarray as xr

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform


class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm1 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.activation = nn.SiLU()
        
        # 如果输入输出通道数或尺寸不同，需要投影
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.activation(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.activation(out + residual)
        return out


class WeatherVAE(nn.Module):
    """天气数据VAE模型
    
    支持两种分布：标准VAE（高斯分布）和S-VAE（超球面分布）
    """
    
    def __init__(
        self,
        spatial_shape=(64, 32),
        n_channels=1,
        latent_channels=4,  # 潜在空间通道数（类似SD VAE）
        hidden_dims=(64, 128, 256, 512),  # 编码器/解码器通道数
        distribution='normal',  # 'normal' 或 'vmf'
        use_residual=True,
    ):
        """
        Args:
            spatial_shape: 空间维度 (Lat, Lon)
            n_channels: 输入通道数
            latent_channels: 潜在空间通道数（空间维度保持不变）
            hidden_dims: 编码器/解码器的隐藏层通道数列表
            distribution: 潜在分布类型 'normal' 或 'vmf'
            use_residual: 是否使用残差连接
        """
        super().__init__()
        
        self.spatial_shape = spatial_shape
        self.n_channels = n_channels
        self.latent_channels = latent_channels
        self.hidden_dims = hidden_dims
        self.distribution = distribution
        self.use_residual = use_residual
        
        H, W = spatial_shape
        self.down_factor = 2 ** len(hidden_dims)
        self.latent_h = H // self.down_factor
        self.latent_w = W // self.down_factor
        
        # 编码器
        encoder_layers = []
        in_channels = n_channels
        
        for i, out_channels in enumerate(hidden_dims):
            if use_residual and i > 0:
                encoder_layers.append(ResidualBlock(in_channels, out_channels, stride=2))
            else:
                encoder_layers.extend([
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(min(32, out_channels), out_channels),
                    nn.SiLU(),
                ])
            in_channels = out_channels
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 潜在空间映射层
        if distribution == 'normal':
            self.conv_mu = nn.Conv2d(hidden_dims[-1], latent_channels, kernel_size=3, padding=1)
            self.conv_logvar = nn.Conv2d(hidden_dims[-1], latent_channels, kernel_size=3, padding=1)
        else:
            self.conv_mean = nn.Conv2d(hidden_dims[-1], latent_channels, kernel_size=3, padding=1)
            self.conv_kappa = nn.Conv2d(hidden_dims[-1], 1, kernel_size=3, padding=1)
        
        # 解码器
        self.proj = nn.Conv2d(latent_channels, hidden_dims[-1], kernel_size=3, padding=1)
        
        decoder_layers = []
        ch = hidden_dims[-1]
        decoder_channels = list(reversed(hidden_dims[:-1]))
        
        for i, out_channels in enumerate(decoder_channels):
            if use_residual:
                decoder_layers.append(ResidualBlock(ch, ch, stride=1))
                decoder_layers.append(nn.ConvTranspose2d(ch, out_channels, 
                                                         kernel_size=4, stride=2, padding=1))
                decoder_layers.append(nn.GroupNorm(min(32, out_channels), out_channels))
                decoder_layers.append(nn.SiLU())
            else:
                decoder_layers.extend([
                    nn.ConvTranspose2d(ch, out_channels, kernel_size=4, stride=2, padding=1),
                    nn.GroupNorm(min(32, out_channels), out_channels),
                    nn.SiLU(),
                ])
            ch = out_channels
        
        if use_residual:
            decoder_layers.append(ResidualBlock(ch, ch, stride=1))
            decoder_layers.append(nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1))
            decoder_layers.append(nn.GroupNorm(min(32, ch), ch))
            decoder_layers.append(nn.SiLU())
        else:
            decoder_layers.extend([
                nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(min(32, ch), ch),
                nn.SiLU(),
            ])
        
        decoder_layers.append(nn.Conv2d(ch, n_channels, kernel_size=3, padding=1))
        decoder_layers.append(nn.Tanh())
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        h = self.encoder(x)
        
        if self.distribution == 'normal':
            mu = self.conv_mu(h)
            logvar = self.conv_logvar(h)
            return mu, logvar
        else:  # vmf
            mean = self.conv_mean(h)  # (B, latent_channels, H//8, W//8)
            # 归一化到单位球面（对每个空间位置）
            mean = mean / (mean.norm(dim=1, keepdim=True) + 1e-8)
            kappa = F.softplus(self.conv_kappa(h)) + 1  # 浓度参数，确保>0
            return mean, kappa
    
    def reparameterize(self, z_mean, z_var_or_kappa):
        if self.distribution == 'normal':
            std = torch.exp(0.5 * z_var_or_kappa)
            eps = torch.randn_like(std)
            z = z_mean + eps * std
            
            q_z = torch.distributions.Normal(z_mean, std)
            p_z = torch.distributions.Normal(
                torch.zeros_like(z_mean),
                torch.ones_like(z_var_or_kappa)
            )
            kl = torch.distributions.kl.kl_divergence(q_z, p_z)
            kl = kl.mean()
            return z, kl
        else:
            B, C, H, W = z_mean.shape
            z_list = []
            kl_list = []
            
            for b in range(B):
                mean_b = z_mean[b]
                kappa_b = z_var_or_kappa[b, 0]
                mean_flat = mean_b.view(C, -1).transpose(0, 1)
                kappa_flat = kappa_b.view(-1)
                
                z_positions = []
                kl_positions = []
                
                for i in range(H * W):
                    mean_i = mean_flat[i:i+1]
                    kappa_i = kappa_flat[i:i+1].view(1, 1)
                    
                    q_z_i = VonMisesFisher(mean_i, kappa_i)
                    p_z_i = HypersphericalUniform(C - 1, device=mean_i.device)
                    z_i = q_z_i.rsample()
                    kl_i = torch.distributions.kl.kl_divergence(q_z_i, p_z_i)
                    if kl_i.dim() > 0:
                        kl_i = kl_i.mean()
                    
                    z_positions.append(z_i)
                    kl_positions.append(kl_i)
                
                z_b = torch.cat(z_positions, dim=0)
                z_b = z_b.transpose(0, 1).view(C, H, W)
                z_list.append(z_b)
                kl_list.append(torch.stack(kl_positions).mean())
            
            z = torch.stack(z_list, dim=0)
            kl = torch.stack(kl_list).mean()
            return z, kl
    
    def decode(self, z):
        h = self.proj(z)
        x_recon = self.decoder(h)
        return x_recon
    
    def forward(self, x):
        z_mean, z_var_or_kappa = self.encode(x)
        z, kl = self.reparameterize(z_mean, z_var_or_kappa)
        x_recon = self.decode(z)
        return x_recon, z_mean, z_var_or_kappa, kl


class WeatherGridDataset(Dataset):
        """天气网格数据集"""
        def __init__(self, data_path, variable, time_slice=None, normalize=True, 
                     norm_stats=None, levels=None, augment=False):
            print(f"加载数据: {data_path}")
            self.ds = xr.open_zarr(data_path)
            if time_slice:
                start, end = time_slice.split(":")
                self.ds = self.ds.sel(time=slice(start, end))
            var_data = self.ds[variable]
            if "level" in var_data.dims:
                if levels is not None:
                    var_data = var_data.sel(level=levels)
                else:
                    self.levels = var_data.level.values.tolist()
            else:
                self.levels = None
            if "latitude" in var_data.dims and "longitude" in var_data.dims:
                dims = list(var_data.dims)
                if "level" in dims:
                    target_dims = ["time", "level", "latitude", "longitude"]
                else:
                    target_dims = ["time", "latitude", "longitude"]
                var_data = var_data.transpose(*target_dims)
            self.data = var_data.values
            if len(self.data.shape) == 4:
                self.data = np.transpose(self.data, (0, 2, 3, 1))
                self.n_levels = self.data.shape[-1]
            else:
                self.data = self.data[:, :, :, np.newaxis]
                self.n_levels = 1
            self.normalize = normalize
            if normalize:
                if norm_stats is None:
                    data_flat = self.data.flatten()
                    self.norm_stats = {
                        'mean': float(data_flat.mean()),
                        'std': float(data_flat.std()),
                        'min': float(data_flat.min()),
                        'max': float(data_flat.max())
                    }
                else:
                    self.norm_stats = norm_stats
            else:
                self.norm_stats = None
            self.augment = augment
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            sample = self.data[idx].astype(np.float32)
            if self.augment and np.random.random() < 0.5:
                sample = np.flip(sample, axis=1)
            if self.normalize and self.norm_stats:
                mean = self.norm_stats['mean']
                std = self.norm_stats['std']
                sample = (sample - mean) / (std + 1e-8)
                sample = np.clip(sample / 3.0, -1.0, 1.0)
            sample = np.transpose(sample, (2, 0, 1))
            return torch.from_numpy(sample).float()
        
        def get_norm_stats(self):
            return self.norm_stats
        
        def get_spatial_shape(self):
            return self.data.shape[1:3]
        
        def get_n_channels(self):
            return self.n_levels


def compute_perceptual_loss(x_recon, x_target):
    if x_recon.shape != x_target.shape:
        x_recon = F.interpolate(x_recon, size=x_target.shape[2:], mode='bilinear', align_corners=False)
    loss_l1 = F.l1_loss(x_recon, x_target, reduction='mean')
    loss_l2 = F.mse_loss(x_recon, x_target, reduction='mean')
    return loss_l1 + loss_l2


def compute_gradient_loss(x_recon, x_target):
    if x_recon.shape != x_target.shape:
        x_recon = F.interpolate(x_recon, size=x_target.shape[2:], mode='bilinear', align_corners=False)
    
    def gradient_x(img):
        return img[:, :, :, :-1] - img[:, :, :, 1:]
    
    def gradient_y(img):
        return img[:, :, :-1, :] - img[:, :, 1:, :]
    
    grad_x_recon = gradient_x(x_recon)
    grad_x_target = gradient_x(x_target)
    grad_y_recon = gradient_y(x_recon)
    grad_y_target = gradient_y(x_target)
    
    loss_x = F.mse_loss(grad_x_recon, grad_x_target)
    loss_y = F.mse_loss(grad_y_recon, grad_y_target)
    
    return loss_x + loss_y


def train(model, train_loader, optimizer, device, epoch, kl_weight=1.0,
          use_advanced_loss=False, grad_loss_weight=0.1, perceptual_weight=1.0):
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for i, x_mb in enumerate(pbar):
        x_mb = x_mb.to(device)
        
        optimizer.zero_grad()
        
        x_recon, z_mean, z_var_or_kappa, kl = model(x_mb)
        
        if use_advanced_loss:
            recon_loss = perceptual_weight * compute_perceptual_loss(x_recon, x_mb)
            recon_loss += grad_loss_weight * compute_gradient_loss(x_recon, x_mb)
        else:
            recon_loss = F.mse_loss(x_recon, x_mb, reduction='mean')
        
        kl_loss = kl
        loss = recon_loss + kl_weight * kl_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'kl': f'{kl_loss.item():.4f}',
            'kl_w': f'{kl_weight:.3f}'
        })
    
    return {
        'loss': total_loss / len(train_loader),
        'recon_loss': total_recon_loss / len(train_loader),
        'kl_loss': total_kl_loss / len(train_loader)
    }


def test(model, test_loader, device):
    model.eval()
    results = defaultdict(list)
    
    with torch.no_grad():
        for x_mb in tqdm(test_loader, desc="Testing"):
            x_mb = x_mb.to(device)
            
            x_recon, z_mean, z_var_or_kappa, kl = model(x_mb)
            
            recon_loss = F.mse_loss(x_recon, x_mb, reduction='mean')
            results['recon_loss'].append(recon_loss.item())
            results['kl_loss'].append(kl.item())
            results['ELBO'].append(-recon_loss.item() - kl.item())
    
    avg_results = {k: np.mean(v) for k, v in results.items()}
    print(f"Test Results: {avg_results}")
    return avg_results


def main():
    parser = argparse.ArgumentParser(description="训练改进版天气网格数据VAE模型")
    
    # 数据参数
    parser.add_argument('--data-path', type=str, required=True,
                       help='zarr数据路径')
    parser.add_argument('--variable', type=str, default='2m_temperature',
                       help='变量名（如 2m_temperature）')
    parser.add_argument('--time-slice', type=str, default=None,
                       help='时间切片，格式: 2020-01-01:2020-12-31')
    parser.add_argument('--levels', type=int, nargs='+', default=None,
                       help='如果变量有level维度，指定要使用的levels')
    parser.add_argument('--train-split', type=float, default=0.8,
                       help='训练集比例')
    parser.add_argument('--normalize', action='store_true', default=True,
                       help='是否归一化数据')
    parser.add_argument('--augment', action='store_true',
                       help='是否启用数据增强（随机水平翻转）')
    
    # 模型参数
    parser.add_argument('--latent-channels', type=int, default=4,
                       help='潜在空间通道数（类似SD VAE）')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[64, 128, 256, 512],
                       help='编码器/解码器隐藏层通道数列表')
    parser.add_argument('--distribution', type=str, default='normal',
                       choices=['normal', 'vmf'],
                       help='潜在分布类型: normal (标准VAE) 或 vmf (S-VAE)')
    parser.add_argument('--use-residual', action='store_true', default=True,
                       help='是否使用残差连接')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='初始学习率')
    parser.add_argument('--device', type=str, default=None,
                       help='设备 (cuda/cuda:0/cuda:1/cpu)，默认自动选择')
    
    # 学习率调度器参数
    parser.add_argument('--lr-scheduler', type=str, default='plateau',
                       choices=['cosine', 'plateau', 'step', 'exponential', None],
                       help='学习率调度器类型')
    parser.add_argument('--lr-scheduler-params', type=str, default=None,
                       help='学习率调度器参数字典（JSON格式）')
    parser.add_argument('--kl-weight', type=float, default=1e-6,
                       help='KL散度损失权重')
    parser.add_argument('--kl-annealing', action='store_true',
                       help='是否使用KL散度退火')
    
    # 损失函数参数
    parser.add_argument('--use-advanced-loss', action='store_true', default=True,
                       help='是否使用改进的损失函数')
    parser.add_argument('--grad-loss-weight', type=float, default=0.1,
                       help='梯度损失权重')
    parser.add_argument('--perceptual-weight', type=float, default=1.0,
                       help='感知损失权重')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, default='outputs/svae_improved',
                       help='输出目录')
    parser.add_argument('--save-model', action='store_true',
                       help='是否保存模型')
    parser.add_argument('--resume', type=str, default=None,
                       help='从checkpoint继续训练的路径')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设备选择
    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"自动选择设备: {device}")
        else:
            device = torch.device('cpu')
            print(f"自动选择设备: {device}")
    else:
        device = torch.device(args.device)
        print(f"使用指定设备: {device}")
    
    # 加载数据集
    print(f"\n加载数据集...")
    full_dataset = WeatherGridDataset(
        args.data_path,
        variable=args.variable,
        time_slice=args.time_slice,
        normalize=args.normalize,
        levels=args.levels,
        augment=args.augment
    )
    
    spatial_shape = full_dataset.get_spatial_shape()
    n_channels = full_dataset.get_n_channels()
    norm_stats = full_dataset.get_norm_stats()
    
    print(f"空间维度: {spatial_shape}")
    print(f"通道数: {n_channels}")
    print(f"数据集大小: {len(full_dataset)}")
    
    # 划分训练集和测试集
    train_size = int(args.train_split * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"训练集: {len(train_dataset)} 样本, 测试集: {len(test_dataset)} 样本")
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # 创建模型
    model = WeatherVAE(
        spatial_shape=spatial_shape,
        n_channels=n_channels,
        latent_channels=args.latent_channels,
        hidden_dims=tuple(args.hidden_dims),
        distribution=args.distribution,
        use_residual=args.use_residual
    ).to(device)
    
    print(f"\n模型参数:")
    print(f"  空间维度: {spatial_shape}")
    print(f"  通道数: {n_channels}")
    print(f"  潜在通道数: {args.latent_channels}")
    print(f"  隐藏层维度: {args.hidden_dims}")
    print(f"  分布类型: {args.distribution}")
    print(f"  下采样因子: {model.down_factor}×")
    print(f"  潜在空间尺寸: {model.latent_h}×{model.latent_w}")
    print(f"  使用残差连接: {args.use_residual}")
    print(f"  总参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = None
    scheduler_type = args.lr_scheduler
    if scheduler_type is not None:
        scheduler_params = {}
        if args.lr_scheduler_params:
            try:
                scheduler_params = json.loads(args.lr_scheduler_params)
            except json.JSONDecodeError:
                print(f"警告: 无法解析lr-scheduler-params，使用默认参数")
        
        if scheduler_type == 'plateau':
            mode = scheduler_params.get('mode', 'min')
            factor = scheduler_params.get('factor', 0.5)
            patience = scheduler_params.get('patience', 10)
            min_lr = scheduler_params.get('min_lr', 1e-6)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=mode, factor=factor, patience=patience, min_lr=min_lr
            )
            print(f"使用 ReduceLROnPlateau 调度器")
        elif scheduler_type == 'cosine':
            T_max = scheduler_params.get('T_max', args.epochs)
            eta_min = scheduler_params.get('eta_min', 0.0)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
            print(f"使用 CosineAnnealingLR 调度器")
    
    # KL散度退火
    kl_annealing = args.kl_annealing
    kl_weight = args.kl_weight
    if kl_annealing:
        print(f"启用KL散度退火: 从0.0逐步增加到{kl_weight}")
    
    # 保存配置
    config = {
        'spatial_shape': spatial_shape,
        'n_channels': n_channels,
        'latent_channels': args.latent_channels,
        'hidden_dims': args.hidden_dims,
        'distribution': args.distribution,
        'use_residual': args.use_residual,
        'norm_stats': norm_stats,
        'args': vars(args)
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"保存配置到: {output_dir / 'config.json'}")
    
    best_model_path = output_dir / "best_model.pth"
    latest_checkpoint_path = output_dir / "checkpoint_latest.pth"
    start_epoch = 1
    best_test_loss = float('inf')
    train_history = []
    last_test_results = None
    
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            if checkpoint.get('optimizer_state_dict') is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and checkpoint.get('scheduler_state_dict') is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            best_test_loss = checkpoint.get('best_test_loss', best_test_loss)
            train_history = checkpoint.get('train_history', [])
            last_test_results = checkpoint.get('test_results') or checkpoint.get('last_test_results')
            start_epoch = checkpoint.get('epoch', 0) + 1
            print(f"从 {resume_path} 恢复训练，起始epoch: {start_epoch}")
        else:
            print(f"警告: 无法找到resume文件 {resume_path}，将从头开始训练")
    
    print(f"\n开始训练...")
    print("=" * 80)
    
    for epoch in range(start_epoch, args.epochs + 1):
        if kl_annealing:
            current_kl_weight = min(kl_weight, kl_weight * (epoch / args.epochs))
        else:
            current_kl_weight = kl_weight
        
        train_results = train(
            model, train_loader, optimizer, device, epoch,
            kl_weight=current_kl_weight,
            use_advanced_loss=args.use_advanced_loss,
            grad_loss_weight=args.grad_loss_weight,
            perceptual_weight=args.perceptual_weight
        )
        
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            if scheduler_type == 'plateau':
                scheduler.step(train_results['loss'])
            else:
                scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != current_lr:
                print(f"学习率更新: {current_lr:.6f} -> {new_lr:.6f}")
        
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"\nEpoch {epoch} 测试结果:")
            test_results = test(model, test_loader, device)
            last_test_results = test_results
            
            train_history.append({
                'epoch': epoch,
                'train_loss': train_results['loss'],
                'train_recon_loss': train_results['recon_loss'],
                'train_kl_loss': train_results['kl_loss'],
                'test_recon_loss': test_results['recon_loss'],
                'test_kl_loss': test_results['kl_loss'],
                'test_ELBO': test_results.get('ELBO', 0),
                'lr': optimizer.param_groups[0]['lr'],
                'kl_weight': current_kl_weight
            })
            
            if args.save_model and test_results['recon_loss'] < best_test_loss:
                best_test_loss = test_results['recon_loss']
                if best_model_path.exists():
                    best_model_path.unlink()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'test_results': test_results,
                    'config': config,
                    'best_test_loss': best_test_loss,
                    'train_history': train_history
                }, best_model_path)
                print(f"保存最佳模型 (epoch {epoch}, loss={best_test_loss:.6f})")
        
        if args.save_model:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'test_results': last_test_results,
                'config': config,
                'best_test_loss': best_test_loss,
                'train_history': train_history
            }, latest_checkpoint_path)
        
        print("-" * 80)
    
    history_path = output_dir / 'train_history.json'
    with open(history_path, 'w') as f:
        json.dump(train_history, f, indent=2)
    print(f"\n训练历史已保存到: {history_path}")
    print("\n训练完成！")


if __name__ == '__main__':
    main()

