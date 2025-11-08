"""
VAE重建测试 - 评估Stable Diffusion VAE对天气数据的重建能力

📋 文件作用：
    测试SD VAE能否准确地重建（encode + decode）天气网格数据。
    这是WeatherDiff流程的第一步（Step 2: E0实验），用于验证VAE是否适用于天气数据。

🔄 重建流程：
    1. 加载ERA5天气数据（64×32网格）
       ├─ 从Google Cloud Storage读取zarr格式数据
       └─ 提取指定变量（如2m_temperature）
    
    2. 数据预处理
       ├─ MinMax归一化到[-1, 1]（关键！必须匹配SD VAE的训练范围）
       ├─ 转换为3通道图像格式（复制单通道到RGB）
       └─ 分割为训练/验证/测试集
    
    3. VAE重建过程
       ├─ Encode: 原始图像 → 潜向量（压缩到1/8尺寸）
       │   输入: (B, 3, H, W)  →  输出: (B, 4, H//8, W//8)
       │
       └─ Decode: 潜向量 → 重建图像（恢复到原始尺寸）
           输入: (B, 4, H//8, W//8)  →  输出: (B, 3, H, W)
    
    4. 评估重建质量
       ├─ 计算7个指标（MAE, RMSE, PSNR, SSIM, 相关系数等）
       ├─ 归一化空间指标（[-1, 1]范围）
       └─ 原始尺度指标（物理单位，开尔文K）
    
    5. 生成可视化
       ├─ 简单对比图（5个样本的原始/重建/误差）
       └─ 世界地图对比（Ground Truth vs Reconstruction）

📊 输出结果：
    - vae_reconstruction_results.json: 详细的评估指标
    - reconstruction_comparison.png: 简单对比图
    - spatial_reconstruction_comparison.png: 世界地图对比（推荐查看）
    - samples_original.npy: 原始样本数据
    - samples_reconstructed.npy: 重建样本数据

🎯 验收标准：
    - RMSE < 10K 且 相关系数 > 0.9: 重建质量良好，可以继续 ✅
    - RMSE < 15K: 重建质量一般，建议微调或训练自定义VAE
    - RMSE > 15K: 重建质量差，强烈建议训练自定义VAE或改用像素空间建模

📖 使用方法:
    # 快速测试（1个月数据，10个样本）
    python test_vae_reconstruction.py --time-slice 2020-01-01:2020-01-31 --n-test-samples 10
    
📚 相关文档:
    - outputs/vae_reconstruction/VAE.md: 评估指标详解
"""

import argparse
import torch
import numpy as np
from pathlib import Path

from weatherdiff.vae import SDVAEWrapper, test_vae_reconstruction
from weatherdiff.utils import WeatherDataModule


def main():
    parser = argparse.ArgumentParser(description='测试VAE重建能力')
    
    # 数据参数
    parser.add_argument('--data-path', type=str, default='gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr',
                       help='数据文件路径')
    parser.add_argument('--variable', type=str, default='2m_temperature',
                       help='变量名')
    parser.add_argument('--time-slice', type=str, default=None,
                       help='时间切片，格式: 2020-01-01:2020-12-31')
    
    # VAE参数
    parser.add_argument('--model-id', type=str, 
                       default='stable-diffusion-v1-5/stable-diffusion-v1-5',
                       help='HuggingFace模型ID')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='设备')
    
    # 数据处理参数
    parser.add_argument('--normalization', type=str, default='minmax',
                       choices=['minmax', 'zscore'],
                       help='归一化方法')
    parser.add_argument('--n-channels', type=int, default=3,
                       help='通道数（1或3）')
    parser.add_argument('--target-size', type=str, default=None,
                       help='目标尺寸，格式: 512,512')
    
    # 测试参数
    parser.add_argument('--n-test-samples', type=int, default=100,
                       help='测试样本数量')
    parser.add_argument('--output-dir', type=str, default='outputs/vae_reconstruction',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 解析target_size
    target_size = None
    if args.target_size:
        h, w = map(int, args.target_size.split(','))
        target_size = (h, w)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("VAE重建测试 - E0实验: 直接使用SD VAE")
    print("=" * 80)
    print(f"数据路径: {args.data_path}")
    print(f"变量: {args.variable}")
    print(f"时间切片: {args.time_slice or '全部'}")
    print(f"归一化: {args.normalization}")
    print(f"通道数: {args.n_channels}")
    print(f"目标尺寸: {target_size or '保持原尺寸'}")
    print(f"设备: {args.device}")
    
    # 加载数据
    print("\n" + "-" * 80)
    print("Step 1: 加载和预处理数据")
    print("-" * 80)
    
    data_module = WeatherDataModule(
        data_path=args.data_path,
        variable=args.variable,
        time_slice=args.time_slice,
        input_length=1,  # 只需要单帧测试重建
        output_length=0,
        batch_size=1,
        normalization=args.normalization,
        n_channels=args.n_channels,
        target_size=target_size
    )
    
    data_module.setup()
    
    # 获取测试数据
    test_data = data_module.test_dataset.data[:args.n_test_samples]
    test_data = torch.from_numpy(test_data).float()
    
    print(f"\n测试数据 shape: {test_data.shape}")
    print(f"数据范围: [{test_data.min():.2f}, {test_data.max():.2f}]")
    
    # 加载VAE
    print("\n" + "-" * 80)
    print("Step 2: 加载Stable Diffusion VAE")
    print("-" * 80)
    
    vae_wrapper = SDVAEWrapper(
        model_id=args.model_id,
        device=args.device,
        dtype=torch.float32
    )
    
    # 测试重建
    print("\n" + "-" * 80)
    print("Step 3: 测试重建能力")
    print("-" * 80)
    
    save_path = output_dir / 'vae_reconstruction_results.json'
    
    metrics = test_vae_reconstruction(
        vae_wrapper=vae_wrapper,
        test_data=test_data,
        normalizer=data_module.normalizer,
        variable=args.variable,
        save_path=str(save_path)
    )
    
    # 保存一些可视化样本
    print("\n" + "-" * 80)
    print("Step 4: 保存可视化样本")
    print("-" * 80)
    
    n_vis_samples = min(5, len(test_data))
    vis_samples = test_data[:n_vis_samples]
    
    with torch.no_grad():
        vis_recons = vae_wrapper.reconstruct(vis_samples)
    
    # 反归一化用于可视化
    vis_samples_orig = data_module.normalizer.inverse_transform(
        vis_samples.numpy(), name=args.variable
    )
    vis_recons_orig = data_module.normalizer.inverse_transform(
        vis_recons.cpu().numpy(), name=args.variable
    )
    
    # 保存numpy数组
    np.save(output_dir / 'samples_original.npy', vis_samples_orig)
    np.save(output_dir / 'samples_reconstructed.npy', vis_recons_orig)
    
    print(f"✓ 原始样本保存到: {output_dir / 'samples_original.npy'}")
    print(f"✓ 重建样本保存到: {output_dir / 'samples_reconstructed.npy'}")
    
    # 创建简单对比图
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(n_vis_samples, 3, figsize=(12, 4 * n_vis_samples))
        if n_vis_samples == 1:
            axes = axes[np.newaxis, :]
        
        for i in range(n_vis_samples):
            # 原始
            axes[i, 0].imshow(vis_samples_orig[i, 0], cmap='RdBu_r')
            axes[i, 0].set_title(f'Sample {i} - Original')
            axes[i, 0].axis('off')
            
            # 重建
            axes[i, 1].imshow(vis_recons_orig[i, 0], cmap='RdBu_r')
            axes[i, 1].set_title(f'Sample {i} - Reconstructed')
            axes[i, 1].axis('off')
            
            # 误差
            error = vis_recons_orig[i, 0] - vis_samples_orig[i, 0]
            im = axes[i, 2].imshow(error, cmap='RdBu_r', 
                                  vmin=-np.abs(error).max(), 
                                  vmax=np.abs(error).max())
            axes[i, 2].set_title(f'Sample {i} - Error (MAE={np.abs(error).mean():.2f})')
            axes[i, 2].axis('off')
            plt.colorbar(im, ax=axes[i, 2])
        
        plt.tight_layout()
        plt.savefig(output_dir / 'reconstruction_comparison.png', dpi=150, bbox_inches='tight')
        print(f"✓ 对比图保存到: {output_dir / 'reconstruction_comparison.png'}")
        plt.close()
    except Exception as e:
        print(f"⚠ 无法生成可视化: {e}")
    
    # 创建世界地图对比
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        
        print("\n生成世界地图对比...")
        
        # 选择一个样本进行世界地图可视化
        sample_idx = 0
        data_true = vis_samples_orig[sample_idx, 0]  # Shape: (H, W)
        data_recon = vis_recons_orig[sample_idx, 0]
        
        H, W = data_true.shape
        
        # 创建经纬度网格 (ERA5数据: 64x32)
        lat = np.linspace(90, -90, H)  # 从北到南
        lon = np.linspace(0, 360, W)  # 从0到360度
        
        # 转换经度到 -180 到 180
        lon_converted = np.where(lon > 180, lon - 360, lon)
        lon_sort_idx = np.argsort(lon_converted)
        lon = lon_converted[lon_sort_idx]
        
        # 重新排列数据
        data_true = data_true[:, lon_sort_idx]
        data_recon = data_recon[:, lon_sort_idx]
        
        # 创建网格
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        
        # 统一颜色范围
        vmin = min(data_true.min(), data_recon.min())
        vmax = max(data_true.max(), data_recon.max())
        
        # 创建图形 (2列: Ground Truth vs Reconstruction)
        fig = plt.figure(figsize=(24, 10))
        
        # ========== 左图: Ground Truth ==========
        ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
        
        im1 = ax1.contourf(lon_grid, lat_grid, data_true,
                          levels=20, cmap='RdYlBu_r',
                          vmin=vmin, vmax=vmax,
                          transform=ccrs.PlateCarree())
        
        ax1.coastlines(linewidth=0.5)
        ax1.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
        ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                     linewidth=0.5, alpha=0.5)
        
        ax1.set_title('Ground Truth', fontsize=16, fontweight='bold', pad=10)
        
        cbar1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar1.set_label('Temperature (K)', fontsize=12)
        
        # ========== 右图: Reconstruction ==========
        ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
        
        im2 = ax2.contourf(lon_grid, lat_grid, data_recon,
                          levels=20, cmap='RdYlBu_r',
                          vmin=vmin, vmax=vmax,
                          transform=ccrs.PlateCarree())
        
        ax2.coastlines(linewidth=0.5)
        ax2.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
        ax2.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                     linewidth=0.5, alpha=0.5)
        
        ax2.set_title('Reconstruction', fontsize=16, fontweight='bold', pad=10)
        
        cbar2 = plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar2.set_label('Temperature (K)', fontsize=12)
        
        # 计算误差统计
        error = data_recon - data_true
        rmse = np.sqrt(np.mean(error**2))
        mae = np.mean(np.abs(error))
        
        # 添加总标题
        fig.suptitle(
            f'{args.variable} - VAE Reconstruction Comparison\n' +
            f'Sample {sample_idx} | RMSE: {rmse:.2f} K | MAE: {mae:.2f} K',
            fontsize=18, fontweight='bold', y=0.98
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(output_dir / 'spatial_reconstruction_comparison.png', 
                   dpi=300, bbox_inches='tight')
        print(f"✓ 世界地图对比保存到: {output_dir / 'spatial_reconstruction_comparison.png'}")
        plt.close()
        
    except ImportError:
        print("⚠ 需要安装cartopy才能生成世界地图: pip install cartopy")
    except Exception as e:
        print(f"⚠ 无法生成世界地图: {e}")
    
    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)
    print(f"\n总结:")
    print(f"  RMSE: {metrics['rmse']:.4f} (原始尺度)")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  SSIM: {metrics['ssim']:.4f}")
    print(f"  相关系数: {metrics['correlation']:.4f}")
    
    # 给出建议
    print("\n下一步建议:")
    if metrics['rmse'] < 5.0 and metrics['correlation'] > 0.9:
        print("  ✓ VAE重建质量良好，可以用于后续步骤")
        print("  → 继续 Step 3: 训练图像到图像预测模型")
    elif metrics['rmse'] < 10.0:
        print("  ⚠ VAE重建质量一般，建议考虑:")
        print("    1. 微调VAE (E1实验)")
        print("    2. 训练自定义VAE (E2实验)")
    else:
        print("  ✗ VAE重建质量较差，强烈建议:")
        print("    1. 训练自定义VAE (E2实验)")
        print("    2. 或直接在像素空间建模")


if __name__ == '__main__':
    main()

