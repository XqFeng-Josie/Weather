"""
训练潜空间U-Net（Step 4: 潜空间预测）

使用方法:
    python train_latent_unet.py --data-path /path/to/data.zarr --variable 2m_temperature
"""

import argparse
import torch
from pathlib import Path

from weatherdiff.unet import LatentUNet, UNetTrainer
from weatherdiff.vae import SDVAEWrapper
from weatherdiff.utils import WeatherDataModule


class LatentUNetTrainer(UNetTrainer):
    """扩展训练器以支持潜空间训练"""
    
    def __init__(self, model, vae_wrapper, vae_batch_size=4, **kwargs):
        """
        Args:
            model: U-Net模型
            vae_wrapper: VAE包装器
            vae_batch_size: VAE编码时的子批次大小（用于控制显存）
            **kwargs: 其他参数传递给UNetTrainer
        """
        super().__init__(model, **kwargs)
        self.vae = vae_wrapper
        self.vae_batch_size = vae_batch_size
    
    def _encode_in_batches(self, images):
        """
        分批编码图像到潜空间（避免显存溢出）
        
        Args:
            images: (N, C, H, W) 图像tensor
        
        Returns:
            latents: (N, 4, H//8, W//8) 潜向量
        """
        N = images.shape[0]
        latent_list = []
        
        for i in range(0, N, self.vae_batch_size):
            end_idx = min(i + self.vae_batch_size, N)
            batch = images[i:end_idx].to(self.device)
            latent_batch = self.vae.encode(batch)
            latent_list.append(latent_batch.cpu())  # 立即移回CPU释放显存
            
            # 清理显存
            del batch, latent_batch
            torch.cuda.empty_cache()
        
        # 合并所有batch
        latents = torch.cat(latent_list, dim=0).to(self.device)
        return latents
    
    def train_epoch(self, train_loader):
        """训练一个epoch（在潜空间）"""
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        from tqdm import tqdm
        pbar = tqdm(train_loader, desc='Training (Latent)')
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            B, T_in, C, H, W = inputs.shape
            T_out = targets.shape[1]
            
            # 编码到潜空间（分批处理避免显存溢出）
            with torch.no_grad():
                # 编码输入（使用分批编码）
                inputs_flat = inputs.reshape(B * T_in, C, H, W)
                latent_inputs = self._encode_in_batches(inputs_flat)
                latent_inputs = latent_inputs.reshape(B, T_in, 4, H // 8, W // 8)
                
                # 编码目标（使用分批编码）
                targets_flat = targets.reshape(B * T_out, C, H, W)
                latent_targets = self._encode_in_batches(targets_flat)
                latent_targets = latent_targets.reshape(B, T_out, 4, H // 8, W // 8)
            
            # 前向传播（在潜空间）
            self.optimizer.zero_grad()
            latent_outputs = self.model(latent_inputs)
            loss = self.criterion(latent_outputs, latent_targets)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # 记录
            total_loss += loss.item()
            n_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / n_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(self, val_loader):
        """验证（在潜空间）"""
        self.model.eval()
        total_loss = 0
        n_batches = 0
        
        from tqdm import tqdm
        for inputs, targets in tqdm(val_loader, desc='Validating (Latent)'):
            B, T_in, C, H, W = inputs.shape
            T_out = targets.shape[1]
            
            # 编码到潜空间（分批处理避免显存溢出）
            inputs_flat = inputs.reshape(B * T_in, C, H, W)
            latent_inputs = self._encode_in_batches(inputs_flat)
            latent_inputs = latent_inputs.reshape(B, T_in, 4, H // 8, W // 8)
            
            targets_flat = targets.reshape(B * T_out, C, H, W)
            latent_targets = self._encode_in_batches(targets_flat)
            latent_targets = latent_targets.reshape(B, T_out, 4, H // 8, W // 8)
            
            # 预测
            latent_outputs = self.model(latent_inputs)
            loss = self.criterion(latent_outputs, latent_targets)
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        return avg_loss


def main():
    parser = argparse.ArgumentParser(description='训练潜空间U-Net')
    
    # 数据参数
    parser.add_argument('--data-path', type=str, default='gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr',
                       help='数据文件路径')
    parser.add_argument('--variable', type=str, default='2m_temperature',
                       help='变量名')
    parser.add_argument('--time-slice', type=str, default="2015-01-01:2019-12-31",
                       help='时间切片')
    parser.add_argument('--preprocessed-data-dir', type=str, default=None,
                       help='预处理数据目录（如果提供，将使用lazy loading）')
    
    # VAE参数
    parser.add_argument('--vae-model-id', type=str,
                       default='stable-diffusion-v1-5/stable-diffusion-v1-5',
                       help='VAE模型ID')
    
    # 模型参数
    parser.add_argument('--input-length', type=int, default=12,
                       help='输入序列长度')
    parser.add_argument('--output-length', type=int, default=4,
                       help='输出序列长度')
    parser.add_argument('--base-channels', type=int, default=128,
                       help='U-Net基础通道数')
    parser.add_argument('--depth', type=int, default=3,
                       help='U-Net深度')
    
    # 训练参数
    parser.add_argument('--batch-size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--vae-batch-size', type=int, default=4,
                       help='VAE编码时的子批次大小（控制显存占用）')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='权重衰减')
    parser.add_argument('--early-stopping', type=int, default=10,
                       help='早停耐心值')
    
    # 数据处理参数
    parser.add_argument('--normalization', type=str, default='minmax',
                       choices=['minmax', 'zscore'],
                       help='归一化方法')
    parser.add_argument('--target-size', type=str, default='512,512',
                       help='目标尺寸（必须是8的倍数）')
    
    # 其他参数
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='设备')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='数据加载线程数')
    parser.add_argument('--output-dir', type=str, default='outputs/latent_unet',
                       help='输出目录')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 解析target_size
    h, w = map(int, args.target_size.split(','))
    target_size = (h, w)
    assert h % 8 == 0 and w % 8 == 0, "尺寸必须是8的倍数"
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("训练潜空间U-Net - Step 4: 潜空间预测")
    print("=" * 80)
    
    # 保存配置
    import json
    config = vars(args)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # 加载VAE
    print("\n" + "-" * 80)
    print("Step 1: 加载VAE")
    print("-" * 80)
    
    vae_wrapper = SDVAEWrapper(
        model_id=args.vae_model_id,
        device=args.device
    )
    
    # 加载数据
    print("\n" + "-" * 80)
    print("Step 2: 加载和预处理数据")
    print("-" * 80)
    
    # 判断使用预处理数据还是实时加载
    if args.preprocessed_data_dir:
        print("使用预处理数据（Lazy Loading模式）")
        from weatherdiff.utils.lazy_dataset import LazyWeatherDataModule
        
        data_module = LazyWeatherDataModule(
            preprocessed_dir=args.preprocessed_data_dir,
            input_length=args.input_length,
            output_length=args.output_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_ratio=0.7,
            val_ratio=0.15
        )
        data_module.setup()
        
        # 从预处理数据获取target_size
        target_size = tuple(data_module.metadata['target_size'])
        
    else:
        print("实时加载数据（内存模式）")
        print("⚠️  警告: 大数据集可能导致内存不足")
        print("   建议先运行 preprocess_data_for_latent_unet.py")
        
        data_module = WeatherDataModule(
            data_path=args.data_path,
            variable=args.variable,
            time_slice=args.time_slice,
            input_length=args.input_length,
            output_length=args.output_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            normalization=args.normalization,
            n_channels=3,
            target_size=target_size
        )
        data_module.setup()
    
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    print(f"\n数据信息:")
    print(f"  图像尺寸: {target_size}")
    print(f"  潜向量尺寸: ({target_size[0]//8}, {target_size[1]//8})")
    print(f"  训练批次数: {len(train_loader)}")
    print(f"  验证批次数: {len(val_loader)}")
    
    # 创建模型
    print("\n" + "-" * 80)
    print("Step 3: 创建潜空间U-Net")
    print("-" * 80)
    
    model = LatentUNet(
        input_length=args.input_length,
        output_length=args.output_length,
        latent_channels=4,
        base_channels=args.base_channels,
        depth=args.depth
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数:")
    print(f"  输入序列长度: {args.input_length}")
    print(f"  输出序列长度: {args.output_length}")
    print(f"  基础通道数: {args.base_channels}")
    print(f"  网络深度: {args.depth}")
    print(f"  总参数量: {n_params:,}")
    
    # 创建训练器
    print("\n" + "-" * 80)
    print("Step 4: 开始训练")
    print("-" * 80)
    
    print(f"训练配置:")
    print(f"  主batch size: {args.batch_size}")
    print(f"  VAE batch size: {args.vae_batch_size} (用于分批编码)")
    print(f"  学习率: {args.lr}")
    print(f"  权重衰减: {args.weight_decay}")
    print(f"  早停耐心: {args.early_stopping}")
    
    trainer = LatentUNetTrainer(
        model=model,
        vae_wrapper=vae_wrapper,
        vae_batch_size=args.vae_batch_size,
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 训练
    # trainer.fit(
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     epochs=args.epochs,
    #     save_dir=str(output_dir),
    #     early_stopping_patience=args.early_stopping
    # )
    
    # 保存归一化统计量
    import pickle
    if args.preprocessed_data_dir:
        # 从预处理数据复制归一化参数
        import shutil
        src_stats = Path(args.preprocessed_data_dir) / 'normalizer_stats.pkl'
        dst_stats = output_dir / 'normalizer_stats.pkl'
        shutil.copy(src_stats, dst_stats)
        
        # 添加VAE信息
        with open(dst_stats, 'rb') as f:
            stats = pickle.load(f)
        stats['vae_model_id'] = args.vae_model_id
        with open(dst_stats, 'wb') as f:
            pickle.dump(stats, f)
    else:
        # 从data_module保存
        with open(output_dir / 'normalizer_stats.pkl', 'wb') as f:
            pickle.dump({
                'method': args.normalization,
                'stats': data_module.normalizer.get_stats(),
                'vae_model_id': args.vae_model_id
            }, f)
    
    print("\n" + "=" * 80)
    print("训练完成!")
    print("=" * 80)
    print(f"\n模型保存在: {output_dir}")
    print("\n下一步:")
    print("  1. 使用 predict_latent_unet.py 进行预测")
    print("  2. 对比像素空间和潜空间模型的性能")
    print("  3. 如果需要概率预测，继续 Step 5: 训练扩散模型")


if __name__ == '__main__':
    main()

