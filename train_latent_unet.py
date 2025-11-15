"""
训练潜空间U-Net（Step 4: 潜空间预测）

使用方法:
    python train_latent_unet.py --data-path /path/to/data.zarr --variable 2m_temperature
"""

import argparse
import torch
from pathlib import Path

from weatherdiff.unet import LatentUNet, UNetTrainer
from weatherdiff.vae import SDVAEWrapper, RAEWrapper
from weatherdiff.utils import WeatherDataModule


class LatentUNetTrainer(UNetTrainer):
    """扩展训练器以支持潜空间训练"""

    def __init__(self, model, vae_wrapper, vae_batch_size=4, use_multi_gpu=False, **kwargs):
        """
        Args:
            model: U-Net模型
            vae_wrapper: VAE包装器（不会被DataParallel包装，保持在主设备上）
            vae_batch_size: VAE编码时的子批次大小（用于控制显存）
            use_multi_gpu: 是否使用多GPU训练（DataParallel）
            **kwargs: 其他参数传递给UNetTrainer
        """
        super().__init__(model, use_multi_gpu=use_multi_gpu, **kwargs)
        self.vae = vae_wrapper
        self.vae_batch_size = vae_batch_size

    def _encode_in_batches(self, images):
        """
        分批编码图像到潜空间（避免显存溢出）

        Args:
            images: (N, C, H, W) 图像tensor

        Returns:
            latents: 潜向量，shape取决于VAE类型
        """
        N = images.shape[0]
        latent_list = []

        # 使用主设备（VAE wrapper保持在主设备上）
        device = self.main_device if hasattr(self, 'main_device') else self.device

        for i in range(0, N, self.vae_batch_size):
            end_idx = min(i + self.vae_batch_size, N)
            batch = images[i:end_idx].to(device)
            latent_batch = self.vae.encode(batch)
            latent_list.append(latent_batch.cpu())  # 立即移回CPU释放显存

            # 清理显存
            del batch, latent_batch
            torch.cuda.empty_cache()

        # 合并所有batch，移回主设备
        latents = torch.cat(latent_list, dim=0).to(device)
        return latents

    def _get_latent_shape(self, image_shape):
        """获取潜向量形状"""
        if hasattr(self.vae, "get_latent_shape"):
            return self.vae.get_latent_shape(image_shape)
        else:
            # 默认SD VAE
            c, h, w = image_shape
            return (4, h // 8, w // 8)

    def train_epoch(self, train_loader):
        """训练一个epoch（在潜空间）"""
        self.model.train()
        # 设置VAE训练模式
        if isinstance(self.vae, RAEWrapper):
            self.vae.train_mode()
        elif isinstance(self.vae, SDVAEWrapper) and not self.vae.freeze_vae:
            self.vae.train_mode()

        total_loss = 0
        n_batches = 0

        from tqdm import tqdm

        pbar = tqdm(train_loader, desc="Training (Latent)")

        for batch_idx, (inputs, targets) in enumerate(pbar):
            B, T_in, C, H, W = inputs.shape
            T_out = targets.shape[1]

            # 获取latent shape
            latent_shape = self._get_latent_shape((C, H, W))
            latent_channels, latent_h, latent_w = latent_shape

            # 编码到潜空间（分批处理避免显存溢出）
            # 检查VAE是否被冻结（所有参数都不需要梯度）
            encode_grad = False
            if isinstance(self.vae, RAEWrapper):
                # 检查encoder是否有任何参数需要梯度
                encoder_requires_grad = any(
                    p.requires_grad for p in self.vae.rae.encoder.parameters()
                )
                encode_grad = encoder_requires_grad  # encoder可训练时，使用梯度
            elif isinstance(self.vae, SDVAEWrapper):
                # 检查VAE是否有任何参数需要梯度
                vae_requires_grad = any(
                    p.requires_grad for p in self.vae.vae.parameters()
                )
                encode_grad = vae_requires_grad  # VAE可训练时，使用梯度
            with torch.set_grad_enabled(encode_grad):
                # 编码输入（使用分批编码）
                inputs_flat = inputs.reshape(B * T_in, C, H, W)
                latent_inputs = self._encode_in_batches(inputs_flat)
                latent_inputs = latent_inputs.reshape(
                    B, T_in, latent_channels, latent_h, latent_w
                )

                # 编码目标（使用分批编码）
                targets_flat = targets.reshape(B * T_out, C, H, W)
                latent_targets = self._encode_in_batches(targets_flat)
                latent_targets = latent_targets.reshape(
                    B, T_out, latent_channels, latent_h, latent_w
                )

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

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / n_batches
        return avg_loss

    @torch.no_grad()
    def validate(self, val_loader):
        """验证（在潜空间）"""
        self.model.eval()
        # 设置VAE评估模式
        if isinstance(self.vae, RAEWrapper):
            self.vae.eval_mode()
        elif isinstance(self.vae, SDVAEWrapper):
            self.vae.eval_mode()

        total_loss = 0
        n_batches = 0

        from tqdm import tqdm

        for inputs, targets in tqdm(val_loader, desc="Validating (Latent)"):
            B, T_in, C, H, W = inputs.shape
            T_out = targets.shape[1]

            # 获取latent shape
            latent_shape = self._get_latent_shape((C, H, W))
            latent_channels, latent_h, latent_w = latent_shape

            # 编码到潜空间（分批处理避免显存溢出）
            inputs_flat = inputs.reshape(B * T_in, C, H, W)
            latent_inputs = self._encode_in_batches(inputs_flat)
            latent_inputs = latent_inputs.reshape(
                B, T_in, latent_channels, latent_h, latent_w
            )

            targets_flat = targets.reshape(B * T_out, C, H, W)
            latent_targets = self._encode_in_batches(targets_flat)
            latent_targets = latent_targets.reshape(
                B, T_out, latent_channels, latent_h, latent_w
            )

            # 预测
            latent_outputs = self.model(latent_inputs)
            loss = self.criterion(latent_outputs, latent_targets)

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        return avg_loss


def main():
    parser = argparse.ArgumentParser(description="训练潜空间U-Net")

    # 数据参数
    parser.add_argument(
        "--data-path",
        type=str,
        default="gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr",
        help="数据文件路径",
    )
    parser.add_argument("--variable", type=str, default="2m_temperature", help="变量名")
    parser.add_argument(
        "--time-slice", type=str, default="2015-01-01:2019-12-31", help="时间切片"
    )
    parser.add_argument(
        "--preprocessed-data-dir",
        type=str,
        default=None,
        help="预处理数据目录（如果提供，将使用lazy loading）",
    )

    # VAE参数
    parser.add_argument(
        "--vae-type",
        type=str,
        default="sd",
        choices=["sd", "rae"],
        help="VAE类型: sd (Stable Diffusion) 或 rae (RAE)",
    )
    parser.add_argument(
        "--vae-model-id",
        type=str,
        default="stable-diffusion-v1-5/stable-diffusion-v1-5",
        help="VAE模型ID（SD VAE使用）",
    )
    parser.add_argument(
        "--vae-train-mode",
        type=str,
        default="pretrained",
        choices=["pretrained", "from_scratch"],
        help="VAE训练模式: pretrained (加载预训练) 或 from_scratch (从头训练)",
    )
    parser.add_argument(
        "--vae-pretrained-path",
        type=str,
        default=None,
        help="可选，预训练VAE权重路径（用于fine-tuning）",
    )
    parser.add_argument(
        "--freeze-vae",
        action="store_true",
        default=False,
        help="冻结VAE参数（默认False，允许训练）",
    )

    # RAE参数
    parser.add_argument(
        "--rae-encoder-cls",
        type=str,
        default="SigLIP2wNorm",
        choices=["Dinov2withNorm", "SigLIP2wNorm", "MAEwNorm"],
        help="RAE encoder类型",
    )
    parser.add_argument(
        "--rae-encoder-config-path",
        type=str,
        default="google/siglip2-base-patch16-256",
        help="RAE encoder配置路径（HuggingFace模型ID）",
    )
    parser.add_argument(
        "--rae-encoder-input-size",
        type=int,
        default=256,
        help="RAE encoder输入图像尺寸",
    )
    parser.add_argument(
        "--rae-decoder-config-path",
        type=str,
        default="facebook/vit-mae-base",
        help="RAE decoder配置路径（HuggingFace模型ID）",
    )
    parser.add_argument(
        "--rae-decoder-patch-size", type=int, default=16, help="RAE decoder patch大小"
    )
    parser.add_argument(
        "--rae-pretrained-decoder-path",
        type=str,
        default=None,
        help="RAE预训练decoder权重路径",
    )
    parser.add_argument(
        "--rae-normalization-stat-path",
        type=str,
        default=None,
        help="RAE归一化统计量路径",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        default=True,
        help="冻结encoder（默认True）",
    )
    parser.add_argument(
        "--freeze-decoder",
        action="store_false",
        dest="freeze_decoder",
        default=False,
        help="冻结decoder（默认False，decoder可微调）",
    )

    # 模型参数
    parser.add_argument("--input-length", type=int, default=12, help="输入序列长度")
    parser.add_argument("--output-length", type=int, default=4, help="输出序列长度")
    parser.add_argument(
        "--base-channels", type=int, default=128, help="U-Net基础通道数"
    )
    parser.add_argument("--depth", type=int, default=3, help="U-Net深度")

    # 训练参数
    parser.add_argument("--batch-size", type=int, default=16, help="批次大小")
    parser.add_argument(
        "--vae-batch-size",
        type=int,
        default=4,
        help="VAE编码时的子批次大小（控制显存占用）",
    )
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--early-stopping", type=int, default=10, help="早停耐心值")

    # 数据处理参数
    parser.add_argument(
        "--normalization",
        type=str,
        default="minmax",
        choices=["minmax", "zscore"],
        help="归一化方法",
    )
    parser.add_argument(
        "--target-size", type=str, default="512,512", help="目标尺寸（必须是8的倍数）"
    )

    # 其他参数
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="设备",
    )
    parser.add_argument(
        "--use-multi-gpu",
        action="store_true",
        default=False,
        help="使用多GPU训练（DataParallel）",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="指定使用的GPU ID（逗号分隔，如 '0,1,2,3'）。如果未指定，使用所有可用GPU",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument(
        "--output-dir", type=str, default="outputs/latent_unet", help="输出目录"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)

    # 处理多GPU设置
    # 注意: CUDA_VISIBLE_DEVICES应该在Python启动前设置（在shell脚本中设置）
    # 如果在这里设置，需要确保在导入torch之前设置，但此时torch已经导入
    # 所以这里只做检查和提示
    use_multi_gpu = args.use_multi_gpu
    if use_multi_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA不可用，无法使用多GPU训练")
        
        # 检查GPU数量
        num_gpus = torch.cuda.device_count()
        if num_gpus < 2:
            print(f"警告: 只有 {num_gpus} 个GPU可用，将使用单GPU训练")
            print(f"提示: 如需使用多GPU，请在shell脚本中设置CUDA_VISIBLE_DEVICES或在命令行中指定--gpu-ids")
            use_multi_gpu = False
        else:
            print(f"使用 {num_gpus} 个GPU进行训练")
            # 显示GPU信息
            for i in range(num_gpus):
                try:
                    gpu_name = torch.cuda.get_device_name(i)
                    print(f"  GPU {i}: {gpu_name}")
                except Exception as e:
                    print(f"  GPU {i}: 无法获取信息 ({e})")
            
            # 如果指定了GPU IDs但没有在环境变量中设置，给出提示
            if args.gpu_ids:
                import os
                env_gpu_ids = os.environ.get('CUDA_VISIBLE_DEVICES', '')
                if env_gpu_ids != args.gpu_ids:
                    print(f"提示: 指定了--gpu-ids={args.gpu_ids}，但CUDA_VISIBLE_DEVICES={env_gpu_ids}")
                    print(f"      建议在shell脚本中设置CUDA_VISIBLE_DEVICES={args.gpu_ids}以获得最佳效果")
    
    # 更新device（多GPU时使用cuda:0作为主设备）
    if use_multi_gpu:
        device = "cuda:0"
    else:
        device = args.device

    # 解析target_size
    h, w = map(int, args.target_size.split(","))
    target_size = (h, w)
    assert h % 8 == 0 and w % 8 == 0, "尺寸必须是8的倍数"

    # 如果使用预处理数据，先读取metadata获取target_size
    if args.preprocessed_data_dir:
        import json

        metadata_path = Path(args.preprocessed_data_dir) / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            if "target_size" in metadata:
                target_size = tuple(metadata["target_size"])
                print(f"从预处理数据读取target_size: {target_size}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("训练潜空间U-Net - Step 4: 潜空间预测")
    print("=" * 80)

    # 加载VAE
    print("\n" + "-" * 80)
    print("Step 1: 加载VAE")
    print("-" * 80)

    if args.vae_type == "sd":
        print(f"使用Stable Diffusion VAE: {args.vae_model_id}")
        print(f"  训练模式: {args.vae_train_mode}")
        if args.vae_pretrained_path:
            print(f"  预训练权重路径: {args.vae_pretrained_path}")
        print(f"  冻结VAE: {args.freeze_vae}")
        vae_wrapper = SDVAEWrapper(
            model_id=args.vae_model_id,
            train_mode=args.vae_train_mode,
            pretrained_path=args.vae_pretrained_path,
            device=args.device,
            freeze_vae=args.freeze_vae,
        )
        latent_channels = 4  # SD VAE固定为4
    elif args.vae_type == "rae":
        print(
            f"使用RAE: encoder={args.rae_encoder_cls}, decoder={args.rae_decoder_config_path}"
        )
        # 构建encoder_params
        encoder_params = {}
        if args.rae_encoder_cls == "Dinov2withNorm":
            encoder_params = {
                "dinov2_path": args.rae_encoder_config_path,
                "normalize": True,
            }
        elif args.rae_encoder_cls == "SigLIP2wNorm":
            encoder_params = {"model_name": args.rae_encoder_config_path}
        elif args.rae_encoder_cls == "MAEwNorm":
            encoder_params = {"model_name": args.rae_encoder_config_path}

        vae_wrapper = RAEWrapper(
            encoder_cls=args.rae_encoder_cls,
            encoder_config_path=args.rae_encoder_config_path,
            encoder_input_size=args.rae_encoder_input_size,
            encoder_params=encoder_params,
            decoder_config_path=args.rae_decoder_config_path,
            decoder_patch_size=args.rae_decoder_patch_size,
            pretrained_decoder_path=args.rae_pretrained_decoder_path,
            normalization_stat_path=args.rae_normalization_stat_path,
            device=args.device,
            freeze_encoder=args.freeze_encoder,
            freeze_decoder=args.freeze_decoder,
        )

        # 获取RAE decoder的输出尺寸（这是固定的，由encoder_input_size和decoder_patch_size决定）
        decoder_output_size = vae_wrapper.get_decoder_output_size()
        print(f"  RAE decoder输出尺寸: {decoder_output_size}")

        # 验证target_size必须等于decoder输出尺寸
        if (
            target_size[0] != decoder_output_size[0]
            or target_size[1] != decoder_output_size[1]
        ):
            raise ValueError(
                f"维度不匹配错误：\n"
                f"  训练时target_size: {target_size}\n"
                f"  RAE decoder输出尺寸: {decoder_output_size}\n"
                f"  原因：RAE decoder的输出尺寸是固定的，由encoder_input_size和decoder_patch_size决定\n"
                f"  当前配置：encoder_input_size={args.rae_encoder_input_size}, "
                f"decoder_patch_size={args.rae_decoder_patch_size}\n"
                f"  解决方案：\n"
                f"    1. 将--target-size设置为 {decoder_output_size[0]},{decoder_output_size[1]}\n"
                f"    2. 或者调整RAE配置（encoder_input_size和decoder_patch_size）以匹配target_size"
            )

        # 获取latent_channels
        latent_shape = vae_wrapper.get_latent_shape((3, target_size[0], target_size[1]))
        latent_channels = latent_shape[0]
    else:
        raise ValueError(f"Unknown VAE type: {args.vae_type}")

    # 保存配置（在加载VAE之后，以便包含latent_channels）
    import json

    config = vars(args)
    config["latent_channels"] = latent_channels  # 添加latent_channels到配置
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

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
            val_ratio=0.15,
        )
        data_module.setup()

        # 从预处理数据获取target_size（如果之前没有读取）
        if "target_size" not in locals() or target_size != tuple(
            data_module.metadata["target_size"]
        ):
            target_size = tuple(data_module.metadata["target_size"])
            print(f"从预处理数据读取target_size: {target_size}")

            # 如果使用RAE，再次验证target_size
            if args.vae_type == "rae":
                if (
                    target_size[0] != decoder_output_size[0]
                    or target_size[1] != decoder_output_size[1]
                ):
                    raise ValueError(
                        f"维度不匹配错误：\n"
                        f"  预处理数据target_size: {target_size}\n"
                        f"  RAE decoder输出尺寸: {decoder_output_size}\n"
                        f"  原因：预处理数据使用的target_size与RAE decoder输出尺寸不匹配\n"
                        f"  解决方案：\n"
                        f"    1. 重新预处理数据，使用target_size={decoder_output_size[0]},{decoder_output_size[1]}\n"
                        f"    2. 或者调整RAE配置（encoder_input_size和decoder_patch_size）以匹配预处理数据的target_size"
                    )

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
            target_size=target_size,
        )
        data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # 获取latent shape（如果使用RAE）
    if args.vae_type == "rae":
        latent_shape = vae_wrapper.get_latent_shape((3, target_size[0], target_size[1]))
        latent_channels, latent_h, latent_w = latent_shape
        print(f"\n数据信息:")
        print(f"  图像尺寸: {target_size}")
        print(f"  潜向量尺寸: ({latent_channels}, {latent_h}, {latent_w})")
    else:
        latent_channels = 4
        latent_h = target_size[0] // 8
        latent_w = target_size[1] // 8
        print(f"\n数据信息:")
        print(f"  图像尺寸: {target_size}")
        print(f"  潜向量尺寸: ({latent_channels}, {latent_h}, {latent_w})")

    print(f"  训练批次数: {len(train_loader)}")
    print(f"  验证批次数: {len(val_loader)}")

    # 创建模型
    print("\n" + "-" * 80)
    print("Step 3: 创建潜空间U-Net")
    print("-" * 80)

    model = LatentUNet(
        input_length=args.input_length,
        output_length=args.output_length,
        latent_channels=latent_channels,
        base_channels=args.base_channels,
        depth=args.depth,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数:")
    print(f"  输入序列长度: {args.input_length}")
    print(f"  输出序列长度: {args.output_length}")
    print(f"  潜向量通道数: {latent_channels}")
    print(f"  基础通道数: {args.base_channels}")
    print(f"  网络深度: {args.depth}")
    print(f"  总参数量: {n_params:,}")

    # 如果使用RAE且decoder可训练，添加decoder参数到优化器
    if args.vae_type == "rae" and not args.freeze_decoder:
        decoder_params = list(vae_wrapper.get_decoder_parameters())
        decoder_param_count = sum(p.numel() for p in decoder_params)
        print(f"  Decoder参数量: {decoder_param_count:,} (可训练)")

    # 创建训练器
    print("\n" + "-" * 80)
    print("Step 4: 开始训练")
    print("-" * 80)

    print(f"训练配置:")
    print(f"  设备: {device}")
    if use_multi_gpu:
        print(f"  多GPU训练: 是 ({torch.cuda.device_count()} 个GPU)")
        print(f"  有效batch size: {args.batch_size * torch.cuda.device_count()} (每个GPU: {args.batch_size})")
    else:
        print(f"  多GPU训练: 否")
    print(f"  VAE类型: {args.vae_type}")
    if args.vae_type == "sd":
        print(f"  VAE训练模式: {args.vae_train_mode}")
        print(f"  VAE冻结: {args.freeze_vae}")
    elif args.vae_type == "rae":
        print(f"  Encoder冻结: {args.freeze_encoder}")
        print(f"  Decoder冻结: {args.freeze_decoder}")
    print(f"  主batch size: {args.batch_size}")
    print(f"  VAE batch size: {args.vae_batch_size} (用于分批编码)")
    print(f"  学习率: {args.lr}")
    print(f"  权重衰减: {args.weight_decay}")
    print(f"  早停耐心: {args.early_stopping}")

    trainer = LatentUNetTrainer(
        model=model,
        vae_wrapper=vae_wrapper,
        vae_batch_size=args.vae_batch_size,
        device=device,
        use_multi_gpu=use_multi_gpu,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # 如果使用SD VAE且可训练，将VAE参数添加到优化器
    if args.vae_type == "sd" and not args.freeze_vae:
        vae_params = list(vae_wrapper.get_vae_parameters())
        vae_param_count = sum(p.numel() for p in vae_params)
        print(f"  VAE参数量: {vae_param_count:,} (可训练)")
        # 获取实际模型参数（如果是DataParallel）
        if use_multi_gpu:
            model_params = list(trainer.model.module.parameters())
        else:
            model_params = list(trainer.model.parameters())
        # 创建新的优化器，包含UNet和VAE参数
        all_params = model_params + vae_params
        trainer.optimizer = torch.optim.AdamW(
            all_params, lr=args.lr, weight_decay=args.weight_decay
        )
        print(f"✓ 优化器已更新：包含UNet和VAE参数")

    # 如果使用RAE且decoder可训练，将decoder参数添加到优化器
    if args.vae_type == "rae" and not args.freeze_decoder:
        decoder_params = list(vae_wrapper.get_decoder_parameters())
        decoder_param_count = sum(p.numel() for p in decoder_params)
        print(f"  Decoder参数量: {decoder_param_count:,} (可训练)")
        # 获取实际模型参数（如果是DataParallel）
        if use_multi_gpu:
            model_params = list(trainer.model.module.parameters())
        else:
            model_params = list(trainer.model.parameters())
        # 创建新的优化器，包含UNet和decoder参数
        all_params = model_params + decoder_params
        trainer.optimizer = torch.optim.AdamW(
            all_params, lr=args.lr, weight_decay=args.weight_decay
        )
        print(f"✓ 优化器已更新：包含UNet和RAE decoder参数")

    # 训练
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_dir=str(output_dir),
        early_stopping_patience=args.early_stopping,
    )

    # 保存归一化统计量
    import pickle

    if args.preprocessed_data_dir:
        # 从预处理数据复制归一化参数
        import shutil

        src_stats = Path(args.preprocessed_data_dir) / "normalizer_stats.pkl"
        dst_stats = output_dir / "normalizer_stats.pkl"
        shutil.copy(src_stats, dst_stats)

        # 添加VAE信息
        with open(dst_stats, "rb") as f:
            stats = pickle.load(f)
        stats["vae_type"] = args.vae_type
        if args.vae_type == "sd":
            stats["vae_model_id"] = args.vae_model_id
        elif args.vae_type == "rae":
            stats["rae_encoder_cls"] = args.rae_encoder_cls
            stats["rae_encoder_config_path"] = args.rae_encoder_config_path
            stats["rae_encoder_input_size"] = args.rae_encoder_input_size
            stats["rae_decoder_config_path"] = args.rae_decoder_config_path
            stats["rae_decoder_patch_size"] = args.rae_decoder_patch_size
            stats["rae_pretrained_decoder_path"] = args.rae_pretrained_decoder_path
            stats["freeze_encoder"] = args.freeze_encoder
            stats["freeze_decoder"] = args.freeze_decoder
            if args.rae_normalization_stat_path:
                stats["rae_normalization_stat_path"] = args.rae_normalization_stat_path
        with open(dst_stats, "wb") as f:
            pickle.dump(stats, f)
    else:
        # 从data_module保存
        vae_info = {"vae_type": args.vae_type}
        if args.vae_type == "sd":
            vae_info["vae_model_id"] = args.vae_model_id
        elif args.vae_type == "rae":
            vae_info["rae_encoder_cls"] = args.rae_encoder_cls
            vae_info["rae_encoder_config_path"] = args.rae_encoder_config_path
            vae_info["rae_encoder_input_size"] = args.rae_encoder_input_size
            vae_info["rae_decoder_config_path"] = args.rae_decoder_config_path
            vae_info["rae_decoder_patch_size"] = args.rae_decoder_patch_size
            vae_info["rae_pretrained_decoder_path"] = args.rae_pretrained_decoder_path
            vae_info["freeze_encoder"] = args.freeze_encoder
            vae_info["freeze_decoder"] = args.freeze_decoder
            if args.rae_normalization_stat_path:
                vae_info["rae_normalization_stat_path"] = (
                    args.rae_normalization_stat_path
                )

        with open(output_dir / "normalizer_stats.pkl", "wb") as f:
            pickle.dump(
                {
                    "method": args.normalization,
                    "stats": data_module.normalizer.get_stats(),
                    **vae_info,
                },
                f,
            )

    print("\n" + "=" * 80)
    print("训练完成!")
    print("=" * 80)
    print(f"\n模型保存在: {output_dir}")
    print("\n下一步:")
    print("  1. 使用 predict_latent_unet.py 进行预测")
    print("  2. 对比像素空间和潜空间模型的性能")
    print("  3. 如果需要概率预测，继续 Step 5: 训练扩散模型")


if __name__ == "__main__":
    main()
