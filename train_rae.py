"""
训练RAE Latent U-Net - 只支持加载预训练权重，不支持从头训练

使用方法:
    # 使用SigLIP2 encoder (推荐)
    python train_rae.py --data-path /path/to/data.zarr --variable 2m_temperature --rae-encoder-cls SigLIP2wNorm --rae-pretrained-decoder-path /path/to/decoder.pt

    # 使用DINOv2 encoder
    python train_rae.py --data-path /path/to/data.zarr --variable 2m_temperature --rae-encoder-cls Dinov2withNorm --rae-pretrained-decoder-path /path/to/decoder.pt

    # 冻结encoder，只训练decoder
    python train_rae.py --data-path /path/to/data.zarr --variable 2m_temperature --rae-pretrained-decoder-path /path/to/decoder.pt --freeze-encoder

    # 冻结decoder，只训练encoder
    python train_rae.py --data-path /path/to/data.zarr --variable 2m_temperature --rae-pretrained-decoder-path /path/to/decoder.pt --freeze-decoder

    # 冻结整个RAE（仅用于推理）
    python train_rae.py --data-path /path/to/data.zarr --variable 2m_temperature --rae-pretrained-decoder-path /path/to/decoder.pt --freeze-encoder --freeze-decoder
"""

import argparse
import torch
from pathlib import Path

from weatherdiff.unet import LatentUNet, UNetTrainer
from weatherdiff.vae import RAEWrapper
from weatherdiff.utils import WeatherDataModule


class RAELatentUNetTrainer(UNetTrainer):
    """RAE Latent U-Net训练器"""

    def __init__(
        self,
        model,
        vae_wrapper,
        vae_batch_size=4,
        use_multi_gpu=False,
        gradient_accumulation_steps=1,
        use_amp=False,
        amp_dtype="float16",
        **kwargs,
    ):
        """
        Args:
            model: U-Net模型
            vae_wrapper: RAE包装器（不会被DataParallel包装，保持在主设备上）
            vae_batch_size: VAE编码时的子批次大小（用于控制显存）
            use_multi_gpu: 是否使用多GPU训练（DataParallel）
            gradient_accumulation_steps: 梯度累积步数（用于减少显存占用）
            use_amp: 是否使用自动混合精度训练
            amp_dtype: 混合精度数据类型，可选 "float16" 或 "bfloat16"
            **kwargs: 其他参数传递给UNetTrainer（但不包括batch_size，因为UNetTrainer不接受）
        """
        # 提取batch_size用于日志，但不传递给UNetTrainer
        batch_size = kwargs.pop("batch_size", None)

        # 只传递UNetTrainer接受的参数
        unet_trainer_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ("device", "lr", "weight_decay", "loss_alpha", "loss_lambda")
        }
        super().__init__(model, use_multi_gpu=use_multi_gpu, **unet_trainer_kwargs)
        self.vae = vae_wrapper
        self.vae_batch_size = vae_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_amp = use_amp

        # 设置混合精度
        if use_amp:
            if amp_dtype == "bfloat16":
                if not torch.cuda.is_bf16_supported():
                    print("警告: 当前GPU不支持bfloat16，将使用float16")
                    amp_dtype = "float16"
                self.amp_dtype = torch.bfloat16
                self.scaler = None  # bfloat16不需要scaler
            else:
                self.amp_dtype = torch.float16
                self.scaler = torch.cuda.amp.GradScaler()
            print(f"✓ 启用混合精度训练: {amp_dtype}")
        else:
            self.amp_dtype = None
            self.scaler = None

        if gradient_accumulation_steps > 1:
            print(f"✓ 启用梯度累积: {gradient_accumulation_steps} 步")
            if batch_size is not None:
                print(
                    f"  有效batch size: {batch_size} × {gradient_accumulation_steps} = {batch_size * gradient_accumulation_steps}"
                )

    def _encode_in_batches(self, images, enable_grad=False):
        """
        分批编码图像到潜空间（避免显存溢出）

        Args:
            images: (N, C, H, W) 图像tensor
            enable_grad: 是否启用梯度计算（仅在encoder可训练时，但RAE encoder通常固定）

        Returns:
            latents: 潜向量，shape取决于RAE配置
        """
        N = images.shape[0]
        latent_list = []

        # 使用主设备（VAE wrapper保持在主设备上）
        device = self.main_device if hasattr(self, "main_device") else self.device

        # RAE encoder通常是固定的，不需要梯度
        with torch.no_grad():
            for i in range(0, N, self.vae_batch_size):
                end_idx = min(i + self.vae_batch_size, N)
                batch = images[i:end_idx].to(device)

                # 使用混合精度编码（如果启用）
                if self.use_amp and self.amp_dtype is not None:
                    with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                        latent_batch = self.vae.encode(batch)
                else:
                    latent_batch = self.vae.encode(batch)

                # 立即移到CPU释放显存
                latent_list.append(latent_batch.cpu())
                del latent_batch
                torch.cuda.empty_cache()

                # 清理显存
                del batch

        # 合并所有batch，移回主设备
        latents = torch.cat(latent_list, dim=0).to(device)
        return latents

    def _get_latent_shape(self, image_shape):
        """获取潜向量形状"""
        if hasattr(self.vae, "get_latent_shape"):
            return self.vae.get_latent_shape(image_shape)
        else:
            raise ValueError("RAE wrapper必须实现get_latent_shape方法")

    def train_epoch(self, train_loader):
        """训练一个epoch（在潜空间）"""
        self.model.train()
        # 设置VAE训练模式（decoder可训练）
        self.vae.train_mode()

        total_loss = 0
        n_batches = 0
        accumulation_counter = 0

        from tqdm import tqdm

        pbar = tqdm(train_loader, desc="Training (RAE Latent)")

        # 在epoch开始时清零梯度（用于梯度累积）
        self.optimizer.zero_grad()

        for batch_idx, (inputs, targets) in enumerate(pbar):
            B, T_in, C, H, W = inputs.shape
            T_out = targets.shape[1]

            # 获取latent shape
            latent_shape = self._get_latent_shape((C, H, W))
            latent_channels, latent_h, latent_w = latent_shape

            # 编码到潜空间（分批处理避免显存溢出）
            # RAE encoder固定，不需要梯度
            inputs_flat = inputs.reshape(B * T_in, C, H, W)
            latent_inputs = self._encode_in_batches(inputs_flat, enable_grad=False)
            latent_inputs = latent_inputs.reshape(
                B, T_in, latent_channels, latent_h, latent_w
            )

            targets_flat = targets.reshape(B * T_out, C, H, W)
            latent_targets = self._encode_in_batches(targets_flat, enable_grad=False)
            latent_targets = latent_targets.reshape(
                B, T_out, latent_channels, latent_h, latent_w
            )

            # 清理输入数据的显存
            del inputs_flat, targets_flat
            if hasattr(inputs, "cpu"):
                inputs = inputs.cpu()
            if hasattr(targets, "cpu"):
                targets = targets.cpu()
            torch.cuda.empty_cache()

            # 前向传播（在潜空间）- 使用混合精度
            if self.use_amp and self.amp_dtype is not None:
                with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                    latent_outputs = self.model(latent_inputs)
                    loss = self.criterion(latent_outputs, latent_targets)
                    # 梯度累积时归一化loss
                    loss = loss / self.gradient_accumulation_steps
            else:
                latent_outputs = self.model(latent_inputs)
                loss = self.criterion(latent_outputs, latent_targets)
                # 梯度累积时归一化loss
                loss = loss / self.gradient_accumulation_steps

            # 反向传播（使用混合精度scaler）
            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulation_counter += 1

            # 只在累积到指定步数或最后一个batch时更新参数
            if accumulation_counter >= self.gradient_accumulation_steps or (
                batch_idx + 1
            ) == len(train_loader):
                # 使用混合精度scaler进行梯度裁剪和优化
                if self.use_amp and self.scaler is not None:
                    # 梯度裁剪：获取实际模型参数（如果是DataParallel）
                    if self.use_multi_gpu:
                        model_params = self.model.module.parameters()
                    else:
                        model_params = self.model.parameters()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(model_params, 1.0)

                    # 如果decoder可训练，也裁剪decoder参数的梯度
                    decoder_params = list(self.vae.get_decoder_parameters())
                    trainable_decoder_params = [
                        p for p in decoder_params if p.requires_grad
                    ]
                    if trainable_decoder_params:
                        torch.nn.utils.clip_grad_norm_(trainable_decoder_params, 1.0)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # 梯度裁剪：获取实际模型参数（如果是DataParallel）
                    if self.use_multi_gpu:
                        model_params = self.model.module.parameters()
                    else:
                        model_params = self.model.parameters()
                    torch.nn.utils.clip_grad_norm_(model_params, 1.0)

                    # 如果decoder可训练，也裁剪decoder参数的梯度
                    decoder_params = list(self.vae.get_decoder_parameters())
                    trainable_decoder_params = [
                        p for p in decoder_params if p.requires_grad
                    ]
                    if trainable_decoder_params:
                        torch.nn.utils.clip_grad_norm_(trainable_decoder_params, 1.0)

                    self.optimizer.step()

                self.optimizer.zero_grad()
                accumulation_counter = 0

            # 记录（恢复真实loss值）
            total_loss += loss.item() * self.gradient_accumulation_steps
            n_batches += 1

            pbar.set_postfix(
                {
                    "loss": f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                    "accum": f"{accumulation_counter}/{self.gradient_accumulation_steps}",
                }
            )

            # 清理显存
            del latent_inputs, latent_targets, latent_outputs, loss
            torch.cuda.empty_cache()

        avg_loss = total_loss / n_batches
        return avg_loss

    @torch.no_grad()
    def validate(self, val_loader):
        """验证（在潜空间）"""
        self.model.eval()
        self.vae.eval_mode()

        total_loss = 0
        n_batches = 0

        from tqdm import tqdm

        for inputs, targets in tqdm(val_loader, desc="Validating (RAE Latent)"):
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
    parser = argparse.ArgumentParser(description="训练RAE Latent U-Net")

    # 数据参数
    parser.add_argument(
        "--data-path",
        type=str,
        default="gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr",
        help="数据文件路径",
    )
    parser.add_argument("--variable", type=str, default="2m_temperature", help="变量名")
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Pressure levels for variables with a level dimension "
            "(e.g. 500 700 850). "
            "如果指定，则只使用这些层的数据进行训练。"
            "如果不指定，默认使用所有可用的 levels。"
            "例如：`--levels 500` 只使用 500hPa，`--levels 500 700 850` 使用多个层。"
        ),
    )
    parser.add_argument(
        "--time-slice", type=str, default="2015-01-01:2019-12-31", help="时间切片"
    )
    parser.add_argument(
        "--preprocessed-data-dir",
        type=str,
        default=None,
        help="预处理数据目录（如果提供，将使用lazy loading）",
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
        required=True,
        help="RAE预训练decoder权重路径（必需，不支持从头训练）",
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
        default=False,
        help="冻结encoder（默认False）",
    )
    parser.add_argument(
        "--freeze-decoder",
        action="store_true",
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
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="梯度累积步数（用于减少显存占用，有效batch size = batch_size × gradient_accumulation_steps）",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        default=False,
        help="使用自动混合精度训练（FP16/BF16），可以显著减少显存占用",
    )
    parser.add_argument(
        "--amp-dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16"],
        help="混合精度数据类型（bfloat16需要GPU支持，但通常更稳定）",
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
        "--output-dir", type=str, default="outputs/rae_latent_unet", help="输出目录"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)

    # 处理多GPU设置
    use_multi_gpu = args.use_multi_gpu
    if use_multi_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA不可用，无法使用多GPU训练")

        num_gpus = torch.cuda.device_count()
        if num_gpus < 2:
            print(f"警告: 只有 {num_gpus} 个GPU可用，将使用单GPU训练")
            use_multi_gpu = False
        else:
            print(f"使用 {num_gpus} 个GPU进行训练")

    # 更新device（多GPU时使用cuda:0作为主设备）
    if use_multi_gpu:
        device = "cuda:0"
    else:
        device = args.device

    vae_device = device

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("训练RAE Latent U-Net")
    print("=" * 80)

    # 加载RAE
    print("\n" + "-" * 80)
    print("Step 1: 加载RAE")
    print("-" * 80)

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
        device=vae_device,
        freeze_encoder=args.freeze_encoder,
        freeze_decoder=args.freeze_decoder,
    )

    # 获取RAE decoder的输出尺寸（这是固定的，由encoder_input_size和decoder_patch_size决定）
    decoder_output_size = vae_wrapper.get_decoder_output_size()
    print(f"  RAE decoder输出尺寸: {decoder_output_size}")
    target_size = decoder_output_size  # RAE decoder输出尺寸决定target_size

    # 注意：配置将在数据加载后更新以包含实际使用的 levels

    # 加载数据
    print("\n" + "-" * 80)
    print("Step 2: 加载和预处理数据")
    print("-" * 80)

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

            # 验证target_size
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

        # 如果用户指定了 levels，则传给数据模块
        levels = args.levels
        if levels is not None:
            print(f"使用指定的levels: {levels}")
        else:
            print("使用所有可用的levels（默认）")

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
            levels=levels,
        )
        data_module.setup()

        # 获取实际使用的levels（可能是用户指定的或自动选择的）
        actual_levels = getattr(data_module, "levels", None)
        if actual_levels is not None:
            print(f"实际使用的levels: {actual_levels}")

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # 保存配置（在数据加载后，以便包含实际使用的levels）
    import json

    config = vars(args)
    config["target_size"] = list(target_size)  # 转换为list以便JSON序列化
    # 获取latent_channels
    latent_shape = vae_wrapper.get_latent_shape((3, target_size[0], target_size[1]))
    latent_channels = latent_shape[0]
    config["latent_channels"] = latent_channels
    config["vae_type"] = "rae"  # 标记为RAE
    # 保存实际使用的levels（从data_module获取，如果使用WeatherDataModule）
    if hasattr(data_module, "levels") and data_module.levels is not None:
        config["levels"] = data_module.levels
    elif args.levels is not None:
        config["levels"] = args.levels
    # 对于预处理数据，从metadata获取levels（如果有）
    if hasattr(data_module, "metadata") and "levels" in data_module.metadata:
        config["levels"] = data_module.metadata["levels"]

    # 打印实际使用的levels
    if "levels" in config:
        print(f"实际使用的levels: {config['levels']}")

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    latent_channels, latent_h, latent_w = latent_shape
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

    # 如果decoder可训练，添加decoder参数到优化器
    if not args.freeze_decoder:
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
        effective_batch = (
            args.batch_size
            * torch.cuda.device_count()
            * args.gradient_accumulation_steps
        )
        print(
            f"  有效batch size: {effective_batch} (每个GPU: {args.batch_size}, 梯度累积: {args.gradient_accumulation_steps})"
        )
    else:
        print(f"  多GPU训练: 否")
        effective_batch = args.batch_size * args.gradient_accumulation_steps
        print(
            f"  有效batch size: {effective_batch} (梯度累积: {args.gradient_accumulation_steps})"
        )
    print(f"  VAE类型: RAE")
    print(f"  Encoder冻结: {args.freeze_encoder}")
    print(f"  Decoder冻结: {args.freeze_decoder}")
    print(f"  主batch size: {args.batch_size}")
    print(f"  VAE batch size: {args.vae_batch_size} (用于分批编码)")
    print(f"  梯度累积步数: {args.gradient_accumulation_steps}")
    if args.use_amp:
        print(f"  混合精度训练: 是 ({args.amp_dtype})")
    else:
        print(f"  混合精度训练: 否 (FP32)")
    print(f"  学习率: {args.lr}")
    print(f"  权重衰减: {args.weight_decay}")
    print(f"  早停耐心: {args.early_stopping}")

    trainer = RAELatentUNetTrainer(
        model=model,
        vae_wrapper=vae_wrapper,
        vae_batch_size=args.vae_batch_size,
        device=device,
        use_multi_gpu=use_multi_gpu,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_amp=args.use_amp,
        amp_dtype=args.amp_dtype,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
    )

    # 如果decoder可训练，将decoder参数添加到优化器
    if not args.freeze_decoder:
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
        import shutil

        src_stats = Path(args.preprocessed_data_dir) / "normalizer_stats.pkl"
        dst_stats = output_dir / "normalizer_stats.pkl"
        shutil.copy(src_stats, dst_stats)

        with open(dst_stats, "rb") as f:
            stats = pickle.load(f)
        stats["vae_type"] = "rae"
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
        vae_info = {
            "vae_type": "rae",
            "rae_encoder_cls": args.rae_encoder_cls,
            "rae_encoder_config_path": args.rae_encoder_config_path,
            "rae_encoder_input_size": args.rae_encoder_input_size,
            "rae_decoder_config_path": args.rae_decoder_config_path,
            "rae_decoder_patch_size": args.rae_decoder_patch_size,
            "rae_pretrained_decoder_path": args.rae_pretrained_decoder_path,
            "freeze_encoder": args.freeze_encoder,
            "freeze_decoder": args.freeze_decoder,
        }
        if args.rae_normalization_stat_path:
            vae_info["rae_normalization_stat_path"] = args.rae_normalization_stat_path

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
    print("  1. 使用 predict_rae.py 进行预测")
    print("  2. 如果需要概率预测，可以继续训练Diffusion模型")


if __name__ == "__main__":
    main()
