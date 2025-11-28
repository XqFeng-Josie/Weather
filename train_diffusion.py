"""
训练扩散模型（Step 5: 扩散式概率预测）

使用方法:
    python train_diffusion.py --preprocessed-data-dir /path/to/preprocessed --variable 2m_temperature
"""

import argparse
import torch
from pathlib import Path
from tqdm import tqdm
import json
import pickle

from weatherdiff.diffusion import WeatherDiffusion, DDPMScheduler
from weatherdiff.diffusion.diffusion_model import DiffusionTrainer
from weatherdiff.vae import SDVAEWrapper
from weatherdiff.utils import WeatherDataModule


def encode_in_batches(vae_wrapper, images, vae_batch_size=4, device="cuda"):
    """
    分批编码图像到潜空间（避免显存溢出）

    Args:
        vae_wrapper: VAE包装器
        images: (N, C, H, W) 图像tensor
        vae_batch_size: VAE编码时的子批次大小
        device: 设备

    Returns:
        latents: (N, 4, H//8, W//8) 潜向量
    """
    N = images.shape[0]
    latent_list = []

    for i in range(0, N, vae_batch_size):
        end_idx = min(i + vae_batch_size, N)
        batch = images[i:end_idx].to(device)
        latent_batch = vae_wrapper.encode(batch)
        latent_list.append(latent_batch.cpu())  # 立即移回CPU释放显存

        # 清理显存
        del batch, latent_batch
        torch.cuda.empty_cache()

    # 合并所有batch
    latents = torch.cat(latent_list, dim=0).to(device)
    return latents


def main():
    parser = argparse.ArgumentParser(description="训练扩散模型")

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

    # VAE参数
    parser.add_argument(
        "--vae-model-id",
        type=str,
        default="stable-diffusion-v1-5/stable-diffusion-v1-5",
        help="VAE模型ID",
    )

    # 模型参数
    parser.add_argument("--input-length", type=int, default=12, help="输入序列长度")
    parser.add_argument("--output-length", type=int, default=4, help="输出序列长度")
    parser.add_argument("--base-channels", type=int, default=128, help="基础通道数")
    parser.add_argument("--depth", type=int, default=3, help="网络深度")

    # 扩散参数
    parser.add_argument(
        "--num-train-timesteps", type=int, default=1000, help="训练时间步数"
    )
    parser.add_argument(
        "--beta-schedule",
        type=str,
        default="linear",
        choices=["linear", "scaled_linear"],
        help="beta调度方式",
    )

    # 训练参数
    parser.add_argument("--batch-size", type=int, default=8, help="批次大小")
    parser.add_argument(
        "--vae-batch-size",
        type=int,
        default=4,
        help="VAE编码时的子批次大小（控制显存占用）",
    )
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--save-interval", type=int, default=10, help="保存间隔")

    # 数据处理参数
    parser.add_argument(
        "--normalization",
        type=str,
        default="minmax",
        choices=["minmax", "zscore"],
        help="归一化方法",
    )
    parser.add_argument("--target-size", type=str, default="512,512", help="目标尺寸")

    # 其他参数
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="设备",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument(
        "--output-dir", type=str, default="outputs/diffusion", help="输出目录"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)

    # 解析target_size
    h, w = map(int, args.target_size.split(","))
    target_size = (h, w)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("训练扩散模型 - Step 5: 扩散式概率预测")
    print("=" * 80)

    # 注意：配置将在数据加载后更新以包含实际使用的 levels

    # ========================================================================
    # Step 1: 加载VAE
    # ========================================================================
    print("\n" + "-" * 80)
    print("Step 1: 加载VAE")
    print("-" * 80)

    vae_wrapper = SDVAEWrapper(model_id=args.vae_model_id, device=args.device)

    # ========================================================================
    # Step 2: 加载和预处理数据
    # ========================================================================
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

        # 从预处理数据获取target_size
        target_size = tuple(data_module.metadata["target_size"])

        # ========================================================================
        # 从预处理数据复制归一化参数
        import shutil

        src_stats = Path(args.preprocessed_data_dir) / "normalizer_stats.pkl"
        dst_stats = output_dir / "normalizer_stats.pkl"
        shutil.copy(src_stats, dst_stats)

        # 添加VAE信息
        with open(dst_stats, "rb") as f:
            stats = pickle.load(f)
        stats["vae_model_id"] = args.vae_model_id
        with open(dst_stats, "wb") as f:
            pickle.dump(stats, f)
        # ========================================================================
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
    config = vars(args)
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

    print(f"\n数据信息:")
    print(f"  图像尺寸: {target_size}")
    print(f"  潜向量尺寸: ({target_size[0]//8}, {target_size[1]//8})")
    print(f"  训练批次数: {len(train_loader)}")
    print(f"  验证批次数: {len(val_loader)}")

    # ========================================================================
    # Step 3: 创建扩散模型和调度器
    # ========================================================================
    print("\n" + "-" * 80)
    print("Step 3: 创建扩散模型和调度器")
    print("-" * 80)

    model = WeatherDiffusion(
        input_length=args.input_length,
        output_length=args.output_length,
        latent_channels=4,
        base_channels=args.base_channels,
        depth=args.depth,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数:")
    print(f"  输入序列长度: {args.input_length}")
    print(f"  输出序列长度: {args.output_length}")
    print(f"  基础通道数: {args.base_channels}")
    print(f"  网络深度: {args.depth}")
    print(f"  总参数量: {n_params:,}")

    # 创建DDPM调度器
    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps, beta_schedule=args.beta_schedule
    )
    print(f"\n调度器参数:")
    print(f"  训练时间步: {args.num_train_timesteps}")
    print(f"  Beta调度: {args.beta_schedule}")

    # 创建训练器
    trainer = DiffusionTrainer(
        model=model,
        vae_wrapper=vae_wrapper,
        scheduler=scheduler,
        device=args.device,
        lr=args.lr,
    )

    # ========================================================================
    # Step 4: 开始训练
    # ========================================================================
    print("\n" + "-" * 80)
    print("Step 4: 开始训练")
    print("-" * 80)

    print(f"训练配置:")
    print(f"  主batch size: {args.batch_size}")
    print(f"  VAE batch size: {args.vae_batch_size} (用于分批编码)")
    print(f"  学习率: {args.lr}")
    print(f"  扩散步数: {args.num_train_timesteps}")

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    vae_batch_size = args.vae_batch_size

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 80)

        # 训练
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc="Training")
        for inputs, targets in pbar:
            B, T_in, C, H, W = inputs.shape
            T_out = targets.shape[1]

            # VAE分批编码（避免显存溢出）
            with torch.no_grad():
                inputs_flat = inputs.reshape(B * T_in, C, H, W)
                condition = encode_in_batches(
                    vae_wrapper, inputs_flat, vae_batch_size, args.device
                )
                condition = condition.reshape(B, T_in, 4, H // 8, W // 8)

                targets_flat = targets.reshape(B * T_out, C, H, W)
                latent_target = encode_in_batches(
                    vae_wrapper, targets_flat, vae_batch_size, args.device
                )
                latent_target = latent_target.reshape(B, T_out, 4, H // 8, W // 8)

            # 采样噪声和时间步
            noise = torch.randn_like(latent_target)
            timesteps = torch.randint(
                0, scheduler.num_train_timesteps, (B,), device=args.device
            ).long()

            # 添加噪声
            noisy_latent = scheduler.add_noise(latent_target, noise, timesteps)

            # 预测噪声（训练）
            trainer.optimizer.zero_grad()
            predicted_noise = model(noisy_latent, timesteps, condition)
            loss = trainer.criterion(predicted_noise, noise)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            trainer.optimizer.step()

            train_losses.append(loss.item())

            # 更新进度条显示当前损失和移动平均损失
            avg_loss = sum(train_losses) / len(train_losses)
            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "avg_loss": f"{avg_loss:.4f}"}
            )

        avg_train_loss = sum(train_losses) / len(train_losses)
        history["train_loss"].append(avg_train_loss)

        # 验证
        model.eval()
        val_losses = []

        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc="Validating")
            for inputs, targets in pbar_val:
                B, T_in, C, H, W = inputs.shape
                T_out = targets.shape[1]

                # VAE分批编码（避免显存溢出）
                inputs_flat = inputs.reshape(B * T_in, C, H, W)
                condition = encode_in_batches(
                    vae_wrapper, inputs_flat, vae_batch_size, args.device
                )
                condition = condition.reshape(B, T_in, 4, H // 8, W // 8)

                targets_flat = targets.reshape(B * T_out, C, H, W)
                latent_target = encode_in_batches(
                    vae_wrapper, targets_flat, vae_batch_size, args.device
                )
                latent_target = latent_target.reshape(B, T_out, 4, H // 8, W // 8)

                # 采样噪声
                noise = torch.randn_like(latent_target)
                timesteps = torch.randint(
                    0, scheduler.num_train_timesteps, (B,), device=args.device
                ).long()

                noisy_latent = scheduler.add_noise(latent_target, noise, timesteps)

                # 预测
                predicted_noise = model(noisy_latent, timesteps, condition)
                loss = torch.nn.functional.mse_loss(predicted_noise, noise)
                val_losses.append(loss.item())

                # 更新进度条显示当前损失和移动平均损失
                avg_val_loss = sum(val_losses) / len(val_losses)
                pbar_val.set_postfix(
                    {"loss": f"{loss.item():.4f}", "avg_loss": f"{avg_val_loss:.4f}"}
                )

        avg_val_loss = sum(val_losses) / len(val_losses)
        history["val_loss"].append(avg_val_loss)

        print(f"训练损失: {avg_train_loss:.6f}")
        print(f"验证损失: {avg_val_loss:.6f}")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "history": history,
                },
                output_dir / "best_model.pt",
            )
            print(f"✓ 保存最佳模型")

        # 定期保存
        if (epoch + 1) % args.save_interval == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "history": history,
                },
                output_dir / f"checkpoint_epoch_{epoch+1}.pt",
            )

    # 保存训练历史
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ========================================================================
    # 保存归一化统计量
    # ========================================================================
    if not args.preprocessed_data_dir:
        # 从data_module保存
        with open(output_dir / "normalizer_stats.pkl", "wb") as f:
            pickle.dump(
                {
                    "method": args.normalization,
                    "stats": data_module.normalizer.get_stats(),
                    "vae_model_id": args.vae_model_id,
                },
                f,
            )

    print("\n" + "=" * 80)
    print("训练完成!")
    print("=" * 80)
    print(f"\n模型保存在: {output_dir}")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print("\n下一步:")
    print("  使用 predict_diffusion.py 进行预测和评估")


if __name__ == "__main__":
    main()
