"""
扩散模型预测脚本 - 概率式天气预测

📋 文件作用:
    使用训练好的扩散模型进行概率式预测，通过采样生成多个可能的未来状态。
    
🔄 预测流程:
    1. 加载训练好的扩散模型和配置
    2. 加载VAE模型
    3. 加载测试数据
    4. VAE分批编码输入序列到潜空间
    5. 从纯噪声开始，逐步去噪生成预测（DDPM/DDIM采样）
    6. VAE分批解码回像素空间
    7. 计算评估指标（单次预测 + 集成预测）
    8. 生成可视化（时间序列图 + 世界地图对比 + 不确定性图）
    9. 保存所有结果
    
⚡ 显存优化:
    - VAE分批编码/解码（与train_diffusion.py一致）
    - 支持多次采样生成集成预测
    - 默认vae_batch_size=4，适合12GB GPU
    
📊 输出文件:
    - prediction_metrics.json: 详细评估指标
    - y_pred_*.npy: 预测数据
    - y_true_*.npy: 真值数据
    - ensemble_*.npy: 集成预测数据（如果num_samples>1）
    - timeseries_*.png: 时间序列对比图
    - spatial_comparison_*.png: 世界地图对比图
    - uncertainty_*.png: 预测不确定性图（集成预测）
    
📖 使用方法:
    # 单次预测
    python predict_diffusion.py \\
        --model-dir outputs/diffusion \\
        --time-slice 2020-01-01:2020-12-31 \\
        --vae-batch-size 4
    
    # 集成预测（生成多个样本）
    python predict_diffusion.py \\
        --model-dir outputs/diffusion \\
        --time-slice 2020-01-01:2020-12-31 \\
        --num-samples 10 \\
        --vae-batch-size 4
    
    # 使用DDIM采样（更快）
    python predict_diffusion.py \\
        --model-dir outputs/diffusion \\
        --sampling-method ddim \\
        --num-inference-steps 50

🎯 预期效果:
    单次预测:
        - RMSE: 3-6K
        - 相关系数: > 0.98
        - SSIM: > 0.98
    
    集成预测:
        - RMSE: 2-4K (集成后更准确)
        - 提供不确定性估计
"""

import argparse
import torch
import numpy as np
import json
import pickle
from pathlib import Path
from tqdm import tqdm

from weatherdiff.diffusion import WeatherDiffusion, DDPMScheduler
from weatherdiff.vae import SDVAEWrapper
from weatherdiff.utils import WeatherDataModule, calculate_metrics, format_metrics
from src.visualization import visualize_predictions_improved


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


def decode_in_batches(vae_wrapper, latents, vae_batch_size=4, device="cuda"):
    """
    分批解码潜向量到像素空间（避免显存溢出）

    Args:
        vae_wrapper: VAE包装器
        latents: (N, 4, H//8, W//8) 潜向量tensor
        vae_batch_size: VAE解码时的子批次大小
        device: 设备

    Returns:
        images: (N, 3, H, W) 图像
    """
    N = latents.shape[0]
    image_list = []

    for i in range(0, N, vae_batch_size):
        end_idx = min(i + vae_batch_size, N)
        batch = latents[i:end_idx].to(device)
        image_batch = vae_wrapper.decode(batch)
        image_list.append(image_batch.cpu())  # 立即移回CPU释放显存

        # 清理显存
        del batch, image_batch
        torch.cuda.empty_cache()

    # 合并所有batch
    images = torch.cat(image_list, dim=0).to(device)
    return images


def ddpm_sample(
    model, condition, latent_shape, scheduler, device, num_inference_steps=None
):
    """
    DDPM采样（逐步去噪）

    Args:
        model: 扩散模型
        condition: 条件（输入序列的潜向量）
        latent_shape: 潜向量形状 (B, T_out, 4, H//8, W//8)
        scheduler: DDPM调度器
        device: 设备
        num_inference_steps: 推理步数（None表示使用全部训练步数）

    Returns:
        samples: 采样结果 (B, T_out, 4, H//8, W//8)
    """
    # 从纯噪声开始
    latent = torch.randn(latent_shape, device=device)

    # 设置推理步数
    if num_inference_steps is None:
        num_inference_steps = scheduler.num_train_timesteps

    timesteps = torch.linspace(
        scheduler.num_train_timesteps - 1,
        0,
        num_inference_steps,
        dtype=torch.long,
        device=device,
    )

    # 逐步去噪
    for t in tqdm(timesteps, desc="DDPM采样"):
        with torch.no_grad():
            # 预测噪声
            noise_pred = model(
                latent, t.unsqueeze(0).expand(latent_shape[0]), condition
            )

            # 去噪一步
            latent = scheduler.step(noise_pred, t, latent)

    return latent


def main():
    parser = argparse.ArgumentParser(description="扩散模型预测")

    # 模型参数
    parser.add_argument(
        "--model-dir",
        type=str,
        default="outputs/diffusion",
        help="模型目录（包含best_model.pt和config.json）",
    )

    # 数据参数
    parser.add_argument(
        "--data-path",
        type=str,
        default="gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr",
        help="数据路径",
    )
    parser.add_argument(
        "--time-slice", type=str, default="2020-01-01:2020-12-31", help="预测时间范围"
    )

    # 采样参数
    parser.add_argument(
        "--sampling-method",
        type=str,
        default="ddpm",
        choices=["ddpm", "ddim"],
        help="采样方法",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="推理步数（None=使用全部训练步数）",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="每个输入生成的样本数（用于集成预测）",
    )

    # 输出参数
    parser.add_argument(
        "--output-dir", type=str, default=None, help="输出目录（默认使用模型目录）"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="预测批次大小")
    parser.add_argument(
        "--vae-batch-size",
        type=int,
        default=4,
        help="VAE编码/解码批次大小（控制显存占用）",
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=None,
        help="选择特定的气压层进行评估/可视化 (e.g. --levels 500 or --levels 500 700 850). "
        "Levels必须与训练时配置中的levels一致。如果不指定，将使用所有levels。",
    )

    # 其他参数
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="设备",
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir) if args.output_dir else model_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("扩散模型预测 - 概率式天气预测")
    print("=" * 80)
    print(f"模型目录: {model_dir}")
    print(f"预测时间: {args.time_slice}")
    print(f"采样方法: {args.sampling_method}")
    print(f"样本数: {args.num_samples}")
    print(f"输出目录: {output_dir}")

    # ========================================================================
    # Step 1: 加载配置
    # ========================================================================
    print("\n" + "-" * 80)
    print("Step 1: 加载配置")
    print("-" * 80)

    config_path = model_dir / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    print(f"✓ 加载配置: {config_path}")
    print(f"  输入序列长度: {config['input_length']}")
    print(f"  输出序列长度: {config['output_length']}")
    print(f"  归一化方法: {config['normalization']}")
    print(f"  VAE模型: {config['vae_model_id']}")

    # 从config读取训练时使用的 levels（如果有的话）
    available_levels = config.get("levels", None)
    if available_levels is not None:
        print(f"  训练时使用的levels: {available_levels}")
    else:
        print(f"  训练时使用的levels: 所有可用的levels（默认）")

    # 处理命令行指定的 --levels 参数
    selected_levels = args.levels
    channel_indices = None  # 用于后续选择指定的 channels
    if selected_levels is not None:
        if available_levels is None:
            raise ValueError(
                "No 'levels' found in config. Cannot select specific levels. "
                "Please use all levels (omit --levels argument)."
            )

        # 确保 available_levels 是列表
        if not isinstance(available_levels, list):
            available_levels = [available_levels]

        # 检查用户指定的 levels 是否在可用的 levels 中
        invalid_levels = [l for l in selected_levels if l not in available_levels]
        if invalid_levels:
            raise ValueError(
                f"Invalid levels: {invalid_levels}. "
                f"Available levels from config: {available_levels}"
            )

        # 将 level 值转换为 channel 索引
        # channel 的顺序与 available_levels 的顺序一致
        channel_indices = []
        for level in selected_levels:
            if level in available_levels:
                channel_indices.append(available_levels.index(level))

        print(
            f"  选择levels {selected_levels} 进行评估/可视化 "
            f"(channel indices: {channel_indices})"
        )
    else:
        print(f"  使用所有levels进行评估/可视化")

    # 加载归一化参数
    normalizer_path = model_dir / "normalizer_stats.pkl"
    with open(normalizer_path, "rb") as f:
        normalizer_data = pickle.load(f)

    print(f"✓ 加载归一化参数: {normalizer_path}")

    # ========================================================================
    # Step 2: 加载VAE
    # ========================================================================
    print("\n" + "-" * 80)
    print("Step 2: 加载VAE")
    print("-" * 80)

    vae_wrapper = SDVAEWrapper(model_id=config["vae_model_id"], device=args.device)
    print(f"✓ VAE加载完成")

    # ========================================================================
    # Step 3: 加载数据（预测模式：不分割数据）
    # ========================================================================
    print("\n" + "-" * 80)
    print("Step 3: 加载数据")
    print("-" * 80)

    target_size = tuple(map(int, config["target_size"].split(",")))

    # 直接加载数据，不使用WeatherDataModule的分割逻辑
    import xarray as xr
    from weatherdiff.utils import (
        prepare_weather_data,
        WeatherSequenceDataset,
        Normalizer,
    )
    from torch.utils.data import DataLoader

    print(f"加载数据: {args.data_path}")
    ds = xr.open_zarr(args.data_path)

    # 时间切片
    start, end = args.time_slice.split(":")
    ds = ds.sel(time=slice(start, end))

    # 获取变量数据
    variable_da = ds[config["variable"]]

    # 如果有level维度，根据config中的levels进行选择（用于数据加载）
    # 注意：数据加载时需要使用训练时使用的所有levels，而不是命令行指定的selected_levels
    # selected_levels只用于后续的评估和可视化
    if "level" in variable_da.dims:
        if available_levels is not None:
            # 使用训练时指定的 levels（所有levels）
            print(f"  变量有level维度，使用训练时的levels: {available_levels}")
            variable_da = variable_da.sel(level=available_levels)
            # 确保顺序与训练时一致
            actual_levels = variable_da.level.values.tolist()
            print(f"  实际加载的levels: {actual_levels}")
        else:
            # 使用所有可用的 levels
            available_levels_from_data = variable_da.level.values.tolist()
            print(
                f"  变量有level维度，使用所有可用的levels: {available_levels_from_data}"
            )
            available_levels = available_levels_from_data
            actual_levels = available_levels_from_data

    data = variable_da.values  # (Time, H, W) 或 (Time, Level, H, W)
    print(f"原始数据 shape: {data.shape}")
    print(f"数据范围: [{data.min():.2f}, {data.max():.2f}]")
    print(f"时间范围: {start} 至 {end}")

    # 准备为图像格式
    data = prepare_weather_data(data, n_channels=3, target_size=target_size)
    print(f"处理后 shape: {data.shape}")
    print(f"  图像尺寸: {target_size}")
    print(f"  潜向量尺寸: ({target_size[0]//8}, {target_size[1]//8})")

    # 验证数据尺寸是否与target_size一致
    _, _, data_H, data_W = data.shape
    if data_H != target_size[0] or data_W != target_size[1]:
        raise ValueError(
            f"数据尺寸不匹配：\n"
            f"  数据加载后的尺寸: ({data_H}, {data_W})\n"
            f"  训练时target_size: {target_size}\n"
            f"  可能原因：prepare_weather_data未正确resize到target_size\n"
            f"  解决方案：检查prepare_weather_data的target_size参数是否正确传递"
        )

    # 归一化（使用训练时保存的参数）
    normalizer = Normalizer(method=config["normalization"])

    # 加载归一化统计量（支持按level归一化）
    if (
        "normalize_per_level" in normalizer_data
        and normalizer_data["normalize_per_level"]
    ):
        # 按level归一化：传递完整的normalizer_data
        normalizer.load_stats(normalizer_data)
        # 从config或metadata中获取n_channels_per_level
        if "n_channels" in config:
            normalizer.n_channels_per_level = config["n_channels"]
        else:
            # 尝试从数据形状推断
            _, C, _, _ = data.shape
            if "levels" in normalizer_data and normalizer_data["levels"]:
                n_levels = len(normalizer_data["levels"])
                if C % n_levels == 0:
                    normalizer.n_channels_per_level = C // n_levels
                    print(f"  推断每个level的通道数: {normalizer.n_channels_per_level}")
                else:
                    raise ValueError(
                        f"无法推断每个level的通道数。总通道数: {C}, Levels数: {n_levels}"
                    )
    else:
        # 全局归一化：只传递stats
        normalizer.load_stats(normalizer_data.get("stats", normalizer_data))

    data = normalizer.transform(data, name=config["variable"])
    print(f"归一化后范围: [{data.min():.2f}, {data.max():.2f}]")

    # 创建完整的序列数据集（不分割）
    full_dataset = WeatherSequenceDataset(
        data, config["input_length"], config["output_length"]
    )

    test_loader = DataLoader(
        full_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"✓ 数据加载完成（预测模式：不分割）")
    print(f"  总样本数: {len(full_dataset)}")
    print(f"  批次数: {len(test_loader)}")

    # ========================================================================
    # Step 4: 加载模型
    # ========================================================================
    print("\n" + "-" * 80)
    print("Step 4: 加载模型")
    print("-" * 80)

    model = WeatherDiffusion(
        input_length=config["input_length"],
        output_length=config["output_length"],
        latent_channels=4,
        base_channels=config["base_channels"],
        depth=config["depth"],
    )

    checkpoint_path = model_dir / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(args.device)
    model.eval()

    # 创建调度器
    scheduler = DDPMScheduler(
        num_train_timesteps=config["num_train_timesteps"],
        beta_schedule=config["beta_schedule"],
    )

    print(f"✓ 模型加载完成: {checkpoint_path}")
    print(f"  训练epoch: {checkpoint['epoch']}")
    print(f"  验证损失: {checkpoint['val_loss']:.6f}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ========================================================================
    # Step 5: 采样预测
    # ========================================================================
    print("\n" + "-" * 80)
    print("Step 5: 扩散采样预测（使用VAE分批编码/解码）")
    print("-" * 80)

    vae_batch_size = args.vae_batch_size
    print(f"  VAE batch size: {vae_batch_size}")
    print(f"  采样方法: {args.sampling_method}")
    print(f"  每个输入生成样本数: {args.num_samples}")

    all_predictions = []
    all_targets = []
    all_inputs = []

    if args.num_samples > 1:
        all_ensemble = []  # 存储所有样本用于集成

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="预测中"):
            B, T_in, C, H, W = inputs.shape
            T_out = targets.shape[1]

            # 编码输入到潜空间
            inputs_flat = inputs.reshape(B * T_in, C, H, W)
            condition = encode_in_batches(
                vae_wrapper, inputs_flat, vae_batch_size, args.device
            )
            condition = condition.reshape(B, T_in, 4, H // 8, W // 8)

            # 生成多个样本
            samples_list = []
            for sample_idx in range(args.num_samples):
                # 扩散采样
                latent_shape = (B, T_out, 4, H // 8, W // 8)
                latent_outputs = ddpm_sample(
                    model,
                    condition,
                    latent_shape,
                    scheduler,
                    args.device,
                    args.num_inference_steps,
                )

                # 解码回像素空间
                latent_outputs_flat = latent_outputs.reshape(
                    B * T_out, 4, H // 8, W // 8
                )
                outputs = decode_in_batches(
                    vae_wrapper, latent_outputs_flat.cpu(), vae_batch_size, args.device
                )

                # 验证解码后的尺寸
                _, _, decoded_H, decoded_W = outputs.shape
                outputs = outputs.reshape(B, T_out, C, decoded_H, decoded_W)

                # 严格验证：解码后的尺寸必须等于target_size（训练时的尺寸）
                if decoded_H != target_size[0] or decoded_W != target_size[1]:
                    raise ValueError(
                        f"维度不匹配错误：\n"
                        f"  VAE解码输出尺寸: ({decoded_H}, {decoded_W})\n"
                        f"  训练时target_size: {target_size}\n"
                        f"  预测时输入尺寸: ({H}, {W})\n"
                        f"  可能原因：\n"
                        f"    1. 预测时数据加载未使用正确的target_size\n"
                        f"    2. VAE配置与训练时不一致\n"
                        f"    3. 数据加载后未正确resize到target_size\n"
                        f"  解决方案：确保预测时使用与训练时相同的target_size和VAE配置"
                    )

                # 验证输入尺寸也等于target_size
                if H != target_size[0] or W != target_size[1]:
                    raise ValueError(
                        f"维度不匹配错误：\n"
                        f"  预测时输入尺寸: ({H}, {W})\n"
                        f"  训练时target_size: {target_size}\n"
                        f"  可能原因：数据加载时未正确resize到target_size\n"
                        f"  解决方案：检查prepare_weather_data是否正确使用target_size参数"
                    )

                samples_list.append(outputs.cpu().numpy())

            # 集成预测（平均）
            samples = np.stack(samples_list, axis=0)  # (num_samples, B, T_out, C, H, W)
            ensemble_pred = samples.mean(axis=0)  # (B, T_out, C, H, W)

            all_predictions.append(ensemble_pred)
            all_targets.append(targets.numpy())
            all_inputs.append(inputs.cpu().numpy())

            if args.num_samples > 1:
                all_ensemble.append(samples)

    y_pred = np.concatenate(all_predictions, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    X = np.concatenate(all_inputs, axis=0)

    if args.num_samples > 1:
        ensemble = np.concatenate(
            all_ensemble, axis=1
        )  # (num_samples, N, T_out, C, H, W)

    print(f"✓ 预测完成")
    print(f"  输入形状: {X.shape}")
    print(f"  预测形状: {y_pred.shape}")
    print(f"  真值形状: {y_true.shape}")
    if args.num_samples > 1:
        print(f"  集成样本形状: {ensemble.shape}")

    # ========================================================================
    # Step 6: 反归一化
    # ========================================================================
    print("\n" + "-" * 80)
    print("Step 6: 评估和反归一化")
    print("-" * 80)

    # 如果指定了 --levels，只选择对应的 channels 进行评估
    if channel_indices is not None:
        print(f"\n选择 channels {channel_indices} (levels {selected_levels}) 进行评估")
        y_pred_selected = y_pred[:, :, channel_indices, :, :]
        y_true_selected = y_true[:, :, channel_indices, :, :]
    else:
        y_pred_selected = y_pred
        y_true_selected = y_true

    # 归一化空间的指标
    print("\n归一化空间的指标:")
    metrics_norm = calculate_metrics(y_pred_selected, y_true_selected, ensemble=False)
    print(format_metrics(metrics_norm))

    # 反归一化到物理单位
    variable = config["variable"]
    C = y_pred.shape[2]
    H = y_pred.shape[3]
    W = y_pred.shape[4]

    y_pred_flat = y_pred.reshape(-1, C, H, W)
    y_true_flat = y_true.reshape(-1, C, H, W)

    y_pred_phys = normalizer.inverse_transform(y_pred_flat, name=variable)
    y_true_phys = normalizer.inverse_transform(y_true_flat, name=variable)

    y_pred_phys = y_pred_phys.reshape(y_pred.shape)
    y_true_phys = y_true_phys.reshape(y_true.shape)

    print("\n✓ 反归一化完成")
    print(f"  预测范围: [{y_pred_phys.min():.2f}, {y_pred_phys.max():.2f}] K")
    print(f"  真值范围: [{y_true_phys.min():.2f}, {y_true_phys.max():.2f}] K")

    # 如果指定了 --levels，只选择对应的 channels 进行评估
    if channel_indices is not None:
        y_pred_phys_selected = y_pred_phys[:, :, channel_indices, :, :]
        y_true_phys_selected = y_true_phys[:, :, channel_indices, :, :]
    else:
        y_pred_phys_selected = y_pred_phys
        y_true_phys_selected = y_true_phys

    # 物理空间的指标
    print("\n物理空间的指标 (原始尺度):")
    metrics_phys = calculate_metrics(
        y_pred_phys_selected, y_true_phys_selected, ensemble=False
    )
    print(format_metrics(metrics_phys))

    # 计算每个lead time的RMSE
    print("\n每个lead time的RMSE:")
    T_out = y_pred_phys_selected.shape[1]
    rmse_per_leadtime = {}
    for t in range(T_out):
        y_pred_t = y_pred_phys_selected[:, t, :, :, :]  # (N, C, H, W)
        y_true_t = y_true_phys_selected[:, t, :, :, :]
        rmse_t = np.sqrt(np.mean((y_pred_t - y_true_t) ** 2))
        rmse_per_leadtime[f"rmse_step_{t+1}"] = float(rmse_t)
        print(f"  Step {t+1} ({(t+1)*6}h): {rmse_t:.4f} K")

    # ========================================================================
    # Step 7: 保存结果
    # ========================================================================
    print("\n" + "-" * 80)
    print("Step 7: 保存结果")
    print("-" * 80)

    # 保存指标
    metrics_all = {
        "mode": "diffusion",
        "sampling_method": args.sampling_method,
        "num_samples": args.num_samples,
        "normalized_space": {k: float(v) for k, v in metrics_norm.items()},
        "physical_space": {k: float(v) for k, v in metrics_phys.items()},
        "physical_space_rmse_per_leadtime": rmse_per_leadtime,
        "time_slice": args.time_slice,
        "n_samples": int(y_pred.shape[0]),
    }

    metrics_path = output_dir / "prediction_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_all, f, indent=2)
    print(f"✓ 指标已保存: {metrics_path}")

    # 保存预测数据
    pred_dir = output_dir / "predictions_data"
    pred_dir.mkdir(exist_ok=True)
    np.save(pred_dir / "y_test_pred_norm.npy", y_pred)
    np.save(pred_dir / "y_test_norm.npy", y_true)
    np.save(pred_dir / "y_test.npy", y_true_phys)
    np.save(pred_dir / "y_test_pred.npy", y_pred_phys)

    if args.num_samples > 1:
        np.save(pred_dir / "ensemble_samples.npy", ensemble)
        print(f"✓ 集成样本已保存: {pred_dir}/ensemble_samples.npy")

    print(f"✓ 预测数据已保存: {pred_dir}/y_*.npy")

    # ========================================================================
    # Step 8: 生成可视化
    # ========================================================================
    print("\n" + "-" * 80)
    print("Step 8: 生成可视化")
    print("-" * 80)

    # 获取空间坐标
    import xarray as xr

    ds = xr.open_zarr(args.data_path)

    spatial_coords = None
    if hasattr(ds, "latitude") and hasattr(ds, "longitude"):
        lat_values = ds.latitude.values
        lon_values = ds.longitude.values

        # 获取预测数据的实际空间形状
        # y_pred_phys shape: (N, T, C, H, W)
        actual_H = y_pred_phys.shape[3]
        actual_W = y_pred_phys.shape[4]

        print(f"\n检查空间坐标:")
        print(f"  数据集坐标: lat={len(lat_values)}, lon={len(lon_values)}")
        print(f"  预测数据形状: H={actual_H}, W={actual_W}")

        # 检查坐标与数据形状的对应关系
        # ERA5 64x32数据集的维度顺序是 (time, longitude, latitude)
        # 所以 H 对应 longitude (64), W 对应 latitude (32)
        if len(lon_values) == actual_H and len(lat_values) == actual_W:
            # ERA5格式: H=64(longitude), W=32(latitude)
            # visualization.py期望: H=latitude, W=longitude
            # 解决方案：转置数据的空间维度
            print(f"  ✓ 坐标匹配 (ERA5格式: H={actual_H}(lon), W={actual_W}(lat))")
            print(f"  转置空间维度以适配visualization (H←→W)")

            # 转置空间维度: (N, T, C, H, W) → (N, T, C, W, H)
            y_pred_phys = np.transpose(y_pred_phys, (0, 1, 2, 4, 3))
            y_true_phys = np.transpose(y_true_phys, (0, 1, 2, 4, 3))

            # 现在 H=32(latitude), W=64(longitude)，符合visualization期望
            spatial_coords = {
                "lat": lat_values,  # 32个纬度值
                "lon": lon_values,  # 64个经度值
            }
            print(
                f"  转置后: H={y_pred_phys.shape[3]}(lat), W={y_pred_phys.shape[4]}(lon)"
            )
        elif len(lat_values) == actual_H and len(lon_values) == actual_W:
            # 标准格式: H=latitude, W=longitude（已经正确）
            spatial_coords = {
                "lat": lat_values,
                "lon": lon_values,
            }
            print(f"  ✓ 坐标匹配 (标准格式: H={actual_H}(lat), W={actual_W}(lon))")
        else:
            # 尺寸完全不匹配，使用默认坐标
            print(
                f"  ⚠ 坐标维度不匹配 (lat:{len(lat_values)}, lon:{len(lon_values)} vs H:{actual_H}, W:{actual_W})"
            )
            print(f"  使用默认坐标...")
            spatial_coords = {
                "lat": np.linspace(-90, 90, actual_H),
                "lon": np.linspace(0, 360, actual_W),
            }

    # 生成可视化
    # 注意：WeatherDiff使用minmax归一化（[-1,1]），与传统模型的zscore不同
    # 这里传递已经反归一化的物理值数据，norm_params=None
    # 可视化函数会生成timeseries_overall（物理值），但不会生成timeseries_physical
    # （因为WeatherDiff的归一化方式与可视化函数预期不兼容）
    visualize_predictions_improved(
        y_true_phys_selected,  # y_test (物理值数据，可能已转置)
        y_pred_phys_selected,  # y_test_pred (物理值数据，可能已转置)
        metrics_phys,  # test_metrics (物理空间指标)
        [variable],  # variables
        "diffusion",  # model_name
        output_dir,  # output_dir
        "spatial",  # data_format
        norm_params=None,  # norm_params (不提供，因为归一化方式不同)
        spatial_coords=spatial_coords,  # spatial_coords
    )

    print(f"✓ 可视化已生成")
    print(f"  - timeseries_overall_{variable}.png (物理值)")
    print(
        f"  - timeseries_physical_{variable}.png (未生成，因为已使用物理值绘制overall)"
    )
    print(f"  - leadtime_independent_{variable}.png")
    print(f"  - rmse_vs_leadtime_{variable}.png")
    print(f"  - spatial_comparison_{variable}.png")

    # ========================================================================
    # 总结
    # ========================================================================
    print("\n" + "=" * 80)
    print("预测完成!")
    print("=" * 80)

    print(f"\n模型: DIFFUSION ({args.sampling_method.upper()})")
    print(f"总结 (物理空间):")
    print(f"  样本数: {y_pred.shape[0]}")
    print(f"  RMSE: {metrics_phys['rmse']:.4f} K")
    print(f"  MAE: {metrics_phys['mae']:.4f} K")
    print(f"  相关系数: {metrics_phys['correlation']:.4f}")
    print(f"  SSIM: {metrics_phys['ssim']:.4f}")

    if args.num_samples > 1:
        print(f"\n集成预测:")
        print(f"  样本数: {args.num_samples}")
        print(f"  不确定性数据已保存")

    print(f"\n结果保存在: {output_dir}")
    print(f"  - prediction_metrics.json: 详细指标")
    print(f"  - y_pred_*.npy: 预测数据")
    print(f"  - *.png: 可视化图片")


if __name__ == "__main__":
    main()
