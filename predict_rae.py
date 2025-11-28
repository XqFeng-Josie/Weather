"""
RAE Latent U-Net预测脚本

使用方法:
    python predict_rae.py --model-dir outputs/rae_latent_unet --time-slice 2020-01-01:2020-12-31
"""

import argparse
import torch
import numpy as np
import json
import pickle
from pathlib import Path
from tqdm import tqdm
import os
from weatherdiff.unet import LatentUNet
from weatherdiff.vae import RAEWrapper
from weatherdiff.utils import WeatherDataModule, calculate_metrics, format_metrics
from src.visualization import visualize_predictions_improved


def detect_lon_first(data_array):
    """
    判断数据的空间维度顺序是否为 (lon, lat)

    Args:
        data_array: xarray DataArray

    Returns:
        True  -> 空间顺序为 (lon, lat)
        False -> 空间顺序为 (lat, lon)
        None  -> 无法判断（缺少维度名称）
    """
    dims = list(getattr(data_array, "dims", []))
    if not dims:
        return None

    lat_dim = next((dim for dim in dims if "lat" in dim.lower()), None)
    lon_dim = next((dim for dim in dims if "lon" in dim.lower()), None)

    if lat_dim is None or lon_dim is None:
        return None

    lat_idx = dims.index(lat_dim)
    lon_idx = dims.index(lon_dim)

    return lon_idx < lat_idx


def reorder_lat_lon_if_needed(data_array):
    """
    Ensure latitude dimension precedes longitude for downstream image prep.
    """
    dims = list(getattr(data_array, "dims", []))
    if not dims:
        print("  未检测到维度信息，跳过纬/经度顺序检查")
        return data_array

    lat_dim = next((dim for dim in dims if "lat" in dim.lower()), None)
    lon_dim = next((dim for dim in dims if "lon" in dim.lower()), None)

    if lat_dim is None or lon_dim is None:
        print(f"  未检测到纬/经度维度，dims={dims}")
        return data_array

    lon_idx = dims.index(lon_dim)
    lat_idx = dims.index(lat_dim)

    if lon_idx < lat_idx:
        print(f"  检测到维度顺序 {dims} (longitude 在 latitude 前)，转置为标准顺序")
        target_dims = [dim for dim in dims if dim not in (lat_dim, lon_dim)]
        target_dims.extend([lat_dim, lon_dim])
        data_array = data_array.transpose(*target_dims)
        print(f"  转置后维度: {data_array.dims}")
    else:
        print(f"  维度顺序符合标准: {dims}")

    return data_array


def encode_in_batches(vae_wrapper, images, vae_batch_size=4, device="cuda"):
    """
    分批编码图像到潜空间（避免显存溢出）

    Args:
        vae_wrapper: RAE包装器
        images: (N, C, H, W) 图像tensor (在CPU上)
        vae_batch_size: VAE编码时的子批次大小
        device: 设备

    Returns:
        latents: (N, latent_dim, H_latent, W_latent) 潜向量
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
        vae_wrapper: RAE包装器
        latents: (N, latent_dim, H_latent, W_latent) 潜向量tensor (在CPU上)
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


def main():
    parser = argparse.ArgumentParser(description="RAE Latent U-Net预测")

    # 模型参数
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
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

    # 输出参数
    parser.add_argument(
        "--output-dir", type=str, default=None, help="输出目录（默认使用模型目录）"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="预测批次大小")
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

    # RAE参数（可选，用于覆盖保存的配置）
    parser.add_argument(
        "--rae-pretrained-decoder-path",
        type=str,
        default=None,
        help="RAE预训练decoder路径（如果未保存在配置中，则必须提供）",
    )
    parser.add_argument(
        "--rae-normalization-stat-path",
        type=str,
        default=None,
        help="RAE归一化统计量路径（可选）",
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
    print("RAE Latent U-Net预测")
    print("=" * 80)
    print(f"模型目录: {model_dir}")
    print(f"预测时间: {args.time_slice}")
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

    # 验证VAE类型
    vae_type = config.get("vae_type", "rae")
    if vae_type != "rae":
        raise ValueError(
            f"此脚本仅支持RAE，但配置中vae_type={vae_type}。请使用predict_vae.py或predict_unet.py"
        )

    # 加载归一化参数
    normalizer_path = model_dir / "normalizer_stats.pkl"
    with open(normalizer_path, "rb") as f:
        normalizer_data = pickle.load(f)

    print(f"✓ 加载归一化参数: {normalizer_path}")

    # ========================================================================
    # Step 2: 加载RAE
    # ========================================================================
    print("\n" + "-" * 80)
    print("Step 2: 加载RAE")
    print("-" * 80)

    print(f"使用RAE: encoder={normalizer_data.get('rae_encoder_cls', 'SigLIP2wNorm')}")
    # 构建encoder_params
    encoder_params = {}
    encoder_cls = normalizer_data.get("rae_encoder_cls", "SigLIP2wNorm")
    encoder_config_path = normalizer_data.get(
        "rae_encoder_config_path", "google/siglip2-base-patch16-256"
    )

    if encoder_cls == "Dinov2withNorm":
        encoder_params = {"dinov2_path": encoder_config_path, "normalize": True}
    elif encoder_cls == "SigLIP2wNorm":
        encoder_params = {"model_name": encoder_config_path}
    elif encoder_cls == "MAEwNorm":
        encoder_params = {"model_name": encoder_config_path}

    # 获取pretrained_decoder_path（优先使用命令行参数，否则使用保存的配置）
    pretrained_decoder_path = args.rae_pretrained_decoder_path
    if pretrained_decoder_path is None:
        pretrained_decoder_path = normalizer_data.get(
            "rae_pretrained_decoder_path", None
        )

    # 验证pretrained_decoder_path是否提供
    if pretrained_decoder_path is None:
        raise ValueError(
            "pretrained_decoder_path 是必需的参数。\n"
            "请通过以下方式之一提供：\n"
            "  1. 使用 --rae-pretrained-decoder-path 命令行参数\n"
            "  2. 确保训练时已保存该路径到 normalizer_stats.pkl\n"
            "例如：python predict_rae.py --model-dir <dir> --rae-pretrained-decoder-path models/decoders/..."
        )

    # 获取normalization_stat_path（优先使用命令行参数，否则使用保存的配置）
    normalization_stat_path = args.rae_normalization_stat_path
    if normalization_stat_path is None:
        normalization_stat_path = normalizer_data.get(
            "rae_normalization_stat_path", None
        )

    vae_wrapper = RAEWrapper(
        encoder_cls=encoder_cls,
        encoder_config_path=encoder_config_path,
        encoder_input_size=normalizer_data.get("rae_encoder_input_size", 256),
        encoder_params=encoder_params,
        decoder_config_path=normalizer_data.get(
            "rae_decoder_config_path", "facebook/vit-mae-base"
        ),
        decoder_patch_size=normalizer_data.get("rae_decoder_patch_size", 16),
        pretrained_decoder_path=pretrained_decoder_path,
        normalization_stat_path=normalization_stat_path,
        device=args.device,
        freeze_encoder=normalizer_data.get("freeze_encoder", True),
        freeze_decoder=normalizer_data.get("freeze_decoder", False),
    )

    print(f"✓ RAE加载完成")

    # ========================================================================
    # Step 3: 加载数据（预测模式：不分割数据）
    # ========================================================================
    print("\n" + "-" * 80)
    print("Step 3: 加载数据")
    print("-" * 80)

    # 解析target_size
    if isinstance(config["target_size"], str):
        target_size = tuple(map(int, config["target_size"].split(",")))
    else:
        target_size = tuple(config["target_size"])

    # 获取latent_channels
    latent_shape = vae_wrapper.get_latent_shape((3, target_size[0], target_size[1]))
    latent_channels, latent_h, latent_w = latent_shape

    # 直接加载数据，不使用WeatherDataModule的分割逻辑
    import xarray as xr
    from torch.utils.data import DataLoader
    from weatherdiff.utils import (
        prepare_weather_data,
        WeatherSequenceDataset,
        Normalizer,
    )

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

    variable_da = reorder_lat_lon_if_needed(variable_da)

    lon_first_flag = detect_lon_first(variable_da)
    if lon_first_flag is not None:
        config["_spatial_lon_first"] = lon_first_flag
        orientation_desc = (
            "Longitude-First (lon->lat)"
            if lon_first_flag
            else "Latitude-First (lat->lon)"
        )
        print(f"  空间维度顺序: {orientation_desc} | dims={variable_da.dims}")
    else:
        print(f"  空间维度顺序: 未检测到纬/经度名称 | dims={variable_da.dims}")
    data = variable_da.values  # (Time, H, W) 或 (Time, Level, H, W)
    print(f"原始数据 shape: {data.shape}")
    print(f"数据范围: [{data.min():.2f}, {data.max():.2f}]")
    print(f"时间范围: {start} 至 {end}")

    # 准备为图像格式
    data = prepare_weather_data(data, n_channels=3, target_size=target_size)
    print(f"处理后 shape: {data.shape}")
    print(f"  图像尺寸: {target_size}")

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

    # 获取latent shape
    latent_shape = vae_wrapper.get_latent_shape((3, target_size[0], target_size[1]))
    latent_channels, latent_h, latent_w = latent_shape
    print(f"  潜向量尺寸: ({latent_channels}, {latent_h}, {latent_w})")

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

    # 获取数据形状信息
    sample_input, _ = full_dataset[0]
    T_in = sample_input.shape[0]
    T_out = config["output_length"]

    # 获取latent_channels
    if "latent_channels" in config:
        latent_channels = config["latent_channels"]
    else:
        latent_shape = vae_wrapper.get_latent_shape((3, target_size[0], target_size[1]))
        latent_channels = latent_shape[0]

    model = LatentUNet(
        input_length=config["input_length"],
        output_length=config["output_length"],
        latent_channels=latent_channels,
        base_channels=config["base_channels"],
        depth=config["depth"],
    )

    checkpoint_path = model_dir / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(args.device)
    model.eval()

    print(f"✓ 模型加载完成: {checkpoint_path}")
    print(f"  训练epoch: {checkpoint['epoch']}")
    print(f"  验证损失: {checkpoint['val_loss']:.6f}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ========================================================================
    # Step 5: 预测（潜空间 -> 像素空间）
    # ========================================================================
    print("\n" + "-" * 80)
    print("Step 5: 生成预测 (使用RAE分批编码/解码)")
    print("-" * 80)

    vae_batch_size = args.vae_batch_size
    print(f"  VAE batch size: {vae_batch_size} (控制显存占用)")

    all_predictions = []
    all_targets = []
    all_inputs = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="预测中(RAE Latent)"):
            B, T_in, C, H, W = inputs.shape
            T_out = targets.shape[1]

            # 编码到潜空间（分批处理避免显存溢出）
            inputs_flat = inputs.reshape(B * T_in, C, H, W)
            latent_inputs = encode_in_batches(
                vae_wrapper, inputs_flat, vae_batch_size, args.device
            )

            # 获取latent shape
            latent_shape = vae_wrapper.get_latent_shape((C, H, W))
            latent_channels_batch, latent_h_batch, latent_w_batch = latent_shape

            latent_inputs = latent_inputs.reshape(
                B, T_in, latent_channels_batch, latent_h_batch, latent_w_batch
            )

            # 潜空间预测（LatentUNet期望5维输入）
            latent_outputs = model(latent_inputs)

            # 解码回像素空间（分批处理避免显存溢出）
            latent_outputs_flat = latent_outputs.reshape(
                B * T_out, latent_channels_batch, latent_h_batch, latent_w_batch
            )
            outputs = decode_in_batches(
                vae_wrapper, latent_outputs_flat.cpu(), vae_batch_size, args.device
            )

            # decode_in_batches返回的是 (B*T_out, C, H, W)
            _, _, decoded_H, decoded_W = outputs.shape
            outputs = outputs.reshape(B, T_out, C, decoded_H, decoded_W)

            # 严格验证：解码后的尺寸必须等于target_size（训练时的尺寸）
            if decoded_H != target_size[0] or decoded_W != target_size[1]:
                raise ValueError(
                    f"维度不匹配错误：\n"
                    f"  RAE解码输出尺寸: ({decoded_H}, {decoded_W})\n"
                    f"  训练时target_size: {target_size}\n"
                    f"  预测时输入尺寸: ({H}, {W})\n"
                    f"  可能原因：\n"
                    f"    1. 预测时数据加载未使用正确的target_size\n"
                    f"    2. RAE配置与训练时不一致（特别是decoder配置）\n"
                    f"    3. 数据加载后未正确resize到target_size\n"
                    f"  解决方案：确保预测时使用与训练时相同的target_size和RAE配置"
                )

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())
            all_inputs.append(inputs.cpu().numpy())

    y_pred = np.concatenate(all_predictions, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    X = np.concatenate(all_inputs, axis=0)

    print(f"✓ 预测完成")
    print(f"  输入形状: {X.shape}")
    print(f"  预测形状: {y_pred.shape}")
    print(f"  真值形状: {y_true.shape}")

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
        "mode": os.path.basename(output_dir),
        "vae_type": "rae",
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
    np.save(pred_dir / "y_test.npy", y_true_phys)  # 真值
    np.save(pred_dir / "y_test_pred.npy", y_pred_phys)  # 预测值
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
    lon_first_flag = config.get("_spatial_lon_first")
    if hasattr(ds, "latitude") and hasattr(ds, "longitude"):
        lat_values = ds.latitude.values
        lon_values = ds.longitude.values

        actual_H = y_pred_phys.shape[3]
        actual_W = y_pred_phys.shape[4]

        print(f"\n检查空间坐标:")
        print(f"  数据集坐标: lat={len(lat_values)}, lon={len(lon_values)}")
        print(f"  预测数据形状: H={actual_H}, W={actual_W}")

        orientation_handled = False
        if lon_first_flag is True:
            print("  检测到训练数据为 Longitude-First（lon->lat），为可视化转置一次")
            y_pred_phys = np.transpose(y_pred_phys, (0, 1, 2, 4, 3))
            y_true_phys = np.transpose(y_true_phys, (0, 1, 2, 4, 3))
            orientation_handled = True
            actual_H = y_pred_phys.shape[3]
            actual_W = y_pred_phys.shape[4]
            print(f"  转置后: H={actual_H}(lat), W={actual_W}(lon)")
        elif lon_first_flag is False:
            print("  检测到训练数据为 Latitude-First（lat->lon），无需额外处理")
            orientation_handled = True
        else:
            print("  未能从配置确定空间顺序，尝试根据坐标长度推断...")

        if not orientation_handled:
            if len(lon_values) == actual_H and len(lat_values) == actual_W:
                print(f"  ✓ 坐标匹配 (ERA5格式: H={actual_H}(lon), W={actual_W}(lat))")
                print(f"  转置空间维度以适配visualization (H<->W)")

                y_pred_phys = np.transpose(y_pred_phys, (0, 1, 2, 4, 3))
                y_true_phys = np.transpose(y_true_phys, (0, 1, 2, 4, 3))
                actual_H = y_pred_phys.shape[3]
                actual_W = y_pred_phys.shape[4]
                print(f"  转置后: H={actual_H}(lat), W={actual_W}(lon)")
            elif len(lat_values) == actual_H and len(lon_values) == actual_W:
                print(f"  ✓ 坐标匹配 (标准格式: H={actual_H}(lat), W={actual_W}(lon))")
            else:
                print(
                    f"  ⚠ 坐标维度不匹配 (lat:{len(lat_values)}, lon:{len(lon_values)} vs H:{actual_H}, W:{actual_W})"
                )

        if (
            len(lat_values) == y_pred_phys.shape[3]
            and len(lon_values) == y_pred_phys.shape[4]
        ):
            spatial_coords = {
                "lat": lat_values,
                "lon": lon_values,
            }
        else:
            print(
                f"  使用默认坐标生成可视化 ({y_pred_phys.shape[3]}x{y_pred_phys.shape[4]}) ..."
            )
            spatial_coords = {
                "lat": np.linspace(-90, 90, y_pred_phys.shape[3]),
                "lon": np.linspace(0, 360, y_pred_phys.shape[4]),
            }

    # 生成可视化（使用选中的 channels）
    visualize_predictions_improved(
        y_true_phys_selected,
        y_pred_phys_selected,
        metrics_phys,
        [variable],
        "rae_latent_unet",
        output_dir,
        "spatial",
        norm_params=None,
        spatial_coords=spatial_coords,
    )

    print(f"✓ 可视化已生成")
    print(f"  - timeseries_overall_{variable}.png (物理值)")
    print(f"  - leadtime_independent_{variable}.png")
    print(f"  - rmse_vs_leadtime_{variable}.png")
    print(f"  - spatial_comparison_{variable}.png")

    # ========================================================================
    # 总结
    # ========================================================================
    print("\n" + "=" * 80)
    print("预测完成!")
    print("=" * 80)

    print(f"\nVAE类型: RAE")
    print(f"总结 (物理空间):")
    print(f"  样本数: {y_pred.shape[0]}")
    print(f"  RMSE: {metrics_phys['rmse']:.4f} K")
    print(f"  MAE: {metrics_phys['mae']:.4f} K")
    print(f"  相关系数: {metrics_phys['correlation']:.4f}")
    print(f"  SSIM: {metrics_phys['ssim']:.4f}")

    print(f"\n结果保存在: {output_dir}")
    print(f"  - prediction_metrics.json: 详细指标")
    print(f"  - y_pred_*.npy: 预测数据")
    print(f"  - *.png: 可视化图片")

    # 性能评价
    if metrics_phys["rmse"] < 3.0:
        print("\n✅ 预测效果优秀！")
    elif metrics_phys["rmse"] < 5.0:
        print("\n✅ 预测效果良好！")
    elif metrics_phys["rmse"] < 10.0:
        print("\n⚠️  预测效果一般")
    else:
        print("\n⚠️  预测效果较差，建议:")
        print("  1. 增加训练数据量")
        print("  2. 增加训练轮数")
        print("  3. 调整模型参数")


if __name__ == "__main__":
    main()
