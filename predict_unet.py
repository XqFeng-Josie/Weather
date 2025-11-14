"""
U-Net统一预测脚本 - 支持像素空间和潜空间两种模式

📋 文件作用:
    对训练好的U-Net模型（像素空间或潜空间）进行预测，生成评估指标和可视化图片。
    
🔄 预测流程:
    1. 根据mode参数选择像素空间或潜空间模式
    2. 加载训练好的模型和配置
    3. 加载归一化参数（与训练时保持一致）
    4. 【潜空间】加载VAE模型
    5. 加载测试数据
    6. 生成预测
       【潜空间】VAE分批编码（避免显存溢出）
       【潜空间】U-Net潜空间预测
       【潜空间】VAE分批解码回像素空间
    7. 计算评估指标（归一化空间 + 物理空间）
    8. 生成可视化（时间序列图 + 世界地图对比）
    9. 保存所有结果
    
⚡ 显存优化:
    使用VAE分批编码/解码策略（与train_latent_unet.py一致）
    - 避免一次性处理大量图像导致显存溢出
    - 默认vae_batch_size=4，适合12GB GPU
    - 可根据GPU显存调整（6GB用2，24GB用8）

📊 输出文件:
    - prediction_metrics.json: 详细评估指标
    - y_pred_*.npy: 预测数据（归一化 + 物理单位）
    - y_true_*.npy: 真值数据
    - timeseries_*.png: 时间序列对比图
    - spatial_comparison_*.png: 世界地图对比图 ⭐
    - rmse_vs_leadtime_*.png: RMSE随预测步长变化

📖 使用方法:
    # 像素空间U-Net预测
    python predict_unet.py --mode pixel \\
        --model-dir outputs/pixel_unet \\
        --time-slice 2020-02-01:2020-02-10
    
    # 潜空间U-Net预测（推荐配置）
    python predict_unet.py --mode latent \\
        --model-dir outputs/latent_unet \\
        --time-slice 2020-02-01:2020-02-10 \\
        --batch-size 32 \\
        --vae-batch-size 4  # 控制显存占用
    
    # 显存不足时调整
    python predict_unet.py --mode latent \\
        --model-dir outputs/latent_unet \\
        --batch-size 16 \\
        --vae-batch-size 2  # 减小VAE批次

🎯 预期效果:
    像素空间U-Net:
        - RMSE: 1-3K
        - 相关系数: > 0.995
        - SSIM: > 0.995
    
    潜空间U-Net:
        - RMSE: 2-5K
        - 相关系数: > 0.99
        - SSIM: > 0.99
"""

import argparse
import torch
import numpy as np
import json
import pickle
from pathlib import Path
from tqdm import tqdm

from weatherdiff.unet import WeatherUNet, LatentUNet
from weatherdiff.vae import SDVAEWrapper, RAEWrapper
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


def encode_in_batches(vae_wrapper, images, vae_batch_size=4, device="cuda"):
    """
    分批编码图像到潜空间（避免显存溢出）

    Args:
        vae_wrapper: VAE包装器
        images: (N, C, H, W) 图像tensor (在CPU上)
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
        latents: (N, 4, H//8, W//8) 潜向量tensor (在CPU上)
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


def predict_pixel_unet(args):
    """像素空间U-Net预测"""

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir) if args.output_dir else model_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("像素空间U-Net预测")
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

    # 加载归一化参数
    normalizer_path = model_dir / "normalizer_stats.pkl"
    with open(normalizer_path, "rb") as f:
        normalizer_data = pickle.load(f)

    print(f"✓ 加载归一化参数: {normalizer_path}")

    # ========================================================================
    # Step 2: 加载数据（预测模式：不分割数据）
    # ========================================================================
    print("\n" + "-" * 80)
    print("Step 2: 加载数据")
    print("-" * 80)

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
    data = variable_da.values  # (Time, H, W)
    print(f"原始数据 shape: {data.shape}")
    print(f"数据范围: [{data.min():.2f}, {data.max():.2f}]")
    print(f"时间范围: {start} 至 {end}")

    # 获取target_size（如果有）
    target_size = None
    if "target_size" in config and config["target_size"]:
        if isinstance(config["target_size"], str):
            target_size = tuple(map(int, config["target_size"].split(",")))
        else:
            target_size = tuple(config["target_size"])

    # 准备为图像格式
    data = prepare_weather_data(
        data, n_channels=config["n_channels"], target_size=target_size
    )
    print(f"处理后 shape: {data.shape}")
    if target_size:
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

    # 归一化（使用训练时保存的参数）
    normalizer = Normalizer(method=config["normalization"])
    normalizer.load_stats(normalizer_data["stats"])
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
    # Step 3: 加载模型
    # ========================================================================
    print("\n" + "-" * 80)
    print("Step 3: 加载模型")
    print("-" * 80)

    # 获取数据形状信息
    sample_input, sample_output = full_dataset[0]
    T_in, C, H, W = sample_input.shape
    T_out = sample_output.shape[0]

    in_channels = T_in * C
    out_channels = T_out * C

    model = WeatherUNet(
        in_channels=in_channels,
        out_channels=out_channels,
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
    # Step 4: 预测
    # ========================================================================
    print("\n" + "-" * 80)
    print("Step 4: 生成预测")
    print("-" * 80)

    all_predictions = []
    all_targets = []
    all_inputs = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="预测中"):
            inputs = inputs.to(args.device)
            B, T_in, C, H, W = inputs.shape
            T_out = targets.shape[1]

            # 展平时间维度
            inputs_flat = inputs.reshape(B, T_in * C, H, W)
            outputs = model(inputs_flat)
            outputs = outputs.reshape(B, T_out, C, H, W)

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

    return y_pred, y_true, normalizer, config, output_dir


def predict_latent_unet(args):
    """潜空间U-Net预测"""

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir) if args.output_dir else model_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("潜空间U-Net预测")
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

    # 检查VAE类型
    vae_type = normalizer_data.get("vae_type", "sd")  # 默认SD VAE
    if vae_type == "sd":
        vae_model_id = config.get(
            "vae_model_id",
            normalizer_data.get("vae_model_id", "stable-diffusion-v1-5"),
        )
        vae_train_mode = config.get(
            "vae_train_mode",
            normalizer_data.get("vae_train_mode", "pretrained"),
        )
        vae_pretrained_path = config.get(
            "vae_pretrained_path",
            normalizer_data.get("vae_pretrained_path", None),
        )
        freeze_vae = config.get(
            "freeze_vae",
            normalizer_data.get("freeze_vae", True),
        )

        print(f"使用Stable Diffusion VAE: {vae_model_id}")
        print(f"  VAE训练模式: {vae_train_mode}")
        if vae_pretrained_path:
            print(f"  预训练权重路径: {vae_pretrained_path}")
        print(f"  VAE冻结: {freeze_vae}")

        vae_wrapper = SDVAEWrapper(
            model_id=vae_model_id,
            device=args.device,
            train_mode=vae_train_mode,
            pretrained_path=vae_pretrained_path,
            freeze_vae=freeze_vae,
        )
        latent_channels = 4
    elif vae_type == "rae":
        print(
            f"使用RAE: encoder={normalizer_data.get('rae_encoder_cls', 'SigLIP2wNorm')}"
        )
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

        vae_wrapper = RAEWrapper(
            encoder_cls=encoder_cls,
            encoder_config_path=encoder_config_path,
            encoder_input_size=normalizer_data.get("rae_encoder_input_size", 256),
            encoder_params=encoder_params,
            decoder_config_path=normalizer_data.get(
                "rae_decoder_config_path", "facebook/vit-mae-base"
            ),
            decoder_patch_size=normalizer_data.get("rae_decoder_patch_size", 16),
            pretrained_decoder_path=normalizer_data.get(
                "rae_pretrained_decoder_path", None
            ),
            normalization_stat_path=normalizer_data.get(
                "rae_normalization_stat_path", None
            ),
            device=args.device,
            freeze_encoder=normalizer_data.get("freeze_encoder", True),
            freeze_decoder=normalizer_data.get("freeze_decoder", False),
        )
        # 获取latent_channels（需要先有target_size，暂时使用默认值）
        # 后面会根据实际target_size更新
        latent_channels = config.get("latent_channels", None)
    else:
        raise ValueError(f"Unknown VAE type: {vae_type}")

    print(f"✓ VAE加载完成")

    # ========================================================================
    # Step 3: 加载数据（预测模式：不分割数据）
    # ========================================================================
    print("\n" + "-" * 80)
    print("Step 3: 加载数据")
    print("-" * 80)

    # 解析target_size
    target_size = tuple(map(int, config["target_size"].split(",")))

    # 如果使用RAE，更新latent_channels
    if vae_type == "rae" and latent_channels is None:
        latent_shape = vae_wrapper.get_latent_shape((3, target_size[0], target_size[1]))
        latent_channels = latent_shape[0]

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
    data = variable_da.values  # (Time, H, W)
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
    if vae_type == "rae":
        latent_shape = vae_wrapper.get_latent_shape((3, target_size[0], target_size[1]))
        latent_channels, latent_h, latent_w = latent_shape
        print(f"  潜向量尺寸: ({latent_channels}, {latent_h}, {latent_w})")
    else:
        print(
            f"  潜向量尺寸: ({latent_channels}, {target_size[0]//8}, {target_size[1]//8})"
        )
        latent_h = target_size[0] // 8
        latent_w = target_size[1] // 8

    # 归一化（使用训练时保存的参数）
    normalizer = Normalizer(method=config["normalization"])
    normalizer.load_stats(normalizer_data["stats"])
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
    elif vae_type == "rae":
        latent_shape = vae_wrapper.get_latent_shape((3, target_size[0], target_size[1]))
        latent_channels = latent_shape[0]
    else:
        latent_channels = 4  # SD VAE固定为4

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
    print("Step 5: 生成预测 (使用VAE分批编码/解码)")
    print("-" * 80)

    vae_batch_size = args.vae_batch_size
    print(f"  VAE batch size: {vae_batch_size} (控制显存占用)")

    all_predictions = []
    all_targets = []
    all_inputs = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="预测中(潜空间)"):
            B, T_in, C, H, W = inputs.shape
            T_out = targets.shape[1]

            # 编码到潜空间（分批处理避免显存溢出）
            inputs_flat = inputs.reshape(B * T_in, C, H, W)
            latent_inputs = encode_in_batches(
                vae_wrapper, inputs_flat, vae_batch_size, args.device
            )

            # 获取latent shape
            if vae_type == "rae":
                latent_shape = vae_wrapper.get_latent_shape((C, H, W))
                latent_channels_batch, latent_h_batch, latent_w_batch = latent_shape
            else:
                latent_channels_batch = 4
                latent_h_batch = H // 8
                latent_w_batch = W // 8

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
            # 如果尺寸不匹配，说明预测时的数据加载或VAE配置有问题，直接报错
            if decoded_H != target_size[0] or decoded_W != target_size[1]:
                raise ValueError(
                    f"维度不匹配错误：\n"
                    f"  VAE解码输出尺寸: ({decoded_H}, {decoded_W})\n"
                    f"  训练时target_size: {target_size}\n"
                    f"  预测时输入尺寸: ({H}, {W})\n"
                    f"  可能原因：\n"
                    f"    1. 预测时数据加载未使用正确的target_size\n"
                    f"    2. VAE配置与训练时不一致（特别是RAE的decoder配置）\n"
                    f"    3. 数据加载后未正确resize到target_size\n"
                    f"  解决方案：确保预测时使用与训练时相同的target_size和VAE配置"
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

    return y_pred, y_true, normalizer, config, output_dir


def main():
    parser = argparse.ArgumentParser(description="U-Net统一预测脚本")

    # 模式选择
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["pixel", "latent"],
        help="预测模式: pixel=像素空间, latent=潜空间",
    )

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
        help="VAE编码/解码批次大小（仅latent模式，控制显存占用）",
    )

    # 其他参数
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="设备",
    )

    args = parser.parse_args()

    # ========================================================================
    # 根据模式选择预测方法
    # ========================================================================
    if args.mode == "pixel":
        y_pred, y_true, normalizer, config, output_dir = predict_pixel_unet(args)
    else:
        y_pred, y_true, normalizer, config, output_dir = predict_latent_unet(args)

    # ========================================================================
    # Step 6: 反归一化
    # ========================================================================
    print("\n" + "-" * 80)
    print("Step 6: 评估和反归一化")
    print("-" * 80)

    # 归一化空间的指标
    print("\n归一化空间的指标:")
    metrics_norm = calculate_metrics(y_pred, y_true, ensemble=False)
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

    # 物理空间的指标
    print("\n物理空间的指标 (原始尺度):")
    metrics_phys = calculate_metrics(y_pred_phys, y_true_phys, ensemble=False)
    print(format_metrics(metrics_phys))

    # 计算每个lead time的RMSE
    print("\n每个lead time的RMSE:")
    T_out = y_pred_phys.shape[1]
    rmse_per_leadtime = {}
    for t in range(T_out):
        y_pred_t = y_pred_phys[:, t, :, :, :]  # (N, C, H, W)
        y_true_t = y_true_phys[:, t, :, :, :]
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
        "mode": args.mode,
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

        # 获取预测数据的实际空间形状
        # y_pred_phys shape: (N, T, C, H, W)
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
            # 使用坐标长度进行回退推断
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

        # 根据最终的空间形状选择坐标
        if (
            len(lat_values) == y_pred_phys.shape[3]
            and len(lon_values) == y_pred_phys.shape[4]
        ):
            spatial_coords = {
                "lat": lat_values,
                "lon": lon_values,
            }
        else:
            # 尺寸完全不匹配，使用默认坐标
            print(
                f"  使用默认坐标生成可视化 ({y_pred_phys.shape[3]}x{y_pred_phys.shape[4]}) ..."
            )
            spatial_coords = {
                "lat": np.linspace(-90, 90, y_pred_phys.shape[3]),
                "lon": np.linspace(0, 360, y_pred_phys.shape[4]),
            }

    # 生成可视化
    # 注意：WeatherDiff使用minmax归一化（[-1,1]），与传统模型的zscore不同
    # 这里传递已经反归一化的物理值数据，norm_params=None
    # 可视化函数会生成timeseries_overall（物理值），但不会生成timeseries_physical
    # （因为WeatherDiff的归一化方式与可视化函数预期不兼容）
    visualize_predictions_improved(
        y_true_phys,  # y_test (物理值数据，可能已转置)
        y_pred_phys,  # y_test_pred (物理值数据，可能已转置)
        metrics_phys,  # test_metrics (物理空间指标)
        [variable],  # variables
        f"{args.mode}_unet",  # model_name
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

    print(f"\n模式: {args.mode.upper()}")
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
