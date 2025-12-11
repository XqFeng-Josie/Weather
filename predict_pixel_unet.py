"""
像素空间U-Net预测脚本

📋 文件作用:
    对训练好的像素空间U-Net模型进行预测，生成评估指标和可视化图片。
    
🔄 预测流程:
    1. 加载训练好的模型和配置
    2. 加载归一化参数（与训练时保持一致）
    3. 加载测试数据
    4. 生成预测
    5. 计算评估指标（归一化空间 + 物理空间）
    6. 生成可视化（时间序列图 + 世界地图对比）
    7. 保存所有结果

📊 输出文件:
    - prediction_metrics.json: 详细评估指标
    - y_pred_*.npy: 预测数据（归一化 + 物理单位）
    - y_true_*.npy: 真值数据
    - timeseries_*.png: 时间序列对比图
    - spatial_comparison_*.png: 世界地图对比图 ⭐
    - rmse_vs_leadtime_*.png: RMSE随预测步长变化

📖 使用方法:
    python predict_pixel_unet.py \\
        --model-dir outputs/pixel_unet \\
        --time-slice 2020-02-01:2020-02-10 \\
        --batch-size 32

🎯 预期效果:
    像素空间U-Net:
        - RMSE: 1-3K
        - 相关系数: > 0.995
        - SSIM: > 0.995
"""

import argparse
import torch
import numpy as np
import json
import pickle
from pathlib import Path
from tqdm import tqdm

from weatherdiff.unet import WeatherUNet
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

    # 从config读取训练时使用的 levels（如果有的话）
    available_levels = config.get("levels", None)
    if available_levels is not None:
        if len(available_levels) == 1:
            print(f"  训练时使用单level模式: level {available_levels[0]}")
        else:
            print(f"  训练时使用多level模式: levels {available_levels}")
    else:
        print(f"  训练时未指定levels（使用所有可用levels或无level变量）")

    # 处理命令行指定的 --levels 参数
    # 注意：对于单level训练的模型，所有通道都来自同一个level
    # 所以不需要channel_indices来选择特定通道
    selected_levels = args.levels
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

        # 对于单level训练，所有通道都来自同一level，无需特殊处理
        print(f"  预测数据将使用levels: {selected_levels}")
    else:
        print(f"  预测数据将使用所有训练时的levels")
        selected_levels = available_levels

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
    variable_da = ds[config["variables"]]

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

    # 统一纬经度维度顺序，与训练/WeatherDataLoader保持一致
    if "latitude" in variable_da.dims and "longitude" in variable_da.dims:
        dims = list(variable_da.dims)
        lon_idx = dims.index("longitude")
        lat_idx = dims.index("latitude")

        if lon_idx < lat_idx:
            print(f"  检测到维度顺序 {dims} (longitude 在 latitude 前)，转置为标准顺序")
            if "level" in dims:
                target_dims = ["time", "level", "latitude", "longitude"]
            else:
                target_dims = ["time", "latitude", "longitude"]
            variable_da = variable_da.transpose(*target_dims)
            print(f"  转置后维度: {variable_da.dims}")
        else:
            print(f"  维度顺序符合标准: {dims}")
    else:
        print(f"  未检测到纬/经度维度，dims={variable_da.dims}")

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

    data = normalizer.transform(data, name=config["variables"])
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

    return (
        y_pred,
        y_true,
        normalizer,
        config,
        output_dir,
        selected_levels,
    )


def main():
    parser = argparse.ArgumentParser(description="像素空间U-Net预测脚本")

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

    # ========================================================================
    # 像素空间U-Net预测
    # ========================================================================
    (
        y_pred,
        y_true,
        normalizer,
        config,
        output_dir,
        selected_levels,
    ) = predict_pixel_unet(args)

    # ========================================================================
    # Step 6: 反归一化
    # ========================================================================
    print("\n" + "-" * 80)
    print("Step 6: 评估和反归一化")
    print("-" * 80)

    # 对于单level训练的模型，使用所有通道（都来自同一level）
    y_pred_selected = y_pred
    y_true_selected = y_true

    # 归一化空间的指标
    print("\n归一化空间的指标:")
    metrics_norm = calculate_metrics(y_pred_selected, y_true_selected, ensemble=False)

    # 添加MSE计算（参考predict.py）
    mse_norm = np.mean((y_pred_selected - y_true_selected) ** 2)
    metrics_norm["mse"] = float(mse_norm)

    print(format_metrics(metrics_norm))

    # 反归一化到物理单位
    variable = config["variables"]
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

    # 对于单level训练的模型，使用所有通道（都来自同一level）
    y_pred_phys_selected = y_pred_phys
    y_true_phys_selected = y_true_phys

    # 物理空间的指标
    print("\n物理空间的指标 (原始尺度):")
    metrics_phys = calculate_metrics(
        y_pred_phys_selected, y_true_phys_selected, ensemble=False
    )

    # 添加MSE计算（参考predict.py）
    mse_phys = np.mean((y_pred_phys_selected - y_true_phys_selected) ** 2)
    metrics_phys["mse"] = float(mse_phys)

    print(format_metrics(metrics_phys))

    # 计算每个lead time的RMSE和MSE
    print("\n每个lead time的RMSE和MSE:")
    T_out = y_pred_phys_selected.shape[1]
    rmse_per_leadtime = {}
    mse_per_leadtime = {}
    for t in range(T_out):
        y_pred_t = y_pred_phys_selected[:, t, :, :, :]  # (N, C, H, W)
        y_true_t = y_true_phys_selected[:, t, :, :, :]
        mse_t = np.mean((y_pred_t - y_true_t) ** 2)
        rmse_t = np.sqrt(mse_t)
        rmse_per_leadtime[f"rmse_step_{t+1}"] = float(rmse_t)
        mse_per_leadtime[f"mse_step_{t+1}"] = float(mse_t)
        print(f"  Step {t+1} ({(t+1)*6}h): RMSE = {rmse_t:.4f} K, MSE = {mse_t:.4f} K²")

    # ========================================================================
    # Step 7: 保存结果
    # ========================================================================
    print("\n" + "-" * 80)
    print("Step 7: 保存结果")
    print("-" * 80)

    # 保存指标
    metrics_all = {
        "mode": "pixel",
        "normalized_space": {k: float(v) for k, v in metrics_norm.items()},
        "physical_space": {k: float(v) for k, v in metrics_phys.items()},
        "physical_space_rmse_per_leadtime": rmse_per_leadtime,
        "physical_space_mse_per_leadtime": mse_per_leadtime,
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

        # 验证坐标长度是否匹配（简单直接的方式，参考predict.py）
        if len(lat_values) == actual_H and len(lon_values) == actual_W:
            spatial_coords = {
                "lat": lat_values,
                "lon": lon_values,
            }
            print("  ✓ 坐标匹配，使用数据集坐标")
        else:
            print(f"  ⚠ 坐标维度不匹配")
            print(f"  使用默认坐标生成可视化...")
            spatial_coords = {
                "lat": np.linspace(-90, 90, actual_H),
                "lon": np.linspace(0, 360, actual_W),
            }
    else:
        # 默认坐标
        actual_H = y_pred_phys.shape[3]
        actual_W = y_pred_phys.shape[4]
        print("  数据集没有坐标信息，使用默认坐标")
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
        "pixel_unet",  # model_name
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

    print(f"\n模式: PIXEL")
    print(f"总结 (物理空间):")
    print(f"  样本数: {y_pred.shape[0]}")
    print(f"  MSE: {metrics_phys['mse']:.4f} K²")
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
