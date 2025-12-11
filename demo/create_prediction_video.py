"""
生成预测对比视频脚本
从预测数据生成左右对比视频（真实 vs 预测）

运行示例:
python create_prediction_video.py \
    --predictions outputs/cnn_2m_temperature/predictions.nc \
    --time-slice "2020-01-01:2020-12-31" \
    --output outputs/cnn_2m_temperature/prediction_video.mp4 \
    --lead-time 0
"""

import argparse
import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
from tqdm import tqdm
import tempfile
import shutil
import subprocess
import sys
import json

# 尝试导入imageio（可选，用于备用方案）
try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


def parse_args():
    parser = argparse.ArgumentParser(description="Generate prediction comparison video")
    
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions.nc file",
    )
    parser.add_argument(
        "--time-slice",
        type=str,
        default=None,
        help="Time slice to use (e.g., '2020-01-01:2020-12-31'). If None, use all data.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="prediction_video.mp4",
        help="Output video file path",
    )
    parser.add_argument(
        "--lead-time",
        type=int,
        default=0,
        help="Lead time index (0-based, 0 = 6h, 1 = 12h, etc.)",
    )
    parser.add_argument(
        "--variable",
        type=str,
        default=None,
        help="Variable name (auto-detect if None)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for video",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="DPI for frames (higher = better quality but slower)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr",
        help="Path to ERA5 data for getting coordinates",
    )
    parser.add_argument(
        "--use-npy",
        action="store_true",
        help="Load data from predictions_data/*.npy files (original scale) instead of predictions.nc",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.json file for denormalization (if predictions.nc contains normalized data)",
    )
    parser.add_argument(
        "--predictions-data-dir",
        type=str,
        default=None,
        help="Path to predictions_data directory (auto-detect if None and --use-npy is set)",
    )
    
    return parser.parse_args()


def load_normalization_params(config_path):
    """从配置文件加载归一化参数"""
    if config_path is None:
        return None
    
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"Warning: Config file not found: {config_path}")
        return None
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    norm_params = config.get("normalization", {})
    if not norm_params:
        return None
    
    return norm_params


def denormalize_data(data, variable, norm_params):
    """反归一化数据"""
    if norm_params is None:
        return data
    
    mean = norm_params.get("mean", {}).get(variable)
    std = norm_params.get("std", {}).get(variable)
    
    if mean is None or std is None:
        print(f"Warning: Normalization params not found for {variable}, using data as-is")
        return data
    
    print(f"Denormalizing {variable} using mean={mean:.2f}, std={std:.2f}")
    return data * std + mean


def load_predictions_from_npy(predictions_data_dir, variable=None):
    """从predictions_data文件夹加载.npy文件（原始尺度）"""
    predictions_data_dir = Path(predictions_data_dir)
    
    pred_file = predictions_data_dir / "y_test_pred.npy"
    true_file = predictions_data_dir / "y_test.npy"
    
    if not pred_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")
    if not true_file.exists():
        raise FileNotFoundError(f"True value file not found: {true_file}")
    
    print(f"Loading predictions from {predictions_data_dir}...")
    y_pred = np.load(pred_file)  # (time, lead_time, channels, lat, lon) or (time, lead_time, lat, lon)
    y_true = np.load(true_file)
    
    print(f"Loaded data shape - pred: {y_pred.shape}, true: {y_true.shape}")
    
    # 处理不同的数据格式
    if y_pred.ndim == 5:  # (time, lead_time, channels, lat, lon)
        # 如果是多通道，取第一个通道（假设是单变量）
        if y_pred.shape[2] == 1:
            y_pred = y_pred[:, :, 0, :, :]  # (time, lead_time, lat, lon)
            y_true = y_true[:, :, 0, :, :]
        else:
            raise ValueError(f"Multi-channel data not supported yet. Shape: {y_pred.shape}")
    elif y_pred.ndim == 4:  # (time, lead_time, lat, lon)
        pass  # 已经是正确格式
    else:
        raise ValueError(f"Unexpected data shape: {y_pred.shape}")
    
    # 创建时间坐标（从2020-01-01开始，每6小时一个时间步）
    n_times = y_pred.shape[0]
    times = np.arange(n_times) * np.timedelta64(6, 'h') + np.datetime64('2020-01-01T00:00:00')
    
    # Lead times (假设是6, 12, 18, 24小时)
    n_lead_times = y_pred.shape[1]
    lead_times = np.arange(1, n_lead_times + 1) * 6  # hours
    
    # 创建经纬度坐标（从原始数据获取，或使用默认值）
    lat = np.linspace(-90, 90, y_pred.shape[2])
    lon = np.linspace(0, 360, y_pred.shape[3])
    
    # 自动检测变量名（从目录名或使用默认值）
    if variable is None:
        # 尝试从目录名推断
        parent_dir = predictions_data_dir.parent.name
        if "temperature" in parent_dir.lower():
            variable = "2m_temperature"
        else:
            variable = "unknown"
    
    print(f"Auto-detected variable: {variable}")
    print(f"Loaded data shape: {y_pred.shape}")
    print(f"Time range: {times[0]} to {times[-1]}")
    print(f"Lead times: {lead_times}")
    print(f"Data range - pred: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
    print(f"Data range - true: [{y_true.min():.2f}, {y_true.max():.2f}]")
    
    return {
        "y_pred": y_pred,
        "y_true": y_true,
        "times": times,
        "lead_times": lead_times,
        "lat": lat,
        "lon": lon,
        "variable": variable,
    }


def load_predictions(pred_path, time_slice=None, variable=None, config_path=None):
    """加载预测数据（从NetCDF文件）"""
    print(f"Loading predictions from {pred_path}...")
    ds = xr.open_dataset(pred_path)
    
    # 自动检测变量名
    if variable is None:
        pred_vars = [v for v in ds.data_vars if v.endswith("_pred")]
        if len(pred_vars) == 0:
            raise ValueError("No prediction variables found in dataset")
        variable = pred_vars[0].replace("_pred", "")
        print(f"Auto-detected variable: {variable}")
    
    pred_var = f"{variable}_pred"
    true_var = f"{variable}_true"
    
    if pred_var not in ds.data_vars:
        raise ValueError(f"Variable {pred_var} not found in dataset")
    if true_var not in ds.data_vars:
        raise ValueError(f"Variable {true_var} not found in dataset")
    
    # 选择时间范围
    if time_slice is not None:
        start, end = time_slice.split(":")
        ds = ds.sel(time=slice(start, end))
    
    y_pred = ds[pred_var].values  # (time, lead_time, lat, lon)
    y_true = ds[true_var].values  # (time, lead_time, lat, lon)
    
    # 检查数据是否已归一化（通过数值范围判断）
    data_range = max(y_pred.max(), y_true.max()) - min(y_pred.min(), y_true.min())
    is_normalized = data_range < 10  # 如果范围小于10，可能是归一化的
    
    # 如果数据看起来是归一化的，尝试反归一化
    if is_normalized and config_path:
        norm_params = load_normalization_params(config_path)
        if norm_params:
            print(f"Data appears normalized (range: [{y_pred.min():.2f}, {y_pred.max():.2f}]), denormalizing...")
            y_pred = denormalize_data(y_pred, variable, norm_params)
            y_true = denormalize_data(y_true, variable, norm_params)
            print(f"After denormalization - pred range: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
            print(f"After denormalization - true range: [{y_true.min():.2f}, {y_true.max():.2f}]")
    
    # 获取坐标
    times = ds.time.values
    lead_times = ds.lead_time.values
    lat = ds.lat.values
    lon = ds.lon.values
    
    print(f"Loaded data shape: {y_pred.shape}")
    print(f"Time range: {times[0]} to {times[-1]}")
    print(f"Lead times: {lead_times}")
    print(f"Lat range: {lat.min():.2f} to {lat.max():.2f}")
    print(f"Lon range: {lon.min():.2f} to {lon.max():.2f}")
    
    return {
        "y_pred": y_pred,
        "y_true": y_true,
        "times": times,
        "lead_times": lead_times,
        "lat": lat,
        "lon": lon,
        "variable": variable,
    }


def get_original_coordinates(data_path):
    """从原始数据获取准确的经纬度坐标"""
    try:
        print(f"Loading original coordinates from {data_path}...")
        ds = xr.open_zarr(data_path)
        lat = ds.latitude.values
        lon = ds.longitude.values
        print(f"Original coordinates loaded: lat={len(lat)}, lon={len(lon)}")
        print(f"  Lat range: {lat.min():.2f} to {lat.max():.2f}")
        print(f"  Lon range: {lon.min():.2f} to {lon.max():.2f}")
        return lat, lon
    except Exception as e:
        print(f"Warning: Could not load original coordinates: {e}")
        print("Using coordinates from predictions file")
        return None, None


def create_frame(
    data_true,
    data_pred,
    lat,
    lon,
    time_str,
    lead_time_hours,
    variable,
    output_path,
    dpi=100,
):
    """创建单帧对比图"""
    # 转换经度：0-360° -> -180° to 180°
    lon_converted = np.where(lon > 180, lon - 360, lon)
    
    # 对经度排序并重新排列数据
    lon_sort_idx = np.argsort(lon_converted)
    lon_sorted = lon_converted[lon_sort_idx]
    
    # 重新排列空间数据（沿着经度维度）
    data_true_sorted = data_true[:, lon_sort_idx]
    data_pred_sorted = data_pred[:, lon_sort_idx]
    
    # 创建2列图（True vs Pred）
    fig = plt.figure(figsize=(24, 10))
    
    # 准备网格
    lon_grid, lat_grid = np.meshgrid(lon_sorted, lat)
    
    # 统一颜色范围
    vmin = min(data_true_sorted.min(), data_pred_sorted.min())
    vmax = max(data_true_sorted.max(), data_pred_sorted.max())
    
    # ========== 左图：Ground Truth ==========
    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    
    im1 = ax1.contourf(
        lon_grid,
        lat_grid,
        data_true_sorted,
        levels=15,
        cmap="RdYlBu_r",
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
    )
    
    # 添加地图特征
    ax1.coastlines(linewidth=0.5)
    ax1.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
    ax1.gridlines(
        draw_labels=True,
        dms=True,
        x_inline=False,
        y_inline=False,
        linewidth=0.5,
        alpha=0.5,
    )
    
    ax1.set_title("Ground Truth", fontsize=14, fontweight="bold", pad=10)
    
    # 添加颜色条
    cbar1 = plt.colorbar(
        im1, ax=ax1, orientation="horizontal", pad=0.05, shrink=0.8
    )
    if "temperature" in variable.lower():
        cbar1.set_label("Temperature (K)", fontsize=11)
    else:
        cbar1.set_label("Value", fontsize=11)
    
    # ========== 右图：Prediction ==========
    ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
    
    im2 = ax2.contourf(
        lon_grid,
        lat_grid,
        data_pred_sorted,
        levels=15,
        cmap="RdYlBu_r",
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
    )
    
    # 添加地图特征
    ax2.coastlines(linewidth=0.5)
    ax2.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
    ax2.gridlines(
        draw_labels=True,
        dms=True,
        x_inline=False,
        y_inline=False,
        linewidth=0.5,
        alpha=0.5,
    )
    
    ax2.set_title("Prediction", fontsize=14, fontweight="bold", pad=10)
    
    # 添加颜色条
    cbar2 = plt.colorbar(
        im2, ax=ax2, orientation="horizontal", pad=0.05, shrink=0.8
    )
    if "temperature" in variable.lower():
        cbar2.set_label("Temperature (K)", fontsize=11)
    else:
        cbar2.set_label("Value", fontsize=11)
    
    # 计算误差统计
    error = data_pred_sorted - data_true_sorted
    rmse = np.sqrt(np.mean(error**2))
    mae = np.mean(np.abs(error))
    
    # 添加总标题
    fig.suptitle(
        f"{variable} - Comparison (Lead Time: {lead_time_hours}h)\n"
        + f"Time: {time_str} | RMSE: {rmse:.2f} K | MAE: {mae:.2f} K",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def create_video(
    predictions_data,
    lead_time_idx,
    output_path,
    fps=10,
    dpi=100,
    original_lat=None,
    original_lon=None,
):
    """创建视频"""
    y_pred = predictions_data["y_pred"]
    y_true = predictions_data["y_true"]
    times = predictions_data["times"]
    lead_times = predictions_data["lead_times"]
    lat = predictions_data["lat"]
    lon = predictions_data["lon"]
    variable = predictions_data["variable"]
    
    # 使用原始坐标（如果可用）
    if original_lat is not None and len(original_lat) == len(lat):
        lat = original_lat
        print(f"Using original latitude coordinates (range: {lat.min():.2f} to {lat.max():.2f})")
    else:
        print(f"Using latitude coordinates from predictions (range: {lat.min():.2f} to {lat.max():.2f})")
    
    if original_lon is not None and len(original_lon) == len(lon):
        lon = original_lon
        print(f"Using original longitude coordinates (range: {lon.min():.2f} to {lon.max():.2f})")
    else:
        print(f"Using longitude coordinates from predictions (range: {lon.min():.2f} to {lon.max():.2f})")
    
    n_times = y_pred.shape[0]
    lead_time_hours = int(lead_times[lead_time_idx])
    
    print(f"\nCreating video with {n_times} frames...")
    print(f"Lead time: {lead_time_hours}h (index {lead_time_idx})")
    print(f"FPS: {fps}, DPI: {dpi}")
    
    # 创建临时目录存储帧
    temp_dir = tempfile.mkdtemp(prefix="video_frames_")
    print(f"Temporary frames directory: {temp_dir}")
    
    try:
        # 生成所有帧
        frame_paths = []
        for t_idx in tqdm(range(n_times), desc="Generating frames"):
            # 获取当前时间步的数据
            data_true = y_true[t_idx, lead_time_idx, :, :]  # (lat, lon)
            data_pred = y_pred[t_idx, lead_time_idx, :, :]  # (lat, lon)
            
            # 格式化时间字符串
            if isinstance(times[t_idx], np.datetime64):
                time_str = str(times[t_idx])[:19]  # 格式: YYYY-MM-DDTHH:MM:SS
            else:
                time_str = str(times[t_idx])
            
            # 保存帧
            frame_path = Path(temp_dir) / f"frame_{t_idx:05d}.png"
            create_frame(
                data_true,
                data_pred,
                lat,
                lon,
                time_str,
                lead_time_hours,
                variable,
                frame_path,
                dpi=dpi,
            )
            frame_paths.append(str(frame_path))
        
        # 创建视频（优先使用ffmpeg，否则使用imageio）
        print(f"\nCreating video from {len(frame_paths)} frames...")
        
        # 检查ffmpeg是否可用
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            use_ffmpeg = True
            print("Using ffmpeg for video encoding")
        except (subprocess.CalledProcessError, FileNotFoundError):
            use_ffmpeg = False
            if HAS_IMAGEIO:
                print("ffmpeg not found, using imageio for video encoding")
            else:
                raise RuntimeError(
                    "Neither ffmpeg nor imageio is available. "
                    "Please install ffmpeg or imageio: pip install imageio[ffmpeg]"
                )
        
        if use_ffmpeg:
            # 使用ffmpeg创建视频
            # 构建ffmpeg命令
            pattern = str(Path(temp_dir) / "frame_%05d.png")
            # 使用 scale 过滤器确保宽度和高度都是偶数（H.264要求）
            # trunc(iw/2)*2 和 trunc(ih/2)*2 确保偶数尺寸
            cmd = [
                "ffmpeg",
                "-y",  # 覆盖输出文件
                "-framerate", str(fps),
                "-i", pattern,
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",  # 高质量（18-28范围，越小质量越高）
                "-preset", "medium",
                str(output_path),
            ]
            
            # 运行ffmpeg
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                print(f"ffmpeg stderr: {stderr.decode()}")
                raise RuntimeError(f"ffmpeg failed with return code {process.returncode}")
        else:
            # 使用imageio创建视频
            writer = imageio.get_writer(
                output_path,
                fps=fps,
                codec="libx264",
                quality=8,
                pixelformat="yuv420p",
            )
            
            for frame_path in tqdm(frame_paths, desc="Writing video"):
                frame = imageio.imread(frame_path)
                writer.append_data(frame)
            
            writer.close()
        
        print(f"\n✓ Video saved to: {output_path}")
        print(f"  Frames: {n_times}")
        print(f"  Duration: {n_times/fps:.1f} seconds")
        print(f"  Lead time: {lead_time_hours}h")
        
    finally:
        # 清理临时文件
        print(f"\nCleaning up temporary files...")
        shutil.rmtree(temp_dir)
        print("✓ Cleanup complete")


def main():
    args = parse_args()
    
    print("=" * 80)
    print("Prediction Video Generator")
    print("=" * 80)
    
    # 1. 加载预测数据
    if args.use_npy:
        # 从predictions_data文件夹加载.npy文件（原始尺度）
        if args.predictions_data_dir:
            predictions_data_dir = Path(args.predictions_data_dir)
        else:
            # 自动检测：从predictions路径推断
            pred_path = Path(args.predictions)
            predictions_data_dir = pred_path.parent / "predictions_data"
        
        if not predictions_data_dir.exists():
            raise FileNotFoundError(
                f"Predictions data directory not found: {predictions_data_dir}\n"
                f"Please specify --predictions-data-dir or ensure the directory exists."
            )
        
        predictions_data = load_predictions_from_npy(
            predictions_data_dir,
            variable=args.variable,
        )
        
        # 如果指定了时间范围，需要过滤数据
        if args.time_slice:
            start, end = args.time_slice.split(":")
            start_dt = np.datetime64(start)
            end_dt = np.datetime64(end)
            time_mask = (predictions_data["times"] >= start_dt) & (predictions_data["times"] <= end_dt)
            predictions_data["y_pred"] = predictions_data["y_pred"][time_mask]
            predictions_data["y_true"] = predictions_data["y_true"][time_mask]
            predictions_data["times"] = predictions_data["times"][time_mask]
            print(f"Filtered to {len(predictions_data['times'])} time steps")
    else:
        # 从NetCDF文件加载
        config_path = args.config
        if config_path is None:
            # 尝试自动检测配置文件
            pred_path = Path(args.predictions)
            possible_config = pred_path.parent / "config.json"
            if possible_config.exists():
                config_path = str(possible_config)
                print(f"Auto-detected config file: {config_path}")
        
        predictions_data = load_predictions(
            args.predictions,
            time_slice=args.time_slice,
            variable=args.variable,
            config_path=config_path,
        )
    
    # 2. 获取原始坐标（可选）
    original_lat, original_lon = get_original_coordinates(args.data_path)
    
    # 3. 检查lead_time索引
    if args.lead_time >= len(predictions_data["lead_times"]):
        raise ValueError(
            f"Lead time index {args.lead_time} out of range. "
            f"Available lead times: {list(range(len(predictions_data['lead_times'])))}"
        )
    
    # 4. 创建输出目录
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 5. 生成视频
    create_video(
        predictions_data,
        args.lead_time,
        str(output_path),
        fps=args.fps,
        dpi=args.dpi,
        original_lat=original_lat,
        original_lon=original_lon,
    )
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()

