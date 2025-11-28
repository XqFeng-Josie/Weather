"""
可视化模块 - 用于预测结果的可视化和评估
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def visualize_predictions_improved(
    y_test,
    y_test_pred,
    test_metrics,
    variables,
    model_name,
    output_dir,
    data_format="flat",
    norm_params=None,
    spatial_coords=None,
):
    """
    改进的可视化方案（非重叠样本）

    生成多种图：
    1. 整体时间序列对比（True vs Pred，汇总所有Lead Time）- 归一化版本
    2. 整体时间序列对比（True vs Pred）- 真实物理值版本
    3. 不同Lead Time的独立对比（使用非重叠样本）
    4. RMSE vs Lead Time 趋势图
    5. （空间数据）固定2个点的4方格世界地图对比

    Args:
        norm_params: 归一化参数字典 {'mean': {var: value}, 'std': {var: value}}
        spatial_coords: 空间坐标字典 {'lat': array, 'lon': array}
    """
    n_variables = len(variables)

    print(f"y_test.shape: {y_test.shape}")
    print(f"y_test_pred.shape: {y_test_pred.shape}")

    for var_idx, var_name in enumerate(variables):
        print(f"\n生成可视化图: {var_name}")

        # 提取该变量的数据
        # 为了确保不同模型和不同分辨率数据的可比性，使用起始点 (0, 0)
        # 这样无论网格分辨率如何，都提取相同的参考点
        if data_format == "flat":
            # flat格式: (samples, lead_times, features)
            # features是展平的空间网格，形状为 H*W
            # 使用起始点 (0, 0)，对应展平索引 0
            n_features = y_test.shape[2]
            grid_points_per_var = n_features // n_variables

            # 起始点在展平数组中的索引
            reference_h, reference_w = 0, 0
            reference_idx_flat = reference_h * 1 + reference_w  # = 0

            feature_idx = var_idx * grid_points_per_var + reference_idx_flat
            print(
                f"  flat格式: 使用起始点 ({reference_h}, {reference_w}), 展平索引={reference_idx_flat}"
            )

            y_true_var = y_test[:, :, feature_idx]  # (samples, lead_times)
            y_pred_var = y_test_pred[:, :, feature_idx]
        else:  # spatial
            # spatial format: (samples, lead_times, channels, H, W)
            # 其中 H=latitude维度, W=longitude维度
            # 使用起始点 (0, 0)
            H, W = y_test.shape[3], y_test.shape[4]
            reference_h, reference_w = 0, 0
            print(
                f"  spatial格式: 使用起始点 ({reference_h}, {reference_w}), 网格大小=({H}, {W})"
            )

            n_channels = y_test.shape[2]

            # 处理通道选择逻辑：
            # 数据流程：(Time, Levels, H, W) -> 选择level -> (Time, 1, H, W) -> 转换为通道数据 -> (Time, Channels, H, W)
            #
            # 情况1：pixel_unet等图像模型的RGB转换
            #   - 选择了1个level，转换为3个RGB通道（3个通道相同）
            #   - 应该选择第一个通道（channel 0）
            #
            # 情况2：多level数据（如geopotential选择了多个levels）
            #   - 选择了多个levels，每个level对应一个通道
            #   - 应该对所有通道求平均，得到所有levels的平均值
            #
            # 情况3：标准情况（每个通道对应一个变量）
            #   - 通道数等于变量数，直接选择对应变量的通道

            # 判断是否为RGB转换情况（单level转换为3通道）
            # CNN和pixel_unet等图像模型：选择单个level后，会转换为3个RGB通道（3个通道相同）
            # 这种情况下，应该选择第一个通道（channel 0）进行可视化
            is_rgb_conversion = n_channels == 3 and n_variables == 1

            if is_rgb_conversion:
                # RGB转换：单level转换为3个相同通道，选择第一个通道
                print(
                    f"  检测到RGB转换（单level转换为3个相同通道），选择第一个通道进行可视化"
                )
                channel_idx = 0
                y_true_var = y_test[
                    :, :, channel_idx, reference_h, reference_w
                ]  # (samples, lead_times)
                y_pred_var = y_test_pred[:, :, channel_idx, reference_h, reference_w]
            elif n_channels > n_variables:
                # 有多个通道（多个levels的情况），对所有通道求平均
                # 这确保了即使训练时选择了多个levels，也能正确显示所有levels的平均值
                print(
                    f"  检测到多通道数据（{n_channels}通道，变量数={n_variables}，多个levels）"
                )
                print(
                    f"  对所有{n_channels}个通道求平均以进行可视化（显示所有levels的平均值）"
                )
                y_true_var = y_test[:, :, :, reference_h, reference_w].mean(
                    axis=2
                )  # (samples, lead_times)
                y_pred_var = y_test_pred[:, :, :, reference_h, reference_w].mean(axis=2)
            else:
                # 标准情况：通道数等于变量数，每个通道对应一个变量
                channel_idx = var_idx if n_channels == n_variables else var_idx
                y_true_var = y_test[
                    :, :, channel_idx, reference_h, reference_w
                ]  # (samples, lead_times)
                y_pred_var = y_test_pred[:, :, channel_idx, reference_h, reference_w]

        safe_var_name = var_name.replace("/", "_").replace(" ", "_")

        # ========================================================================
        # 图1：整体时间序列对比（将所有预测汇总到时间轴）
        # ========================================================================
        fig, ax = plt.subplots(1, 1, figsize=(20, 6))

        n_samples, n_lead_times = y_true_var.shape

        # 使用非重叠采样构建时间序列
        time_series_true = []
        time_series_pred = []
        time_indices = []

        stride_for_viz = n_lead_times  # 使用output_length作为stride
        for i in range(0, min(200, n_samples), stride_for_viz):  # 限制显示前200个样本
            for t in range(n_lead_times):
                if i + t < n_samples:
                    time_idx = i + t
                    time_series_true.append(y_true_var[i, t])
                    time_series_pred.append(y_pred_var[i, t])
                    time_indices.append(time_idx)

        ax.plot(
            time_indices, time_series_true, "b-", label="True", alpha=0.7, linewidth=1.5
        )
        ax.plot(
            time_indices, time_series_pred, "r-", label="Pred", alpha=0.7, linewidth=1.5
        )
        ax.set_xlabel("Time Step Index", fontsize=12)

        # 设置Y轴标签：如果没有norm_params，说明传入的是物理值
        if norm_params is None:
            # 根据变量名设置ylabel（物理值）
            if "temperature" in var_name.lower():
                ax.set_ylabel("Temperature (K)", fontsize=12)
            elif "wind" in var_name.lower():
                ax.set_ylabel("Wind Speed (m/s)", fontsize=12)
            elif "pressure" in var_name.lower():
                ax.set_ylabel("Pressure (Pa)", fontsize=12)
            else:
                ax.set_ylabel("Physical Value", fontsize=12)
        else:
            # 归一化值
            ax.set_ylabel("Normalized Value", fontsize=12)

        ax.set_title(
            f"{var_name} - Overall Time Series Prediction",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)

        plt.tight_layout()
        plt.savefig(
            output_dir / f"timeseries_overall_{safe_var_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print(f"  ✓ 保存: timeseries_overall_{safe_var_name}.png")

        # ========================================================================
        # 图2：不同Lead Time的独立对比（非重叠样本）
        # ========================================================================
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            f"{var_name} - Independent Lead Time Comparison\n(Non-overlapping samples)",
            fontsize=16,
            fontweight="bold",
        )

        # 使用非重叠的样本
        stride_samples = n_lead_times  # 每OUTPUT_LENGTH个样本取一个
        selected_samples = np.arange(
            0, min(50 * stride_samples, n_samples), stride_samples
        )
        selected_samples = selected_samples[:50]  # 最多50个

        for i, ax in enumerate(axes.flat):
            if i >= n_lead_times:
                ax.axis("off")
                continue

            # 提取该Lead Time的数据（非重叠样本）
            y_true_lt = y_true_var[selected_samples, i]
            y_pred_lt = y_pred_var[selected_samples, i]

            # 绘制
            x_indices = np.arange(len(selected_samples))
            ax.plot(x_indices, y_true_lt, "b-o", label="True", alpha=0.7, markersize=4)
            ax.plot(x_indices, y_pred_lt, "r-s", label="Pred", alpha=0.7, markersize=4)
            ax.set_title(f"Lead Time {i+1} ({(i+1)*6}h ahead)", fontsize=12)
            ax.set_xlabel("Independent Sample Index", fontsize=10)

            # 设置Y轴标签
            if norm_params is None:
                # 物理值
                if "temperature" in var_name.lower():
                    ax.set_ylabel("Temperature (K)", fontsize=10)
                elif "wind" in var_name.lower():
                    ax.set_ylabel("Wind Speed (m/s)", fontsize=10)
                else:
                    ax.set_ylabel("Physical Value", fontsize=10)
            else:
                # 归一化值
                ax.set_ylabel("Normalized Value", fontsize=10)

            ax.set_xlim(left=0)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            # 添加RMSE信息
            rmse = np.sqrt(np.mean((y_true_lt - y_pred_lt) ** 2))
            ax.text(
                0.02,
                0.98,
                f"RMSE: {rmse:.4f}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        plt.tight_layout()
        plt.savefig(
            output_dir / f"leadtime_independent_{safe_var_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print(f"  ✓ 保存: leadtime_independent_{safe_var_name}.png")

        # ========================================================================
        # 图3：Lead Time vs RMSE
        # ========================================================================
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        rmse_per_leadtime = []
        for t in range(n_lead_times):
            y_true_lt = y_true_var[selected_samples, t]
            y_pred_lt = y_pred_var[selected_samples, t]
            rmse = np.sqrt(np.mean((y_true_lt - y_pred_lt) ** 2))
            rmse_per_leadtime.append(rmse)

        lead_hours = np.arange(1, n_lead_times + 1) * 6
        ax.plot(
            lead_hours,
            rmse_per_leadtime,
            "o-",
            linewidth=2,
            markersize=8,
            color="darkblue",
        )
        ax.set_xlabel("Lead Time (hours)", fontsize=12)
        ax.set_ylabel("RMSE", fontsize=12)
        ax.set_title(
            f"{var_name} - RMSE vs Lead Time\n(Non-overlapping samples)",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)

        # 标注数值
        for i, (hour, rmse) in enumerate(zip(lead_hours, rmse_per_leadtime)):
            ax.text(hour, rmse, f"{rmse:.4f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        plt.savefig(
            output_dir / f"rmse_vs_leadtime_{safe_var_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print(f"  ✓ 保存: rmse_vs_leadtime_{safe_var_name}.png")

        # ========================================================================
        # 图4：非归一化的时间序列图（真实物理值）
        # ========================================================================
        if norm_params is not None and var_name in norm_params["mean"]:
            mean_val = norm_params["mean"][var_name]
            std_val = norm_params["std"][var_name]

            # 反归一化
            time_series_true_denorm = [v * std_val + mean_val for v in time_series_true]
            time_series_pred_denorm = [v * std_val + mean_val for v in time_series_pred]

            fig, ax = plt.subplots(1, 1, figsize=(20, 6))

            ax.plot(
                time_indices,
                time_series_true_denorm,
                "b-",
                label="True",
                alpha=0.7,
                linewidth=1.5,
            )
            ax.plot(
                time_indices,
                time_series_pred_denorm,
                "r-",
                label="Pred",
                alpha=0.7,
                linewidth=1.5,
            )
            ax.set_xlabel("Time Step Index", fontsize=12)

            # 根据变量名设置ylabel
            if "temperature" in var_name.lower():
                ax.set_ylabel("Temperature (K)", fontsize=12)
            elif "wind" in var_name.lower():
                ax.set_ylabel("Wind Speed (m/s)", fontsize=12)
            elif "pressure" in var_name.lower():
                ax.set_ylabel("Pressure (Pa)", fontsize=12)
            else:
                ax.set_ylabel("Physical Value", fontsize=12)

            ax.set_title(
                f"{var_name} - Overall Time Series (Physical Values)",
                fontsize=14,
                fontweight="bold",
            )
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(left=0)

            plt.tight_layout()
            plt.savefig(
                output_dir / f"timeseries_physical_{safe_var_name}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            print(f"  ✓ 保存: timeseries_physical_{safe_var_name}.png (物理值)")

        # ========================================================================
        # 图5：世界地图对比（True vs Pred，2张并排）
        # ========================================================================
        if data_format == "spatial" and spatial_coords is not None:
            print(f"  生成空间对比图...")

            # 获取完整的空间数据
            # spatial format: (samples, lead_times, channels, H, W)
            # 其中 H=latitude维度(纬度数), W=longitude维度(经度数)
            n_channels = y_test.shape[2]

            # 处理通道选择逻辑（与时间序列图保持一致）：
            # 数据流程：(Time, Levels, H, W) -> 选择level -> (Time, 1, H, W) -> 转换为通道数据 -> (Time, Channels, H, W)
            #
            # 情况1：pixel_unet等图像模型的RGB转换
            #   - 选择了1个level，转换为3个RGB通道（3个通道相同）
            #   - 应该选择第一个通道（channel 0）
            #
            # 情况2：多level数据（如geopotential选择了多个levels）
            #   - 选择了多个levels，每个level对应一个通道
            #   - 应该对所有通道求平均，得到所有levels的平均值
            #
            # 情况3：标准情况（每个通道对应一个变量）
            #   - 通道数等于变量数，直接选择对应变量的通道

            # 判断是否为RGB转换情况（单level转换为3通道）
            # CNN和pixel_unet等图像模型：选择单个level后，会转换为3个RGB通道（3个通道相同）
            # 这种情况下，应该选择第一个通道（channel 0）进行可视化
            is_rgb_conversion = n_channels == 3 and n_variables == 1

            if is_rgb_conversion:
                # RGB转换：单level转换为3个相同通道，选择第一个通道
                print(
                    f"  空间对比图：检测到RGB转换（单level转换为3个相同通道），选择第一个通道进行可视化"
                )
                channel_idx = 0
                y_true_spatial = y_test[
                    :, :, channel_idx, :, :
                ]  # (samples, lead_times, H, W)
                y_pred_spatial = y_test_pred[:, :, channel_idx, :, :]
            elif n_channels > n_variables:
                # 有多个通道（多个levels的情况），对所有通道求平均
                # 这确保了即使训练时选择了多个levels，也能正确显示所有levels的平均值
                print(
                    f"  空间对比图：检测到多通道数据（{n_channels}通道，变量数={n_variables}，多个levels）"
                )
                print(
                    f"  对所有{n_channels}个通道求平均以进行可视化（显示所有levels的平均值）"
                )
                y_true_spatial = y_test[:, :, :, :, :].mean(
                    axis=2
                )  # (samples, lead_times, H, W)
                y_pred_spatial = y_test_pred[:, :, :, :, :].mean(axis=2)
            else:
                # 标准情况：通道数等于变量数，每个通道对应一个变量
                channel_idx = var_idx if n_channels == n_variables else var_idx
                y_true_spatial = y_test[
                    :, :, channel_idx, :, :
                ]  # (samples, lead_times, H, W)
                y_pred_spatial = y_test_pred[:, :, channel_idx, :, :]

            H, W = y_true_spatial.shape[2], y_true_spatial.shape[3]
            print(f"  空间数据形状: H={H}(latitude), W={W}(longitude)")

            # 获取经纬度数组
            lat = spatial_coords.get("lat", np.linspace(-90, 90, H))
            lon = spatial_coords.get("lon", np.linspace(0, 360, W))

            # 确保 lat 和 lon 的长度与 H, W 匹配
            if len(lat) != H or len(lon) != W:
                print(
                    f"  警告: 坐标维度不匹配 (lat:{len(lat)} vs H:{H}, lon:{len(lon)} vs W:{W})"
                )
                print(f"  使用默认坐标...")
                lat = np.linspace(-90, 90, H)
                lon = np.linspace(0, 360, W)

            # 转换经度：0-360° -> -180° to 180° (更直观的显示顺序)
            # 0-180° 保持不变，180-360° 转换为 -180 到 0°
            lon_converted = np.where(lon > 180, lon - 360, lon)

            # 对经度排序并重新排列数据
            lon_sort_idx = np.argsort(lon_converted)
            lon = lon_converted[lon_sort_idx]

            # 重新排列空间数据（沿着经度维度）
            y_true_spatial = y_true_spatial[:, :, :, lon_sort_idx]
            y_pred_spatial = y_pred_spatial[:, :, :, lon_sort_idx]

            # 固定选择2个有代表性的点（使用实际数组长度）
            n_lat, n_lon = len(lat), len(lon)
            # 点1：赤道附近（热带） - 纬度中间，经度的1/4处
            # 点2：北半球中纬度 - 纬度的3/4处，经度的3/4处
            point1_h, point1_w = n_lat // 2, n_lon // 4
            point2_h, point2_w = (n_lat * 3) // 4, (n_lon * 3) // 4

            # 边界检查
            point1_h = min(point1_h, n_lat - 1)
            point1_w = min(point1_w, n_lon - 1)
            point2_h = min(point2_h, n_lat - 1)
            point2_w = min(point2_w, n_lon - 1)

            point1_lat, point1_lon = lat[point1_h], lon[point1_w]
            point2_lat, point2_lon = lat[point2_h], lon[point2_w]

            # 选择一个时间点和lead time用于对比
            sample_idx = n_samples // 2  # 中间时刻
            lead_idx = 0  # 第一个lead time (6小时预测)

            # 获取数据
            data_true = y_true_spatial[sample_idx, lead_idx, :, :]
            data_pred = y_pred_spatial[sample_idx, lead_idx, :, :]

            # 反归一化（如果有参数）
            if norm_params is not None and var_name in norm_params["mean"]:
                mean_val = norm_params["mean"][var_name]
                std_val = norm_params["std"][var_name]
                data_true = data_true * std_val + mean_val
                data_pred = data_pred * std_val + mean_val

            # 创建2列图（True vs Pred）
            fig = plt.figure(figsize=(24, 10))

            # 准备网格
            lon_grid, lat_grid = np.meshgrid(lon, lat)

            # 统一颜色范围
            vmin = min(data_true.min(), data_pred.min())
            vmax = max(data_true.max(), data_pred.max())

            # ========== 左图：Ground Truth ==========
            ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())

            im1 = ax1.contourf(
                lon_grid,
                lat_grid,
                data_true,
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

            # 标记2个固定点
            ax1.plot(
                point1_lon,
                point1_lat,
                "ro",
                markersize=12,
                transform=ccrs.PlateCarree(),
                markeredgecolor="white",
                markeredgewidth=2,
            )
            ax1.plot(
                point2_lon,
                point2_lat,
                "go",
                markersize=12,
                transform=ccrs.PlateCarree(),
                markeredgecolor="white",
                markeredgewidth=2,
            )

            # 添加点的标签
            val1_true = data_true[point1_h, point1_w]
            val2_true = data_true[point2_h, point2_w]

            ax1.text(
                point1_lon,
                point1_lat + 5,
                f"P1: {val1_true:.1f}",
                transform=ccrs.PlateCarree(),
                fontsize=10,
                ha="center",
                va="bottom",
                color="red",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )
            ax1.text(
                point2_lon,
                point2_lat + 5,
                f"P2: {val2_true:.1f}",
                transform=ccrs.PlateCarree(),
                fontsize=10,
                ha="center",
                va="bottom",
                color="green",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

            ax1.set_title(f"Ground Truth", fontsize=14, fontweight="bold", pad=10)

            # 添加颜色条
            cbar1 = plt.colorbar(
                im1, ax=ax1, orientation="horizontal", pad=0.05, shrink=0.8
            )
            if "temperature" in var_name.lower():
                cbar1.set_label("Temperature (K)", fontsize=11)
            else:
                cbar1.set_label("Value", fontsize=11)

            # ========== 右图：Prediction ==========
            ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())

            im2 = ax2.contourf(
                lon_grid,
                lat_grid,
                data_pred,
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

            # 标记2个固定点
            ax2.plot(
                point1_lon,
                point1_lat,
                "ro",
                markersize=12,
                transform=ccrs.PlateCarree(),
                markeredgecolor="white",
                markeredgewidth=2,
            )
            ax2.plot(
                point2_lon,
                point2_lat,
                "go",
                markersize=12,
                transform=ccrs.PlateCarree(),
                markeredgecolor="white",
                markeredgewidth=2,
            )

            # 添加点的标签
            val1_pred = data_pred[point1_h, point1_w]
            val2_pred = data_pred[point2_h, point2_w]

            ax2.text(
                point1_lon,
                point1_lat + 5,
                f"P1: {val1_pred:.1f}",
                transform=ccrs.PlateCarree(),
                fontsize=10,
                ha="center",
                va="bottom",
                color="red",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )
            ax2.text(
                point2_lon,
                point2_lat + 5,
                f"P2: {val2_pred:.1f}",
                transform=ccrs.PlateCarree(),
                fontsize=10,
                ha="center",
                va="bottom",
                color="green",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

            ax2.set_title(f"Prediction", fontsize=14, fontweight="bold", pad=10)

            # 添加颜色条
            cbar2 = plt.colorbar(
                im2, ax=ax2, orientation="horizontal", pad=0.05, shrink=0.8
            )
            if "temperature" in var_name.lower():
                cbar2.set_label("Temperature (K)", fontsize=11)
            else:
                cbar2.set_label("Value", fontsize=11)

            # 计算误差统计
            error = data_pred - data_true
            rmse = np.sqrt(np.mean(error**2))
            mae = np.mean(np.abs(error))

            # 添加总标题
            fig.suptitle(
                f"{var_name} - Spatial Comparison (Lead Time: {(lead_idx+1)*6}h)\n"
                + f"Sample: {sample_idx} | Point 1 (Red): ({point1_lat:.1f}°, {point1_lon:.1f}°) | "
                + f"Point 2 (Green): ({point2_lat:.1f}°, {point2_lon:.1f}°)\n"
                + f"RMSE: {rmse:.2f} | MAE: {mae:.2f}",
                fontsize=16,
                fontweight="bold",
                y=0.98,
            )

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(
                output_dir / f"spatial_comparison_{safe_var_name}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            print(
                f"  ✓ 保存: spatial_comparison_{safe_var_name}.png (True vs Pred 世界地图对比)"
            )


def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Dict:
    """
    计算预测指标

    Args:
        y_pred: 预测值
        y_true: 真实值

    Returns:
        指标字典
    """
    metrics = {}

    # 整体指标
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y_true))

    metrics["mse"] = float(mse)
    metrics["rmse"] = float(rmse)
    metrics["mae"] = float(mae)

    # 按时间步分别计算
    for t in range(y_true.shape[1]):
        mse_t = np.mean((y_pred[:, t] - y_true[:, t]) ** 2)
        metrics[f"rmse_step_{t+1}"] = float(np.sqrt(mse_t))

    return metrics


def compute_variable_wise_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    n_variables: int,
    format: str = "flat",
) -> Dict:
    """
    计算每个变量的独立指标

    Args:
        y_pred: 预测值
        y_true: 真实值
        n_variables: 变量数量
        format: 'flat' 或 'spatial'

    Returns:
        dict: 每个变量的RMSE和MAE
    """
    metrics = {}

    if format == "flat":
        # (n_samples, time, features)
        n_features = y_pred.shape[2]
        features_per_var = n_features // n_variables

        for i in range(n_variables):
            start = i * features_per_var
            end = (i + 1) * features_per_var

            var_pred = y_pred[:, :, start:end]
            var_true = y_true[:, :, start:end]

            mse = np.mean((var_pred - var_true) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(var_pred - var_true))

            metrics[f"var_{i}_rmse"] = float(rmse)
            metrics[f"var_{i}_mae"] = float(mae)

    elif format == "spatial":
        # (n_samples, time, channels, H, W)
        n_channels = y_pred.shape[2]

        if n_channels == n_variables:
            for i in range(n_variables):
                var_pred = y_pred[:, :, i, :, :]
                var_true = y_true[:, :, i, :, :]

                mse = np.mean((var_pred - var_true) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(var_pred - var_true))

                metrics[f"var_{i}_rmse"] = float(rmse)
                metrics[f"var_{i}_mae"] = float(mae)
        else:
            channels_per_var = n_channels // n_variables
            for i in range(n_variables):
                start_ch = i * channels_per_var
                end_ch = (i + 1) * channels_per_var

                var_pred = y_pred[:, :, start_ch:end_ch, :, :]
                var_true = y_true[:, :, start_ch:end_ch, :, :]

                mse = np.mean((var_pred - var_true) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(var_pred - var_true))

                metrics[f"var_{i}_rmse"] = float(rmse)
                metrics[f"var_{i}_mae"] = float(mae)

    return metrics
