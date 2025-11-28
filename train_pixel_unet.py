"""
训练像素空间U-Net（Step 3: 图像到图像预测基线）

使用方法:
    python train_pixel_unet.py --data-path /path/to/data.zarr --variable 2m_temperature
"""

import argparse
import torch
from pathlib import Path

from weatherdiff.unet import WeatherUNet, UNetTrainer
from weatherdiff.utils import WeatherDataModule


def main():
    parser = argparse.ArgumentParser(description="训练像素空间U-Net")

    # 数据参数
    parser.add_argument(
        "--data-path",
        type=str,
        default="gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr",
        help="数据文件路径",
    )
    parser.add_argument("--variables", type=str, default="2m_temperature", help="变量名")
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
        "--time-slice",
        type=str,
        default="2015-01-01:2019-12-31",
        help="时间切片，格式: 2015-01-01:2019-12-31",
    )

    # 模型参数
    parser.add_argument("--input-length", type=int, default=12, help="输入序列长度")
    parser.add_argument("--output-length", type=int, default=4, help="输出序列长度")
    parser.add_argument("--base-channels", type=int, default=64, help="U-Net基础通道数")
    parser.add_argument("--depth", type=int, default=4, help="U-Net深度")

    # 训练参数
    parser.add_argument("--batch-size", type=int, default=16, help="批次大小")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument(
        "--loss-alpha", type=float, default=0.8, help="损失函数alpha参数 (L1和L2权重)"
    )
    parser.add_argument("--loss-lambda", type=float, default=0.1, help="梯度损失权重")
    parser.add_argument("--early-stopping", type=int, default=10, help="早停耐心值")

    # 数据处理参数
    parser.add_argument(
        "--normalization",
        type=str,
        default="minmax",
        choices=["minmax", "zscore"],
        help="归一化方法",
    )
    parser.add_argument("--n-channels", type=int, default=3, help="通道数（1或3）")
    parser.add_argument(
        "--target-size", type=str, default=None, help="目标尺寸，格式: 256,256"
    )

    # 其他参数
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="设备",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument(
        "--output-dir", type=str, default="outputs/pixel_unet", help="输出目录"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()
    print(args)

    # 设置随机种子
    torch.manual_seed(args.seed)

    # 解析target_size
    target_size = None
    if args.target_size:
        h, w = map(int, args.target_size.split(","))
        target_size = (h, w)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("训练像素空间U-Net - Step 3: 图像到图像预测基线")
    print("=" * 80)

    # 注意：配置将在数据加载后更新以包含实际使用的 levels

    # 加载数据
    print("\n" + "-" * 80)
    print("Step 1: 加载和预处理数据")
    print("-" * 80)

    # 如果用户指定了 levels，则传给数据模块
    levels = args.levels
    if levels is not None:
        print(f"使用指定的levels: {levels}")
        if len(levels) == 1:
            print(
                f"  → 单level模式：将从原始数据选择level {levels[0]}，转换为{args.n_channels}通道图像"
            )
        else:
            print(
                f"  → 多level模式：将从原始数据选择{len(levels)}个levels，每个level转换为{args.n_channels}通道"
            )
            print(f"  → 注意：推荐使用单level模式以减少复杂度")
    else:
        print("使用所有可用的levels（默认）")

    data_module = WeatherDataModule(
        data_path=args.data_path,
        variable=args.variables,
        time_slice=args.time_slice,
        input_length=args.input_length,
        output_length=args.output_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalization=args.normalization,
        n_channels=args.n_channels,
        target_size=target_size,
        levels=levels,
    )

    data_module.setup()

    # 获取数据加载器
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # 获取样本以确定输入输出尺寸
    sample_input, sample_output = data_module.get_sample("train", 0)
    T_in, C, H, W = sample_input.shape
    T_out = sample_output.shape[0]

    print(f"\n数据信息:")
    print(f"  输入形状: ({T_in}, {C}, {H}, {W})")
    print(f"  输出形状: ({T_out}, {C}, {H}, {W})")
    print(f"  训练批次数: {len(train_loader)}")
    print(f"  验证批次数: {len(val_loader)}")

    # 保存配置（在数据加载后，以便包含实际使用的levels）
    import json

    config = vars(args).copy()
    # 记录实际使用的 levels
    if hasattr(data_module, "levels") and data_module.levels is not None:
        config["levels"] = data_module.levels
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"✓ 配置已保存到: {output_dir / 'config.json'}")

    # 创建模型
    print("\n" + "-" * 80)
    print("Step 2: 创建U-Net模型")
    print("-" * 80)

    in_channels = T_in * C
    out_channels = T_out * C

    model = WeatherUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=args.base_channels,
        depth=args.depth,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数:")
    print(f"  输入通道: {in_channels}")
    print(f"  输出通道: {out_channels}")
    print(f"  基础通道数: {args.base_channels}")
    print(f"  网络深度: {args.depth}")
    print(f"  总参数量: {n_params:,}")

    # 创建训练器
    print("\n" + "-" * 80)
    print("Step 3: 开始训练")
    print("-" * 80)

    trainer = UNetTrainer(
        model=model,
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        loss_alpha=args.loss_alpha,
        loss_lambda=args.loss_lambda,
    )

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

    with open(output_dir / "normalizer_stats.pkl", "wb") as f:
        pickle.dump(
            {"method": args.normalization, "stats": data_module.normalizer.get_stats()},
            f,
        )
    print(f"✓ 归一化统计量已保存到: {output_dir / 'normalizer_stats.pkl'}")

    print("\n" + "=" * 80)
    print("训练完成!")
    print("=" * 80)
    print(f"\n模型和结果保存在: {output_dir}")
    print("\n下一步:")
    print("  1. 使用 predict_pixel_unet.py 进行预测")
    print("  2. 评估模型性能")
    print("  3. 如果效果好，继续 Step 4: 训练潜空间模型")


if __name__ == "__main__":
    main()
