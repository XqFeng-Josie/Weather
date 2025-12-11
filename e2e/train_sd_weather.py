# scripts/train_sd_weather.py
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.data_loader import WeatherDataLoader
from weatherdiff.diffusion.latent_sd_weather import (
    LDMConfig, WeatherLDM, SDTrainConfig, SDTrainer
)


def parse_args():
    ap = argparse.ArgumentParser("Train Stable Diffusion (Latent) for Weather Prediction")
    # Data
    ap.add_argument("--data-path", type=str,
                    default="gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr")
    ap.add_argument("--variables", type=str, default="2m_temperature")
    ap.add_argument("--time-slice", type=str, default="2020-01-01:2020-12-31")
    ap.add_argument("--input-length", type=int, default=12)
    ap.add_argument("--output-length", type=int, default=4)

    # Model
    ap.add_argument("--base-channels", type=int, default=64)
    ap.add_argument("--vae-latent-channels", type=int, default=4)
    ap.add_argument("--vae-kl-weight", type=float, default=1e-6)
    ap.add_argument("--cross-attn-dim", type=int, default=512)
    ap.add_argument("--patch-size", type=int, default=8)
    ap.add_argument("--num-diffusion-steps", type=int, default=1000)
    ap.add_argument("--beta-schedule", type=str, default="squaredcos_cap_v2",
                    choices=["linear", "squaredcos_cap_v2"])
    ap.add_argument("--num-inference-steps", type=int, default=50)

    # Train
    ap.add_argument("--epochs-vae", type=int, default=20)
    ap.add_argument("--epochs-ldm", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--early-stop", type=int, default=20)
    ap.add_argument("--use-amp", type=lambda s: str(s).lower() == "true", default=True)
    ap.add_argument("--stage", type=str, default="both", choices=["vae", "ldm", "both"])
    ap.add_argument("--output-dir", type=str, default="./outputs_sd")

    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Stable Diffusion (Latent) Weather")
    print("=" * 80)

    # -------------------------------
    # 1) 加载 ERA5 → 构造 (X, y)
    # -------------------------------
    variables = [v.strip() for v in args.variables.split(",")]
    loader = WeatherDataLoader(data_path=args.data_path, variables=variables)

    start, end = args.time_slice.split(":")
    ds = loader.load_data(time_slice=slice(start, end))
    feats = loader.prepare_features(normalize=True)

    X, y = loader.create_sequences(
        feats,
        input_length=args.input_length,
        output_length=args.output_length,
        format="spatial",
    )

    print("\nData shapes:")
    print(f"  X: {X.shape}  (N, T_in, C_in, H, W)")
    print(f"  y: {y.shape}  (N, T_out, C_out, H, W)")

    splits = loader.split_data(X, y, train_ratio=0.7, val_ratio=0.15)
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    # -------------------------------
    # 2) Dataloader
    # -------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # -------------------------------
    # 3) 模型
    # -------------------------------
    in_channels_history = X.shape[2]
    out_channels_future = y.shape[2]  # 与历史变量数相同（按你的管线）
    cfg_ldm = LDMConfig(
        input_length=args.input_length,
        output_length=args.output_length,
        in_channels_history=in_channels_history,
        out_channels_future=out_channels_future,
        base_channels=args.base_channels,
        vae_latent_channels=args.vae_latent_channels,
        vae_kl_weight=args.vae_kl_weight,
        patch_size=args.patch_size,
        cross_attn_dim=args.cross_attn_dim,
        num_diffusion_steps=args.num_diffusion_steps,
        beta_schedule=args.beta_schedule,
    )
    model = WeatherLDM(cfg_ldm).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel params (trainable): {n_params:,}")

    cfg_train = SDTrainConfig(
        device=device,
        lr=args.lr,
        epochs_vae=args.epochs_vae,
        epochs_ldm=args.epochs_ldm,
        batch_size=args.batch_size,
        early_stop=args.early_stop,
        num_inference_steps_val=args.num_inference_steps,
        checkpoint_path=str(out_dir / "best_sd.pth"),
        use_amp=args.use_amp,
    )

    trainer = SDTrainer(model, cfg_train)

    # -------------------------------
    # 4) 训练
    # -------------------------------
    history = trainer.fit(train_loader, val_loader, stage=args.stage)

    # -------------------------------
    # 5) 评估（采样 → 指标）
    # -------------------------------
    print("\nEvaluating on test set (deterministic: 1 sample run) ...")
    y_preds = []

    with torch.no_grad():
        for Xb, _ in tqdm(test_loader):
            Xb = Xb.to(device)
            yp = model.sample(Xb, num_inference_steps=args.num_inference_steps)
            y_preds.append(yp.cpu().numpy())

    y_pred = np.concatenate(y_preds, axis=0)
    np.save(out_dir / "y_test_pred_sd.npy", y_pred)
    np.save(out_dir / "y_test_sd.npy", y_test)

    mse = np.mean((y_test - y_pred) ** 2)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_test - y_pred)))

    # 每个预报步的 RMSE
    rmse_per_step = {}
    for i in range(args.output_length):
        rm = np.sqrt(np.mean((y_test[:, i] - y_pred[:, i]) ** 2))
        rmse_per_step[f"rmse_step_{i+1}"] = float(rm)

    metrics = {"mse": float(mse), "rmse": rmse, "mae": mae, **rmse_per_step}
    with open(out_dir / "metrics_sd.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nTest Metrics (deterministic):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # -------------------------------
    # 6) 可视化
    # -------------------------------
    if len(history["train_loss"]) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(history["train_loss"], label="Train")
        plt.plot(history["val_loss"], label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("LDM Training History")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(out_dir / "training_history_sd.png", dpi=160, bbox_inches="tight")
        plt.close()

    # RMSE vs lead time
    steps = list(range(1, args.output_length + 1))
    vals = [metrics[f"rmse_step_{i}"] for i in steps]
    plt.figure(figsize=(8, 5))
    plt.plot(steps, vals, "o-", linewidth=2)
    plt.xlabel("Lead Time Step")
    plt.ylabel("RMSE")
    plt.title("Stable Diffusion (Latent) - RMSE vs Lead Time")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / "rmse_vs_leadtime_sd.png", dpi=160, bbox_inches="tight")
    plt.close()

    # 空间展示（第一个样本/变量，最多4步）
    sample_idx, var_idx = 0, 0
    H, W = y_test.shape[-2], y_test.shape[-1]
    plt.figure(figsize=(18, 8))
    Tplot = min(4, args.output_length)
    for t in range(Tplot):
        plt.subplot(2, Tplot, t + 1)
        plt.imshow(y_test[sample_idx, t, var_idx], cmap="RdBu_r")
        plt.title(f"True t+{t+1}")
        plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(2, Tplot, Tplot + t + 1)
        plt.imshow(y_pred[sample_idx, t, var_idx], cmap="RdBu_r")
        plt.title(f"Pred t+{t+1}")
        plt.colorbar(fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_dir / "spatial_examples_sd.png", dpi=160, bbox_inches="tight")
    plt.close()

    print(f"\n✓ Results saved to {out_dir}")
    print(f"✓ Checkpoint: {out_dir/'best_sd.pth'}")


if __name__ == "__main__":
    main()
