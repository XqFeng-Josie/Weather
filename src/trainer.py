"""
模型训练器
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Optional
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt


class WeatherTrainer:
    """天气预测模型训练器"""

    def __init__(
        self,
        model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        learning_rate=1e-3,
        weight_decay=1e-5,
        gradient_accumulation_steps=1,
        criterion=None,
    ):
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # 如果是PyTorch模型，移到GPU
        if hasattr(model, "to"):
            self.model = model.to(device)
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=5
            )
            # 使用自定义损失函数或默认MSE
            if criterion is not None:
                self.criterion = criterion
                if hasattr(criterion, "to"):
                    self.criterion = criterion.to(device)
                print(f"Using custom criterion: {type(criterion).__name__}")
            else:
                self.criterion = nn.MSELoss()

        self.history = {"train_loss": [], "val_loss": []}

        if gradient_accumulation_steps > 1:
            print(f"Using gradient accumulation: {gradient_accumulation_steps} steps")

    def train_sklearn_model(self, X_train, y_train, X_val, y_val):
        """训练sklearn模型（LR等）"""
        print("Training sklearn model...")
        self.model.fit(X_train, y_train)

        # 评估
        train_score = self.model.score(X_train, y_train)
        val_score = self.model.score(X_val, y_val)

        print(f"Train R²: {train_score:.4f}")
        print(f"Val R²: {val_score:.4f}")

        return {"train_r2": train_score, "val_r2": val_score}

    def train_pytorch_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
        checkpoint_path: str = "best_model.pth",
    ):
        """训练PyTorch模型"""
        print(f"Training PyTorch model on {self.device}...")

        # 创建DataLoader
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

        # 使用多进程加载和pinned memory加速
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,  # 多进程加载数据
            pin_memory=True,  # 加速CPU到GPU传输
            persistent_workers=True,  # 保持worker进程
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=2,  # 验证集用较少worker
            pin_memory=True,
        )

        # Early stopping
        best_val_loss = float("inf")
        patience_counter = 0

        # 训练循环
        for epoch in range(epochs):
            # Train
            train_loss = self._train_epoch(train_loader)

            # Validate
            val_loss = self._validate_epoch(val_loader)

            # 记录
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            # 学习率调度
            self.scheduler.step(val_loss)

            print(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # 加载最佳模型
        self.load_checkpoint(checkpoint_path)

        return self.history

    def _train_epoch(self, train_loader):
        """训练一个epoch（支持梯度累积）"""
        self.model.train()
        total_loss = 0

        self.optimizer.zero_grad()  # 在epoch开始时清零

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Forward
            y_pred = self.model(X_batch)
            loss = self.criterion(y_pred, y_batch)

            # 对梯度累积步数进行归一化
            loss = loss / self.gradient_accumulation_steps

            # Backward
            loss.backward()

            # 每accumulation_steps步或最后一个batch更新一次
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (
                batch_idx + 1
            ) == len(train_loader):
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.gradient_accumulation_steps  # 恢复真实loss

        return total_loss / len(train_loader)

    def _validate_epoch(self, val_loader):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)

                total_loss += loss.item()

        return total_loss / len(val_loader)

    def predict(self, X: np.ndarray, batch_size: int = 32):
        """预测"""
        # sklearn模型
        if hasattr(self.model, "predict") and not isinstance(self.model, nn.Module):
            return self.model.predict(X)

        # PyTorch模型
        self.model.eval()
        X_tensor = torch.FloatTensor(X)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(
            dataset, batch_size=batch_size, num_workers=2, pin_memory=True
        )

        predictions = []
        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device)
                y_pred = self.model(X_batch)
                predictions.append(y_pred.cpu().numpy())

        return np.concatenate(predictions, axis=0)

    def evaluate(self, X: np.ndarray, y: np.ndarray):
        """
        评估模型

        支持两种数据格式:
        - Flat: (n_samples, output_length, n_features)
        - Spatial: (n_samples, output_length, n_channels, H, W)
        """
        y_pred = self.predict(X)

        # 计算各种指标
        mse = np.mean((y_pred - y) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_pred - y))

        # 按时间步分别计算
        metrics_per_step = {}
        for t in range(y.shape[1]):
            mse_t = np.mean((y_pred[:, t] - y[:, t]) ** 2)
            metrics_per_step[f"rmse_step_{t+1}"] = np.sqrt(mse_t)

        metrics = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            **metrics_per_step,
        }

        return metrics, y_pred

    def save_checkpoint(self, path: str):
        """保存模型"""
        if isinstance(self.model, nn.Module):
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "history": self.history,
                },
                path,
            )
        else:
            import pickle

            with open(path, "wb") as f:
                pickle.dump(self.model, f)

    def load_checkpoint(self, path: str):
        """加载模型"""
        if isinstance(self.model, nn.Module):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "history" in checkpoint:
                self.history = checkpoint["history"]
        else:
            import pickle

            with open(path, "rb") as f:
                self.model = pickle.load(f)

    def plot_history(self, save_path: Optional[str] = None):
        """绘制训练历史"""
        if not self.history["train_loss"]:
            print("No training history to plot")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.history["train_loss"], label="Train Loss")
        plt.plot(self.history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.title("Training History")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()


def calculate_weatherbench_metrics(y_true, y_pred, climatology_mean=None):
    """
    计算WeatherBench风格的指标

    Args:
        y_true: (n_samples, lead_times, n_features)
        y_pred: (n_samples, lead_times, n_features)
        climatology_mean: 气候态均值（可选）
    """
    metrics = {}

    # RMSE per lead time
    for t in range(y_true.shape[1]):
        mse = np.mean((y_pred[:, t] - y_true[:, t]) ** 2)
        metrics[f"rmse_lead_{t+1}"] = np.sqrt(mse)

        # MAE per lead time
        metrics[f"mae_lead_{t+1}"] = np.mean(np.abs(y_pred[:, t] - y_true[:, t]))

        # Bias per lead time
        metrics[f"bias_lead_{t+1}"] = np.mean(y_pred[:, t] - y_true[:, t])

    # Overall metrics
    mse_overall = np.mean((y_pred - y_true) ** 2)
    metrics["rmse_overall"] = np.sqrt(mse_overall)
    metrics["mae_overall"] = np.mean(np.abs(y_pred - y_true))
    metrics["bias_overall"] = np.mean(y_pred - y_true)

    # R2 score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    metrics["r2_score"] = 1 - (ss_res / (ss_tot + 1e-8))

    # ACC (如果有气候态)
    if climatology_mean is not None:
        # 确保climatology_mean的形状匹配
        if climatology_mean.shape != y_true.shape:
            climatology_mean = np.broadcast_to(
                climatology_mean.mean(axis=0, keepdims=True), y_true.shape
            )

        pred_anomaly = y_pred - climatology_mean
        true_anomaly = y_true - climatology_mean

        # ACC per lead time
        for t in range(y_true.shape[1]):
            pred_anom_t = pred_anomaly[:, t]
            true_anom_t = true_anomaly[:, t]

            numerator = np.mean(pred_anom_t * true_anom_t)
            denominator = np.sqrt(np.mean(pred_anom_t**2) * np.mean(true_anom_t**2))
            metrics[f"acc_lead_{t+1}"] = numerator / (denominator + 1e-8)

        # Overall ACC
        numerator = np.mean(pred_anomaly * true_anomaly)
        denominator = np.sqrt(np.mean(pred_anomaly**2) * np.mean(true_anomaly**2))
        metrics["acc_overall"] = numerator / (denominator + 1e-8)

    # Skill score (相对于持续性预测)
    # 持续性预测：假设未来值等于最后观测值
    # 这里简化处理，使用整体均值作为基准
    persistence_mse = np.mean(
        (y_true - np.mean(y_true, axis=(0, 1), keepdims=True)) ** 2
    )
    skill_score = 1 - (mse_overall / (persistence_mse + 1e-8))
    metrics["skill_score"] = skill_score

    return metrics


if __name__ == "__main__":
    # 测试训练器
    from models import LSTMModel

    # 模拟数据
    X_train = np.random.randn(1000, 12, 10)
    y_train = np.random.randn(1000, 4, 10)
    X_val = np.random.randn(200, 12, 10)
    y_val = np.random.randn(200, 4, 10)

    # 创建模型
    model = LSTMModel(input_size=10, output_length=4)

    # 创建训练器
    trainer = WeatherTrainer(model)

    # 训练
    history = trainer.train_pytorch_model(
        X_train, y_train, X_val, y_val, epochs=5, batch_size=32
    )

    # 评估
    metrics, y_pred = trainer.evaluate(X_val, y_val)
    print("\nValidation Metrics:")
    for key, val in metrics.items():
        print(f"  {key}: {val:.4f}")
