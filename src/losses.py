"""
多变量损失函数模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


class MultiVariableLoss(nn.Module):
    """
    多变量加权损失函数

    为每个变量分配独立的权重，解决不同变量尺度和难度差异导致的训练不平衡问题。

    支持两种数据格式：
    - Flat: (batch, time, features) - 用于LSTM/Transformer
    - Spatial: (batch, time, channels, H, W) - 用于CNN/ConvLSTM
    """

    def __init__(
        self,
        n_variables: int,
        weights: Optional[List[float]] = None,
        variable_ranges: Optional[List[Tuple[int, int]]] = None,
        format: str = "flat",
        auto_balance: bool = True,
    ):
        """
        Args:
            n_variables: 变量数量
            weights: 每个变量的权重列表，如果为None则使用均等权重
            variable_ranges: flat格式下每个变量的特征范围 [(start, end), ...]
            format: 'flat' 或 'spatial'
            auto_balance: 是否自动平衡权重（根据各变量的loss动态调整）
        """
        super().__init__()
        self.n_variables = n_variables
        self.format = format
        self.auto_balance = auto_balance
        self.variable_ranges = variable_ranges

        # 初始化权重
        if weights is None:
            weights = [1.0] * n_variables

        # 归一化权重
        total = sum(weights)
        weights = [w / total for w in weights]

        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

        # 用于自动平衡的移动平均loss
        if auto_balance:
            self.register_buffer(
                "loss_ema", torch.ones(n_variables, dtype=torch.float32)
            )
            self.ema_decay = 0.9
            self.update_count = 0
            self.warmup_steps = 100  # 前100步不调整权重

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算加权损失

        Args:
            pred: 预测值
                - flat: (batch, time, features)
                - spatial: (batch, time, channels, H, W)
            target: 真实值，形状同pred

        Returns:
            加权后的总损失
        """
        if self.format == "flat":
            return self._compute_flat_loss(pred, target)
        elif self.format == "spatial":
            return self._compute_spatial_loss(pred, target)
        else:
            raise ValueError(f"Unknown format: {self.format}")

    def _compute_flat_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """计算flat格式的加权损失"""
        if self.variable_ranges is None:
            # 如果没有指定范围，均分特征维度
            n_features = pred.shape[2]
            features_per_var = n_features // self.n_variables
            self.variable_ranges = [
                (i * features_per_var, (i + 1) * features_per_var)
                for i in range(self.n_variables)
            ]

        var_losses = []
        for i, (start, end) in enumerate(self.variable_ranges):
            var_pred = pred[:, :, start:end]
            var_target = target[:, :, start:end]
            var_loss = F.mse_loss(var_pred, var_target)
            var_losses.append(var_loss)

        # 堆叠为tensor
        var_losses = torch.stack(var_losses)

        # 自动平衡权重
        if self.auto_balance and self.training:
            self._update_auto_balance(var_losses)

        # 加权求和
        weighted_loss = torch.sum(self.weights * var_losses)

        return weighted_loss

    def _compute_spatial_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """计算spatial格式的加权损失"""
        # pred shape: (batch, time, channels, H, W)
        n_channels = pred.shape[2]

        if n_channels == self.n_variables:
            # 简单情况：每个channel对应一个变量
            var_losses = []
            for i in range(self.n_variables):
                var_pred = pred[:, :, i : i + 1, :, :]
                var_target = target[:, :, i : i + 1, :, :]
                var_loss = F.mse_loss(var_pred, var_target)
                var_losses.append(var_loss)
        else:
            # 复杂情况：可能有多层级的变量（如pressure levels）
            # 均分channels
            channels_per_var = n_channels // self.n_variables
            var_losses = []
            for i in range(self.n_variables):
                start_ch = i * channels_per_var
                end_ch = (i + 1) * channels_per_var
                var_pred = pred[:, :, start_ch:end_ch, :, :]
                var_target = target[:, :, start_ch:end_ch, :, :]
                var_loss = F.mse_loss(var_pred, var_target)
                var_losses.append(var_loss)

        var_losses = torch.stack(var_losses)

        # 自动平衡权重
        if self.auto_balance and self.training:
            self._update_auto_balance(var_losses)

        # 加权求和
        weighted_loss = torch.sum(self.weights * var_losses)

        return weighted_loss

    def _update_auto_balance(self, var_losses: torch.Tensor):
        """根据各变量的loss动态调整权重"""
        self.update_count += 1

        # Warmup期间不调整
        if self.update_count <= self.warmup_steps:
            return

        # 更新EMA
        with torch.no_grad():
            self.loss_ema = (
                self.ema_decay * self.loss_ema
                + (1 - self.ema_decay) * var_losses.detach()
            )

            # 每50步调整一次权重
            if self.update_count % 50 == 0:
                # 权重与loss成正比（loss越大，权重越大）
                # 这样可以让模型更关注难学的变量
                new_weights = self.loss_ema / (self.loss_ema.sum() + 1e-8)

                # 平滑更新
                self.weights = 0.9 * self.weights + 0.1 * new_weights

    def get_variable_losses(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        计算并返回每个变量的独立loss（用于监控）

        Returns:
            dict: {var_0: loss, var_1: loss, ...}
        """
        var_losses = {}

        if self.format == "flat":
            if self.variable_ranges is None:
                n_features = pred.shape[2]
                features_per_var = n_features // self.n_variables
                variable_ranges = [
                    (i * features_per_var, (i + 1) * features_per_var)
                    for i in range(self.n_variables)
                ]
            else:
                variable_ranges = self.variable_ranges

            for i, (start, end) in enumerate(variable_ranges):
                var_pred = pred[:, :, start:end]
                var_target = target[:, :, start:end]
                var_loss = F.mse_loss(var_pred, var_target)
                var_losses[f"var_{i}"] = var_loss.item()

        elif self.format == "spatial":
            n_channels = pred.shape[2]
            if n_channels == self.n_variables:
                for i in range(self.n_variables):
                    var_pred = pred[:, :, i : i + 1, :, :]
                    var_target = target[:, :, i : i + 1, :, :]
                    var_loss = F.mse_loss(var_pred, var_target)
                    var_losses[f"var_{i}"] = var_loss.item()
            else:
                channels_per_var = n_channels // self.n_variables
                for i in range(self.n_variables):
                    start_ch = i * channels_per_var
                    end_ch = (i + 1) * channels_per_var
                    var_pred = pred[:, :, start_ch:end_ch, :, :]
                    var_target = target[:, :, start_ch:end_ch, :, :]
                    var_loss = F.mse_loss(var_pred, var_target)
                    var_losses[f"var_{i}"] = var_loss.item()

        return var_losses

    def get_current_weights(self) -> np.ndarray:
        """获取当前权重"""
        return self.weights.cpu().numpy()


def compute_variable_wise_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    n_variables: int,
    format: str = "flat",
) -> dict:
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

            metrics[f"var_{i}_rmse"] = rmse
            metrics[f"var_{i}_mae"] = mae

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

                metrics[f"var_{i}_rmse"] = rmse
                metrics[f"var_{i}_mae"] = mae
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

                metrics[f"var_{i}_rmse"] = rmse
                metrics[f"var_{i}_mae"] = mae

    return metrics
