"""
线性回归模型
包括：标准LR和多输出独立LR（每个变量独立模型）
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from typing import List


class LinearRegressionModel:
    """标准线性回归baseline - 所有变量共享一个模型"""

    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def fit(self, X, y):
        """
        Args:
            X: (n_samples, input_length, n_features)
            y: (n_samples, output_length, n_features)
        """
        # 展平为2D: (n_samples, input_length * n_features)
        X_flat = X.reshape(X.shape[0], -1)
        y_flat = y.reshape(y.shape[0], -1)

        # 标准化
        X_scaled = self.scaler_X.fit_transform(X_flat)
        y_scaled = self.scaler_y.fit_transform(y_flat)

        # 训练
        print(f"Training Linear Regression on {X_scaled.shape} -> {y_scaled.shape}")
        self.model.fit(X_scaled, y_scaled)

        return self

    def predict(self, X):
        """预测"""
        original_shape = X.shape
        X_flat = X.reshape(X.shape[0], -1)
        X_scaled = self.scaler_X.transform(X_flat)
        y_pred_scaled = self.model.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)

        # 还原形状: (n_samples, output_length, n_features)
        # 推断output_length
        output_length = y_pred.shape[1] // original_shape[2]
        return y_pred.reshape(X.shape[0], output_length, original_shape[2])

    def score(self, X, y):
        """R2 score"""
        X_flat = X.reshape(X.shape[0], -1)
        y_flat = y.reshape(y.shape[0], -1)
        X_scaled = self.scaler_X.transform(X_flat)
        y_scaled = self.scaler_y.transform(y_flat)
        return self.model.score(X_scaled, y_scaled)


class MultiOutputLinearRegression:
    """
    多变量独立线性回归 - 每个**变量**一个模型

    例如：2m_temperature一个模型，geopotential一个模型
    每个模型预测该变量的所有网格点
    """

    def __init__(self, alpha=10.0, n_variables=1):
        """
        Args:
            alpha: Ridge正则化参数
            n_variables: 变量数量（如2m_temperature, geopotential算2个）
        """
        self.alpha = alpha
        self.n_variables = n_variables
        self.models = []  # 每个变量一个模型
        self.scalers_X = []
        self.scalers_y = []
        self.grid_points_per_var = None
        self.output_length = None

    def fit(self, X, y):
        """
        为每个变量训练独立模型

        Args:
            X: (n_samples, input_length, n_features)
               n_features = n_variables × grid_points
            y: (n_samples, output_length, n_features)
        """
        n_samples, input_length, total_features = X.shape
        self.output_length = y.shape[1]

        # 计算每个变量的网格点数
        self.grid_points_per_var = total_features // self.n_variables

        print(f"\nTraining Multi-Variable LR")
        print(f"  Variables: {self.n_variables}")
        print(f"  Grid points per variable: {self.grid_points_per_var}")
        print(f"  Total features: {total_features}")

        # 展平输入: (n_samples, input_length * total_features)
        X_flat = X.reshape(n_samples, -1)

        # 为每个变量训练一个模型
        for var_idx in range(self.n_variables):
            # 提取该变量对应的网格点
            start_idx = var_idx * self.grid_points_per_var
            end_idx = (var_idx + 1) * self.grid_points_per_var

            # y中该变量的数据: (n_samples, output_length, grid_points)
            y_var = y[:, :, start_idx:end_idx]
            # 展平为: (n_samples, output_length * grid_points)
            y_var_flat = y_var.reshape(n_samples, -1)

            # 标准化
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_scaled = scaler_X.fit_transform(X_flat)
            y_scaled = scaler_y.fit_transform(y_var_flat)

            # 训练模型
            model = Ridge(alpha=self.alpha, solver="cholesky")
            model.fit(X_scaled, y_scaled)

            self.models.append(model)
            self.scalers_X.append(scaler_X)
            self.scalers_y.append(scaler_y)

            r2 = model.score(X_scaled, y_scaled)
            print(f"  Variable {var_idx+1}/{self.n_variables}: R²={r2:.4f}")

        return self

    def predict(self, X):
        """
        使用各变量的模型预测

        Args:
            X: (n_samples, input_length, n_features)
        Returns:
            (n_samples, output_length, n_features)
        """
        n_samples = X.shape[0]
        total_features = X.shape[2]
        X_flat = X.reshape(n_samples, -1)

        # 初始化输出
        y_pred = np.zeros((n_samples, self.output_length, total_features))

        # 每个变量用对应的模型预测
        for var_idx in range(self.n_variables):
            start_idx = var_idx * self.grid_points_per_var
            end_idx = (var_idx + 1) * self.grid_points_per_var

            # 标准化
            X_scaled = self.scalers_X[var_idx].transform(X_flat)

            # 预测
            y_pred_scaled = self.models[var_idx].predict(X_scaled)

            # 反标准化
            y_pred_var = self.scalers_y[var_idx].inverse_transform(y_pred_scaled)

            # 重塑并放回对应位置
            y_pred_var = y_pred_var.reshape(
                n_samples, self.output_length, self.grid_points_per_var
            )
            y_pred[:, :, start_idx:end_idx] = y_pred_var

        return y_pred

    def score(self, X, y):
        """计算平均R2 score"""
        y_pred = self.predict(X)

        # 计算整体R2
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - (ss_res / (ss_tot + 1e-8))


if __name__ == "__main__":
    # 测试
    np.random.seed(42)

    # 模拟数据
    n_samples = 1000
    input_length = 12
    output_length = 4
    n_features = 5

    X = np.random.randn(n_samples, input_length, n_features)
    y = np.random.randn(n_samples, output_length, n_features)

    print("=" * 60)
    print("Testing Standard LR")
    print("=" * 60)
    model1 = LinearRegressionModel(alpha=1.0)
    model1.fit(X, y)
    y_pred1 = model1.predict(X)
    print(f"Prediction shape: {y_pred1.shape}")
    print(f"R2 score: {model1.score(X, y):.4f}")

    print("\n" + "=" * 60)
    print("Testing Multi-Output LR (2 variables)")
    print("=" * 60)
    # 模拟2个变量，每个变量2.5个特征
    model2 = MultiOutputLinearRegression(alpha=10.0, n_variables=2)
    # n_features=5, 假设前2.5个是var1，后2.5个是var2（实际会是整数）
    X2 = np.random.randn(n_samples, input_length, 4)  # 4个特征，2个变量
    y2 = np.random.randn(n_samples, output_length, 4)
    model2.fit(X2, y2)
    y_pred2 = model2.predict(X2)
    print(f"Prediction shape: {y_pred2.shape}")
    print(f"R2 score: {model2.score(X2, y2):.4f}")
