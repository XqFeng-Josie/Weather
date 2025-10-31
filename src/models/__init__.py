"""
天气预测模型模块
提供各种模型实现：LR, CNN, LSTM, ConvLSTM, Transformer
"""

from .linear_regression import LinearRegressionModel, MultiOutputLinearRegression
from .cnn import SimpleCNN
from .lstm import LSTMModel
from .convlstm import ConvLSTM, ConvLSTMModel
from .transformer import TransformerModel

__all__ = [
    "LinearRegressionModel",
    "MultiOutputLinearRegression",
    "SimpleCNN",
    "LSTMModel",
    "ConvLSTM",
    "ConvLSTMModel",
    "TransformerModel",
]


def get_model(model_name: str, **kwargs):
    """
    模型工厂函数

    Args:
        model_name: 模型名称 ['lr', 'lr_multi', 'cnn', 'lstm', 'convlstm', 'transformer']
        **kwargs: 模型参数
    """
    models = {
        "lr": LinearRegressionModel,
        "lr_multi": MultiOutputLinearRegression,
        "cnn": SimpleCNN,
        "lstm": LSTMModel,
        "convlstm": ConvLSTMModel,
        "transformer": TransformerModel,
    }

    if model_name.lower() not in models:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(models.keys())}"
        )

    return models[model_name.lower()](**kwargs)


def count_parameters(model):
    """统计模型参数量"""
    if hasattr(model, "models"):  # MultiOutputLinearRegression
        total = 0
        for m in model.models:
            if hasattr(m, "coef_"):
                total += m.coef_.size
        return total
    elif hasattr(model, "model") and hasattr(
        model.model, "coef_"
    ):  # LinearRegressionModel
        return model.model.coef_.size
    elif hasattr(model, "parameters"):  # PyTorch models
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return 0
