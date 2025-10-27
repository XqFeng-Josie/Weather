"""
快速测试脚本 - 验证pipeline各个组件
"""
import numpy as np
import torch
import sys
from pathlib import Path

def test_models():
    """测试模型"""
    print("="*80)
    print("Testing Models")
    print("="*80)
    
    from src.models import LSTMModel, TransformerModel, LinearRegressionModel, count_parameters
    
    batch_size = 32
    input_length = 12
    output_length = 4
    n_features = 10
    
    # 测试数据
    X = np.random.randn(batch_size, input_length, n_features)
    y = np.random.randn(batch_size, output_length, n_features)
    
    # 1. Linear Regression
    print("\n1. Linear Regression:")
    lr_model = LinearRegressionModel(alpha=1.0)
    lr_model.fit(X, y)
    y_pred = lr_model.predict(X)
    print(f"   ✓ Input: {X.shape}, Output: {y_pred.shape}")
    assert y_pred.shape == y.shape, "Shape mismatch!"
    
    # 2. LSTM
    print("\n2. LSTM:")
    lstm = LSTMModel(input_size=n_features, output_length=output_length)
    X_tensor = torch.FloatTensor(X)
    y_pred = lstm(X_tensor)
    print(f"   ✓ Input: {X_tensor.shape}, Output: {y_pred.shape}")
    print(f"   ✓ Parameters: {count_parameters(lstm):,}")
    assert y_pred.shape == (batch_size, output_length, n_features)
    
    # 3. Transformer
    print("\n3. Transformer:")
    transformer = TransformerModel(input_size=n_features, output_length=output_length)
    y_pred = transformer(X_tensor)
    print(f"   ✓ Input: {X_tensor.shape}, Output: {y_pred.shape}")
    print(f"   ✓ Parameters: {count_parameters(transformer):,}")
    assert y_pred.shape == (batch_size, output_length, n_features)
    
    print("\n✓ All models working correctly!")
    return True


def test_data_loader():
    """测试数据加载器（使用模拟数据）"""
    print("\n" + "="*80)
    print("Testing Data Loader")
    print("="*80)
    
    from src.data_loader import WeatherDataLoader
    
    # 创建模拟数据
    import xarray as xr
    import pandas as pd
    
    times = pd.date_range('2020-01-01', '2020-12-31', freq='6H')
    lats = np.linspace(-90, 90, 32)
    lons = np.linspace(0, 360, 64)
    
    temp = np.random.randn(len(times), len(lats), len(lons))
    
    ds = xr.Dataset({
        'temperature': (['time', 'latitude', 'longitude'], temp),
    }, coords={
        'time': times,
        'latitude': lats,
        'longitude': lons,
    })
    
    print(f"   Mock data shape: {ds['temperature'].shape}")
    print(f"   Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
    
    # 测试序列创建
    print("\n   Testing sequence creation...")
    data_flat = temp[:, 15, 30]  # 单点
    X, y = WeatherDataLoader.create_sequences(
        data_flat[:, None],  # (time, 1)
        input_length=12,
        output_length=4
    )
    print(f"   ✓ X: {X.shape}, y: {y.shape}")
    assert X.shape[1] == 12
    assert y.shape[1] == 4
    
    print("\n✓ Data loader working correctly!")
    return True


def test_trainer():
    """测试训练器"""
    print("\n" + "="*80)
    print("Testing Trainer")
    print("="*80)
    
    from src.models import LSTMModel
    from src.trainer import WeatherTrainer
    
    # 模拟数据
    X_train = np.random.randn(100, 12, 5)
    y_train = np.random.randn(100, 4, 5)
    X_val = np.random.randn(20, 12, 5)
    y_val = np.random.randn(20, 4, 5)
    
    print(f"   Train: X={X_train.shape}, y={y_train.shape}")
    print(f"   Val:   X={X_val.shape}, y={y_val.shape}")
    
    # 创建模型和训练器
    model = LSTMModel(input_size=5, output_length=4, hidden_size=32, num_layers=1)
    trainer = WeatherTrainer(model, learning_rate=0.01)
    
    print(f"\n   Training for 3 epochs (quick test)...")
    history = trainer.train_pytorch_model(
        X_train, y_train, X_val, y_val,
        epochs=3,
        batch_size=16,
        early_stopping_patience=10
    )
    
    print(f"   ✓ Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"   ✓ Final val loss: {history['val_loss'][-1]:.4f}")
    
    # 测试预测
    print(f"\n   Testing prediction...")
    y_pred = trainer.predict(X_val)
    print(f"   ✓ Prediction shape: {y_pred.shape}")
    assert y_pred.shape == y_val.shape
    
    # 测试评估
    print(f"\n   Testing evaluation...")
    metrics, _ = trainer.evaluate(X_val, y_val)
    print(f"   ✓ RMSE: {metrics['rmse']:.4f}")
    print(f"   ✓ MAE: {metrics['mae']:.4f}")
    
    print("\n✓ Trainer working correctly!")
    return True


def test_end_to_end():
    """端到端测试"""
    print("\n" + "="*80)
    print("End-to-End Test")
    print("="*80)
    
    from src.models import LSTMModel
    from src.trainer import WeatherTrainer
    
    # 1. 准备数据
    print("\n1. Preparing data...")
    n_samples = 200
    X = np.random.randn(n_samples, 12, 8)
    y = X[:, -4:, :] + np.random.randn(n_samples, 4, 8) * 0.1  # 简单的预测任务
    
    train_size = int(n_samples * 0.7)
    val_size = int(n_samples * 0.85)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:val_size], y[train_size:val_size]
    X_test, y_test = X[val_size:], y[val_size:]
    
    print(f"   ✓ Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # 2. 创建模型
    print("\n2. Creating model...")
    model = LSTMModel(input_size=8, output_length=4, hidden_size=64, num_layers=2)
    trainer = WeatherTrainer(model)
    
    # 3. 训练
    print("\n3. Training (5 epochs)...")
    trainer.train_pytorch_model(
        X_train, y_train, X_val, y_val,
        epochs=5,
        batch_size=32
    )
    
    # 4. 评估
    print("\n4. Evaluating...")
    test_metrics, y_pred = trainer.evaluate(X_test, y_test)
    
    print(f"\n   Test Metrics:")
    for key, val in test_metrics.items():
        if 'step' in key:
            continue
        print(f"     {key}: {val:.4f}")
    
    # 5. 保存/加载
    print("\n5. Testing save/load...")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        temp_path = f.name
    
    trainer.save_checkpoint(temp_path)
    print(f"   ✓ Saved to {temp_path}")
    
    # 加载
    model2 = LSTMModel(input_size=8, output_length=4, hidden_size=64, num_layers=2)
    trainer2 = WeatherTrainer(model2)
    trainer2.load_checkpoint(temp_path)
    print(f"   ✓ Loaded from {temp_path}")
    
    # 验证预测一致
    y_pred2 = trainer2.predict(X_test)
    assert np.allclose(y_pred, y_pred2), "Predictions don't match after load!"
    print(f"   ✓ Predictions match after load")
    
    # 清理
    Path(temp_path).unlink()
    
    print("\n✓ End-to-end test passed!")
    return True


def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("Weather Prediction Pipeline Tests")
    print("="*80)
    
    tests = [
        ("Models", test_models),
        ("Data Loader", test_data_loader),
        ("Trainer", test_trainer),
        ("End-to-End", test_end_to_end),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success, None))
        except Exception as e:
            print(f"\n❌ {name} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False, str(e)))
    
    # 总结
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    
    for name, success, error in results:
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")
        if error:
            print(f"       Error: {error}")
    
    n_pass = sum(1 for _, s, _ in results if s)
    n_total = len(results)
    
    print(f"\nResult: {n_pass}/{n_total} tests passed")
    
    if n_pass == n_total:
        print("\n🎉 All tests passed! Pipeline is ready to use.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())

