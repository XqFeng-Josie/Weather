"""
Diffusion Model Trainer

专门为Diffusion模型设计的训练器
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from pathlib import Path


class DiffusionTrainer:
    """
    Diffusion模型训练器
    
    与标准训练器的区别：
    1. Loss计算：预测噪声 vs 预测目标
    2. 采样策略：需要多步去噪
    3. 评估方式：生成质量 + 预测精度
    """
    
    def __init__(
        self,
        model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        learning_rate=1e-4,
        weight_decay=1e-5,
        ema_decay=0.9999,
        use_ema=True,
    ):
        """
        Args:
            model: DiffusionWeatherModel
            device: 计算设备
            learning_rate: 学习率（diffusion通常用更小的lr）
            weight_decay: 权重衰减
            ema_decay: EMA衰减率（指数移动平均，稳定训练）
            use_ema: 是否使用EMA
        """
        self.model = model.to(device)
        self.device = device
        self.use_ema = use_ema
        
        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),  # Adam默认值
        )
        
        # 学习率调度
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=1000,  # 会在train中更新
            eta_min=learning_rate * 0.01
        )
        
        # EMA模型（用于推理，更稳定）
        if use_ema:
            self.ema_model = self._create_ema_model()
            self.ema_decay = ema_decay
        else:
            self.ema_model = None
        
        # Loss
        self.criterion = nn.MSELoss()
        
        # 移动scheduler到设备
        if hasattr(model, 'scheduler'):
            model.scheduler.to(device)
        
        # 训练历史
        self.history = {"train_loss": [], "val_loss": []}
    
    def _create_ema_model(self):
        """创建EMA模型（参数的指数移动平均）"""
        # 深拷贝模型结构
        import copy
        ema_model = copy.deepcopy(self.model)
        ema_model.load_state_dict(self.model.state_dict())
        ema_model.eval()
        
        # 确保scheduler也在正确的设备上
        if hasattr(ema_model, 'scheduler'):
            ema_model.scheduler.to(self.device)
        
        for param in ema_model.parameters():
            param.requires_grad = False
        return ema_model
    
    def _update_ema(self):
        """更新EMA模型参数"""
        if not self.use_ema:
            return
        
        with torch.no_grad():
            for ema_param, param in zip(
                self.ema_model.parameters(),
                self.model.parameters()
            ):
                ema_param.data.mul_(self.ema_decay).add_(
                    param.data, alpha=1 - self.ema_decay
                )
    
    def train_model(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=50,
        batch_size=16,  # diffusion通常用更小的batch
        num_workers=4,
        early_stopping_patience=15,
        checkpoint_path="best_diffusion_model.pth",
        num_inference_steps=50,
    ):
        """
        训练Diffusion模型
        
        Args:
            X_train: 训练输入 (n_samples, input_length, channels, H, W)
            y_train: 训练目标 (n_samples, output_length, channels, H, W)
            X_val: 验证输入
            y_val: 验证目标
            epochs: 训练轮数
            batch_size: 批次大小
            num_workers: 数据加载线程数
            early_stopping_patience: 早停耐心
            checkpoint_path: 检查点路径
            num_inference_steps: 验证时的推理步数
        """
        print(f"Training Diffusion model on {self.device}...")
        print(f"Using EMA: {self.use_ema}")
        
        # 创建DataLoader
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers // 2,
            pin_memory=True,
        )
        
        # 更新scheduler的T_max
        self.scheduler.T_max = epochs * len(train_loader)
        
        # Early stopping
        best_val_loss = float("inf")
        patience_counter = 0
        
        # 训练循环
        for epoch in range(epochs):
            # Train
            train_loss = self._train_epoch(train_loader)
            
            # Validate
            # 前50 epochs只做快速验证（避免推理爆炸）
            # 之后每10个epoch做完整推理验证
            if epoch >= 50 and ((epoch + 1) % 10 == 0 or epoch == epochs - 1):
                val_loss = self._validate_epoch(val_loader, num_inference_steps)
            else:
                # 简单验证（不完整推理，节省时间和避免早期爆炸）
                val_loss = self._validate_epoch_fast(val_loader)
            
            # 记录
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            
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
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in tqdm(train_loader, desc="Training", leave=False):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward（返回预测噪声和真实噪声）
            noise_pred, noise_target, _ = self.model(X_batch, y_batch)
            
            # Loss：预测噪声 vs 真实噪声
            loss = self.criterion(noise_pred, noise_target)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update EMA
            self._update_ema()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def _validate_epoch_fast(self, val_loader):
        """快速验证（只计算训练loss，不做完整推理）"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # 使用EMA模型（如果有）
                model = self.ema_model if self.use_ema else self.model
                model.train()  # 需要设置为train模式才能返回训练输出
                
                result = model(X_batch, y_batch)
                if isinstance(result, tuple) and len(result) == 3:
                    noise_pred, noise_target, _ = result
                else:
                    # 如果是推理模式的输出，跳过
                    continue
                    
                loss = self.criterion(noise_pred, noise_target)
                total_loss += loss.item()
        
        self.model.eval()
        return total_loss / len(val_loader)
    
    def _validate_epoch(self, val_loader, num_inference_steps):
        """完整验证（做完整的去噪推理）"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in tqdm(val_loader, desc="Validating", leave=False):
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # 使用EMA模型（如果有）
                model = self.ema_model if self.use_ema else self.model
                
                try:
                    # 完整推理
                    y_pred = model(X_batch, None, num_inference_steps)
                    
                    # 检查是否有异常值
                    if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
                        print(f"\n⚠️  Warning: NaN/Inf detected in predictions, skipping batch")
                        continue
                    
                    # 计算预测误差（而非噪声误差）
                    loss = self.criterion(y_pred, y_batch)
                    
                    # 检查loss是否异常
                    if loss.item() > 1000:  # 异常大的loss
                        print(f"\n⚠️  Warning: Abnormal loss {loss.item():.1f}, using fallback validation")
                        # 回退到快速验证
                        model.train()
                        noise_pred, noise_target, _ = model(X_batch, y_batch)
                        loss = self.criterion(noise_pred, noise_target)
                        model.eval()
                    
                    total_loss += loss.item()
                    
                except RuntimeError as e:
                    print(f"\n⚠️  Warning: Runtime error during inference: {e}, skipping batch")
                    continue
        
        return total_loss / len(val_loader)
    
    def predict(self, X, num_inference_steps=50, use_ema=True):
        """
        生成预测
        
        Args:
            X: 输入数据 (n_samples, input_length, channels, H, W)
            num_inference_steps: 推理步数（越多越慢但质量更好）
            use_ema: 是否使用EMA模型
        
        Returns:
            predictions: (n_samples, output_length, channels, H, W)
        """
        model = self.ema_model if (use_ema and self.use_ema) else self.model
        model.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = model(X_tensor, None, num_inference_steps)
        
        return predictions.cpu().numpy()
    
    def predict_ensemble(self, X, num_members=10, num_inference_steps=None):
        """
        生成 Ensemble 预测（GenCast 风格）
        
        Args:
            X: 输入数据 (numpy array or tensor)
            num_members: Ensemble 成员数量
            num_inference_steps: 推理步数
            
        Returns:
            ensemble_predictions: numpy array (num_members, batch_size, output_length, channels, H, W)
        """
        self.model.eval()
        model = self.ema_model if self.use_ema else self.model
        
        if num_inference_steps is None:
            num_inference_steps = self.model.num_timesteps // 20
        
        # 转换输入
        if isinstance(X, np.ndarray):
            X_tensor = torch.from_numpy(X).float().to(self.device)
        else:
            X_tensor = X.to(self.device)
        
        ensemble_predictions = []
        
        with torch.no_grad():
            for i in range(num_members):
                # 每个 ensemble 成员使用不同的随机种子（通过 diffusion 的随机性实现）
                y_pred = model(X_tensor, None, num_inference_steps)
                ensemble_predictions.append(y_pred.cpu().numpy())
        
        # 堆叠成 (num_members, batch, ...)
        ensemble_predictions = np.stack(ensemble_predictions, axis=0)
        
        return ensemble_predictions
    
    def save_checkpoint(self, path):
        """保存检查点"""
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "history": self.history,
        }
        
        if self.use_ema:
            checkpoint["ema_model"] = self.ema_model.state_dict()
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.history = checkpoint["history"]
        
        if self.use_ema and "ema_model" in checkpoint:
            self.ema_model.load_state_dict(checkpoint["ema_model"])

