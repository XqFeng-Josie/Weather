"""U-Net训练器"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
import json
from typing import Optional, Dict


class CombinedLoss(nn.Module):
    """组合损失函数"""
    
    def __init__(self, alpha: float = 0.8, lambda_grad: float = 0.1):
        """
        Args:
            alpha: L1和L2的权重，loss = alpha * L1 + (1-alpha) * L2
            lambda_grad: 梯度损失的权重
        """
        super().__init__()
        self.alpha = alpha
        self.lambda_grad = lambda_grad
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
    
    def forward(self, pred, target):
        # 像素损失
        l1_loss = self.l1(pred, target)
        l2_loss = self.l2(pred, target)
        pixel_loss = self.alpha * l1_loss + (1 - self.alpha) * l2_loss
        
        # 梯度损失（空间平滑度）
        grad_pred_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        grad_target_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        grad_pred_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        grad_target_y = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        grad_loss = (self.l1(grad_pred_x, grad_target_x) + 
                    self.l1(grad_pred_y, grad_target_y))
        
        total_loss = pixel_loss + self.lambda_grad * grad_loss
        
        return total_loss


class UNetTrainer:
    """U-Net训练器"""
    
    def __init__(self,
                 model: nn.Module,
                 device: str = 'cuda',
                 lr: float = 1e-4,
                 weight_decay: float = 0.01,
                 loss_alpha: float = 0.8,
                 loss_lambda: float = 0.1):
        """
        Args:
            model: U-Net模型
            device: 设备
            lr: 学习率
            weight_decay: 权重衰减
            loss_alpha: 损失函数alpha参数
            loss_lambda: 梯度损失权重
        """
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        self.criterion = CombinedLoss(alpha=loss_alpha, lambda_grad=loss_lambda)
        self.scheduler = None
        
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            # 数据移到设备
            # inputs: (B, T_in, C, H, W)
            # targets: (B, T_out, C, H, W)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            B, T_in, C, H, W = inputs.shape
            T_out = targets.shape[1]
            
            # 展平时间维度
            inputs = inputs.reshape(B, T_in * C, H, W)
            targets = targets.reshape(B, T_out * C, H, W)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # 记录
            total_loss += loss.item()
            n_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / n_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(self, val_loader):
        """验证"""
        self.model.eval()
        total_loss = 0
        n_batches = 0
        
        for inputs, targets in tqdm(val_loader, desc='Validating'):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            B, T_in, C, H, W = inputs.shape
            T_out = targets.shape[1]
            
            inputs = inputs.reshape(B, T_in * C, H, W)
            targets = targets.reshape(B, T_out * C, H, W)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        return avg_loss
    
    def fit(self, 
            train_loader, 
            val_loader,
            epochs: int,
            save_dir: str,
            early_stopping_patience: int = 10):
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            save_dir: 保存目录
            early_stopping_patience: 早停耐心值
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建学习率调度器
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("\n" + "=" * 80)
        print(f"开始训练 - 总共 {epochs} 个epochs")
        print("=" * 80)
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 80)
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # 验证
            val_loss = self.validate(val_loader)
            self.history['val_loss'].append(val_loss)
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            print(f"Epoch {epoch + 1}/100 - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'history': self.history
                }
                
                torch.save(checkpoint, save_dir / 'best_model.pt')
                print(f"  ✓ 保存最佳模型 (验证损失: {val_loss:.6f})")
            else:
                patience_counter += 1
                print(f"  未改进 ({patience_counter}/{early_stopping_patience})")
            
            # 早停
            if patience_counter >= early_stopping_patience:
                print(f"\n早停触发 - 验证损失已 {patience_counter} 个epoch未改进")
                break
            
            # 定期保存
            if (epoch + 1) % 10 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'history': self.history
                }
                torch.save(checkpoint, save_dir / f'checkpoint_epoch_{epoch+1}.pt')
        
        # 保存训练历史
        with open(save_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print("\n" + "=" * 80)
        print("训练完成!")
        print(f"最佳验证损失: {best_val_loss:.6f}")
        print(f"模型保存在: {save_dir}")
        print("=" * 80)
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        print(f"✓ 加载检查点: {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  验证损失: {checkpoint['val_loss']:.6f}")

