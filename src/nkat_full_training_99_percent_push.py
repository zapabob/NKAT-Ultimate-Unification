#!/usr/bin/env python3
# nkat_full_training_99_percent_push.py
"""
NKAT-Transformer フル訓練 99%精度プッシュ
TPE=0.7113のベストパラメータ + 最新テクニック

目標:
- ValAcc ≥ 99.0%
- λ_theory ≈ 20 (軽量性維持)
- TPE ≥ 0.75 (99%/log10(1+20k) ≈ 0.75)

最新テクニック:
- EMA (Exponential Moving Average)
- Mixup + CutMix
- Stochastic Depth (DropPath)
- Lookahead Optimizer
- Gradient Clipping
- 電源断リカバリーシステム
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import json
import time
import os
import pickle
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 英語グラフ設定（文字化け防止）
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# CUDA最適化
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

from nkat_transformer.model import NKATVisionTransformer
from utils.metrics import tpe_metric, count_nkat_parameters


class EMAModel:
    """Exponential Moving Average for model parameters"""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model: nn.Module):
        """EMAパラメータ更新"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self, model: nn.Module):
        """EMAパラメータを適用"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self, model: nn.Module):
        """元のパラメータを復元"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class LookaheadOptimizer:
    """Lookahead Optimizer wrapper"""
    
    def __init__(self, optimizer: optim.Optimizer, k: int = 5, alpha: float = 0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.step_count = 0
        
        self.slow_weights = {}
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.slow_weights[p] = p.data.clone()
    
    def step(self, closure=None):
        """最適化ステップ"""
        loss = self.optimizer.step(closure)
        self.step_count += 1
        
        if self.step_count % self.k == 0:
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    if p.requires_grad and p in self.slow_weights:
                        self.slow_weights[p] += self.alpha * (p.data - self.slow_weights[p])
                        p.data = self.slow_weights[p]
        
        return loss
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def state_dict(self):
        return {
            'optimizer': self.optimizer.state_dict(),
            'slow_weights': self.slow_weights,
            'step_count': self.step_count
        }
    
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.slow_weights = state_dict['slow_weights']
        self.step_count = state_dict['step_count']


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Mixup data augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """CutMix data augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    y_a, y_b = y, y[index]
    
    # CutMix box
    W, H = x.size(2), x.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    return x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss calculation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class PowerRecoverySystem:
    """電源断リカバリーシステム"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints/recovery"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.recovery_file = os.path.join(checkpoint_dir, "recovery_state.pkl")
    
    def save_state(self, epoch: int, model: nn.Module, optimizer, scheduler, 
                   ema_model: EMAModel, train_losses: List[float], 
                   val_accuracies: List[float], best_acc: float):
        """状態保存"""
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'ema_state_dict': ema_model.shadow if ema_model else None,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'best_acc': best_acc,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.recovery_file, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self) -> Optional[Dict]:
        """状態復元"""
        if os.path.exists(self.recovery_file):
            try:
                with open(self.recovery_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"WARNING: Recovery file corrupted: {e}")
                return None
        return None
    
    def cleanup(self):
        """リカバリーファイル削除"""
        if os.path.exists(self.recovery_file):
            os.remove(self.recovery_file)


class FullTraining99PercentPush:
    """フル訓練99%プッシュシステム"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.recovery_system = PowerRecoverySystem()
        
        # ベストパラメータ（TPE=0.7113）
        self.best_params = {
            'temperature': 0.5469995537778101,
            'top_k': 6,
            'top_p': 0.7554317362491326,
            'nkat_strength': 0.0023912273756024186,
            'nkat_decay': 0.992297354758566,
            'lr': 0.00023485677334823308,
            'label_smoothing': 0.06691314688896288,
            'dropout_attn': 0.17479367920079422,
            'dropout_embed': 0.07883647026878872,
            'embed_dim': 384,
            'depth': 5
        }
        
        # 拡張設定
        self.config = {
            'epochs': 30,
            'batch_size': 256,
            'early_stopping_patience': 8,
            'ema_decay': 0.9999,
            'mixup_alpha': 0.1,
            'cutmix_alpha': 0.1,
            'drop_path_rate': 0.1,
            'lookahead_k': 5,
            'lookahead_alpha': 0.5,
            'grad_clip_norm': 1.0,
            'checkpoint_every': 5
        }
    
    def create_model(self) -> nn.Module:
        """最適化モデル作成"""
        model = NKATVisionTransformer(
            img_size=28,
            patch_size=4,
            num_classes=10,
            embed_dim=self.best_params['embed_dim'],
            depth=self.best_params['depth'],
            num_heads=8,
            temperature=self.best_params['temperature'],
            top_k=self.best_params['top_k'],
            top_p=self.best_params['top_p'],
            nkat_strength=self.best_params['nkat_strength'],
            nkat_decay=self.best_params['nkat_decay'],
            dropout_attn=self.best_params['dropout_attn'],
            dropout_embed=self.best_params['dropout_embed'],
            drop_path_rate=self.config['drop_path_rate']
        ).to(self.device)
        
        return model
    
    def create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """データローダー作成"""
        # 強化されたデータ拡張
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=train_transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=test_transform
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True, 
            num_workers=4,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, test_loader
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   optimizer, criterion, ema_model: EMAModel, epoch: int) -> float:
        """1エポック訓練"""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        
        for batch_idx, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(self.device), targets.to(self.device).long()
            
            # データ拡張選択
            use_mixup = np.random.random() < 0.5
            use_cutmix = np.random.random() < 0.3
            
            if use_mixup and not use_cutmix:
                data, targets_a, targets_b, lam = mixup_data(data, targets, self.config['mixup_alpha'])
            elif use_cutmix:
                data, targets_a, targets_b, lam = cutmix_data(data, targets, self.config['cutmix_alpha'])
            else:
                targets_a, targets_b, lam = targets, targets, 1.0
            
            optimizer.zero_grad()
            outputs = model(data)
            
            # Loss計算
            if lam != 1.0:
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, targets)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['grad_clip_norm'])
            
            optimizer.step()
            
            # EMA更新
            if ema_model:
                ema_model.update(model)
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg': f'{total_loss/num_batches:.4f}'
            })
        
        return total_loss / num_batches
    
    def validate(self, model: nn.Module, test_loader: DataLoader, 
                criterion, ema_model: EMAModel = None) -> Tuple[float, float]:
        """検証"""
        model.eval()
        
        # EMA適用
        if ema_model:
            ema_model.apply_shadow(model)
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device).long()
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        # EMA復元
        if ema_model:
            ema_model.restore(model)
        
        accuracy = correct / total
        avg_loss = total_loss / len(test_loader)
        
        return accuracy, avg_loss
    
    def run_full_training(self) -> Dict[str, Any]:
        """フル訓練実行"""
        print("Starting Full Training 99% Push")
        print("="*60)
        print(f"Target: ValAcc >= 99.0%, TPE >= 0.75")
        print(f"Device: {self.device}")
        print("="*60)
        
        # リカバリー確認
        recovery_state = self.recovery_system.load_state()
        start_epoch = 0
        
        # モデル・オプティマイザー作成
        model = self.create_model()
        train_loader, test_loader = self.create_data_loaders()
        
        # Loss function with label smoothing
        criterion = nn.CrossEntropyLoss(label_smoothing=self.best_params['label_smoothing'])
        
        # Optimizer (AdamW + Lookahead)
        base_optimizer = optim.AdamW(
            model.parameters(),
            lr=self.best_params['lr'],
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        optimizer = LookaheadOptimizer(
            base_optimizer,
            k=self.config['lookahead_k'],
            alpha=self.config['lookahead_alpha']
        )
        
        # Scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            base_optimizer, 
            T_max=self.config['epochs'],
            eta_min=1e-6
        )
        
        # EMA
        ema_model = EMAModel(model, decay=self.config['ema_decay'])
        
        # 履歴
        train_losses = []
        val_accuracies = []
        val_losses = []
        best_acc = 0.0
        patience_counter = 0
        
        # リカバリー復元
        if recovery_state:
            print(f"Recovering from epoch {recovery_state['epoch']}")
            start_epoch = recovery_state['epoch'] + 1
            model.load_state_dict(recovery_state['model_state_dict'])
            optimizer.load_state_dict(recovery_state['optimizer_state_dict'])
            if recovery_state['scheduler_state_dict']:
                scheduler.load_state_dict(recovery_state['scheduler_state_dict'])
            if recovery_state['ema_state_dict']:
                ema_model.shadow = recovery_state['ema_state_dict']
            train_losses = recovery_state['train_losses']
            val_accuracies = recovery_state['val_accuracies']
            best_acc = recovery_state['best_acc']
        
        # 訓練ループ
        start_time = time.time()
        
        for epoch in range(start_epoch, self.config['epochs']):
            # 訓練
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion, ema_model, epoch)
            train_losses.append(train_loss)
            
            # 検証
            val_acc, val_loss = self.validate(model, test_loader, criterion, ema_model)
            val_accuracies.append(val_acc)
            val_losses.append(val_loss)
            
            # スケジューラー更新
            scheduler.step()
            
            # ベストモデル保存
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                
                # ベストモデル保存
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'ema_state_dict': ema_model.shadow,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_accuracy': val_acc,
                    'val_loss': val_loss,
                    'config': self.config,
                    'best_params': self.best_params
                }, 'checkpoints/best_99_percent_model.pt')
                
                print(f"New best: {val_acc:.4f}")
            else:
                patience_counter += 1
            
            # 進捗表示
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1:2d}/{self.config['epochs']:2d} | "
                  f"Loss: {train_loss:.4f} | "
                  f"ValAcc: {val_acc:.4f} | "
                  f"ValLoss: {val_loss:.4f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Best: {best_acc:.4f}")
            
            # リカバリー状態保存
            if epoch % self.config['checkpoint_every'] == 0:
                self.recovery_system.save_state(
                    epoch, model, optimizer, scheduler, ema_model,
                    train_losses, val_accuracies, best_acc
                )
            
            # Early stopping
            if patience_counter >= self.config['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            # 99%達成チェック
            if val_acc >= 0.99:
                print(f"Target 99% achieved at epoch {epoch+1}!")
                break
        
        training_time = time.time() - start_time
        
        # 最終評価
        param_analysis = count_nkat_parameters(model)
        final_tpe = tpe_metric(best_acc, param_analysis['nkat_params'])
        
        # 結果サマリー
        results = {
            'timestamp': datetime.now().isoformat(),
            'best_accuracy': best_acc,
            'final_tpe_score': final_tpe,
            'lambda_theory': param_analysis['nkat_params'],
            'nkat_ratio': param_analysis['nkat_ratio'],
            'total_params': param_analysis['total_params'],
            'training_time': training_time,
            'epochs_completed': len(train_losses),
            'config': self.config,
            'best_params': self.best_params,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'val_losses': val_losses,
            'target_achieved': best_acc >= 0.99,
            'tpe_target_achieved': final_tpe >= 0.75
        }
        
        # リカバリーファイル削除
        self.recovery_system.cleanup()
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str = None):
        """結果保存"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"logs/full_training_99_percent_{timestamp}.json"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {output_path}")
    
    def create_visualization(self, results: Dict[str, Any], output_path: str = None):
        """結果可視化"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"logs/full_training_99_percent_viz_{timestamp}.png"
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        epochs = range(1, len(results['train_losses']) + 1)
        
        # 訓練Loss
        ax1.plot(epochs, results['train_losses'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, results['val_losses'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 検証精度
        ax2.plot(epochs, [acc*100 for acc in results['val_accuracies']], 'g-', linewidth=2)
        ax2.axhline(y=99.0, color='r', linestyle='--', alpha=0.7, label='99% Target')
        ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([90, 100])
        
        # TPE進化
        tpe_scores = []
        for acc in results['val_accuracies']:
            tpe = tpe_metric(acc, results['lambda_theory'])
            tpe_scores.append(tpe)
        
        ax3.plot(epochs, tpe_scores, 'purple', linewidth=2)
        ax3.axhline(y=0.75, color='r', linestyle='--', alpha=0.7, label='TPE Target 0.75')
        ax3.set_title('TPE Score Evolution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('TPE Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 最終メトリクス
        metrics = ['Best Accuracy', 'Final TPE', 'Lambda Theory', 'Training Hours']
        values = [
            results['best_accuracy'] * 100,
            results['final_tpe_score'],
            results['lambda_theory'] / 1000,  # K単位
            results['training_time'] / 3600
        ]
        colors = ['green', 'purple', 'blue', 'orange']
        
        bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
        ax4.set_title('Final Metrics Summary', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Value')
        
        # 値をバーの上に表示
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to {output_path}")
    
    def print_summary(self, results: Dict[str, Any]):
        """結果サマリー表示"""
        print("\n" + "="*60)
        print("FULL TRAINING 99% PUSH SUMMARY")
        print("="*60)
        print(f"Best Accuracy: {results['best_accuracy']:.4f} ({results['best_accuracy']*100:.2f}%)")
        print(f"Final TPE Score: {results['final_tpe_score']:.6f}")
        print(f"Lambda Theory: {results['lambda_theory']}")
        print(f"NKAT Ratio: {results['nkat_ratio']:.2e}")
        print(f"Training Time: {results['training_time']/3600:.2f} hours")
        print(f"Epochs Completed: {results['epochs_completed']}")
        
        print(f"\nTARGET ACHIEVEMENT:")
        print(f"  99% Accuracy: {'ACHIEVED' if results['target_achieved'] else 'NOT ACHIEVED'}")
        print(f"  TPE >= 0.75: {'ACHIEVED' if results['tpe_target_achieved'] else 'NOT ACHIEVED'}")
        
        if results['target_achieved'] and results['tpe_target_achieved']:
            print(f"\nMISSION ACCOMPLISHED! 99% + TPE >= 0.75")
        elif results['target_achieved']:
            print(f"\n99% ACHIEVED! TPE needs improvement")
        else:
            print(f"\nProgress made, continue optimization")
        
        print("="*60)


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="NKAT Full Training 99% Push")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--output_dir", type=str, default="logs", help="Output directory")
    
    args = parser.parse_args()
    
    # 訓練実行
    trainer = FullTraining99PercentPush(device=args.device)
    
    # 設定上書き
    if args.epochs != 30:
        trainer.config['epochs'] = args.epochs
    if args.batch_size != 256:
        trainer.config['batch_size'] = args.batch_size
    
    try:
        results = trainer.run_full_training()
        
        # 結果表示・保存
        trainer.print_summary(results)
        trainer.save_results(results)
        trainer.create_visualization(results)
        
        print(f"\nFull training completed!")
        print(f"Best model saved to: checkpoints/best_99_percent_model.pt")
        
    except KeyboardInterrupt:
        print(f"\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 