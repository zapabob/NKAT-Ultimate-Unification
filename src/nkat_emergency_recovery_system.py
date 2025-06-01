#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔋 NKAT 緊急電源断復旧システム
RTX3080 CUDA最適化 + 段階的復旧 + tqdm進捗表示
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import time
from datetime import datetime
from tqdm import tqdm
import glob
import warnings
warnings.filterwarnings('ignore')

# 英語表記設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class NKATEmergencyTransformer(nn.Module):
    """緊急復旧用軽量NKAT Transformer"""
    
    def __init__(self, img_size=28, patch_size=4, num_classes=10, 
                 embed_dim=512, depth=6, num_heads=8, nkat_strength=0.02):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.nkat_strength = nkat_strength
        
        # パッチ埋め込み（軽量化）
        self.patch_embed = nn.Conv2d(
            in_channels=1, out_channels=embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
        # 位置埋め込み
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, depth)
        
        # 分類ヘッド
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        B = x.shape[0]
        
        # パッチ埋め込み
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        
        # CLS token追加
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 位置埋め込み
        x = x + self.pos_embed
        
        # NKAT強化
        if self.nkat_strength > 0:
            mean_activation = x.mean(dim=-1, keepdim=True)
            nkat_factor = 1.0 + self.nkat_strength * torch.tanh(mean_activation)
            x = x * nkat_factor
        
        # Transformer
        x = self.transformer(x)
        
        # 分類
        cls_output = self.norm(x[:, 0])
        return self.head(cls_output)

class EmergencyRecoverySystem:
    """緊急復旧システム"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recovery_dir = f"recovery_data/emergency_{self.timestamp}"
        os.makedirs(self.recovery_dir, exist_ok=True)
        
        print("🔋 NKAT Emergency Recovery System")
        print("="*50)
        print(f"🕐 Recovery Time: {self.timestamp}")
        print(f"🎯 Device: {device}")
        print(f"📁 Recovery Dir: {self.recovery_dir}")
        
        if torch.cuda.is_available():
            print(f"🚀 RTX3080 Detected: {torch.cuda.get_device_name()}")
            torch.backends.cudnn.benchmark = True
        
        print("="*50)
    
    def scan_available_checkpoints(self):
        """利用可能チェックポイントスキャン"""
        
        print("\n🔍 Scanning Available Checkpoints...")
        
        checkpoint_patterns = [
            "checkpoints/*.pth",
            "checkpoints/*/*.pth", 
            "*.pth",
            "recovery_data/*/*.pth"
        ]
        
        found_checkpoints = []
        for pattern in checkpoint_patterns:
            files = glob.glob(pattern)
            for file in files:
                if os.path.getsize(file) > 1024 * 1024:  # 1MB以上
                    found_checkpoints.append({
                        'path': file,
                        'size_mb': os.path.getsize(file) / (1024*1024),
                        'modified': os.path.getmtime(file)
                    })
        
        # 更新日時でソート
        found_checkpoints.sort(key=lambda x: x['modified'], reverse=True)
        
        print(f"✅ Found {len(found_checkpoints)} checkpoints:")
        for i, cp in enumerate(found_checkpoints[:5]):  # 上位5個表示
            print(f"  {i+1}. {cp['path']} ({cp['size_mb']:.1f}MB)")
        
        return found_checkpoints
    
    def attempt_checkpoint_recovery(self, checkpoint_path):
        """チェックポイントからの復旧試行"""
        
        print(f"\n🔄 Attempting Recovery: {checkpoint_path}")
        
        try:
            # モデル作成
            model = NKATEmergencyTransformer().to(self.device)
            
            # チェックポイント読み込み
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                elif 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            
            print("✅ Checkpoint loaded successfully")
            
            # クイック評価
            accuracy = self.quick_evaluation(model)
            
            recovery_info = {
                'checkpoint_path': checkpoint_path,
                'accuracy': accuracy,
                'recovery_successful': accuracy > 0.5,  # 50%以上で成功とみなす
                'timestamp': self.timestamp
            }
            
            return model, recovery_info
            
        except Exception as e:
            print(f"❌ Recovery failed: {e}")
            return None, {'error': str(e), 'recovery_successful': False}
    
    def quick_evaluation(self, model):
        """クイック評価"""
        
        print("⚡ Quick Evaluation...")
        
        # テストデータ準備
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
        
        # サブセット（1000サンプル）
        indices = torch.randperm(len(test_dataset))[:1000]
        test_subset = Subset(test_dataset, indices)
        test_loader = DataLoader(test_subset, batch_size=128, shuffle=False, num_workers=0)
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Quick Eval", leave=False):
                data, target = data.to(self.device), target.to(self.device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = correct / total
        print(f"⚡ Quick Accuracy: {accuracy:.3f}")
        
        return accuracy
    
    def emergency_training(self, model=None, epochs=15):
        """緊急訓練"""
        
        print(f"\n🚨 Emergency Training ({epochs} epochs)")
        
        if model is None:
            model = NKATEmergencyTransformer().to(self.device)
        
        # データローダー
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1, 0.1))
        ])
        
        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        
        # 高速訓練のため10000サンプルに制限
        indices = torch.randperm(len(train_dataset))[:10000]
        train_subset = Subset(train_dataset, indices)
        train_loader = DataLoader(train_subset, batch_size=256, shuffle=True, num_workers=2)
        
        # オプティマイザー
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # 訓練ループ
        model.train()
        training_history = []
        
        print("🔥 Starting Emergency Training...")
        
        for epoch in tqdm(range(epochs), desc="Emergency Training"):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            
            for batch_idx, (data, target) in enumerate(progress_bar):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, target)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                epoch_total += target.size(0)
                epoch_correct += (predicted == target).sum().item()
                
                if batch_idx % 10 == 0:
                    progress_bar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{100.*epoch_correct/epoch_total:.1f}%'
                    })
            
            scheduler.step()
            
            epoch_accuracy = epoch_correct / epoch_total
            avg_loss = epoch_loss / len(train_loader)
            
            training_history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'accuracy': epoch_accuracy
            })
            
            # チェックポイント保存（5エポックごと）
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(self.recovery_dir, f"emergency_checkpoint_epoch_{epoch+1}.pth")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'accuracy': epoch_accuracy,
                    'loss': avg_loss
                }, checkpoint_path)
        
        # 最終評価
        final_accuracy = self.quick_evaluation(model)
        
        # 最終チェックポイント保存
        final_checkpoint = os.path.join(self.recovery_dir, "emergency_final_model.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'training_history': training_history,
            'final_accuracy': final_accuracy,
            'timestamp': self.timestamp
        }, final_checkpoint)
        
        print(f"✅ Emergency Training Complete!")
        print(f"🎯 Final Accuracy: {final_accuracy:.3f}")
        print(f"💾 Checkpoint: {final_checkpoint}")
        
        return model, training_history, final_accuracy
    
    def create_recovery_report(self, recovery_info, training_history, final_accuracy):
        """復旧レポート作成"""
        
        print("\n📊 Creating Recovery Report...")
        
        # 可視化
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 訓練履歴
        epochs = [h['epoch'] for h in training_history]
        losses = [h['loss'] for h in training_history]
        accuracies = [h['accuracy'] for h in training_history]
        
        ax1.plot(epochs, losses, 'r-', label='Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Emergency Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(epochs, accuracies, 'b-', label='Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Emergency Training Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 復旧メトリクス
        metrics = ['Initial Recovery', 'Final Training', 'Target (99%)', 'Minimum (80%)']
        values = [
            recovery_info.get('accuracy', 0) * 100,
            final_accuracy * 100,
            99.0,
            80.0
        ]
        colors = ['orange', 'green', 'red', 'blue']
        
        bars = ax3.bar(metrics, values, color=colors, alpha=0.7)
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_title('Recovery Performance Metrics')
        ax3.axhline(y=80, color='blue', linestyle='--', alpha=0.5)
        ax3.axhline(y=99, color='red', linestyle='--', alpha=0.5)
        
        for bar, val in zip(bars, values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom')
        
        # 4. 復旧タイムライン
        ax4.axis('off')
        timeline_text = f"""Emergency Recovery Timeline
        
🔋 Power Outage Detected
⚡ Recovery System Activated: {self.timestamp}
🔍 Checkpoint Scan: {len(glob.glob('**/*.pth', recursive=True))} files
🔄 Recovery Attempt: {'✅ Success' if recovery_info.get('recovery_successful', False) else '❌ Failed'}
🚨 Emergency Training: {len(training_history)} epochs
🎯 Final Accuracy: {final_accuracy:.1%}

Status: {'🟢 RECOVERED' if final_accuracy > 0.8 else '🟡 PARTIAL' if final_accuracy > 0.5 else '🔴 FAILED'}
        """
        
        ax4.text(0.1, 0.9, timeline_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', family='monospace')
        
        plt.tight_layout()
        
        # 保存
        report_path = os.path.join(self.recovery_dir, f"emergency_recovery_report_{self.timestamp}.png")
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # JSON レポート
        json_report = {
            'recovery_timestamp': self.timestamp,
            'initial_recovery': recovery_info,
            'training_history': training_history,
            'final_accuracy': final_accuracy,
            'recovery_status': 'SUCCESS' if final_accuracy > 0.8 else 'PARTIAL' if final_accuracy > 0.5 else 'FAILED',
            'recommendations': self._generate_recommendations(final_accuracy)
        }
        
        json_path = os.path.join(self.recovery_dir, f"emergency_recovery_report_{self.timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Recovery Report: {report_path}")
        print(f"📋 JSON Report: {json_path}")
        
        return report_path, json_path
    
    def _generate_recommendations(self, final_accuracy):
        """改善提案生成"""
        
        recommendations = []
        
        if final_accuracy < 0.8:
            recommendations.append({
                'priority': 'HIGH',
                'action': 'Extend training epochs to 50+',
                'reason': f'Current accuracy {final_accuracy:.1%} < 80% minimum'
            })
        
        if final_accuracy < 0.95:
            recommendations.append({
                'priority': 'MEDIUM', 
                'action': 'Enable data augmentation and regularization',
                'reason': 'Improve generalization performance'
            })
        
        recommendations.append({
            'priority': 'LOW',
            'action': 'Run full Stage2-5 validation pipeline',
            'reason': 'Comprehensive performance verification'
        })
        
        return recommendations
    
    def run_full_recovery(self):
        """完全復旧実行"""
        
        print("🔋 Starting Full Emergency Recovery...")
        
        # 1. チェックポイントスキャン
        checkpoints = self.scan_available_checkpoints()
        
        # 2. 復旧試行
        model = None
        recovery_info = {'recovery_successful': False}
        
        if checkpoints:
            for cp in checkpoints[:3]:  # 最大3個試行
                model, recovery_info = self.attempt_checkpoint_recovery(cp['path'])
                if recovery_info.get('recovery_successful', False):
                    print(f"✅ Recovery successful from: {cp['path']}")
                    break
        
        # 3. 緊急訓練
        model, training_history, final_accuracy = self.emergency_training(model)
        
        # 4. レポート作成
        report_path, json_path = self.create_recovery_report(recovery_info, training_history, final_accuracy)
        
        # 5. サマリー表示
        print("\n" + "="*50)
        print("🔋 EMERGENCY RECOVERY COMPLETE")
        print("="*50)
        print(f"🎯 Final Accuracy: {final_accuracy:.1%}")
        print(f"📊 Status: {'🟢 SUCCESS' if final_accuracy > 0.8 else '🟡 PARTIAL' if final_accuracy > 0.5 else '🔴 FAILED'}")
        print(f"📁 Recovery Dir: {self.recovery_dir}")
        print(f"📊 Report: {report_path}")
        print("="*50)
        
        return final_accuracy > 0.8

def main():
    """メイン実行関数"""
    
    # RTX3080 CUDA設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 緊急復旧システム起動
    recovery_system = EmergencyRecoverySystem(device)
    
    # 完全復旧実行
    success = recovery_system.run_full_recovery()
    
    if success:
        print("🎉 Emergency recovery completed successfully!")
        print("🚀 Ready to continue normal operations.")
    else:
        print("⚠️ Partial recovery completed.")
        print("📋 Check recommendations for next steps.")

if __name__ == "__main__":
    main() 