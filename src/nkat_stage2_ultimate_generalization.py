#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT Stage ② Ultimate Generalization Test
RTX3080最適化 + tqdm進捗表示 + 英語グラフ表記
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import json
import time
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 英語表記設定（文字化け防止）
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class NKATPatchEmbedding(nn.Module):
    """段階的パッチ埋め込み（nkat_core_standalone.py参考）"""
    
    def __init__(self, img_size=28, patch_size=4, channels=1, embed_dim=512):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        self.conv_layers = nn.Sequential(
            # Stage 1: 低レベル特徴
            nn.Conv2d(channels, embed_dim // 4, 3, padding=1),
            nn.BatchNorm2d(embed_dim // 4),
            nn.GELU(),
            
            # Stage 2: 中レベル特徴
            nn.Conv2d(embed_dim // 4, embed_dim // 2, 3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            
            # Stage 3: 高レベル特徴 + パッチ化
            nn.Conv2d(embed_dim // 2, embed_dim, patch_size, stride=patch_size)
        )
        
    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, H//patch_size, W//patch_size)
        x = self.conv_layers(x)
        # Flatten: (B, embed_dim, num_patches)
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

class NKATTransformerPractical(nn.Module):
    """NKAT Transformer Practical Edition - nkat_core_standalone.py準拠"""
    
    def __init__(self, img_size=28, patch_size=4, num_classes=10, 
                 embed_dim=512, depth=8, num_heads=8, mlp_ratio=4.0, 
                 nkat_strength=0.0, dropout=0.1):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.nkat_strength = nkat_strength
        
        # 入力チャンネル数を動的に決定
        channels = 1 if num_classes <= 47 else 3
        
        # 段階的パッチ埋め込み（実証済み手法）
        self.patch_embedding = NKATPatchEmbedding(img_size, patch_size, channels, embed_dim)
        
        # 位置埋め込み
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim) * 0.02
        )
        
        # クラストークン
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # 入力正規化
        self.input_norm = nn.LayerNorm(embed_dim)
        
        # 標準的なTransformer Encoder（実証済み）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, depth)
        
        # 改良された分類ヘッド
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout * 0.5),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(embed_dim // 4, num_classes)
        )
        
        # 重み初期化
        self.apply(self._init_weights)
        
        # RTX3080最適化
        self.use_amp = torch.cuda.is_available()
    
    def _init_weights(self, m):
        """重み初期化（nkat_core_standalone.py準拠）"""
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
        
        # 段階的パッチ埋め込み
        x = self.patch_embedding(x)  # (B, num_patches, embed_dim)
        
        # クラストークン追加
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches + 1, embed_dim)
        
        # 位置埋め込み
        x = x + self.pos_embedding
        x = self.input_norm(x)
        
        # 軽微なNKAT適応（オプション）
        if self.nkat_strength > 0:
            # 数学的に安全な適応的調整（サイズ不一致修正）
            mean_activation = x.mean(dim=-1, keepdim=True)  # (B, S, 1)
            nkat_factor = 1.0 + self.nkat_strength * 0.01 * torch.tanh(mean_activation)
            x = x * nkat_factor
        
        # Transformer処理
        x = self.transformer(x)
        
        # 分類
        cls_output = x[:, 0]  # (B, embed_dim)
        logits = self.classifier(cls_output)
        
        return logits

def create_enhanced_transforms(dataset_name):
    """nkat_core_standalone.py風の強化されたデータ変換"""
    
    if dataset_name == 'CIFAR10':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15, fill=0),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        # MNIST, FashionMNIST, EMNIST用
        train_transform = transforms.Compose([
            transforms.RandomRotation(degrees=15, fill=0),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=5,
                fill=0
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    return train_transform, test_transform

def get_dataset_config(dataset_name):
    """データセット設定を取得"""
    configs = {
        'MNIST': {
            'num_classes': 10,
            'channels': 1,
            'size': 28,
            'dataset_func': torchvision.datasets.MNIST,
            'target_accuracy': 99.0
        },
        'FashionMNIST': {
            'num_classes': 10,
            'channels': 1,
            'size': 28,
            'dataset_func': torchvision.datasets.FashionMNIST,
            'target_accuracy': 95.0
        },
        'EMNIST': {
            'num_classes': 47,
            'channels': 1,
            'size': 28,
            'dataset_func': lambda root, train, transform, download: 
                torchvision.datasets.EMNIST(root, split='balanced', train=train, transform=transform, download=download),
            'target_accuracy': 92.0
        },
        'CIFAR10': {
            'num_classes': 10,
            'channels': 3,
            'size': 32,
            'dataset_func': torchvision.datasets.CIFAR10,
            'target_accuracy': 90.0
        }
    }
    return configs.get(dataset_name, configs['MNIST'])

def train_and_evaluate(model, train_loader, test_loader, num_epochs, device, dataset_name):
    """RTX3080最適化トレーニング & 評価（nkat_core_standalone.py準拠）"""
    
    # nkat_core_standalone.pyスタイルの設定
    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)  # ラベルスムージング
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=1e-4,  # より保守的な学習率
        weight_decay=2e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )
    
    # Mixed precision training (RTX3080最適化)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    model.train()
    train_losses = []
    train_accuracies = []
    best_accuracy = 0
    
    print(f"\n🚀 {dataset_name} Training Starting on {device}")
    print(f"📊 Training Strategy: nkat_core_standalone.py準拠")
    
    for epoch in tqdm(range(num_epochs), desc=f"{dataset_name} Training"):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            try:
                if scaler and device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        output = model(data)
                        loss = criterion(output, target)
                    
                    # 数値安定性チェック
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"❌ NaN/Inf detected at epoch {epoch+1}, batch {batch_idx}")
                        continue
                    
                    scaler.scale(loss).backward()
                    # 勾配クリッピング（重要）
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(data)
                    loss = criterion(output, target)
                    
                    # 数値安定性チェック
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"❌ NaN/Inf detected at epoch {epoch+1}, batch {batch_idx}")
                        continue
                        
                    loss.backward()
                    # 勾配クリッピング
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
            except RuntimeError as e:
                print(f"❌ Runtime error at epoch {epoch+1}, batch {batch_idx}: {e}")
                # メモリクリア
                torch.cuda.empty_cache() if device.type == 'cuda' else None
                continue
            
            epoch_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # リアルタイム進捗更新
            if batch_idx % 50 == 0:
                current_acc = 100. * correct / total
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.2f}%',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
                
                # メモリクリア（定期的）
                if batch_idx % 200 == 0 and device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        # 学習率更新
        scheduler.step()
        
        epoch_accuracy = 100. * correct / total
        train_losses.append(epoch_loss / len(train_loader))
        train_accuracies.append(epoch_accuracy)
        
        # 定期評価
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            model.eval()
            test_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    
                    if scaler and device.type == 'cuda':
                        with torch.amp.autocast('cuda'):
                            output = model(data)
                            test_loss += criterion(output, target).item()
                    else:
                        output = model(data)
                        test_loss += criterion(output, target).item()
                    
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
            
            test_accuracy = 100. * correct / total
            best_accuracy = max(best_accuracy, test_accuracy)
            
            print(f"Epoch {epoch+1:3d}: Train: {epoch_accuracy:.2f}%, Test: {test_accuracy:.2f}%, Best: {best_accuracy:.2f}%")
        else:
            print(f"Epoch {epoch+1:3d}: Train Acc: {epoch_accuracy:.2f}%")
    
    # 最終評価
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    print(f"\n🔍 {dataset_name} Final Evaluation...")
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc=f"{dataset_name} Final Evaluation"):
            data, target = data.to(device), target.to(device)
            
            if scaler and device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    output = model(data)
                    test_loss += criterion(output, target).item()
            else:
                output = model(data)
                test_loss += criterion(output, target).item()
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    final_accuracy = 100. * correct / total
    avg_test_loss = test_loss / len(test_loader)
    
    print(f"✅ {dataset_name} Final Results:")
    print(f"   📊 Test Accuracy: {final_accuracy:.2f}%")
    print(f"   📉 Test Loss: {avg_test_loss:.4f}")
    print(f"   🎯 Best Accuracy: {max(best_accuracy, final_accuracy):.2f}%")
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracy': final_accuracy,
        'test_loss': avg_test_loss,
        'best_accuracy': max(best_accuracy, final_accuracy)
    }

def create_visualization(results, dataset_name, timestamp):
    """結果可視化（英語表記）"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'NKAT Stage II: {dataset_name} Generalization Results', fontsize=16, fontweight='bold')
    
    # Training Loss
    ax1.plot(results['train_losses'], 'b-', linewidth=2, label='Training Loss')
    ax1.set_title('Training Loss Progression', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Training Accuracy
    ax2.plot(results['train_accuracies'], 'g-', linewidth=2, label='Training Accuracy')
    ax2.axhline(y=results['test_accuracy'], color='r', linestyle='--', 
               label=f'Test Accuracy: {results["test_accuracy"]:.2f}%')
    ax2.set_title('Training Accuracy Progression', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Performance Summary
    config = get_dataset_config(dataset_name)
    target_acc = config['target_accuracy']
    
    categories = ['Target', 'Achieved', 'Improvement']
    values = [target_acc, results['test_accuracy'], 
              max(0, results['test_accuracy'] - target_acc)]
    colors = ['lightcoral', 'lightgreen', 'gold']
    
    bars = ax3.bar(categories, values, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_title(f'{dataset_name} Performance Analysis', fontweight='bold')
    ax3.set_ylabel('Accuracy (%)')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 値をバーの上に表示
    for bar, value in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # Stage II Progress
    datasets = ['MNIST', 'FashionMNIST', 'EMNIST', 'CIFAR10']
    if dataset_name in datasets:
        progress = (datasets.index(dataset_name) + 1) / len(datasets) * 100
        ax4.pie([progress, 100-progress], 
               labels=[f'Completed: {progress:.0f}%', f'Remaining: {100-progress:.0f}%'],
               colors=['lightgreen', 'lightgray'],
               autopct='%1.1f%%',
               startangle=90)
        ax4.set_title('Stage II Progress Overview', fontweight='bold')
    
    plt.tight_layout()
    
    # 高解像度保存
    filename = f'nkat_stage2_{dataset_name.lower()}_results_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    return filename

def main():
    """メイン実行関数"""
    
    # RTX3080 CUDA最適化設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(0)  # RTX3080選択
        torch.backends.cudnn.benchmark = True  # 最適化
        torch.backends.cudnn.deterministic = False  # 高速化
        print(f"🚀 RTX3080 CUDA Optimization Enabled: {torch.cuda.get_device_name()}")
    
    # シード設定
    torch.manual_seed(1337)
    np.random.seed(1337)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Stage II Target Datasets
    target_datasets = ['MNIST', 'FashionMNIST', 'EMNIST', 'CIFAR10']
    all_results = {}
    
    print("🌟 NKAT Stage II Ultimate Generalization Test Starting...")
    print(f"📅 Timestamp: {timestamp}")
    print(f"🔧 Device: {device}")
    print(f"📊 Target Datasets: {', '.join(target_datasets)}")
    
    for dataset_name in target_datasets:
        print(f"\n{'='*60}")
        print(f"🎯 Processing Dataset: {dataset_name}")
        print(f"{'='*60}")
        
        config = get_dataset_config(dataset_name)
        
        # データ変換
        train_transform, test_transform = create_enhanced_transforms(dataset_name)
        
        # データセット読み込み
        try:
            train_dataset = config['dataset_func']('./data', train=True, 
                                                 transform=train_transform, download=True)
            test_dataset = config['dataset_func']('./data', train=False, 
                                                transform=test_transform, download=True)
        except Exception as e:
            print(f"❌ Failed to load {dataset_name}: {e}")
            continue
        
        # DataLoader作成
        batch_size = 128 if dataset_name != 'CIFAR10' else 64  # CIFAR10は重いので小さく
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                               shuffle=False, num_workers=4, pin_memory=True)
        
        # モデル初期化
        model = NKATTransformerPractical(
            img_size=config['size'],
            patch_size=4,
            num_classes=config['num_classes'],
            embed_dim=384 if dataset_name != 'EMNIST' else 512,  # EMNISTは大きく
            depth=6 if dataset_name != 'CIFAR10' else 8,         # CIFAR10は深く
            num_heads=8,
            mlp_ratio=4.0,
            nkat_strength=0.0,
            dropout=0.1
        ).to(device)
        
        print(f"📋 Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # トレーニング実行
        num_epochs = 30 if dataset_name != 'EMNIST' else 40  # EMNISTは長めに
        results = train_and_evaluate(model, train_loader, test_loader, 
                                   num_epochs, device, dataset_name)
        
        # NaN発生などでトレーニングが失敗した場合
        if results is None:
            print(f"❌ Training failed for {dataset_name} due to numerical instability")
            # メモリクリア
            del model
            torch.cuda.empty_cache() if device.type == 'cuda' else None
            continue
        
        # 結果保存
        all_results[dataset_name] = {
            'config': config,
            'results': results,
            'model_params': sum(p.numel() for p in model.parameters()),
            'device': str(device),
            'timestamp': timestamp
        }
        
        # 可視化
        viz_filename = create_visualization(results, dataset_name, timestamp)
        all_results[dataset_name]['visualization'] = viz_filename
        
        # 目標達成チェック
        target_acc = config['target_accuracy']
        achieved_acc = results['test_accuracy']
        
        if achieved_acc >= target_acc:
            print(f"🎉 {dataset_name}: TARGET ACHIEVED! {achieved_acc:.2f}% >= {target_acc:.2f}%")
        else:
            print(f"⚠️  {dataset_name}: TARGET MISSED. {achieved_acc:.2f}% < {target_acc:.2f}%")
        
        # メモリクリア
        del model
        torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    # 最終結果保存
    results_filename = f'nkat_stage2_ultimate_results_{timestamp}.json'
    with open(results_filename, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # サマリー表示
    print(f"\n{'='*80}")
    print("🏆 NKAT Stage II Ultimate Generalization Summary")
    print(f"{'='*80}")
    
    for dataset_name, data in all_results.items():
        results = data['results']
        config = data['config']
        status = "✅ PASSED" if results['test_accuracy'] >= config['target_accuracy'] else "❌ FAILED"
        print(f"{dataset_name:12} | Target: {config['target_accuracy']:5.1f}% | "
              f"Achieved: {results['test_accuracy']:5.2f}% | {status}")
    
    total_passed = sum(1 for dataset_name, data in all_results.items() 
                      if data['results']['test_accuracy'] >= data['config']['target_accuracy'])
    
    print(f"\n🎯 Overall Stage II Success Rate: {total_passed}/{len(all_results)} ({total_passed/len(all_results)*100:.1f}%)")
    print(f"📁 Results saved to: {results_filename}")
    print("🚀 Ready for Stage III: Robustness & Security Testing!")

if __name__ == "__main__":
    main() 