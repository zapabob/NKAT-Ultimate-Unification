#!/usr/bin/env python3
"""
🧪 NKAT 包括的アブレーション実験システム
理論⇔工学バランス最適化と TPE (Theory-Practical Equilibrium) 指標

提案された実験プラン:
A0: Full model (=99.20％) - 基線
A1: 段階的パッチ埋め込み → 単層ConvPatch
A2: NKAT strength=0 → 0.02 (微追加)
A3: Standard -> カスタムTransformerBlock
A4: label_smoothing=0, clip_grad なし
A5: CosineLR → 固定1e-4

RTX3080最適化、電源断リカバリーシステム付き
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# CUDA最適化
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    print(f"🚀 RTX3080 CUDA Optimization Enabled: {torch.cuda.get_device_name(0)}")

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Hiragino Sans']
plt.rcParams['axes.unicode_minus'] = False

class NKATPatchEmbedding(nn.Module):
    """段階的パッチ埋め込み（A1実験対象）"""
    
    def __init__(self, img_size=28, patch_size=4, channels=1, embed_dim=512, use_gradual=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.use_gradual = use_gradual
        
        if use_gradual:
            # 段階的ConvStem（99%の要因）
            self.conv_layers = nn.Sequential(
                nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.GELU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.GELU(),
                nn.Conv2d(128, embed_dim, kernel_size=patch_size, stride=patch_size)
            )
        else:
            # 単層ConvPatch（A1実験）
            self.conv_layers = nn.Conv2d(channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        self.num_patches = (img_size // patch_size) ** 2
        
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.conv_layers(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

class NKATTransformerV1(nn.Module):
    """NKAT-Transformer 実験用バリエーション"""
    
    def __init__(self, 
                 img_size=28, 
                 patch_size=4, 
                 num_classes=10,
                 embed_dim=512, 
                 depth=8, 
                 num_heads=8, 
                 mlp_ratio=4.0,
                 nkat_strength=0.0,  # A2実験対象
                 dropout=0.1,
                 use_gradual_patch=True,  # A1実験対象
                 use_custom_transformer=False,  # A3実験対象
                 use_label_smoothing=True,  # A4実験対象
                 use_cosine_lr=True):  # A5実験対象
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.nkat_strength = nkat_strength
        self.use_custom_transformer = use_custom_transformer
        
        # 実験設定保存
        self.config = {
            'use_gradual_patch': use_gradual_patch,
            'use_custom_transformer': use_custom_transformer,
            'use_label_smoothing': use_label_smoothing,
            'use_cosine_lr': use_cosine_lr,
            'nkat_strength': nkat_strength
        }
        
        # 入力チャンネル数決定
        channels = 1 if num_classes <= 47 else 3
        
        # パッチ埋め込み（A1実験）
        self.patch_embedding = NKATPatchEmbedding(
            img_size, patch_size, channels, embed_dim, use_gradual_patch
        )
        
        # 位置埋め込み
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim) * 0.02
        )
        
        # クラストークン
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # 入力正規化
        self.input_norm = nn.LayerNorm(embed_dim)
        
        # Transformerエンコーダー（A3実験）
        if use_custom_transformer:
            # カスタムTransformerBlock（旧版）
            self.transformer_layers = nn.ModuleList([
                self._create_custom_block(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ])
        else:
            # 標準TransformerEncoder（99%実現版）
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
        
        # 分類ヘッド
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
    
    def _create_custom_block(self, embed_dim, num_heads, mlp_ratio, dropout):
        """カスタムTransformerBlock（不安定版）"""
        return nn.ModuleDict({
            'attention': nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True),
            'norm1': nn.LayerNorm(embed_dim),
            'mlp': nn.Sequential(
                nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
                nn.Dropout(dropout)
            ),
            'norm2': nn.LayerNorm(embed_dim)
        })
    
    def _init_weights(self, m):
        """重み初期化"""
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
        x = self.patch_embedding(x)  # (B, num_patches, embed_dim)
        
        # クラストークン追加
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches + 1, embed_dim)
        
        # 位置埋め込み
        if x.shape[1] == self.pos_embedding.shape[1]:
            x = x + self.pos_embedding
        else:
            # サイズ不一致時の安全な処理
            pos_emb = self.pos_embedding[:, :x.shape[1], :]
            x = x + pos_emb
        
        x = self.input_norm(x)
        
        # NKAT適応（A2実験）
        if self.nkat_strength > 0:
            # 安全な適応的調整
            mean_activation = x.mean(dim=-1, keepdim=True)
            nkat_factor = 1.0 + self.nkat_strength * 0.01 * torch.tanh(mean_activation)
            x = x * nkat_factor
        
        # Transformer処理
        if self.use_custom_transformer:
            # カスタムTransformer
            for layer in self.transformer_layers:
                # Self-attention
                residual = x
                x = layer['norm1'](x)
                attn_out, _ = layer['attention'](x, x, x)
                x = residual + attn_out
                
                # MLP
                residual = x
                x = layer['norm2'](x)
                x = residual + layer['mlp'](x)
        else:
            # 標準Transformer
            x = self.transformer(x)
        
        # 分類
        cls_output = x[:, 0]  # (B, embed_dim)
        logits = self.classifier(cls_output)
        
        return logits

def calculate_tpe_score(val_accuracy, theory_params, total_params):
    """
    TPE (Theory-Practical Equilibrium) スコア計算
    
    TPE = ValAcc / log10(1 + λ_theory)
    λ_theory = 理論的パラメータ数（NKAT専用モジュール重み数）
    """
    lambda_theory = theory_params
    if lambda_theory < 1:
        lambda_theory = 1  # log(1) = 0を避ける
    
    tpe = val_accuracy / np.log10(1 + lambda_theory)
    
    return {
        'tpe_score': tpe,
        'val_accuracy': val_accuracy,
        'theory_params': theory_params,
        'total_params': total_params,
        'theory_ratio': theory_params / total_params if total_params > 0 else 0
    }

def count_theory_parameters(model):
    """理論的パラメータ数（NKAT関連）をカウント"""
    theory_params = 0
    
    # NKAT関連のパラメータを特定
    for name, param in model.named_parameters():
        if any(keyword in name.lower() for keyword in ['nkat', 'gauge', 'quantum', 'theory']):
            theory_params += param.numel()
    
    # NKATパッチ埋め込みの段階的部分
    if hasattr(model, 'patch_embedding') and model.config.get('use_gradual_patch', True):
        # 段階的ConvStemの追加パラメータ
        for param in model.patch_embedding.parameters():
            theory_params += param.numel() * 0.3  # 理論的寄与分として30%
    
    return int(theory_params)

def run_single_experiment(exp_config, dataset_name='MNIST', num_epochs=20, device='cuda'):
    """単一実験実行"""
    
    print(f"\n🧪 実験 {exp_config['name']}: {exp_config['description']}")
    print(f"📊 設定: {exp_config['params']}")
    
    # データ準備
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
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
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4)
    
    # モデル作成
    model = NKATTransformerV1(**exp_config['params']).to(device)
    
    # パラメータカウント
    total_params = sum(p.numel() for p in model.parameters())
    theory_params = count_theory_parameters(model)
    
    print(f"📊 Total Parameters: {total_params:,}")
    print(f"📊 Theory Parameters: {theory_params:,}")
    
    # 損失関数（A4実験）
    if exp_config['params'].get('use_label_smoothing', True):
        criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # オプティマイザー
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=2e-4)
    
    # スケジューラー（A5実験）
    if exp_config['params'].get('use_cosine_lr', True):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    else:
        scheduler = None
    
    # 勾配クリッピング設定（A4実験）
    use_grad_clip = exp_config['params'].get('use_label_smoothing', True)
    
    # Mixed precision
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    # トレーニング
    model.train()
    train_losses = []
    train_accuracies = []
    grad_norms = []
    
    for epoch in tqdm(range(num_epochs), desc=f"Training {exp_config['name']}"):
        epoch_loss = 0.0
        correct = 0
        total = 0
        epoch_grad_norms = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    output = model(data)
                    loss = criterion(output, target)
                
                # 数値安定性チェック
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"❌ Numerical instability detected at epoch {epoch+1}, batch {batch_idx}")
                    continue
                
                scaler.scale(loss).backward()
                
                # 勾配ノルム記録
                if batch_idx % 50 == 0:
                    grad_norm = 0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            if torch.isfinite(param_norm):
                                grad_norm += param_norm.item() ** 2
                    epoch_grad_norms.append(grad_norm ** 0.5)
                
                # 勾配クリッピング（A4実験）
                if use_grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = criterion(output, target)
                
                # 数値安定性チェック
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"❌ Numerical instability detected at epoch {epoch+1}, batch {batch_idx}")
                    continue
                
                loss.backward()
                
                if use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
            
            epoch_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        if scheduler is not None:
            scheduler.step()
        
        # エポック統計
        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100. * correct / total
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        
        if epoch_grad_norms:
            grad_norms.append(np.mean(epoch_grad_norms))
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%")
    
    # テスト評価
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    test_accuracy = 100. * correct / total
    
    # TPEスコア計算
    tpe_metrics = calculate_tpe_score(test_accuracy / 100.0, theory_params, total_params)
    
    # 結果まとめ
    results = {
        'experiment': exp_config['name'],
        'description': exp_config['description'],
        'test_accuracy': test_accuracy,
        'train_accuracy_final': train_accuracies[-1] if train_accuracies else 0,
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'grad_norms': grad_norms,
        'total_params': total_params,
        'theory_params': theory_params,
        'tpe_metrics': tpe_metrics,
        'config': exp_config['params']
    }
    
    return results

def create_experiment_configs():
    """実験設定A0-A5を作成"""
    
    base_config = {
        'img_size': 28,
        'patch_size': 4,
        'num_classes': 10,
        'embed_dim': 512,
        'depth': 8,
        'num_heads': 8,
        'mlp_ratio': 4.0,
        'dropout': 0.1
    }
    
    experiments = [
        {
            'name': 'A0',
            'description': 'Full model (基線) - 99.20%目標',
            'params': {
                **base_config,
                'nkat_strength': 0.0,
                'use_gradual_patch': True,
                'use_custom_transformer': False,
                'use_label_smoothing': True,
                'use_cosine_lr': True
            }
        },
        {
            'name': 'A1',
            'description': '段階的パッチ埋め込み → 単層ConvPatch',
            'params': {
                **base_config,
                'nkat_strength': 0.0,
                'use_gradual_patch': False,  # 変更点
                'use_custom_transformer': False,
                'use_label_smoothing': True,
                'use_cosine_lr': True
            }
        },
        {
            'name': 'A2',
            'description': 'NKAT strength=0 → 0.02 (微追加)',
            'params': {
                **base_config,
                'nkat_strength': 0.02,  # 変更点
                'use_gradual_patch': True,
                'use_custom_transformer': False,
                'use_label_smoothing': True,
                'use_cosine_lr': True
            }
        },
        {
            'name': 'A3',
            'description': 'Standard → カスタムTransformerBlock',
            'params': {
                **base_config,
                'nkat_strength': 0.0,
                'use_gradual_patch': True,
                'use_custom_transformer': True,  # 変更点
                'use_label_smoothing': True,
                'use_cosine_lr': True
            }
        },
        {
            'name': 'A4',
            'description': 'label_smoothing=0, clip_grad なし',
            'params': {
                **base_config,
                'nkat_strength': 0.0,
                'use_gradual_patch': True,
                'use_custom_transformer': False,
                'use_label_smoothing': False,  # 変更点
                'use_cosine_lr': True
            }
        },
        {
            'name': 'A5',
            'description': 'CosineLR → 固定1e-4',
            'params': {
                **base_config,
                'nkat_strength': 0.0,
                'use_gradual_patch': True,
                'use_custom_transformer': False,
                'use_label_smoothing': True,
                'use_cosine_lr': False  # 変更点
            }
        }
    ]
    
    return experiments

def create_comprehensive_visualization(all_results, timestamp):
    """包括的可視化とTPE分析"""
    
    # 結果データフレーム作成
    df = pd.DataFrame([
        {
            'Experiment': r['experiment'],
            'Description': r['description'],
            'Test_Accuracy': r['test_accuracy'],
            'TPE_Score': r['tpe_metrics']['tpe_score'],
            'Theory_Params': r['theory_params'],
            'Total_Params': r['total_params'],
            'Theory_Ratio': r['tpe_metrics']['theory_ratio']
        }
        for r in all_results
    ])
    
    # 図作成
    fig = plt.figure(figsize=(20, 12))
    
    # 1. アブレーション結果比較
    plt.subplot(2, 3, 1)
    bars = plt.bar(df['Experiment'], df['Test_Accuracy'], 
                   color=plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(df))))
    plt.ylabel('Test Accuracy (%)')
    plt.title('🧪 Ablation Study Results')
    plt.xticks(rotation=45)
    
    # 基線からの差分表示
    baseline_acc = df[df['Experiment'] == 'A0']['Test_Accuracy'].iloc[0]
    for i, (bar, acc) in enumerate(zip(bars, df['Test_Accuracy'])):
        diff = acc - baseline_acc
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{diff:+.2f}', ha='center', va='bottom', fontsize=10)
    
    # 2. TPEスコア比較
    plt.subplot(2, 3, 2)
    plt.bar(df['Experiment'], df['TPE_Score'], color='lightgreen', alpha=0.7)
    plt.ylabel('TPE Score')
    plt.title('🎯 Theory-Practical Equilibrium')
    plt.xticks(rotation=45)
    
    # 3. パラメータ効率分析
    plt.subplot(2, 3, 3)
    plt.scatter(df['Theory_Ratio'] * 100, df['Test_Accuracy'], 
                s=100, c=df['TPE_Score'], cmap='viridis', alpha=0.8)
    plt.xlabel('Theory Parameter Ratio (%)')
    plt.ylabel('Test Accuracy (%)')
    plt.title('📊 Parameter Efficiency')
    plt.colorbar(label='TPE Score')
    
    # 4. 実験詳細テーブル
    plt.subplot(2, 3, 4)
    plt.axis('tight')
    plt.axis('off')
    table_data = df[['Experiment', 'Test_Accuracy', 'TPE_Score']].round(3)
    table = plt.table(cellText=table_data.values,
                     colLabels=table_data.columns,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.title('📋 Detailed Results')
    
    # 5. 学習曲線比較
    plt.subplot(2, 3, 5)
    for result in all_results:
        if 'train_accuracies' in result and result['train_accuracies']:
            plt.plot(result['train_accuracies'], 
                    label=f"{result['experiment']}: {result['test_accuracy']:.2f}%",
                    alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy (%)')
    plt.title('📈 Training Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 6. TPE vs Accuracy散布図
    plt.subplot(2, 3, 6)
    colors = ['red' if exp == 'A0' else 'blue' for exp in df['Experiment']]
    plt.scatter(df['TPE_Score'], df['Test_Accuracy'], c=colors, s=100, alpha=0.7)
    for i, exp in enumerate(df['Experiment']):
        plt.annotate(exp, (df['TPE_Score'].iloc[i], df['Test_Accuracy'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points')
    plt.xlabel('TPE Score')
    plt.ylabel('Test Accuracy (%)')
    plt.title('🎯 TPE vs Performance')
    
    plt.tight_layout()
    
    # 保存
    result_file = f'nkat_comprehensive_ablation_analysis_{timestamp}.png'
    plt.savefig(result_file, dpi=300, bbox_inches='tight')
    print(f"📊 包括的分析結果を保存: {result_file}")
    
    return result_file

def generate_comprehensive_report(all_results, timestamp):
    """包括的実験レポート生成"""
    
    report = f"""
# 🧪 NKAT包括的アブレーション実験レポート
**実行日時**: {timestamp}
**RTX3080最適化**: 有効

## 📊 実験概要
提案された理論⇔工学バランス検証実験A0-A5を実施。
TPE (Theory-Practical Equilibrium) 指標による定量的評価を導入。

## 🎯 主要結果

### 精度比較
"""
    
    # 結果テーブル
    for result in all_results:
        tpe = result['tpe_metrics']
        report += f"""
**{result['experiment']}**: {result['description']}
- テスト精度: {result['test_accuracy']:.2f}%
- TPEスコア: {tpe['tpe_score']:.4f}
- 理論パラメータ: {tpe['theory_params']:,}
- 理論比率: {tpe['theory_ratio']*100:.2f}%
"""
    
    # 基線との比較
    baseline = next(r for r in all_results if r['experiment'] == 'A0')
    baseline_acc = baseline['test_accuracy']
    
    report += f"""
## 🔍 詳細分析

### 基線(A0)からの変化量
"""
    
    for result in all_results:
        if result['experiment'] != 'A0':
            diff = result['test_accuracy'] - baseline_acc
            report += f"- **{result['experiment']}**: {diff:+.2f} pt\n"
    
    # TPE分析
    tpe_scores = [r['tpe_metrics']['tpe_score'] for r in all_results]
    best_tpe_idx = np.argmax(tpe_scores)
    best_tpe_exp = all_results[best_tpe_idx]
    
    report += f"""
### 🏆 TPE最優秀実験
**{best_tpe_exp['experiment']}** - {best_tpe_exp['description']}
- TPEスコア: {best_tpe_exp['tpe_metrics']['tpe_score']:.4f}
- テスト精度: {best_tpe_exp['test_accuracy']:.2f}%

これは理論的複雑さと実用性能の最適バランスを示している。

## 🚀 次のアクション
1. 最優秀TPE実験の詳細メカニズム解析
2. Attention Entropy分析の実施
3. NKAT θ微調整の閾値探索実験
4. 論文化に向けたTable 1データ確定
"""
    
    # レポート保存
    report_file = f'NKAT_Comprehensive_Ablation_Report_{timestamp}.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📋 包括的レポートを保存: {report_file}")
    return report_file

def main():
    parser = argparse.ArgumentParser(description='NKAT包括的アブレーション実験')
    parser.add_argument('--experiments', nargs='+', default=['A0', 'A1', 'A2', 'A3', 'A4', 'A5'],
                       help='実行する実験 (A0-A5)')
    parser.add_argument('--epochs', type=int, default=20, help='トレーニングエポック数')
    parser.add_argument('--dataset', default='MNIST', help='データセット')
    parser.add_argument('--device', default='cuda', help='デバイス')
    parser.add_argument('--recovery', action='store_true', help='電源断リカバリーモード')
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("🧪 NKAT 包括的アブレーション実験システム")
    print(f"📅 実行開始: {timestamp}")
    print(f"🔧 デバイス: {args.device}")
    print(f"📊 実行実験: {args.experiments}")
    
    # GPU確認
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"🚀 CUDA デバイス: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU メモリ: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 実験設定作成
    all_experiments = create_experiment_configs()
    selected_experiments = [exp for exp in all_experiments if exp['name'] in args.experiments]
    
    # 結果保存用
    all_results = []
    checkpoint_file = f'ablation_checkpoint_{timestamp}.json'
    
    # 電源断リカバリー
    if args.recovery and os.path.exists(checkpoint_file):
        print("🔄 電源断リカバリーモード: チェックポイントを読み込み")
        with open(checkpoint_file, 'r') as f:
            all_results = json.load(f)
        completed_experiments = [r['experiment'] for r in all_results]
        selected_experiments = [exp for exp in selected_experiments 
                              if exp['name'] not in completed_experiments]
    
    # 実験実行
    for exp_config in selected_experiments:
        try:
            print(f"\n{'='*60}")
            print(f"🔬 実験 {exp_config['name']} 開始")
            print(f"{'='*60}")
            
            result = run_single_experiment(
                exp_config, 
                dataset_name=args.dataset,
                num_epochs=args.epochs,
                device=device
            )
            
            all_results.append(result)
            
            # チェックポイント保存
            with open(checkpoint_file, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            
            print(f"✅ 実験 {exp_config['name']} 完了")
            print(f"📊 テスト精度: {result['test_accuracy']:.2f}%")
            print(f"🎯 TPEスコア: {result['tpe_metrics']['tpe_score']:.4f}")
            
        except Exception as e:
            print(f"❌ 実験 {exp_config['name']} でエラー: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if all_results:
        print(f"\n{'='*60}")
        print("📊 最終分析と可視化")
        print(f"{'='*60}")
        
        # 包括的可視化
        viz_file = create_comprehensive_visualization(all_results, timestamp)
        
        # 包括的レポート
        report_file = generate_comprehensive_report(all_results, timestamp)
        
        # 結果JSON保存
        final_results_file = f'nkat_ablation_final_results_{timestamp}.json'
        with open(final_results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\n🎉 全実験完了!")
        print(f"📊 可視化: {viz_file}")
        print(f"📋 レポート: {report_file}")
        print(f"💾 結果データ: {final_results_file}")
        
        # 最優秀TPE実験発表
        tpe_scores = [r['tpe_metrics']['tpe_score'] for r in all_results]
        best_idx = np.argmax(tpe_scores)
        best_result = all_results[best_idx]
        
        print(f"\n🏆 TPE最優秀実験: {best_result['experiment']}")
        print(f"📈 精度: {best_result['test_accuracy']:.2f}%")
        print(f"🎯 TPEスコア: {best_result['tpe_metrics']['tpe_score']:.4f}")
        print(f"💡 {best_result['description']}")
        
    else:
        print("❌ 実行された実験がありません")

if __name__ == "__main__":
    main() 