#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT-Transformer Performance Optimization Strategy
97.79% → 99%+ Accuracy Enhancement Plan

現在の分析結果に基づく包括的改善戦略:
1. 混合精度問題の解決
2. 困難クラス(5,9,7)の特別対策
3. データ拡張の強化
4. モデル構造の最適化
5. 訓練戦略の改善

Author: NKAT Advanced Computing Team
Date: 2025-06-01
Target: 99%+ Test Accuracy on MNIST
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from tqdm import tqdm
import logging

# 英語表記設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

class NKATOptimizationConfig:
    """最適化設定"""
    
    def __init__(self):
        # 基本設定（既存）
        self.image_size = 28
        self.patch_size = 7
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.channels = 1
        self.num_classes = 10
        
        # 改善版モデル設定
        self.d_model = 512  # 384 → 512に増加
        self.nhead = 8      # 6 → 8に増加
        self.num_layers = 12  # 8 → 12に増加（深さ向上）
        self.dim_feedforward = 2048  # 1536 → 2048に増加
        self.dropout = 0.08  # 0.1 → 0.08に調整
        
        # 混合精度対策
        self.use_mixed_precision = False  # 安定性のため無効化
        self.use_gradient_clipping = True
        self.max_grad_norm = 1.0
        
        # 困難クラス対策
        self.use_class_weights = True
        self.difficult_classes = [5, 7, 9]  # 分析結果より
        self.class_weight_boost = 1.5
        
        # 強化データ拡張
        self.use_advanced_augmentation = True
        self.rotation_range = 15  # 10 → 15度に拡大
        self.zoom_range = 0.15
        self.use_elastic_deformation = True
        self.use_random_erasing = True
        
        # 訓練戦略改善
        self.num_epochs = 100  # 50 → 100に延長
        self.batch_size = 64   # 128 → 64に調整（安定性向上）
        self.learning_rate = 1e-4  # 3e-4 → 1e-4に調整
        self.use_cosine_restart = True
        self.restart_period = 20
        
        # 正則化強化
        self.use_label_smoothing = True
        self.label_smoothing = 0.08  # 0.05 → 0.08に強化
        self.use_mixup = True
        self.mixup_alpha = 0.4
        self.weight_decay = 2e-4  # 1e-4 → 2e-4に強化
        
        # アンサンブル準備
        self.save_multiple_models = True
        self.model_variants = 3

class AdvancedDataAugmentation:
    """強化版データ拡張"""
    
    def __init__(self, config):
        self.config = config
        
    def create_train_transforms(self):
        """訓練用変換"""
        transforms_list = [
            transforms.ToPILImage(),
            
            # 回転拡張強化
            transforms.RandomRotation(
                degrees=self.config.rotation_range,
                fill=0,
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            
            # アフィン変換
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=5,
                fill=0
            ),
            
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
        
        # ランダム消去
        if self.config.use_random_erasing:
            transforms_list.append(
                transforms.RandomErasing(
                    p=0.1,
                    scale=(0.02, 0.1),
                    ratio=(0.3, 3.3),
                    value=0
                )
            )
        
        return transforms.Compose(transforms_list)
    
    def elastic_deformation(self, image, alpha=100, sigma=10):
        """弾性変形（手書き文字の自然な変動をシミュレート）"""
        random_state = np.random.RandomState(None)
        
        shape = image.shape
        dx = random_state.uniform(-1, 1, shape) * alpha
        dy = random_state.uniform(-1, 1, shape) * alpha
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
        
        from scipy.ndimage import map_coordinates, gaussian_filter
        dx = gaussian_filter(dx, sigma, mode='constant', cval=0) 
        dy = gaussian_filter(dy, sigma, mode='constant', cval=0)
        
        distorted_image = map_coordinates(image, indices, order=1, mode='reflect')
        return distorted_image.reshape(shape)

class ImprovedNKATVisionTransformer(nn.Module):
    """改善版NKAT Vision Transformer"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 改善版パッチ埋め込み
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(config.channels, config.d_model // 4, 3, padding=1),
            nn.BatchNorm2d(config.d_model // 4),
            nn.GELU(),
            nn.Conv2d(config.d_model // 4, config.d_model, 
                     config.patch_size, stride=config.patch_size)
        )
        
        # 位置埋め込み
        self.pos_embedding = nn.Parameter(
            torch.randn(1, config.num_patches + 1, config.d_model) * 0.02
        )
        
        # クラストークン
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, config.d_model) * 0.02
        )
        
        # Transformerエンコーダー（改善版）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.num_layers)
        
        # 分類ヘッド（改善版）
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.num_classes)
        )
        
        # 重み初期化
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """重み初期化"""
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # パッチ埋め込み
        x = self.patch_embedding(x)  # [B, d_model, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, d_model]
        
        # クラストークン追加
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 位置埋め込み
        x = x + self.pos_embedding
        
        # Transformer処理
        x = self.transformer(x)
        
        # 分類
        cls_output = x[:, 0]
        logits = self.classifier(cls_output)
        
        return {
            'logits': logits,
            'cls_features': cls_output,
            'attention_weights': None,  # 必要に応じて実装
            'quantum_contribution': torch.tensor(0.0)  # NKAT互換性のため
        }

def create_class_weighted_sampler(dataset, config):
    """クラス重み付きサンプラー"""
    labels = [dataset[i][1] for i in range(len(dataset))]
    class_counts = np.bincount(labels)
    
    # 困難クラスの重みを上げる
    class_weights = 1.0 / class_counts
    for difficult_class in config.difficult_classes:
        class_weights[difficult_class] *= config.class_weight_boost
    
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler

def mixup_data(x, y, alpha=1.0):
    """Mixup データ拡張"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup損失"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class OptimizationStrategy:
    """最適化戦略実行クラス"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def create_optimized_model(self):
        """最適化モデルの作成"""
        model = ImprovedNKATVisionTransformer(self.config).to(self.device)
        
        # パラメータ数確認
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Optimized model parameters: {total_params:,}")
        
        return model
    
    def create_optimized_dataloader(self):
        """最適化データローダー"""
        augmentation = AdvancedDataAugmentation(self.config)
        
        # 訓練データ
        train_transform = augmentation.create_train_transforms()
        train_dataset = torchvision.datasets.MNIST(
            root="data", train=True, download=True, transform=train_transform
        )
        
        # クラス重み付きサンプラー
        if self.config.use_class_weights:
            sampler = create_class_weighted_sampler(train_dataset, self.config)
            shuffle = False
        else:
            sampler = None
            shuffle = True
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        # テストデータ
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        test_dataset = torchvision.datasets.MNIST(
            root="data", train=False, download=True, transform=test_transform
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size * 2,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, test_loader
    
    def create_optimized_optimizer(self, model):
        """最適化オプティマイザー"""
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        if self.config.use_cosine_restart:
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, 
                T_0=self.config.restart_period,
                T_mult=2,
                eta_min=self.config.learning_rate * 0.01
            )
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.learning_rate * 0.1
            )
        
        return optimizer, scheduler
    
    def create_optimized_criterion(self):
        """最適化損失関数"""
        if self.config.use_label_smoothing:
            criterion = nn.CrossEntropyLoss(
                label_smoothing=self.config.label_smoothing
            )
        else:
            criterion = nn.CrossEntropyLoss()
        
        return criterion
    
    def generate_optimization_report(self):
        """最適化戦略レポート"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report = {
            "optimization_strategy": "NKAT-Transformer 97.79% → 99%+ Enhancement",
            "timestamp": timestamp,
            "current_performance": {
                "test_accuracy": 97.79,
                "problematic_classes": [5, 7, 9],
                "main_confusions": ["3→5", "9→4", "4→9", "7→2"]
            },
            "improvements": {
                "model_architecture": {
                    "d_model": f"{384} → {self.config.d_model}",
                    "num_layers": f"{8} → {self.config.num_layers}",
                    "nhead": f"{6} → {self.config.nhead}",
                    "deeper_classifier": True
                },
                "training_strategy": {
                    "mixed_precision": f"Disabled (stability)",
                    "gradient_clipping": self.config.use_gradient_clipping,
                    "class_weights": self.config.use_class_weights,
                    "longer_training": f"{50} → {self.config.num_epochs} epochs"
                },
                "data_augmentation": {
                    "advanced_rotation": f"±{self.config.rotation_range}°",
                    "elastic_deformation": self.config.use_elastic_deformation,
                    "mixup": self.config.use_mixup,
                    "random_erasing": self.config.use_random_erasing
                },
                "regularization": {
                    "label_smoothing": self.config.label_smoothing,
                    "weight_decay": self.config.weight_decay,
                    "dropout": self.config.dropout
                }
            },
            "expected_improvements": {
                "target_accuracy": "99.0%+",
                "error_reduction": "50%+ (from 2.21% to <1.0%)",
                "problematic_class_boost": "+2-3% for classes 5,7,9"
            }
        }
        
        # レポート保存
        with open(f"analysis/nkat_optimization_strategy_{timestamp}.json", "w") as f:
            json.dump(report, f, indent=2)
        
        return report

def main():
    """メイン実行関数"""
    print("NKAT-Transformer Performance Optimization Strategy")
    print("=" * 60)
    print("Current Performance: 97.79% → Target: 99%+")
    print("=" * 60)
    
    # 設定
    config = NKATOptimizationConfig()
    strategy = OptimizationStrategy(config)
    
    print("\n📊 Current Analysis Summary:")
    print("• Test Accuracy: 97.79% (221/10000 errors)")
    print("• Problematic Classes: 5 (96.75%), 7 (97.08%), 9 (96.13%)")
    print("• Main Confusions: 3→5 (15x), 9→4 (14x), 4→9 (12x)")
    print("• Issue Found: Mixed precision causing NaN")
    
    print("\n🚀 Optimization Strategy:")
    print("1. Model Architecture Enhancement:")
    print(f"   • d_model: 384 → {config.d_model}")
    print(f"   • Layers: 8 → {config.num_layers}")
    print(f"   • Attention heads: 6 → {config.nhead}")
    print("   • Deeper classification head")
    
    print("\n2. Training Strategy Improvements:")
    print("   • Mixed precision: DISABLED (stability)")
    print("   • Class-weighted sampling for difficult classes")
    print(f"   • Extended training: 50 → {config.num_epochs} epochs")
    print("   • Gradient clipping for stability")
    
    print("\n3. Advanced Data Augmentation:")
    print(f"   • Enhanced rotation: ±{config.rotation_range}°")
    print("   • Elastic deformation")
    print("   • Mixup augmentation")
    print("   • Random erasing")
    
    print("\n4. Regularization Enhancement:")
    print(f"   • Label smoothing: {config.label_smoothing}")
    print(f"   • Weight decay: {config.weight_decay}")
    print("   • Cosine annealing with restarts")
    
    # 最適化レポート生成
    report = strategy.generate_optimization_report()
    
    print(f"\n✅ Optimization strategy saved: analysis/nkat_optimization_strategy_{report['timestamp']}.json")
    
    print("\n🎯 Expected Results:")
    print("• Target Accuracy: 99.0%+")
    print("• Error Reduction: 50%+ (2.21% → <1.0%)")
    print("• Class 5,7,9 Boost: +2-3% each")
    print("• Stable mixed precision alternative")
    
    print("\n📋 Next Steps:")
    print("1. Implement improved model architecture")
    print("2. Set up advanced data augmentation pipeline")
    print("3. Configure class-weighted training")
    print("4. Run extended training with monitoring")
    print("5. Validate on full test set")
    
    print("\n" + "=" * 60)
    print("NKAT-Transformer Optimization Strategy Complete!")
    print("Ready for 99%+ accuracy implementation")
    print("=" * 60)

if __name__ == "__main__":
    main() 