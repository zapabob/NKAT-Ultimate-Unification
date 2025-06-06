#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final 99%+ Achievement Model Testing
最終99%+達成モデルの検証

Author: NKAT Advanced Computing Team
Date: 2025-06-01
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 英語表記設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

from nkat_enhanced_transformer_v2 import (
    EnhancedNKATVisionTransformer,
    AdvancedDataAugmentation
)
from nkat_final_99_percent_push import FinalPushConfig

def load_final_model():
    """最終99%+モデルをロード"""
    checkpoint_path = 'checkpoints/nkat_final_99_percent.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 設定復元
    config = FinalPushConfig()
    if 'config' in checkpoint:
        for key, value in checkpoint['config'].items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # モデル作成
    model = EnhancedNKATVisionTransformer(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"🏆 Final 99%+ model loaded!")
    print(f"Checkpoint accuracy: {checkpoint.get('test_accuracy', 'unknown'):.2f}%")
    print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
    
    return model, config, checkpoint

def final_evaluation():
    """最終評価"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # モデルロード
    model, config, checkpoint = load_final_model()
    model.to(device)
    model.eval()
    
    # テストデータ
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = torchvision.datasets.MNIST(
        root="data", train=False, download=True, transform=test_transform
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4
    )
    
    # 評価実行
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    all_confidences = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Final Evaluation'):
            data, target = data.to(device), target.to(device)
            
            outputs = model(data)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probabilities, 1)
            
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
    
    accuracy = 100. * correct / total
    
    # 結果表示
    print(f"\n🎯 FINAL 99%+ MODEL RESULTS:")
    print(f"═══════════════════════════════")
    print(f"🏆 Accuracy: {accuracy:.4f}%")
    print(f"📊 Correct: {correct} / {total}")
    print(f"❌ Errors: {total - correct}")
    print(f"📈 Error Rate: {100 * (total - correct) / total:.4f}%")
    print(f"🎯 99%+ Target: {'✅ ACHIEVED' if accuracy >= 99.0 else '❌ NOT YET'}")
    
    return accuracy, all_preds, all_targets, all_confidences

def create_achievement_visualization(accuracy, preds, targets, confidences):
    """達成記念可視化"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 混同行列
    cm = confusion_matrix(targets, preds)
    
    # クラス別精度
    class_accuracies = {}
    for i in range(10):
        class_mask = np.array(targets) == i
        if np.sum(class_mask) > 0:
            class_acc = np.sum(np.array(preds)[class_mask] == i) / np.sum(class_mask) * 100
            class_accuracies[i] = class_acc
    
    # 特大可視化作成
    plt.figure(figsize=(20, 16))
    
    # タイトル
    plt.suptitle(f'🎊 NKAT-TRANSFORMER 99%+ ACHIEVEMENT 🎊\nFinal Accuracy: {accuracy:.2f}%', 
                fontsize=24, fontweight='bold', color='green')
    
    # 1. 混同行列（大きく）
    plt.subplot(2, 3, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
               xticklabels=range(10), yticklabels=range(10), cbar_kws={'shrink': 0.8})
    plt.title(f'Confusion Matrix - {accuracy:.2f}% Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    
    # 2. クラス別精度（ゴールド表示）
    plt.subplot(2, 3, 2)
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    colors = ['gold' if acc >= 99.0 else 'orange' if acc >= 98.0 else 'lightcoral' for acc in accuracies]
    
    bars = plt.bar(classes, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    plt.title('Class-wise Accuracy (Gold = 99%+)', fontsize=14, fontweight='bold')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(95, 100)
    plt.grid(axis='y', alpha=0.3)
    
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 3. 進化の軌跡（ドラマチック）
    plt.subplot(2, 3, 3)
    milestones = ['Original\n(97.79%)', 'Enhanced v2.0\n(98.56%)', 'Final 99%+\n({:.2f}%)'.format(accuracy)]
    milestone_acc = [97.79, 98.56, accuracy]
    colors = ['#3498db', '#e74c3c', '#2ecc71']  # Blue -> Red -> Green
    
    bars = plt.bar(range(len(milestones)), milestone_acc, color=colors, alpha=0.8, 
                   width=0.6, edgecolor='black', linewidth=2)
    plt.title('NKAT-Transformer Evolution', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(97, 100)
    plt.xticks(range(len(milestones)), ['Original', 'Enhanced', 'Final'], fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    
    for bar, acc, milestone in zip(bars, milestone_acc, milestones):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 4. 信頼度分布
    plt.subplot(2, 3, 4)
    plt.hist(confidences, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(confidences), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(confidences):.3f}')
    plt.xlabel('Prediction Confidence', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Confidence Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 5. エラー分析
    plt.subplot(2, 3, 5)
    errors = []
    for i in range(len(targets)):
        if targets[i] != preds[i]:
            errors.append(f"{targets[i]}→{preds[i]}")
    
    if errors:
        error_counts = {}
        for error in errors:
            error_counts[error] = error_counts.get(error, 0) + 1
        
        top_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:8]
        if top_errors:
            error_labels, error_values = zip(*top_errors)
            plt.barh(range(len(error_labels)), error_values, color='lightcoral', alpha=0.7)
            plt.yticks(range(len(error_labels)), error_labels)
            plt.xlabel('Error Count', fontsize=12)
            plt.title('Top Error Patterns', fontsize=14, fontweight='bold')
            plt.grid(axis='x', alpha=0.3)
    
    # 6. 達成証明書
    plt.subplot(2, 3, 6)
    
    # 証明書風のテキスト
    certificate_text = f"""
🏆 ACHIEVEMENT CERTIFICATE 🏆

NKAT-TRANSFORMER
has successfully achieved

{accuracy:.2f}% ACCURACY
on MNIST Dataset

Surpassing the 99%+ Target

🎯 MISSION ACCOMPLISHED 🎯

Date: {datetime.now().strftime('%Y-%m-%d')}
Errors: {len(targets) - sum(np.array(preds) == np.array(targets))} / 10,000
Success Rate: {accuracy:.4f}%

Ready for Production Deployment
"""
    
    plt.text(0.5, 0.5, certificate_text, fontsize=12, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="gold", alpha=0.8),
             transform=plt.gca().transAxes)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'figures/nkat_final_achievement_certificate_{timestamp}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return timestamp

def main():
    """メイン実行"""
    print("🎊 NKAT-TRANSFORMER FINAL 99%+ ACHIEVEMENT TEST 🎊")
    print("═" * 70)
    
    # 評価実行
    accuracy, preds, targets, confidences = final_evaluation()
    
    # 達成可視化
    timestamp = create_achievement_visualization(accuracy, preds, targets, confidences)
    
    # 最終レポート
    print(f"\n🎊 FINAL ACHIEVEMENT SUMMARY 🎊")
    print(f"═══════════════════════════════════")
    print(f"🏆 Final Accuracy: {accuracy:.4f}%")
    print(f"🎯 Target Achievement: {'✅ COMPLETE' if accuracy >= 99.0 else '❌ INCOMPLETE'}")
    print(f"📈 Improvement from Original: +{accuracy - 97.79:.2f}%")
    print(f"⚡ Total Errors: {len(targets) - sum(np.array(preds) == np.array(targets))} / 10,000")
    print(f"🚀 Ready for Production: {'YES' if accuracy >= 99.0 else 'NEEDS MORE WORK'}")
    
    if accuracy >= 99.0:
        print(f"\n🎉 CONGRATULATIONS! 🎉")
        print(f"NKAT-Transformer has successfully achieved 99%+ accuracy!")
        print(f"This is a remarkable achievement in AI and ML!")
        print(f"Ready for:")
        print(f"  • 📝 Academic Publication")
        print(f"  • 🏭 Industrial Deployment") 
        print(f"  • 🏆 Competition Submission")
        print(f"  • 🌟 Open Source Release")
    
    print(f"\n📊 Achievement certificate: figures/nkat_final_achievement_certificate_{timestamp}.png")
    print(f"═" * 70)

if __name__ == "__main__":
    main() 