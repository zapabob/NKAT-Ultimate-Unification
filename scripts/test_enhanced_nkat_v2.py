#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced NKAT-Transformer v2.0 Model Testing
保存されたモデルの性能評価と詳細分析

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
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# 英語表記設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

# Enhanced NKAT-Transformer v2.0のクラス定義をインポート
from nkat_enhanced_transformer_v2 import (
    EnhancedNKATConfig, 
    EnhancedNKATVisionTransformer,
    AdvancedDataAugmentation
)

def load_model(checkpoint_path):
    """モデルとチェックポイントをロード"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 設定の復元
    config = EnhancedNKATConfig()
    if 'config' in checkpoint:
        for key, value in checkpoint['config'].items():
            setattr(config, key, value)
    
    # モデルの作成と重みのロード
    model = EnhancedNKATVisionTransformer(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Checkpoint accuracy: {checkpoint.get('test_accuracy', 'unknown'):.2f}%")
    
    return model, config, checkpoint

def evaluate_model(model, config, device):
    """モデルの詳細評価"""
    model.to(device)
    model.eval()
    
    # テストデータローダー
    augmentation = AdvancedDataAugmentation(config)
    test_transform = augmentation.create_test_transforms()
    
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
        for data, target in tqdm(test_loader, desc='Evaluating'):
            data, target = data.to(device), target.to(device)
            
            outputs = model(data)
            
            # 予測と信頼度
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probabilities, 1)
            
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
    
    accuracy = 100. * correct / total
    print(f"\n🎯 Test Results:")
    print(f"• Accuracy: {accuracy:.4f}%")
    print(f"• Errors: {total - correct} / {total}")
    print(f"• Error Rate: {100 * (total - correct) / total:.4f}%")
    
    return accuracy, all_preds, all_targets, all_confidences

def detailed_analysis(preds, targets, confidences, config):
    """詳細分析と可視化"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 混同行列
    cm = confusion_matrix(targets, preds)
    
    # クラス別精度と統計
    class_stats = {}
    for i in range(10):
        class_mask = np.array(targets) == i
        if np.sum(class_mask) > 0:
            class_accuracy = np.sum(np.array(preds)[class_mask] == i) / np.sum(class_mask) * 100
            class_confidence = np.mean(np.array(confidences)[class_mask])
            class_stats[i] = {
                'accuracy': class_accuracy,
                'confidence': class_confidence,
                'samples': np.sum(class_mask)
            }
    
    # エラー分析
    errors = []
    for i in range(len(targets)):
        if targets[i] != preds[i]:
            errors.append({
                'true': targets[i],
                'pred': preds[i],
                'confidence': confidences[i]
            })
    
    # 可視化
    plt.figure(figsize=(20, 15))
    
    # 1. 混同行列
    plt.subplot(2, 3, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # 2. クラス別精度
    plt.subplot(2, 3, 2)
    classes = list(class_stats.keys())
    accuracies = [class_stats[c]['accuracy'] for c in classes]
    colors = ['red' if c in config.difficult_classes else 'blue' for c in classes]
    
    bars = plt.bar(classes, accuracies, color=colors, alpha=0.7)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.ylim(90, 100)
    
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 3. クラス別信頼度
    plt.subplot(2, 3, 3)
    confidences_by_class = [class_stats[c]['confidence'] for c in classes]
    plt.bar(classes, confidences_by_class, color='green', alpha=0.7)
    plt.title('Class-wise Confidence')
    plt.xlabel('Class')
    plt.ylabel('Average Confidence')
    
    # 4. エラー分布
    plt.subplot(2, 3, 4)
    error_pairs = {}
    for error in errors:
        pair = f"{error['true']}→{error['pred']}"
        error_pairs[pair] = error_pairs.get(pair, 0) + 1
    
    # トップ10エラーペア
    top_errors = sorted(error_pairs.items(), key=lambda x: x[1], reverse=True)[:10]
    if top_errors:
        pairs, counts = zip(*top_errors)
        plt.barh(range(len(pairs)), counts)
        plt.yticks(range(len(pairs)), pairs)
        plt.xlabel('Error Count')
        plt.title('Top 10 Error Patterns')
    
    # 5. 信頼度分布
    plt.subplot(2, 3, 5)
    plt.hist(confidences, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(np.mean(confidences), color='red', linestyle='--', 
                label=f'Mean: {np.mean(confidences):.3f}')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Confidence Distribution')
    plt.legend()
    
    # 6. 困難クラス分析
    plt.subplot(2, 3, 6)
    difficult_acc = [class_stats[c]['accuracy'] for c in config.difficult_classes]
    difficult_labels = [f'Class {c}' for c in config.difficult_classes]
    
    bars = plt.bar(difficult_labels, difficult_acc, color='red', alpha=0.7)
    plt.title('Difficult Classes Performance')
    plt.ylabel('Accuracy (%)')
    plt.ylim(95, 100)
    
    for bar, acc in zip(bars, difficult_acc):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{acc:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'figures/nkat_enhanced_v2_test_analysis_{timestamp}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # レポート生成
    overall_accuracy = np.mean(np.array(preds) == np.array(targets)) * 100
    
    report = {
        'timestamp': timestamp,
        'model_type': 'Enhanced NKAT-Transformer v2.0',
        'overall_accuracy': float(overall_accuracy),
        'total_samples': len(targets),
        'total_errors': len(errors),
        'error_rate': float(len(errors) / len(targets) * 100),
        'average_confidence': float(np.mean(confidences)),
        'class_statistics': {
            str(k): {
                'accuracy': float(v['accuracy']),
                'confidence': float(v['confidence']),
                'samples': int(v['samples'])
            } for k, v in class_stats.items()
        },
        'difficult_classes_performance': {
            str(c): float(class_stats[c]['accuracy']) 
            for c in config.difficult_classes
        },
        'top_error_patterns': dict(top_errors[:10]) if top_errors else {},
        'confusion_matrix': cm.tolist(),
        'target_achievement': {
            '99_percent': bool(overall_accuracy >= 99.0),
            'vs_previous': f"{overall_accuracy:.2f}% vs 97.79% (previous best)"
        }
    }
    
    # レポート保存
    with open(f'analysis/nkat_enhanced_v2_test_report_{timestamp}.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    return report

def main():
    """メイン実行関数"""
    print("🔍 Enhanced NKAT-Transformer v2.0 Model Testing")
    print("=" * 60)
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # モデルロード
    checkpoint_path = 'checkpoints/nkat_enhanced_v2_best.pth'
    try:
        model, config, checkpoint = load_model(checkpoint_path)
        print(f"\n✅ Model loaded successfully")
        
        # パラメータ数
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # 評価実行
    print(f"\n🧪 Starting evaluation...")
    accuracy, preds, targets, confidences = evaluate_model(model, config, device)
    
    # 詳細分析
    print(f"\n📊 Generating detailed analysis...")
    report = detailed_analysis(preds, targets, confidences, config)
    
    # 結果サマリー
    print(f"\n" + "=" * 60)
    print(f"🎯 ENHANCED NKAT-TRANSFORMER V2.0 TEST RESULTS")
    print(f"=" * 60)
    print(f"• Overall Accuracy: {accuracy:.4f}%")
    print(f"• Previous Best: 97.79%")
    print(f"• Improvement: {accuracy - 97.79:+.2f}%")
    print(f"• Target (99%+): {'✅ ACHIEVED' if accuracy >= 99.0 else '🔄 IN PROGRESS'}")
    
    # 困難クラス性能
    print(f"\n📈 Difficult Classes Performance:")
    for cls in config.difficult_classes:
        cls_acc = report['class_statistics'][str(cls)]['accuracy']
        print(f"• Class {cls}: {cls_acc:.2f}%")
    
    # エラー分析
    print(f"\n❌ Error Analysis:")
    print(f"• Total Errors: {len(targets) - sum(np.array(preds) == np.array(targets))}")
    print(f"• Error Rate: {100 * (1 - accuracy/100):.4f}%")
    
    if report['top_error_patterns']:
        print(f"• Top Error Pattern: {list(report['top_error_patterns'].keys())[0]} " +
              f"({list(report['top_error_patterns'].values())[0]} times)")
    
    print(f"\n💾 Analysis saved: analysis/nkat_enhanced_v2_test_report_{report['timestamp']}.json")
    print(f"📊 Visualization: figures/nkat_enhanced_v2_test_analysis_{report['timestamp']}.png")
    
    if accuracy >= 99.0:
        print(f"\n🎉 CONGRATULATIONS!")
        print(f"Enhanced NKAT-Transformer v2.0 has achieved 99%+ accuracy!")
        print(f"Ready for production deployment and publication.")
    else:
        improvement_needed = 99.0 - accuracy
        print(f"\n📈 Progress towards 99%:")
        print(f"Current: {accuracy:.2f}% | Target: 99.00% | Gap: {improvement_needed:.2f}%")
    
    print(f"\n" + "=" * 60)
    print(f"Enhanced NKAT-Transformer v2.0 Testing Complete!")
    print(f"=" * 60)

if __name__ == "__main__":
    main() 