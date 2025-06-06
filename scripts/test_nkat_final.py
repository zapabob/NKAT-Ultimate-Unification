#!/usr/bin/env python3
"""
NKAT-Transformer 最終評価・可視化スクリプト
RTX3080対応 + 高精度分析
"""

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report

# 既存実装をインポート
from nkat_transformer_mnist_recognition import NKATVisionTransformer, NKATVisionConfig

# 設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams['font.family'] = 'DejaVu Sans'

def main():
    print("🚀 NKAT-Transformer Final Performance Evaluation")
    print("=" * 60)
    print(f"Device: {device}")
    
    # モデル読み込み
    config = NKATVisionConfig()
    model = NKATVisionTransformer(config).to(device)
    
    checkpoint = torch.load("nkat_mnist_checkpoints/latest_checkpoint.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded from epoch {checkpoint['epoch']}")
    print(f"✅ Best validation accuracy: {checkpoint.get('best_val_acc', 'N/A'):.4f}%")
    
    # テストデータ
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
    
    # 評価実行
    print("\n📊 Running comprehensive evaluation...")
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            probs = F.softmax(logits, dim=1)
            _, preds = torch.max(probs, 1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    accuracy = 100.0 * correct / total
    
    print(f"\n🎯 Final Results:")
    print(f"Test Accuracy: {accuracy:.4f}% ({correct}/{total})")
    print(f"Error Rate: {100-accuracy:.4f}%")
    
    # 混同行列作成
    cm = confusion_matrix(all_labels, all_preds)
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. 混同行列
    im1 = axes[0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[0].set_title('NKAT-ViT Confusion Matrix')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    
    # 数値を表示
    for i in range(10):
        for j in range(10):
            axes[0].text(j, i, str(cm[i, j]), ha="center", va="center")
    
    # 2. クラス別精度
    class_acc = []
    for i in range(10):
        mask = np.array(all_labels) == i
        if mask.sum() > 0:
            class_correct = (np.array(all_preds)[mask] == i).sum()
            class_acc.append(100.0 * class_correct / mask.sum())
        else:
            class_acc.append(0)
    
    axes[1].bar(range(10), class_acc, alpha=0.7, color='skyblue')
    axes[1].set_title('NKAT-ViT Class-wise Accuracy')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, alpha=0.3)
    
    # 値をバーの上に表示
    for i, v in enumerate(class_acc):
        axes[1].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('analysis/nkat_final_evaluation.png', dpi=300, bbox_inches='tight')
    print("📈 Analysis plot saved to analysis/nkat_final_evaluation.png")
    
    # 詳細レポート生成
    report = {
        "timestamp": datetime.now().isoformat(),
        "model": "NKAT-Vision Transformer",
        "test_accuracy": accuracy,
        "total_samples": total,
        "errors": total - correct,
        "class_accuracies": {str(i): acc for i, acc in enumerate(class_acc)},
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(all_labels, all_preds, output_dict=True),
        "nkat_features": {
            "non_commutative_geometry": True,
            "gauge_invariance": True,
            "quantum_gravity_correction": True,
            "super_convergence_factor": True
        }
    }
    
    with open('analysis/nkat_final_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("📄 Detailed report saved to analysis/nkat_final_report.json")
    
    # サマリー
    print(f"\n🏆 NKAT-Transformer Achievement Summary:")
    print(f"   ✅ Test Accuracy: {accuracy:.4f}%")
    print(f"   ✅ Theoretical Target (95%+): {'ACHIEVED' if accuracy >= 95 else 'NOT ACHIEVED'}")
    print(f"   ✅ Best Class: {class_acc.index(max(class_acc))} ({max(class_acc):.2f}%)")
    print(f"   ⚠️  Worst Class: {class_acc.index(min(class_acc))} ({min(class_acc):.2f}%)")
    
    error_samples = total - correct
    print(f"\n🔍 Error Analysis:")
    print(f"   Total Errors: {error_samples}")
    print(f"   Error Rate: {100-accuracy:.4f}%")
    
    if accuracy >= 97.0:
        print("\n🎉 EXCELLENT PERFORMANCE! NKAT theory demonstrates strong effectiveness!")
    elif accuracy >= 95.0:
        print("\n👍 GOOD PERFORMANCE! NKAT enhancements working well!")
    else:
        print("\n🔧 IMPROVEMENT NEEDED: Consider enhanced GE-Conv implementation!")
    
    print("\n📋 Next Steps:")
    print("   1. Implement Gauge-Equivariant Convolution for +2% boost")
    print("   2. Add learnable θ(x) parameters")
    print("   3. Optimize super-convergence factor γ scheduling")
    print("   4. Consider CutMix + Label Smoothing for robustness")

if __name__ == "__main__":
    main() 