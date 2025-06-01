#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT-Transformer混同行列・誤分類分析スクリプト
高解像度可視化・詳細エラー分析対応

Author: NKAT Advanced Computing Team
Date: 2025-01-26
CUDA Requirement: RTX3080 or higher
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
import json
import os
from datetime import datetime
from tqdm import tqdm

# 日本語文字化け防止・英語表記設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

def load_test_results():
    """テスト結果の読み込み"""
    results_path = "analysis/test_results.pt"
    if not os.path.exists(results_path):
        raise FileNotFoundError(
            "Test results not found. Please run eval_nkat_transformer.py first."
        )
    
    data = torch.load(results_path)
    return data

def plot_confusion_matrix(y_true, y_pred, save_path="analysis/confusion_matrix.png"):
    """混同行列の可視化"""
    print("Generating confusion matrix...")
    
    # 混同行列計算
    cm = confusion_matrix(y_true, y_pred)
    
    # 正規化版も作成
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # サブプロット作成
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. 絶対数での混同行列
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=range(10), yticklabels=range(10),
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title("NKAT-ViT Confusion Matrix (Counts)", fontsize=14, weight='bold')
    axes[0].set_xlabel("Predicted Label", fontsize=12)
    axes[0].set_ylabel("True Label", fontsize=12)
    
    # 2. 正規化された混同行列
    sns.heatmap(cm_normalized, annot=True, fmt=".3f", cmap="Oranges",
                xticklabels=range(10), yticklabels=range(10),
                ax=axes[1], cbar_kws={'label': 'Normalized Rate'})
    axes[1].set_title("NKAT-ViT Confusion Matrix (Normalized)", fontsize=14, weight='bold')
    axes[1].set_xlabel("Predicted Label", fontsize=12)
    axes[1].set_ylabel("True Label", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    
    return cm, cm_normalized

def analyze_classification_errors(cm):
    """分類エラーの詳細分析"""
    print("\n=== Error Analysis ===")
    
    # 各クラスのエラー統計
    total_per_class = cm.sum(axis=1)
    correct_per_class = np.diag(cm)
    error_per_class = total_per_class - correct_per_class
    
    print("Class-wise Error Statistics:")
    for i in range(10):
        if total_per_class[i] > 0:
            error_rate = error_per_class[i] / total_per_class[i] * 100
            print(f"Class {i}: {error_per_class[i]}/{total_per_class[i]} errors ({error_rate:.2f}%)")
    
    # 最も混同しやすいペア
    print("\nMost Confused Pairs:")
    confusion_pairs = []
    for i in range(10):
        for j in range(10):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append((i, j, cm[i, j]))
    
    # 混同数でソート
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    
    for i, (true_class, pred_class, count) in enumerate(confusion_pairs[:10]):
        print(f"{i+1}. {true_class} → {pred_class}: {count} times")
    
    return confusion_pairs

def plot_class_performance(y_true, y_pred, confidences, save_path="analysis/class_performance.png"):
    """クラス別性能の詳細可視化"""
    print("Generating class performance analysis...")
    
    # メトリクス計算
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    # クラス別信頼度統計
    class_confidences = []
    for class_id in range(10):
        mask = y_true == class_id
        if mask.sum() > 0:
            class_conf = confidences[mask].mean()
            class_confidences.append(class_conf)
        else:
            class_confidences.append(0)
    
    # データフレーム作成
    df = pd.DataFrame({
        'Class': range(10),
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support,
        'Avg_Confidence': class_confidences
    })
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Precision, Recall, F1-Score
    x = np.arange(10)
    width = 0.25
    
    axes[0, 0].bar(x - width, precision, width, label='Precision', alpha=0.8)
    axes[0, 0].bar(x, recall, width, label='Recall', alpha=0.8)
    axes[0, 0].bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    axes[0, 0].set_xlabel('Class')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('NKAT-ViT: Precision, Recall, F1-Score by Class')
    axes[0, 0].set_xticks(x)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Support (サンプル数)
    axes[0, 1].bar(range(10), support, color='skyblue', alpha=0.8)
    axes[0, 1].set_xlabel('Class')
    axes[0, 1].set_ylabel('Sample Count')
    axes[0, 1].set_title('Test Set Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 平均信頼度
    axes[1, 0].bar(range(10), class_confidences, color='lightgreen', alpha=0.8)
    axes[1, 0].set_xlabel('Class')
    axes[1, 0].set_ylabel('Average Confidence')
    axes[1, 0].set_title('Average Prediction Confidence by Class')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Accuracy vs Confidence散布図
    class_accuracy = recall  # Recall = Class-wise accuracy
    axes[1, 1].scatter(class_confidences, class_accuracy, s=100, alpha=0.7)
    for i in range(10):
        axes[1, 1].annotate(str(i), (class_confidences[i], class_accuracy[i]), 
                           xytext=(5, 5), textcoords='offset points')
    axes[1, 1].set_xlabel('Average Confidence')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Confidence vs Accuracy Relationship')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Class performance analysis saved to {save_path}")
    
    return df

def visualize_error_samples(y_true, y_pred, confidences, save_path="analysis/error_samples.png"):
    """誤分類サンプルの可視化"""
    print("Loading test dataset for error visualization...")
    
    # テストデータセット読み込み
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    
    # 誤分類インデックス
    error_indices = np.where(y_true != y_pred)[0]
    
    if len(error_indices) == 0:
        print("No errors found! Perfect classification!")
        return
    
    # 信頼度でソート（高信頼度エラーが最も問題）
    error_confidences = confidences[error_indices]
    sorted_indices = error_indices[np.argsort(error_confidences)[-16:]]  # 上位16個
    
    # 可視化
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle('NKAT-ViT: High-Confidence Misclassifications', fontsize=16, weight='bold')
    
    for i, idx in enumerate(sorted_indices):
        row, col = i // 4, i % 4
        
        # 画像取得（非正規化）
        image, _ = test_dataset[idx]
        image_np = image.squeeze().numpy()
        
        # 表示
        axes[row, col].imshow(image_np, cmap='gray')
        axes[row, col].set_title(f'True: {y_true[idx]}, Pred: {y_pred[idx]}\nConf: {confidences[idx]:.3f}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Error samples visualization saved to {save_path}")

def generate_detailed_report(results, cm, class_df):
    """詳細レポートの生成"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    y_true = results["preds"].numpy()
    y_pred = results["labels"].numpy()
    
    # 分類レポート
    class_report = classification_report(y_true, y_pred, output_dict=True)
    
    # 統計サマリー
    total_errors = len(y_true) - np.sum(y_true == y_pred)
    error_rate = total_errors / len(y_true) * 100
    
    report = {
        "timestamp": timestamp,
        "model": "NKAT-Vision Transformer",
        "test_accuracy": results["accuracy"],
        "total_samples": len(y_true),
        "total_errors": int(total_errors),
        "error_rate_percent": error_rate,
        "classification_report": class_report,
        "confusion_matrix": cm.tolist(),
        "class_performance": class_df.to_dict('records'),
        "throughput_samples_per_sec": results["throughput"],
        "inference_time_per_batch": results["inference_time"]
    }
    
    # レポート保存
    report_path = f"analysis/detailed_analysis_report_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nDetailed analysis report saved to {report_path}")
    
    # サマリー表示
    print(f"\n{'='*60}")
    print("NKAT-Transformer Performance Summary")
    print(f"{'='*60}")
    print(f"Test Accuracy: {results['accuracy']:.4f}%")
    print(f"Total Errors: {total_errors}/{len(y_true)} ({error_rate:.2f}%)")
    print(f"Throughput: {results['throughput']:.2f} samples/sec")
    print(f"Most Problematic Classes: {class_df.nsmallest(3, 'F1-Score')['Class'].tolist()}")
    print(f"Best Performing Classes: {class_df.nlargest(3, 'F1-Score')['Class'].tolist()}")
    
    return report

def main():
    """メイン実行関数"""
    print("NKAT-Transformer Confusion Matrix & Error Analysis")
    print("=" * 60)
    
    # テスト結果読み込み
    try:
        results = load_test_results()
        print(f"Loaded test results: {len(results['labels'])} samples")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # データ準備
    y_true = results["labels"].numpy()
    y_pred = results["preds"].numpy()
    confidences = results["confidences"].numpy()
    
    print(f"True labels shape: {y_true.shape}")
    print(f"Predicted labels shape: {y_pred.shape}")
    print(f"Confidences shape: {confidences.shape}")
    
    # 1. 混同行列生成
    cm, cm_normalized = plot_confusion_matrix(y_true, y_pred)
    
    # 2. エラー分析
    confusion_pairs = analyze_classification_errors(cm)
    
    # 3. クラス別性能分析
    class_df = plot_class_performance(y_true, y_pred, confidences)
    
    # 4. エラーサンプル可視化
    visualize_error_samples(y_true, y_pred, confidences)
    
    # 5. 詳細レポート生成
    report = generate_detailed_report(results, cm, class_df)
    
    print(f"\n{'='*60}")
    print("Analysis completed successfully!")
    print("Generated files:")
    print("- analysis/confusion_matrix.png")
    print("- analysis/class_performance.png") 
    print("- analysis/error_samples.png")
    print(f"- analysis/detailed_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    print("Next step: Run plot_curves.py for training history visualization")

if __name__ == "__main__":
    main() 