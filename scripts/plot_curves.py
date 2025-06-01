#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT-Transformer学習曲線可視化スクリプト
Loss/Accuracy/Learning Rate履歴の詳細分析

Author: NKAT Advanced Computing Team
Date: 2025-01-26
CUDA Requirement: RTX3080 or higher
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from datetime import datetime
import seaborn as sns
from scipy.signal import savgol_filter

# 日本語文字化け防止・英語表記設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

def load_training_history():
    """学習履歴データの読み込み"""
    possible_paths = [
        "nkat_mnist_checkpoints/train_history.json",
        "nkat_mnist_checkpoints/training_log.json",
        "logs/training_history.json"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loading training history from: {path}")
            with open(path, 'r') as f:
                return json.load(f)
    
    # 履歴ファイルが見つからない場合、チェックポイントから情報を抽出
    print("Training history file not found, extracting from checkpoints...")
    return extract_history_from_checkpoints()

def extract_history_from_checkpoints():
    """チェックポイントファイルから学習履歴を抽出"""
    checkpoint_dir = "nkat_mnist_checkpoints"
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError("No checkpoint directory found")
    
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    # チェックポイントファイルを走査
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith("checkpoint_epoch_") and filename.endswith(".pt"):
            try:
                checkpoint_path = os.path.join(checkpoint_dir, filename)
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                
                if 'epoch' in checkpoint:
                    history["epoch"].append(checkpoint['epoch'])
                    history["train_loss"].append(checkpoint.get('train_loss', 0))
                    history["train_acc"].append(checkpoint.get('train_acc', 0))
                    history["val_loss"].append(checkpoint.get('val_loss', 0))
                    history["val_acc"].append(checkpoint.get('val_acc', 0))
            except:
                continue
    
    # エポック順にソート
    if history["epoch"]:
        sorted_indices = np.argsort(history["epoch"])
        for key in history:
            history[key] = [history[key][i] for i in sorted_indices]
    
    return history

def plot_training_curves(history, save_path="analysis/training_curves.png"):
    """学習曲線の可視化"""
    print("Generating training curves...")
    
    if not history["epoch"]:
        print("No training history data found")
        return
    
    epochs = history["epoch"]
    train_loss = history.get("train_loss", [])
    train_acc = history.get("train_acc", [])
    val_loss = history.get("val_loss", [])
    val_acc = history.get("val_acc", [])
    
    # 2x2サブプロット
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('NKAT-Transformer Training History', fontsize=16, weight='bold')
    
    # 1. Loss曲線
    if train_loss and val_loss:
        axes[0, 0].plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training & Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 最低損失点をマーク
        if val_loss:
            min_val_loss_epoch = epochs[np.argmin(val_loss)]
            min_val_loss = min(val_loss)
            axes[0, 0].plot(min_val_loss_epoch, min_val_loss, 'ro', markersize=8)
            axes[0, 0].annotate(f'Best: {min_val_loss:.4f}@{min_val_loss_epoch}', 
                               xy=(min_val_loss_epoch, min_val_loss), 
                               xytext=(10, 10), textcoords='offset points')
    
    # 2. Accuracy曲線
    if train_acc and val_acc:
        axes[0, 1].plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Training & Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 最高精度点をマーク
        if val_acc:
            max_val_acc_epoch = epochs[np.argmax(val_acc)]
            max_val_acc = max(val_acc)
            axes[0, 1].plot(max_val_acc_epoch, max_val_acc, 'go', markersize=8)
            axes[0, 1].annotate(f'Best: {max_val_acc:.2f}%@{max_val_acc_epoch}', 
                               xy=(max_val_acc_epoch, max_val_acc), 
                               xytext=(10, -15), textcoords='offset points')
    
    # 3. Learning Rate (推定)
    if len(epochs) > 1:
        # Learning rate scheduleの推定（一般的なパターン）
        warmup_epochs = min(10, len(epochs) // 4)
        lr_schedule = []
        initial_lr = 1e-3
        
        for i, epoch in enumerate(epochs):
            if epoch < warmup_epochs:
                lr = initial_lr * (epoch + 1) / warmup_epochs
            else:
                # Cosine annealing
                lr = initial_lr * 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (max(epochs) - warmup_epochs)))
            lr_schedule.append(lr)
        
        axes[1, 0].plot(epochs, lr_schedule, 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Estimated Learning Rate Schedule')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
    
    # 4. Overfitting分析
    if train_acc and val_acc and len(train_acc) == len(val_acc):
        gap = np.array(train_acc) - np.array(val_acc)
        axes[1, 1].plot(epochs, gap, 'purple', linewidth=2)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].fill_between(epochs, gap, 0, alpha=0.3, color='purple')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Train - Val Accuracy (%)')
        axes[1, 1].set_title('Overfitting Analysis (Train-Val Gap)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 平均オーバーフィッティング度を表示
        avg_gap = np.mean(gap[-10:]) if len(gap) >= 10 else np.mean(gap)
        axes[1, 1].text(0.7, 0.9, f'Avg Gap: {avg_gap:.2f}%', 
                       transform=axes[1, 1].transAxes, fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    
    return fig

def plot_detailed_metrics(history, save_path="analysis/detailed_metrics.png"):
    """詳細メトリクス分析"""
    print("Generating detailed metrics analysis...")
    
    if not history["epoch"] or len(history["epoch"]) < 5:
        print("Insufficient data for detailed analysis")
        return
    
    epochs = np.array(history["epoch"])
    val_acc = np.array(history.get("val_acc", []))
    
    if len(val_acc) == 0:
        print("No validation accuracy data found")
        return
    
    # データの平滑化
    if len(val_acc) > 5:
        smoothed_acc = savgol_filter(val_acc, min(5, len(val_acc)//2*2-1), 3)
    else:
        smoothed_acc = val_acc
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('NKAT-Transformer Detailed Performance Analysis', fontsize=16, weight='bold')
    
    # 1. 収束分析
    axes[0, 0].plot(epochs, val_acc, 'b-', alpha=0.6, label='Raw Validation Accuracy')
    axes[0, 0].plot(epochs, smoothed_acc, 'r-', linewidth=2, label='Smoothed')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Validation Accuracy (%)')
    axes[0, 0].set_title('Convergence Analysis')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 収束点を推定
    if len(smoothed_acc) > 10:
        last_10 = smoothed_acc[-10:]
        convergence_std = np.std(last_10)
        axes[0, 0].text(0.05, 0.95, f'Convergence STD: {convergence_std:.3f}%', 
                       transform=axes[0, 0].transAxes, fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    # 2. 改善率分析
    if len(val_acc) > 1:
        improvement = np.diff(val_acc)
        axes[0, 1].plot(epochs[1:], improvement, 'g-', linewidth=2)
        axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy Improvement (%)')
        axes[0, 1].set_title('Epoch-to-Epoch Improvement')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 最大改善点
        max_improvement_epoch = epochs[1:][np.argmax(improvement)]
        max_improvement = max(improvement)
        axes[0, 1].plot(max_improvement_epoch, max_improvement, 'ro', markersize=8)
    
    # 3. パフォーマンス分布
    axes[1, 0].hist(val_acc, bins=min(20, len(val_acc)//2), alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 0].axvline(np.mean(val_acc), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(val_acc):.2f}%')
    axes[1, 0].axvline(np.median(val_acc), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(val_acc):.2f}%')
    axes[1, 0].set_xlabel('Validation Accuracy (%)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Performance Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 目標達成分析
    target_accuracies = [90, 95, 97, 98, 99]
    achievement_epochs = []
    
    for target in target_accuracies:
        achieved_epochs = epochs[val_acc >= target]
        if len(achieved_epochs) > 0:
            achievement_epochs.append(achieved_epochs[0])
        else:
            achievement_epochs.append(None)
    
    # 達成済みの目標のみプロット
    achieved_targets = [target for target, epoch in zip(target_accuracies, achievement_epochs) if epoch is not None]
    achieved_epochs_clean = [epoch for epoch in achievement_epochs if epoch is not None]
    
    if achieved_targets:
        axes[1, 1].bar(range(len(achieved_targets)), achieved_epochs_clean, alpha=0.7, color='lightgreen')
        axes[1, 1].set_xlabel('Target Accuracy (%)')
        axes[1, 1].set_ylabel('Epoch Achieved')
        axes[1, 1].set_title('Target Achievement Timeline')
        axes[1, 1].set_xticks(range(len(achieved_targets)))
        axes[1, 1].set_xticklabels([f'{t}%' for t in achieved_targets])
        axes[1, 1].grid(True, alpha=0.3)
        
        # 値をバーの上に表示
        for i, (target, epoch) in enumerate(zip(achieved_targets, achieved_epochs_clean)):
            axes[1, 1].text(i, epoch + 0.5, f'{epoch}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Detailed metrics saved to {save_path}")

def generate_training_report(history):
    """学習レポートの生成"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if not history["epoch"]:
        print("No training data available for report generation")
        return
    
    epochs = history["epoch"]
    val_acc = history.get("val_acc", [])
    train_acc = history.get("train_acc", [])
    
    # 統計計算
    total_epochs = len(epochs)
    final_val_acc = val_acc[-1] if val_acc else 0
    best_val_acc = max(val_acc) if val_acc else 0
    best_epoch = epochs[np.argmax(val_acc)] if val_acc else 0
    
    # 収束分析
    if len(val_acc) >= 10:
        last_10_std = np.std(val_acc[-10:])
        converged = last_10_std < 0.1  # 0.1%未満の変動で収束と判定
    else:
        last_10_std = np.std(val_acc) if val_acc else 0
        converged = False
    
    # オーバーフィッティング分析
    if train_acc and val_acc and len(train_acc) == len(val_acc):
        final_gap = train_acc[-1] - val_acc[-1]
        avg_gap = np.mean(np.array(train_acc) - np.array(val_acc))
        overfitting = avg_gap > 2.0  # 2%以上の差でオーバーフィッティングと判定
    else:
        final_gap = 0
        avg_gap = 0
        overfitting = False
    
    report = {
        "timestamp": timestamp,
        "model": "NKAT-Vision Transformer",
        "training_summary": {
            "total_epochs": total_epochs,
            "final_validation_accuracy": final_val_acc,
            "best_validation_accuracy": best_val_acc,
            "best_epoch": best_epoch,
            "convergence_std_last_10": last_10_std,
            "converged": converged,
            "final_train_val_gap": final_gap,
            "average_train_val_gap": avg_gap,
            "overfitting_detected": overfitting
        },
        "milestones": {
            "epochs_to_90_percent": next((e for e, acc in zip(epochs, val_acc) if acc >= 90), None),
            "epochs_to_95_percent": next((e for e, acc in zip(epochs, val_acc) if acc >= 95), None),
            "epochs_to_97_percent": next((e for e, acc in zip(epochs, val_acc) if acc >= 97), None),
        },
        "training_efficiency": {
            "average_improvement_per_epoch": (best_val_acc - val_acc[0]) / total_epochs if val_acc and total_epochs > 0 else 0,
            "training_stability": "High" if last_10_std < 0.1 else "Medium" if last_10_std < 0.5 else "Low"
        }
    }
    
    # レポート保存
    report_path = f"analysis/training_report_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nTraining report saved to {report_path}")
    
    # サマリー表示
    print(f"\n{'='*60}")
    print("NKAT-Transformer Training Summary")
    print(f"{'='*60}")
    print(f"Total Epochs: {total_epochs}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}% (Epoch {best_epoch})")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}%")
    print(f"Training Convergence: {'Yes' if converged else 'No'} (STD: {last_10_std:.3f}%)")
    print(f"Overfitting Status: {'Detected' if overfitting else 'Not Detected'} (Gap: {avg_gap:.2f}%)")
    
    return report

def main():
    """メイン実行関数"""
    print("NKAT-Transformer Training Curves Analysis")
    print("=" * 60)
    
    try:
        # 学習履歴読み込み
        history = load_training_history()
        
        if not history or not history.get("epoch"):
            print("No training history found")
            return
            
        print(f"Loaded training history: {len(history['epoch'])} epochs")
        
        # 1. 基本学習曲線
        plot_training_curves(history)
        
        # 2. 詳細メトリクス分析
        plot_detailed_metrics(history)
        
        # 3. 学習レポート生成
        report = generate_training_report(history)
        
        print(f"\n{'='*60}")
        print("Training curves analysis completed successfully!")
        print("Generated files:")
        print("- analysis/training_curves.png")
        print("- analysis/detailed_metrics.png")
        print(f"- analysis/training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import torch  # torchが必要な場合のimport
    main() 