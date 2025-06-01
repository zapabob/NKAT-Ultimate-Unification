#!/usr/bin/env python3
import torch
import json

# チェックポイント読み込み
ckpt = torch.load('nkat_mnist_checkpoints/latest_checkpoint.pt', map_location='cpu')

print("🌟 NKAT-Transformer MNIST 訓練状況レポート")
print("=" * 50)
print(f"📊 現在のエポック: {ckpt['epoch']}")
print(f"🏆 最高検証精度: {ckpt['best_val_acc']:.4f}%")
print(f"📈 訓練履歴数: {len(ckpt['train_history']['epoch'])}")
print()

# 最近の訓練結果
history = ckpt['train_history']
if len(history['epoch']) > 0:
    print("📋 最近の訓練結果:")
    start_idx = max(0, len(history['epoch'])-5)
    for i in range(start_idx, len(history['epoch'])):
        epoch = history['epoch'][i]
        train_acc = history['train_acc'][i]
        val_acc = history['val_acc'][i]
        train_loss = history['train_loss'][i]
        val_loss = history['val_loss'][i]
        print(f"  エポック {epoch:03d}: 訓練精度 {train_acc:.2f}%, 検証精度 {val_acc:.2f}% (損失: {train_loss:.4f}/{val_loss:.4f})")

print()
print("🎯 NKAT理論的特徴:")
for key, value in ckpt['config'].items():
    if 'theta_nc' in key or 'gamma_conv' in key or 'quantum' in key:
        print(f"  {key}: {value}")

print()
print(f"💾 保存時刻: {ckpt['timestamp']}") 