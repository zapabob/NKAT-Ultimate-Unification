#!/usr/bin/env python3
"""
NKAT-Transformer 簡単動作テスト
"""

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# 既存のNKAT実装をインポート
from nkat_transformer_mnist_recognition import NKATVisionTransformer, NKATVisionConfig

def test_nkat_model():
    """NKAT-Transformerの簡単テスト"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 設定とモデル
    config = NKATVisionConfig()
    model = NKATVisionTransformer(config).to(device)
    
    # チェックポイント読み込み
    checkpoint_path = "nkat_mnist_checkpoints/latest_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    # テストデータ
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    
    # 小さなサンプルでテスト
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            if i >= 10:  # 最初の1000サンプルのみ
                break
                
            images, labels = images.to(device), labels.to(device)
            
            # 予測
            outputs = model(images)
            print(f"Batch {i}: Output type: {type(outputs)}")
            
            if isinstance(outputs, dict):
                logits = outputs['logits']
                print(f"Logits shape: {logits.shape}")
                print(f"Logits sample: {logits[0].detach().cpu().numpy()}")
                
                # NaNチェック
                if torch.isnan(logits).any():
                    print("WARNING: NaN detected in logits!")
                    print(f"NaN count: {torch.isnan(logits).sum()}")
                    
                # 極値チェック
                print(f"Logits min/max: {logits.min():.4f} / {logits.max():.4f}")
                
                pred = logits.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
                
                accuracy = 100.0 * correct / total
                print(f"Running accuracy: {accuracy:.2f}% ({correct}/{total})")
                print("-" * 50)
            
    print(f"Final test accuracy on {total} samples: {100.0 * correct / total:.2f}%")

if __name__ == "__main__":
    test_nkat_model() 