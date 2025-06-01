#!/usr/bin/env python3
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
import sys
import os

# パス追加
sys.path.append('.')

from nkat_transformer.model import NKATVisionTransformer
from utils.metrics import tpe_metric, count_nkat_parameters

def quick_eval(checkpoint_path, device='cuda'):
    """クイック評価"""
    print("Starting quick evaluation...")
    
    # データローダー
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)
    
    # モデル作成
    model = NKATVisionTransformer(
        img_size=28,
        patch_size=4,
        num_classes=10,
        embed_dim=384,
        depth=5,
        num_heads=8
    ).to(device)
    
    # チェックポイントロード
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint: {checkpoint_path}")
    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")
        return
    
    # 評価
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device).long()
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = correct / total
    
    # パラメータ分析
    param_analysis = count_nkat_parameters(model)
    tpe_score = tpe_metric(accuracy, param_analysis['nkat_params'])
    
    # 結果出力
    print(f"EVAL_RESULT_START")
    print(f"accuracy={accuracy:.6f}")
    print(f"tpe_score={tpe_score:.6f}")
    print(f"lambda_theory={param_analysis['nkat_params']}")
    print(f"nkat_ratio={param_analysis['nkat_ratio']:.6f}")
    print(f"total_params={param_analysis['total_params']}")
    print(f"target_99_achieved={accuracy >= 0.99}")
    print(f"EVAL_RESULT_END")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    quick_eval(args.checkpoint, args.device)
