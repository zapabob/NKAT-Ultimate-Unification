#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
なんJ風 依存関係インストールスクリプト
RTX3080のCUDA環境でNKAT理論に必要なライブラリをインストールするぜ！
"""

import subprocess
import sys
import os
from pathlib import Path

def install_package(package):
    """パッケージをインストール"""
    try:
        print(f"📦 {package} をインストール中...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} インストール完了")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {package} インストール失敗: {e}")
        return False

def main():
    """メイン関数"""
    print("🚀 なんJ風 依存関係インストール開始！")
    
    # 必要なパッケージリスト
    packages = [
        "torch",  # PyTorch
        "torchvision",  # PyTorch Vision
        "torchaudio",  # PyTorch Audio
        "numpy",  # 数値計算
        "matplotlib",  # 可視化
        "tqdm",  # プログレスバー
        "scipy",  # 科学技術計算
        "pandas",  # データ分析
        "seaborn",  # 統計可視化
        "jupyter",  # Jupyter Notebook
        "ipykernel",  # IPython Kernel
    ]
    
    # CUDA対応のPyTorchをインストール
    print("🎮 CUDA対応PyTorchをインストール中...")
    
    # Windows用のCUDA対応PyTorch
    torch_cuda_command = [
        sys.executable, "-m", "pip", "install", 
        "torch", "torchvision", "torchaudio", 
        "--index-url", "https://download.pytorch.org/whl/cu118"
    ]
    
    try:
        subprocess.check_call(torch_cuda_command)
        print("✅ CUDA対応PyTorchインストール完了")
    except subprocess.CalledProcessError as e:
        print(f"❌ CUDA対応PyTorchインストール失敗: {e}")
        print("⚠️ CPU版PyTorchをインストールします")
        install_package("torch")
        install_package("torchvision")
        install_package("torchaudio")
    
    # その他のパッケージをインストール
    print("\n📦 その他のパッケージをインストール中...")
    success_count = 0
    total_count = len(packages) - 3  # torch関連は既にインストール済み
    
    for package in packages[3:]:  # torch関連以外
        if install_package(package):
            success_count += 1
    
    print(f"\n📊 インストール結果:")
    print(f"  成功: {success_count}/{total_count}")
    print(f"  失敗: {total_count - success_count}/{total_count}")
    
    # 環境確認
    print("\n🔍 環境確認中...")
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"🎮 CUDA利用可能: {torch.cuda.get_device_name(0)}")
            print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        else:
            print("⚠️ CUDA利用不可")
    except ImportError:
        print("❌ PyTorchインポート失敗")
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError:
        print("❌ NumPyインポート失敗")
    
    try:
        import matplotlib
        print(f"✅ Matplotlib: {matplotlib.__version__}")
    except ImportError:
        print("❌ Matplotlibインポート失敗")
    
    print("\n🎉 なんJ風 依存関係インストール完了！")
    print("🚀 NKAT理論の数値解析を開始できます！")

if __name__ == "__main__":
    main() 