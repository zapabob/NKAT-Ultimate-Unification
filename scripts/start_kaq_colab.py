#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌌 KAQ統合理論 Google Colab起動スクリプト
Kolmogorov-Arnold-Quantum Unified Theory for Google Colab

Author: 峯岸　亮 (Ryo Minegishi)
Institution: 放送大学 (The Open University of Japan)
Date: 2025-05-28
Version: Colab Optimized v1.0
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_colab_environment():
    """Google Colab環境チェック"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def install_dependencies():
    """依存関係のインストール"""
    print("📦 依存関係をインストール中...")
    
    packages = [
        "pykan",
        "torch",
        "torchvision", 
        "torchaudio",
        "numpy",
        "matplotlib",
        "scipy",
        "tqdm",
        "plotly",
        "ipywidgets"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} インストール完了")
        except subprocess.CalledProcessError:
            print(f"❌ {package} インストール失敗")
            return False
    
    return True

def setup_japanese_fonts():
    """日本語フォントセットアップ"""
    if check_colab_environment():
        print("🔤 日本語フォントをセットアップ中...")
        try:
            subprocess.check_call(["apt-get", "update"])
            subprocess.check_call(["apt-get", "install", "-y", "fonts-noto-cjk"])
            subprocess.check_call(["fc-cache", "-fv"])
            print("✅ 日本語フォントセットアップ完了")
            return True
        except subprocess.CalledProcessError:
            print("❌ 日本語フォントセットアップ失敗")
            return False
    return True

def create_colab_config():
    """Google Colab設定ファイル作成"""
    config = {
        "kaq_settings": {
            "ka_dimension": 8,
            "qft_qubits": 8,
            "theta": 1e-20,
            "use_gpu": True,
            "mixed_precision": True,
            "memory_efficient": True
        },
        "visualization": {
            "enable_plotly": True,
            "enable_matplotlib": True,
            "japanese_fonts": True
        },
        "experiment": {
            "default_experiments": 3,
            "save_results": True,
            "interactive_mode": True
        }
    }
    
    with open("kaq_colab_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print("⚙️ 設定ファイル作成完了")
    return config

def check_gpu_availability():
    """GPU利用可能性チェック"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🚀 GPU利用可能: {gpu_name}")
            print(f"💾 VRAM: {gpu_memory:.1f} GB")
            return True
        else:
            print("⚠️ GPU利用不可 - CPUモードで実行")
            return False
    except ImportError:
        print("❌ PyTorchが利用できません")
        return False

def create_notebook_launcher():
    """ノートブック起動用コード生成"""
    launcher_code = '''
# KAQ統合理論 Google Colab起動コード
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm.notebook import tqdm

# 設定読み込み
with open("kaq_colab_config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

print("🌌 KAQ統合理論 Google Colab版")
print("=" * 50)
print(f"📊 K-A次元: {config['kaq_settings']['ka_dimension']}")
print(f"⚛️ 量子ビット: {config['kaq_settings']['qft_qubits']}")
print(f"🔧 非可換パラメータ: {config['kaq_settings']['theta']:.2e}")

# GPU設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ デバイス: {device}")

# 日本語フォント設定
plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("✅ 初期化完了 - 実験を開始できます！")
'''
    
    with open("kaq_launcher.py", "w", encoding="utf-8") as f:
        f.write(launcher_code)
    
    print("🚀 起動コード生成完了")

def display_instructions():
    """使用方法表示"""
    instructions = """
🌌 KAQ統合理論 Google Colab版 - セットアップ完了！

📋 次のステップ:

1. 📓 ノートブックを開く:
   - NKAT_Colab_Notebook.ipynb (メイン実装)
   - NKAT_Colab_Notebook_Part2.ipynb (量子フーリエ変換)
   - NKAT_Colab_Visualization.ipynb (可視化)

2. 🚀 実行方法:
   - ランタイム → ランタイムのタイプを変更 → GPU選択
   - セルを順番に実行

3. 🎮 インタラクティブ実験:
   - パラメータ調整パネルを使用
   - リアルタイム可視化を確認

4. 📊 結果確認:
   - 忠実度とワームホール効果を分析
   - 3D可視化で幾何学構造を確認

🔗 詳細情報: README_KAQ_Colab.md

🎉 実験を楽しんでください！
"""
    print(instructions)

def main():
    """メイン実行関数"""
    print("🌌 KAQ統合理論 Google Colab セットアップ開始")
    print("=" * 60)
    
    # 環境チェック
    is_colab = check_colab_environment()
    if is_colab:
        print("✅ Google Colab環境を検出")
    else:
        print("ℹ️ ローカル環境で実行")
    
    # 依存関係インストール
    if not install_dependencies():
        print("❌ セットアップ失敗: 依存関係インストールエラー")
        return False
    
    # 日本語フォントセットアップ
    if not setup_japanese_fonts():
        print("⚠️ 日本語フォントセットアップに問題がありました")
    
    # 設定ファイル作成
    config = create_colab_config()
    
    # GPU確認
    gpu_available = check_gpu_availability()
    
    # 起動コード生成
    create_notebook_launcher()
    
    # 使用方法表示
    display_instructions()
    
    print("\n🎉 KAQ統合理論 Google Colab セットアップ完了！")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 