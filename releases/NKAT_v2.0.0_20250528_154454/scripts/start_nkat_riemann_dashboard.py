#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 NKAT リーマン予想解析ダッシュボード起動スクリプト
NKAT Riemann Hypothesis Analysis Dashboard Launcher

Author: NKAT Research Team
Date: 2025-01-28
Version: 1.0

機能:
- 依存関係の自動チェック
- GPU環境の確認
- Streamlitダッシュボードの起動
- ブラウザの自動オープン
"""

import sys
import os
import subprocess
import importlib
import webbrowser
import time
from pathlib import Path

def check_python_version():
    """Python バージョンチェック"""
    print("🐍 Python バージョンチェック...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8以上が必要です")
        print(f"現在のバージョン: {sys.version}")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def check_dependencies():
    """依存関係チェック"""
    print("\n📦 依存関係チェック...")
    
    # 基本パッケージ（軽量）
    basic_packages = [
        'numpy',
        'matplotlib',
        'plotly',
        'streamlit',
        'pandas',
        'scipy',
        'psutil',
        'h5py',
        'tqdm'
    ]
    
    # オプションパッケージ
    optional_packages = ['GPUtil']
    
    missing_packages = []
    
    # 基本パッケージチェック
    for package in basic_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (未インストール)")
            missing_packages.append(package)
    
    # PyTorchの特別処理（時間がかかる可能性があるため）
    print("🔍 PyTorch チェック中（時間がかかる場合があります）...")
    try:
        import torch
        print(f"✅ torch (バージョン: {torch.__version__})")
    except ImportError:
        print("❌ torch (未インストール)")
        missing_packages.append('torch')
    except Exception as e:
        print(f"⚠️ torch 読み込みエラー: {e}")
        print("   続行しますが、GPU機能が制限される可能性があります")
    
    # オプションパッケージチェック
    for package in optional_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} (オプション)")
        except ImportError:
            print(f"⚠️ {package} (オプション - GPU監視機能が制限されます)")
    
    if missing_packages:
        print(f"\n⚠️ 不足パッケージ: {', '.join(missing_packages)}")
        print("以下のコマンドでインストールしてください:")
        if 'torch' in missing_packages:
            print("# CUDA対応PyTorchのインストール:")
            print("pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121")
            print("# その他のパッケージ:")
            other_packages = [pkg for pkg in missing_packages if pkg != 'torch']
            if other_packages:
                print(f"pip install {' '.join(other_packages)}")
        else:
            print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_gpu_environment():
    """GPU環境チェック"""
    print("\n🎮 GPU環境チェック...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU検出: {gpu_name}")
            print(f"✅ VRAM: {total_memory:.1f} GB")
            
            # RTX3080チェック
            if "RTX 3080" in gpu_name or "RTX3080" in gpu_name:
                print("🚀 RTX3080検出 - 専用最適化が有効になります")
            
            return True
        else:
            print("⚠️ CUDA対応GPUが見つかりません")
            print("CPUモードで動作します")
            return True
    except Exception as e:
        print(f"❌ GPU環境チェックエラー: {e}")
        return False

def check_streamlit():
    """Streamlit チェック"""
    print("\n🌐 Streamlit チェック...")
    
    try:
        import streamlit as st
        print("✅ Streamlit インストール済み")
        return True
    except ImportError:
        print("❌ Streamlit が見つかりません")
        print("pip install streamlit でインストールしてください")
        return False

def find_dashboard_script():
    """ダッシュボードスクリプトの検索"""
    print("\n🔍 ダッシュボードスクリプト検索...")
    
    # 可能なパス（優先順位順）
    possible_paths = [
        Path("src/simple_nkat_dashboard.py"),  # 簡単版を優先
        Path("src/nkat_riemann_rtx3080_dashboard.py"),  # フル版
        Path("../src/simple_nkat_dashboard.py"),
        Path("../src/nkat_riemann_rtx3080_dashboard.py"),
        Path("simple_nkat_dashboard.py"),
        Path("nkat_riemann_rtx3080_dashboard.py")
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"✅ ダッシュボードスクリプト発見: {path}")
            if "simple" in str(path):
                print("   📝 簡単版ダッシュボードを使用します")
            else:
                print("   🚀 フル版ダッシュボードを使用します")
            return path
    
    print("❌ ダッシュボードスクリプトが見つかりません")
    print("以下のパスを確認してください:")
    for path in possible_paths:
        print(f"  - {path}")
    
    return None

def create_results_directories():
    """結果保存ディレクトリの作成"""
    print("\n📁 ディレクトリ作成...")
    
    directories = [
        "Results/checkpoints",
        "Results/logs",
        "Results/images",
        "Results/json"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ {directory}")

def start_streamlit_dashboard(script_path):
    """Streamlit ダッシュボード起動"""
    print(f"\n🚀 Streamlit ダッシュボード起動中...")
    print(f"スクリプト: {script_path}")
    
    try:
        # Streamlit コマンド構築
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(script_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ]
        
        print(f"実行コマンド: {' '.join(cmd)}")
        
        # ブラウザを少し遅れて開く
        def open_browser():
            time.sleep(3)
            webbrowser.open("http://localhost:8501")
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Streamlit 起動
        print("\n" + "="*60)
        print("🌌 NKAT リーマン予想解析ダッシュボード")
        print("="*60)
        print("📍 URL: http://localhost:8501")
        print("🛑 停止: Ctrl+C")
        print("="*60)
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n\n👋 ダッシュボードを停止しました")
    except Exception as e:
        print(f"\n❌ 起動エラー: {e}")
        return False
    
    return True

def install_cuda_pytorch():
    """CUDA対応PyTorchの自動インストール"""
    print("\n🔧 CUDA対応PyTorchを自動インストール中...")
    try:
        import subprocess
        cmd = [
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio",
            "--extra-index-url", "https://download.pytorch.org/whl/cu121"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CUDA対応PyTorchのインストール完了")
            return True
        else:
            print(f"❌ インストールエラー: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ インストール中にエラー: {e}")
        return False

def main():
    """メイン関数"""
    print("🌌 NKAT リーマン予想解析ダッシュボード起動スクリプト")
    print("=" * 60)
    
    # 環境チェック
    if not check_python_version():
        sys.exit(1)
    
    if not check_dependencies():
        print("\n❌ 依存関係が不足しています")
        
        # PyTorchの自動インストールを試行
        response = input("\n🤖 CUDA対応PyTorchを自動インストールしますか？ (y/n): ")
        if response.lower() in ['y', 'yes', 'はい']:
            if install_cuda_pytorch():
                print("✅ 再度依存関係をチェックします...")
                if not check_dependencies():
                    print("❌ まだ不足しているパッケージがあります")
                    sys.exit(1)
            else:
                print("❌ 自動インストールに失敗しました")
                print("手動でインストールしてから再実行してください")
                sys.exit(1)
        else:
            print("必要なパッケージをインストールしてから再実行してください")
            sys.exit(1)
    
    if not check_gpu_environment():
        print("\n⚠️ GPU環境に問題がありますが、続行します")
    
    if not check_streamlit():
        sys.exit(1)
    
    # ダッシュボードスクリプト検索
    script_path = find_dashboard_script()
    if not script_path:
        sys.exit(1)
    
    # ディレクトリ作成
    create_results_directories()
    
    # ダッシュボード起動
    print("\n✅ 全ての前提条件が満たされました")
    print("🚀 ダッシュボードを起動します...")
    
    if not start_streamlit_dashboard(script_path):
        sys.exit(1)

if __name__ == "__main__":
    main() 