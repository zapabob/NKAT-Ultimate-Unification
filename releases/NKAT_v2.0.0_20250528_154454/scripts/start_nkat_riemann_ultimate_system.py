#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 NKAT リーマン予想解析システム - 起動スクリプト
NKAT Riemann Hypothesis Analysis System Launcher

非可換コルモゴロフアーノルド表現理論による最高精度リーマン予想解析システムの起動

Author: NKAT Research Team
Date: 2025-05-28
Version: 1.0
"""

import sys
import os
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import importlib.util
import platform

def print_header():
    """ヘッダー表示"""
    print("🌌" + "=" * 80)
    print("  NKAT リーマン予想解析システム - 最高精度版")
    print("  Non-Commutative Kolmogorov-Arnold Theory Riemann Analysis")
    print("  Ultimate Precision Implementation")
    print("=" * 82)
    print()

def check_python_version() -> bool:
    """Python バージョンチェック"""
    print("🐍 Python バージョンチェック...")
    
    version = sys.version_info
    required_major, required_minor = 3, 8
    
    if version.major < required_major or (version.major == required_major and version.minor < required_minor):
        print(f"❌ Python {required_major}.{required_minor}+ が必要です")
        print(f"   現在のバージョン: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_required_packages() -> Tuple[bool, List[str]]:
    """必要パッケージのチェック"""
    print("📦 必要パッケージチェック...")
    
    required_packages = [
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('matplotlib', 'matplotlib'),
        ('torch', 'torch'),
        ('plotly', 'plotly'),
        ('streamlit', 'streamlit'),
        ('tqdm', 'tqdm'),
        ('h5py', 'h5py'),
        ('psutil', 'psutil'),
        ('GPUtil', 'GPUtil'),
        ('mpmath', 'mpmath'),
        ('pandas', 'pandas'),
        ('seaborn', 'seaborn')
    ]
    
    missing_packages = []
    installed_packages = []
    
    for package_name, import_name in required_packages:
        try:
            spec = importlib.util.find_spec(import_name)
            if spec is not None:
                module = importlib.import_module(import_name)
                version = getattr(module, '__version__', 'unknown')
                print(f"✅ {package_name}: {version}")
                installed_packages.append(package_name)
            else:
                print(f"❌ {package_name}: 未インストール")
                missing_packages.append(package_name)
        except ImportError:
            print(f"❌ {package_name}: インポートエラー")
            missing_packages.append(package_name)
    
    return len(missing_packages) == 0, missing_packages

def install_missing_packages(missing_packages: List[str]) -> bool:
    """不足パッケージのインストール"""
    if not missing_packages:
        return True
    
    print(f"\n📥 不足パッケージのインストール: {', '.join(missing_packages)}")
    
    try:
        # requirements.txt からインストール
        requirements_file = Path(__file__).parent.parent / "requirements.txt"
        if requirements_file.exists():
            print("requirements.txt からインストール中...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ パッケージインストール完了")
                return True
            else:
                print(f"❌ インストールエラー: {result.stderr}")
                return False
        else:
            print("❌ requirements.txt が見つかりません")
            return False
            
    except Exception as e:
        print(f"❌ インストール中にエラー: {e}")
        return False

def check_gpu_environment() -> Dict[str, any]:
    """GPU環境チェック"""
    print("🎮 GPU環境チェック...")
    
    gpu_info = {
        'cuda_available': False,
        'gpu_count': 0,
        'gpu_name': None,
        'gpu_memory_gb': 0,
        'rtx3080_detected': False,
        'cuda_version': None
    }
    
    try:
        import torch
        
        gpu_info['cuda_available'] = torch.cuda.is_available()
        
        if gpu_info['cuda_available']:
            gpu_info['gpu_count'] = torch.cuda.device_count()
            gpu_info['gpu_name'] = torch.cuda.get_device_name(0)
            gpu_info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
            gpu_info['cuda_version'] = torch.version.cuda
            
            # RTX3080検出
            if "RTX 3080" in gpu_info['gpu_name'] or "RTX3080" in gpu_info['gpu_name']:
                gpu_info['rtx3080_detected'] = True
                print(f"⚡ RTX3080検出: {gpu_info['gpu_name']}")
                print(f"💾 VRAM: {gpu_info['gpu_memory_gb']:.1f} GB")
            else:
                print(f"🎮 GPU: {gpu_info['gpu_name']}")
                print(f"💾 VRAM: {gpu_info['gpu_memory_gb']:.1f} GB")
            
            print(f"🔧 CUDA: {gpu_info['cuda_version']}")
            
        else:
            print("⚠️ CUDA が利用できません（CPU モードで動作）")
            
    except ImportError:
        print("❌ PyTorch がインストールされていません")
    except Exception as e:
        print(f"❌ GPU環境チェックエラー: {e}")
    
    return gpu_info

def create_config_file(gpu_info: Dict[str, any]) -> bool:
    """設定ファイル作成"""
    print("⚙️ 設定ファイル作成...")
    
    config = {
        'system_info': {
            'platform': platform.system(),
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'gpu_info': gpu_info
        },
        'nkat_parameters': {
            'nkat_dimension': 32,
            'nkat_precision': 150,
            'nkat_max_terms': 4096,
            'nkat_epsilon': 1e-50,
            'riemann_critical_line_start': 0.5,
            'riemann_critical_line_end': 100.0,
            'riemann_zero_search_precision': 1e-30,
            'riemann_max_zeros': 1000,
            'theta_ij': 1e-35,
            'c_star_algebra_dim': 256,
            'hilbert_space_dim': 512,
            'spectral_triple_dim': 128
        },
        'gpu_optimization': {
            'gpu_batch_size': 2048 if gpu_info['rtx3080_detected'] else 1024,
            'gpu_memory_limit_gb': 9.0 if gpu_info['rtx3080_detected'] else 6.0,
            'use_mixed_precision': True,
            'cuda_streams': 4,
            'rtx3080_optimizations': gpu_info['rtx3080_detected']
        },
        'monitoring': {
            'monitoring_interval_seconds': 1.0,
            'enable_gpu_monitoring': gpu_info['cuda_available'],
            'enable_cpu_monitoring': True,
            'log_level': 20  # INFO
        },
        'checkpoint': {
            'checkpoint_interval_seconds': 300,
            'auto_save_enabled': True,
            'max_checkpoint_files': 10,
            'checkpoint_compression': True
        }
    }
    
    try:
        config_file = Path(__file__).parent.parent / "nkat_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 設定ファイル作成: {config_file}")
        return True
        
    except Exception as e:
        print(f"❌ 設定ファイル作成エラー: {e}")
        return False

def test_system_components() -> bool:
    """システムコンポーネントテスト"""
    print("🧪 システムコンポーネントテスト...")
    
    try:
        # 基本インポートテスト
        print("  📦 基本モジュールインポートテスト...")
        import numpy as np
        import torch
        import matplotlib.pyplot as plt
        import streamlit as st
        print("  ✅ 基本モジュールインポート成功")
        
        # GPU テスト
        if torch.cuda.is_available():
            print("  🎮 GPU計算テスト...")
            x = torch.randn(100, 100, device='cuda')
            y = torch.matmul(x, x.T)
            result = y.cpu().numpy()
            print(f"  ✅ GPU計算テスト成功 (結果形状: {result.shape})")
        
        # 高精度計算テスト
        print("  🔢 高精度計算テスト...")
        import mpmath as mp
        mp.mp.dps = 50
        pi_high_precision = mp.pi
        print(f"  ✅ 高精度π計算: {str(pi_high_precision)[:20]}...")
        
        # ファイルI/Oテスト
        print("  💾 ファイルI/Oテスト...")
        import h5py
        test_file = Path(__file__).parent.parent / "test_checkpoint.h5"
        with h5py.File(test_file, 'w') as f:
            f.create_dataset('test_data', data=np.random.randn(10, 10))
        test_file.unlink()  # テストファイル削除
        print("  ✅ ファイルI/Oテスト成功")
        
        print("✅ 全システムコンポーネントテスト成功")
        return True
        
    except Exception as e:
        print(f"❌ システムコンポーネントテストエラー: {e}")
        return False

def start_streamlit_server() -> bool:
    """Streamlit サーバー起動"""
    print("🌐 Streamlit ダッシュボード起動...")
    
    try:
        # メインシステムファイルのパス
        main_system_file = Path(__file__).parent.parent / "src" / "nkat_riemann_ultimate_precision_system.py"
        
        if not main_system_file.exists():
            print(f"❌ メインシステムファイルが見つかりません: {main_system_file}")
            return False
        
        # Streamlit コマンド構築
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(main_system_file),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ]
        
        print("Streamlit サーバー起動中...")
        print(f"URL: http://localhost:8501")
        print("終了するには Ctrl+C を押してください")
        
        # サーバー起動
        subprocess.run(cmd)
        
        return True
        
    except KeyboardInterrupt:
        print("\n⏹️ ユーザーによる停止")
        return True
    except Exception as e:
        print(f"❌ Streamlit サーバー起動エラー: {e}")
        return False

def main():
    """メイン関数"""
    print_header()
    
    # 1. Python バージョンチェック
    if not check_python_version():
        print("❌ Python バージョンが不適切です")
        sys.exit(1)
    
    print()
    
    # 2. 必要パッケージチェック
    packages_ok, missing_packages = check_required_packages()
    
    if not packages_ok:
        print(f"\n⚠️ 不足パッケージ: {', '.join(missing_packages)}")
        
        # 自動インストール試行
        if input("自動インストールを試行しますか？ (y/N): ").lower() == 'y':
            if install_missing_packages(missing_packages):
                print("✅ パッケージインストール完了")
                # 再チェック
                packages_ok, _ = check_required_packages()
            else:
                print("❌ パッケージインストール失敗")
                sys.exit(1)
        else:
            print("手動でパッケージをインストールしてください:")
            print(f"pip install {' '.join(missing_packages)}")
            sys.exit(1)
    
    print()
    
    # 3. GPU環境チェック
    gpu_info = check_gpu_environment()
    
    print()
    
    # 4. 設定ファイル作成
    if not create_config_file(gpu_info):
        print("❌ 設定ファイル作成失敗")
        sys.exit(1)
    
    print()
    
    # 5. システムコンポーネントテスト
    if not test_system_components():
        print("❌ システムコンポーネントテスト失敗")
        sys.exit(1)
    
    print()
    
    # 6. RTX3080最適化有効化
    if gpu_info['rtx3080_detected']:
        print("⚡ RTX3080専用最適化を有効化...")
        os.environ['NKAT_RTX3080_OPTIMIZATION'] = '1'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # 非同期実行
        print("✅ RTX3080最適化設定完了")
        print()
    
    # 7. システム起動
    print("🚀 NKAT リーマン予想解析システム起動...")
    print()
    
    if not start_streamlit_server():
        print("❌ システム起動失敗")
        sys.exit(1)

if __name__ == "__main__":
    main() 