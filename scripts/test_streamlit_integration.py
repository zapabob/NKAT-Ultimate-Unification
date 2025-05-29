#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 Streamlit監視ダッシュボード統合テスト
NKAT GPU解析とStreamlit監視の統合動作確認

Author: NKAT Research Team
Date: 2025-01-24
Version: 1.0
"""

import sys
import os
import time
import subprocess
import threading
import requests
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

def test_dependencies():
    """依存関係のテスト"""
    print("📦 依存関係テスト開始")
    print("=" * 40)
    
    required_modules = [
        ('streamlit', 'Streamlit'),
        ('plotly', 'Plotly'),
        ('psutil', 'psutil'),
        ('GPUtil', 'GPUtil'),
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('tqdm', 'tqdm')
    ]
    
    success_count = 0
    
    for module_name, display_name in required_modules:
        try:
            __import__(module_name)
            print(f"✅ {display_name} - インストール済み")
            success_count += 1
        except ImportError:
            print(f"❌ {display_name} - 未インストール")
    
    print(f"\n📊 結果: {success_count}/{len(required_modules)} モジュールが利用可能")
    
    if success_count == len(required_modules):
        print("🎉 全ての依存関係が満たされています！")
        return True
    else:
        print("⚠️  一部の依存関係が不足しています")
        return False

def test_system_monitoring():
    """システム監視機能のテスト"""
    print("\n🖥️ システム監視機能テスト")
    print("=" * 40)
    
    try:
        import psutil
        import GPUtil
        import torch
        
        # CPU情報
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        print(f"💻 CPU使用率: {cpu_percent:.1f}%")
        print(f"💻 CPUコア数: {cpu_count}")
        
        # メモリ情報
        memory = psutil.virtual_memory()
        print(f"💾 メモリ使用率: {memory.percent:.1f}%")
        print(f"💾 メモリ使用量: {memory.used / 1e9:.1f} GB / {memory.total / 1e9:.1f} GB")
        
        # GPU情報
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🎮 GPU: {gpu_name}")
            print(f"🎮 GPU VRAM: {gpu_memory:.1f} GB")
            
            # GPUtilでの詳細情報
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    print(f"🎮 GPU使用率: {gpu.load * 100:.1f}%")
                    print(f"🎮 GPU温度: {gpu.temperature}°C")
                    print(f"🎮 GPUメモリ使用率: {(gpu.memoryUsed / gpu.memoryTotal) * 100:.1f}%")
            except Exception as e:
                print(f"⚠️  GPUtil情報取得エラー: {e}")
        else:
            print("⚠️  CUDA GPUが利用できません")
        
        print("✅ システム監視機能テスト完了")
        return True
        
    except Exception as e:
        print(f"❌ システム監視テストエラー: {e}")
        return False

def test_nkat_integration():
    """NKAT解析統合テスト"""
    print("\n🚀 NKAT解析統合テスト")
    print("=" * 40)
    
    try:
        from src.gpu.dirac_laplacian_analysis_gpu_recovery import (
            RecoveryGPUOperatorParameters,
            RecoveryGPUDiracLaplacianAnalyzer
        )
        
        # 軽量テスト設定
        params = RecoveryGPUOperatorParameters(
            dimension=3,
            lattice_size=4,  # 非常に小さなサイズ
            theta=0.01,
            kappa=0.05,
            mass=0.1,
            coupling=1.0,
            recovery_enabled=False,  # テスト用にRecovery無効
            max_eigenvalues=5,       # 少ない固有値数
        )
        
        print(f"📊 テスト設定: {params.dimension}次元, 格子{params.lattice_size}")
        
        # アナライザー初期化テスト
        analyzer = RecoveryGPUDiracLaplacianAnalyzer(params)
        print("✅ アナライザー初期化成功")
        
        # ガンマ行列構築テスト
        gamma_matrices = analyzer.gamma_matrices
        print(f"✅ ガンマ行列構築成功: {len(gamma_matrices)}個の{gamma_matrices[0].shape}行列")
        
        print("✅ NKAT解析統合テスト完了")
        return True
        
    except Exception as e:
        print(f"❌ NKAT解析統合テストエラー: {e}")
        return False

def test_streamlit_app():
    """Streamlitアプリのテスト"""
    print("\n🌐 Streamlitアプリテスト")
    print("=" * 40)
    
    app_path = Path(__file__).parent.parent / "src" / "gpu" / "streamlit_gpu_monitor.py"
    
    if not app_path.exists():
        print(f"❌ Streamlitアプリが見つかりません: {app_path}")
        return False
    
    print(f"📁 アプリパス確認: {app_path}")
    
    # Streamlitアプリの構文チェック
    try:
        import ast
        with open(app_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        ast.parse(source_code)
        print("✅ Streamlitアプリ構文チェック成功")
        
    except SyntaxError as e:
        print(f"❌ Streamlitアプリ構文エラー: {e}")
        return False
    except Exception as e:
        print(f"❌ Streamlitアプリチェックエラー: {e}")
        return False
    
    print("✅ Streamlitアプリテスト完了")
    return True

def test_file_structure():
    """ファイル構造のテスト"""
    print("\n📁 ファイル構造テスト")
    print("=" * 40)
    
    project_root = Path(__file__).parent.parent
    
    required_files = [
        "src/gpu/streamlit_gpu_monitor.py",
        "src/gpu/dirac_laplacian_analysis_gpu_recovery.py",
        "scripts/start_streamlit_dashboard.py",
        "scripts/test_tqdm_gpu_recovery.py",
        "start_nkat_dashboard.bat",
        "requirements.txt",
        "README.md"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - 見つかりません")
            missing_files.append(file_path)
    
    if not missing_files:
        print("✅ 全ての必要ファイルが存在します")
        return True
    else:
        print(f"⚠️  {len(missing_files)}個のファイルが不足しています")
        return False

def main():
    """メインテスト関数"""
    print("🧪 NKAT Streamlit監視ダッシュボード統合テスト")
    print("=" * 80)
    
    test_results = []
    
    # 1. ファイル構造テスト
    test_results.append(("ファイル構造", test_file_structure()))
    
    # 2. 依存関係テスト
    test_results.append(("依存関係", test_dependencies()))
    
    # 3. システム監視テスト
    test_results.append(("システム監視", test_system_monitoring()))
    
    # 4. NKAT解析統合テスト
    test_results.append(("NKAT解析統合", test_nkat_integration()))
    
    # 5. Streamlitアプリテスト
    test_results.append(("Streamlitアプリ", test_streamlit_app()))
    
    # 結果サマリー
    print("\n" + "=" * 80)
    print("📊 テスト結果サマリー")
    print("=" * 80)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 成功" if result else "❌ 失敗"
        print(f"{test_name:20} : {status}")
        if result:
            passed_tests += 1
    
    print(f"\n📈 総合結果: {passed_tests}/{total_tests} テスト成功")
    
    if passed_tests == total_tests:
        print("🎉 全てのテストが成功しました！")
        print("\n🚀 次のステップ:")
        print("1. start_nkat_dashboard.bat を実行してダッシュボードを起動")
        print("2. http://localhost:8501 でダッシュボードにアクセス")
        print("3. GPU/CPU監視とNKAT解析を体験")
        return True
    else:
        print("⚠️  一部のテストが失敗しました")
        print("\n🔧 トラブルシューティング:")
        print("1. requirements.txt の依存関係をインストール")
        print("2. Python 3.8以上を使用")
        print("3. 必要なファイルが存在するか確認")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 