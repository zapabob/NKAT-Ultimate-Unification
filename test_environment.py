#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTX3080環境テストスクリプト
========================

NKAT統一理論システムがRTX3080で正常動作するかテストします
- CUDA環境確認
- メモリテスト
- エンコーディングテスト
- 基本機能テスト

"""

import sys
import os
import torch
import platform
import psutil
from datetime import datetime

def test_python_environment():
    """Python環境テスト"""
    print("=" * 50)
    print("🐍 Python環境テスト")
    print("=" * 50)
    
    print(f"Python バージョン: {sys.version}")
    print(f"プラットフォーム: {platform.platform()}")
    print(f"システム: {platform.system()} {platform.release()}")
    print(f"アーキテクチャ: {platform.architecture()[0]}")
    
    # エンコーディング確認
    print(f"デフォルトエンコーディング: {sys.getdefaultencoding()}")
    print(f"ファイルシステムエンコーディング: {sys.getfilesystemencoding()}")
    
    # メモリ情報
    memory = psutil.virtual_memory()
    print(f"システムメモリ: {memory.total / (1024**3):.2f} GB")
    print(f"使用可能メモリ: {memory.available / (1024**3):.2f} GB")
    
    return True


def test_cuda_environment():
    """CUDA環境テスト"""
    print("\n" + "=" * 50)
    print("🚀 CUDA環境テスト")
    print("=" * 50)
    
    # PyTorch CUDA確認
    print(f"PyTorch バージョン: {torch.__version__}")
    print(f"CUDA利用可能: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDAが利用できません")
        return False
    
    # GPU情報
    device_count = torch.cuda.device_count()
    print(f"GPU数: {device_count}")
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  総メモリ: {props.total_memory / (1024**3):.2f} GB")
        print(f"  マルチプロセッサ数: {props.multi_processor_count}")
        print(f"  CUDA Capability: {props.major}.{props.minor}")
    
    # 現在のGPUメモリ使用量
    torch.cuda.empty_cache()
    current_memory = torch.cuda.memory_allocated(0) / (1024**3)
    cached_memory = torch.cuda.memory_reserved(0) / (1024**3)
    
    print(f"\n現在のメモリ使用量:")
    print(f"  割り当て済み: {current_memory:.2f} GB")
    print(f"  キャッシュ: {cached_memory:.2f} GB")
    
    return True


def test_memory_allocation():
    """メモリ割り当てテスト"""
    print("\n" + "=" * 50)
    print("💾 メモリ割り当てテスト")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("⚠️ CUDA未対応のためスキップ")
        return True
    
    try:
        # 小さなテンソルから開始
        sizes = [100, 500, 1000, 2000, 4000]
        
        for size in sizes:
            # complex128テンソル作成
            test_tensor = torch.zeros((size, size), dtype=torch.complex128, device='cuda')
            memory_used = test_tensor.element_size() * test_tensor.numel() / (1024**3)
            
            print(f"  {size}×{size} 行列: {memory_used:.3f} GB - ✅")
            
            # メモリ解放
            del test_tensor
            torch.cuda.empty_cache()
        
        # RTX3080限界テスト
        max_size = 16000  # 約4GB相当
        print(f"\n最大サイズテスト: {max_size}×{max_size}")
        
        try:
            large_tensor = torch.zeros((max_size, max_size), dtype=torch.complex64, device='cuda')
            memory_used = large_tensor.element_size() * large_tensor.numel() / (1024**3)
            print(f"  大型テンソル作成成功: {memory_used:.2f} GB - ✅")
            
            del large_tensor
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print(f"  大型テンソル作成失敗: メモリ不足 - ⚠️")
            torch.cuda.empty_cache()
    
    except Exception as e:
        print(f"❌ メモリテストエラー: {e}")
        return False
    
    return True


def test_encoding():
    """エンコーディングテスト"""
    print("\n" + "=" * 50)
    print("🔤 エンコーディングテスト")
    print("=" * 50)
    
    # 日本語・絵文字テスト
    test_strings = [
        "基本日本語テスト",
        "NKAT統一理論システム",
        "🚀 絵文字テスト 🔬",
        "数式: ∫ ∂μ Aμ = 0",
        "特殊記号: ∇ × ∂ ⊗ ≠ ≈"
    ]
    
    for i, test_str in enumerate(test_strings, 1):
        try:
            # ファイル書き込みテスト
            test_file = f"encoding_test_{i}.txt"
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(test_str)
            
            # ファイル読み込みテスト
            with open(test_file, 'r', encoding='utf-8') as f:
                read_str = f.read()
            
            if read_str == test_str:
                print(f"  テスト{i}: ✅ - {test_str[:20]}...")
            else:
                print(f"  テスト{i}: ❌ - 不一致")
            
            # テストファイル削除
            os.remove(test_file)
            
        except Exception as e:
            print(f"  テスト{i}: ❌ - {e}")
    
    return True


def test_basic_nkat_functionality():
    """基本NKAT機能テスト"""
    print("\n" + "=" * 50)
    print("🧮 基本NKAT機能テスト")
    print("=" * 50)
    
    try:
        # 基本的なテンソル操作
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 小さなハミルトニアン作成
        H_size = 100
        H = torch.zeros((H_size, H_size), dtype=torch.complex128, device=device)
        
        # 対角項設定
        H.fill_diagonal_(1.0)
        
        # 隣接項設定
        for i in range(H_size - 1):
            H[i, i+1] = -0.5
            H[i+1, i] = -0.5
        
        print(f"  ハミルトニアン作成: {H.shape} - ✅")
        
        # 固有値計算テスト
        eigenvals, _ = torch.linalg.eigh(H)
        eigenvals = torch.real(eigenvals)
        
        E_0 = float(eigenvals[0])
        E_1 = float(eigenvals[1])
        mass_gap = E_1 - E_0
        
        print(f"  固有値計算: ✅")
        print(f"    基底状態: {E_0:.6f}")
        print(f"    質量ギャップ: {mass_gap:.6f}")
        
        # メモリクリーンアップ
        del H, eigenvals
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ NKAT機能テストエラー: {e}")
        return False


def main():
    """メインテスト実行"""
    print("RTX3080環境適合性テスト")
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # テスト実行
    tests = [
        ("Python環境", test_python_environment),
        ("CUDA環境", test_cuda_environment),
        ("メモリ割り当て", test_memory_allocation),
        ("エンコーディング", test_encoding),
        ("NKAT基本機能", test_basic_nkat_functionality),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name}テストで例外発生: {e}")
            results[test_name] = False
    
    # 結果サマリー
    print("\n" + "=" * 50)
    print("📊 テスト結果サマリー")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 全テスト合格！RTX3080環境でNKATシステムが動作可能です")
        print("💡 rtx3080_run_config.py で本格実行してください")
    else:
        print("⚠️ 一部テスト失敗。環境設定を確認してください")
        print("💡 失敗したテストの詳細を確認し、必要に応じて設定を調整してください")
    
    print("=" * 50)


if __name__ == "__main__":
    main() 