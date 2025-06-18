#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTX3080最適化実行設定
==================

NKAT統一理論システムのRTX3080向け最適化設定
- メモリ使用量制限
- バッチサイズ調整
- エンコーディング問題対策
- 電源断保護強化

"""

import os
import sys
import torch
from pathlib import Path

# RTX3080最適化設定
def setup_rtx3080_environment():
    """RTX3080環境設定"""
    
    # PyTorch CUDA設定
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # デバッグ用
    
    # Python エンコーディング設定
    if sys.platform.startswith('win'):
        # Windows環境でのUTF-8強制
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        os.environ['PYTHONUTF8'] = '1'
    
    # CUDA初期化とメモリクリア
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # GPU情報表示
        device = torch.cuda.get_device_properties(0)
        total_memory = device.total_memory / (1024**3)  # GB
        print(f"[INFO] GPU: {device.name}")
        print(f"[INFO] 総メモリ: {total_memory:.2f} GB")
        print(f"[INFO] 使用可能メモリ: {torch.cuda.get_device_properties(0).total_memory / (1024**3) * 0.8:.2f} GB")
    
    print("[INFO] RTX3080環境設定完了")


def get_rtx3080_config():
    """RTX3080向け設定パラメータ"""
    
    # プロジェクトルート設定
    current_dir = Path(__file__).parent
    sys.path.append(str(current_dir / 'src' / 'nkat_v12'))
    
    from clay_millennium_solver import ClayMillenniumConfig
    
    # RTX3080最適化設定
    config = ClayMillenniumConfig(
        # 基本設定
        device='cuda',
        dtype=torch.complex64,  # メモリ節約：complex128 → complex64
        
        # メモリ制限設定
        N_gauge=2,              # SU(2)群でメモリ節約
        lattice_sizes=[8, 12, 16],  # 格子サイズ縮小
        K_max=100,              # URT最大モード数削減
        batch_size=4,           # バッチサイズ削減
        memory_limit=0.8,       # GPUメモリ使用率80%制限
        
        # 精度設定（計算負荷軽減）
        proof_precision=1e-12,          # 1e-15 → 1e-12
        spectral_gap_threshold=0.05,    # 0.1 → 0.05
        target_nilpotency_precision=1e-12,  # 1e-14 → 1e-12
        
        # 計算回数制限
        max_iterations=500,     # 1000 → 500
        verification_levels=3,  # 5 → 3
        
        # 電源断保護強化
        checkpoint_interval=180,  # 3分間隔
        backup_count=5,          # バックアップ数削減
        
        # 非可換パラメータ（軽量化）
        theta=1e-69,            # 6.58e-70 → 1e-69
        alpha=0.2,              # 0.3 → 0.2
        
        # クロス検証削減
        cross_validation_folds=3,  # 5 → 3
        stability_tests=5,         # 10 → 5
    )
    
    return config


def main():
    """メイン実行関数"""
    print("="*60)
    print("🎯 NKAT統一理論 RTX3080最適化実行システム")
    print("="*60)
    
    # 環境設定
    setup_rtx3080_environment()
    
    # 設定取得
    config = get_rtx3080_config()
    
    try:
        # Clay Millennium Solver実行
        print("\n[INFO] Clay Millennium Problem解決システム開始...")
        
        from clay_millennium_solver import run_clay_millennium_solver
        results = run_clay_millennium_solver(config)
        
        # 結果保存
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f"rtx3080_results_{timestamp}.json"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n[OK] 結果保存完了: {result_file}")
        print(f"[OK] 証明レベル: {results.get('proof_level', 'Unknown')}")
        print(f"[OK] 総合スコア: {results.get('total_score', 0.0):.4f}")
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n[ERROR] CUDAメモリ不足: {e}")
        print("[SUGGESTION] lattice_sizes をさらに小さくしてください")
        print("[SUGGESTION] K_max を50以下に設定してください")
        
    except Exception as e:
        print(f"\n[ERROR] 実行エラー: {e}")
        print("[INFO] ログファイルを確認してください")
        
    finally:
        # メモリクリーンアップ
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\n[INFO] RTX3080実行システム終了")


if __name__ == "__main__":
    main() 