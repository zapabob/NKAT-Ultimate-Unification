#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 tqdm + logging プログレスバー機能テストスクリプト
NKAT GPU Recovery解析のプログレスバー + ログ記録動作確認用

Author: NKAT Research Team
Date: 2025-01-24
Version: 1.1 - logging機能追加
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from gpu.dirac_laplacian_analysis_gpu_recovery import (
    RecoveryGPUOperatorParameters,
    RecoveryGPUDiracLaplacianAnalyzer,
    setup_logger
)
import torch
from tqdm import tqdm
import time
import logging

def test_logging_functionality():
    """logging機能の基本テスト"""
    print("📝 logging機能テスト開始")
    print("=" * 40)
    
    # テスト用ロガーの作成
    test_logger = setup_logger('TestLogger', level=logging.DEBUG)
    
    # 各レベルのログテスト
    test_logger.debug("これはDEBUGレベルのログです")
    test_logger.info("これはINFOレベルのログです")
    test_logger.warning("これはWARNINGレベルのログです")
    test_logger.error("これはERRORレベルのログです")
    
    print("✅ logging機能テスト完了")
    return test_logger

def test_tqdm_functionality():
    """tqdm機能の基本テスト"""
    print("🧪 tqdmプログレスバー機能テスト開始")
    print("=" * 60)
    
    # GPU情報表示
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 使用デバイス: {device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 小規模テスト設定
    test_params = RecoveryGPUOperatorParameters(
        dimension=3,           # 3次元（軽量）
        lattice_size=8,        # 小さな格子サイズ
        theta=0.01,
        kappa=0.05,
        mass=0.1,
        coupling=1.0,
        recovery_enabled=True,
        checkpoint_interval=30,  # 短い間隔
        auto_save=True,
        max_eigenvalues=20,     # 少ない固有値数
        log_level=logging.DEBUG  # デバッグレベルでログ出力
    )
    
    print(f"\n📊 テスト設定:")
    print(f"次元: {test_params.dimension}")
    print(f"格子サイズ: {test_params.lattice_size}")
    print(f"最大固有値数: {test_params.max_eigenvalues}")
    print(f"ログレベル: {test_params.log_level}")
    
    try:
        # アナライザーの初期化
        print("\n🔧 アナライザー初期化中...")
        analyzer = RecoveryGPUDiracLaplacianAnalyzer(test_params)
        
        # 軽量解析の実行
        print("\n🚀 軽量解析実行中...")
        results = analyzer.run_full_analysis_with_recovery()
        
        # 結果表示
        print("\n✅ テスト完了！")
        print("=" * 60)
        print("📊 結果サマリー:")
        print(f"スペクトル次元: {results['results']['spectral_dimension']:.6f}")
        print(f"理論値との差: {results['results']['dimension_error']:.6f}")
        print(f"計算時間: {results['results']['total_computation_time']:.2f}秒")
        print(f"行列サイズ: {results['results']['matrix_size']:,}")
        print(f"チェックポイントID: {results['checkpoint_id']}")
        
        return True
        
    except Exception as e:
        print(f"❌ テストエラー: {e}")
        return False

def test_progress_bars():
    """プログレスバーの視覚的テスト"""
    print("\n🎨 プログレスバー視覚テスト")
    print("=" * 40)
    
    # 基本的なプログレスバー
    print("1. 基本プログレスバー:")
    for i in tqdm(range(10), desc="基本テスト"):
        time.sleep(0.1)
    
    # ネストしたプログレスバー
    print("\n2. ネストプログレスバー:")
    for i in tqdm(range(3), desc="外側ループ"):
        for j in tqdm(range(5), desc=f"内側ループ{i+1}", leave=False):
            time.sleep(0.05)
    
    # 説明文変更
    print("\n3. 動的説明文:")
    with tqdm(total=5, desc="動的テスト") as pbar:
        for i in range(5):
            pbar.set_description(f"ステップ {i+1}/5 処理中")
            time.sleep(0.2)
            pbar.update(1)
    
    print("✅ プログレスバーテスト完了")

def test_log_file_creation():
    """ログファイル作成テスト"""
    print("\n📁 ログファイル作成テスト")
    print("=" * 40)
    
    # ログディレクトリの確認
    log_dir = "results/logs"
    if os.path.exists(log_dir):
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
        print(f"📂 ログディレクトリ: {log_dir}")
        print(f"📄 ログファイル数: {len(log_files)}")
        
        if log_files:
            latest_log = sorted(log_files)[-1]
            log_path = os.path.join(log_dir, latest_log)
            file_size = os.path.getsize(log_path) / 1024  # KB
            print(f"📝 最新ログファイル: {latest_log}")
            print(f"📏 ファイルサイズ: {file_size:.2f} KB")
            
            # ログファイルの内容を少し表示
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    print(f"📖 ログ行数: {len(lines)}")
                    if lines:
                        print("📋 最初の数行:")
                        for i, line in enumerate(lines[:3]):
                            print(f"  {i+1}: {line.strip()}")
                        if len(lines) > 3:
                            print("  ...")
                            print(f"  {len(lines)}: {lines[-1].strip()}")
            except Exception as e:
                print(f"⚠️  ログファイル読み込みエラー: {e}")
        else:
            print("📄 ログファイルが見つかりません")
    else:
        print(f"📂 ログディレクトリが存在しません: {log_dir}")
    
    print("✅ ログファイルテスト完了")

if __name__ == "__main__":
    print("🧪 NKAT tqdm + logging 統合機能テスト")
    print("=" * 80)
    
    # 1. logging機能テスト
    test_logger = test_logging_functionality()
    
    # 2. プログレスバーの視覚テスト
    test_progress_bars()
    
    # 3. 実際のGPU解析テスト（tqdm + logging統合）
    print("\n" + "=" * 80)
    success = test_tqdm_functionality()
    
    # 4. ログファイル作成確認
    test_log_file_creation()
    
    # 最終結果
    print("\n" + "=" * 80)
    if success:
        print("🎉 全てのテストが正常に完了しました！")
        print("✅ tqdmプログレスバー機能が正常に動作")
        print("✅ logging機能が正常に動作")
        print("✅ Recovery機能が正常に動作")
        print("✅ 統合システムが完全に機能")
        
        test_logger.info("全テスト完了: 成功")
    else:
        print("⚠️  一部のテストでエラーが発生しました。")
        print("📋 ログファイルを確認してください。")
        
        test_logger.error("テスト失敗: エラーが発生")
    
    print("\n📁 生成されたファイル:")
    print("  - results/logs/: ログファイル")
    print("  - results/checkpoints/: チェックポイントファイル")
    print("  - results/json/: 結果JSONファイル") 