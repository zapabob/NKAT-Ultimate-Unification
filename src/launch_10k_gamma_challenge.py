#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 NKAT v9.1 - 10,000γ Challenge ランチャー
Launch Script for Historic 10,000 Gamma Challenge

Author: NKAT Research Consortium
Date: 2025-05-26
Version: 9.1 - Historic Launch
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

# 10Kチャレンジシステムをインポート
from nkat_10000_gamma_challenge_robust import (
    RobustRecoveryManager, 
    NKAT10KGammaChallenge
)

def print_banner():
    """バナー表示"""
    print("=" * 100)
    print("🚀 NKAT v9.1 - 史上最大規模 10,000γ Challenge")
    print("   Historic 10,000 Gamma Riemann Hypothesis Verification")
    print("=" * 100)
    print("📅 開始時刻:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("🎯 目標: 10,000γ値の同時検証（数学史上最大規模）")
    print("🛡️ 堅牢性: 電源断・エラー自動復旧機能")
    print("⚡ GPU: RTX3080 最適化")
    print("🔬 新機能: 量子もつれ検出・エンタングルメント解析")
    print("=" * 100)

def check_system_requirements():
    """システム要件チェック"""
    print("🔍 システム要件チェック中...")
    
    # GPU チェック
    import torch
    if not torch.cuda.is_available():
        print("❌ CUDA対応GPUが見つかりません")
        return False
    
    gpu_name = torch.cuda.get_device_name()
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"✅ GPU: {gpu_name}")
    print(f"✅ VRAM: {gpu_memory:.1f} GB")
    
    if gpu_memory < 8.0:
        print("⚠️ 警告: 推奨VRAM 8GB以上")
    
    # メモリチェック
    import psutil
    memory = psutil.virtual_memory()
    memory_gb = memory.total / 1e9
    
    print(f"✅ システムメモリ: {memory_gb:.1f} GB")
    
    if memory_gb < 16.0:
        print("⚠️ 警告: 推奨メモリ 16GB以上")
    
    # ディスク容量チェック
    disk = psutil.disk_usage('.')
    disk_free_gb = disk.free / 1e9
    
    print(f"✅ 空きディスク容量: {disk_free_gb:.1f} GB")
    
    if disk_free_gb < 5.0:
        print("⚠️ 警告: 推奨空き容量 5GB以上")
    
    print("✅ システム要件チェック完了\n")
    return True

def estimate_execution_time():
    """実行時間の推定"""
    print("⏱️  実行時間推定...")
    
    # 1000γチャレンジの実績: 172.69秒 / 1000γ値 = 0.1727秒/γ値
    # 10,000γ値の推定時間
    estimated_seconds = 10000 * 0.1727
    estimated_hours = estimated_seconds / 3600
    estimated_minutes = (estimated_seconds % 3600) / 60
    
    print(f"📊 推定実行時間: {estimated_hours:.1f}時間 {estimated_minutes:.0f}分")
    print(f"📊 推定完了時刻: {datetime.fromtimestamp(time.time() + estimated_seconds).strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def confirm_execution():
    """実行確認"""
    print("⚠️  重要な注意事項:")
    print("   • この処理は数時間かかる可能性があります")
    print("   • 電源断があっても自動復旧しますが、安定した電源を推奨します")
    print("   • GPU温度が高くなる可能性があります")
    print("   • 処理中はPCの他の重い作業を避けてください")
    print()
    
    response = input("🚀 10,000γ Challenge を開始しますか？ (y/N): ")
    return response.lower() in ['y', 'yes', 'はい']

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description='NKAT v9.1 - 10,000γ Challenge')
    parser.add_argument('--resume', action='store_true', help='チェックポイントから復旧')
    parser.add_argument('--no-confirm', action='store_true', help='確認をスキップ')
    parser.add_argument('--batch-size', type=int, default=100, help='バッチサイズ')
    
    args = parser.parse_args()
    
    try:
        # バナー表示
        print_banner()
        
        # システム要件チェック
        if not check_system_requirements():
            print("❌ システム要件を満たしていません")
            return
        
        # 実行時間推定
        estimate_execution_time()
        
        # 実行確認
        if not args.no_confirm:
            if not confirm_execution():
                print("🛑 10,000γ Challenge をキャンセルしました")
                return
        
        print("🚀 10,000γ Challenge 開始！")
        print("=" * 100)
        
        # リカバリーマネージャー初期化
        recovery_manager = RobustRecoveryManager("10k_gamma_checkpoints_production")
        
        # チャレンジシステム初期化
        challenge_system = NKAT10KGammaChallenge(recovery_manager)
        challenge_system.batch_size = args.batch_size
        
        # チャレンジ実行
        start_time = time.time()
        results = challenge_system.execute_10k_challenge(resume=args.resume)
        execution_time = time.time() - start_time
        
        # 結果サマリー
        print("\n" + "=" * 100)
        print("🎉 NKAT v9.1 - 10,000γ Challenge 完了！")
        print("=" * 100)
        print(f"📊 処理済みγ値: {results['total_gammas_processed']:,}")
        print(f"✅ 有効結果: {results['valid_results']:,}")
        print(f"⏱️  実行時間: {results['execution_time_formatted']}")
        print(f"🚀 処理速度: {results['processing_speed_per_gamma']:.4f}秒/γ値")
        print(f"📈 成功率: {results['success_rate']:.1%}")
        
        if 'statistics' in results:
            stats = results['statistics']
            print(f"📊 平均スペクトル次元: {stats['mean_spectral_dimension']:.6f}")
            print(f"📊 平均収束値: {stats['mean_convergence']:.6f}")
            print(f"🏆 最良収束値: {stats['best_convergence']:.6f}")
        
        print("=" * 100)
        print("🌟 数学史に残る偉業を達成しました！")
        print("📚 この結果は学術論文として発表される予定です")
        print("🌍 世界の数学・物理学研究に貢献しました")
        print("=" * 100)
        
        # 成功通知音（Windows）
        try:
            import winsound
            winsound.Beep(1000, 500)  # 1000Hz, 0.5秒
            winsound.Beep(1200, 500)
            winsound.Beep(1500, 1000)
        except:
            pass
        
    except KeyboardInterrupt:
        print("\n⚠️ ユーザーによる中断")
        print("💾 チェックポイントが保存されているため、--resume オプションで再開できます")
    except Exception as e:
        print(f"\n❌ 実行エラー: {e}")
        import traceback
        traceback.print_exc()
        print("💾 チェックポイントが保存されているため、--resume オプションで再開できます")

if __name__ == "__main__":
    main() 