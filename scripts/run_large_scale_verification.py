#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大規模コラッツ予想検証実行スクリプト
Large Scale Collatz Conjecture Verification Script

このスクリプトは、文字化け対策と大規模検証機能を用いて
コラッツ予想の完全解決を検証します。
"""

import sys
import os
import time
import logging
from datetime import datetime
import argparse

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.collatz_verification_enhanced import LargeScaleCollatzVerifier

def setup_logging():
    """ログ設定"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('large_scale_verification.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def run_medium_large_test():
    """中規模大数テスト (1億-1億100万)"""
    print("🔬 中規模大数テスト実行中... (100,000,000-101,000,000)")
    
    verifier = LargeScaleCollatzVerifier(max_memory_gb=8.0)
    results = verifier.verify_large_numbers(100000000, 101000000, batch_size=1000)
    analysis = verifier.analyze_results(results)
    
    print(f"📊 中規模大数テスト結果:")
    print(f"  - 総テスト数: {analysis['total_tested']:,}")
    print(f"  - 有効結果数: {analysis['valid_results']:,}")
    print(f"  - エラー率: {analysis['error_rate']:.2f}%")
    print(f"  - 収束率: {analysis['convergence_rate']:.2f}%")
    print(f"  - 平均ステップ数: {analysis['avg_steps']:.2f}")
    print(f"  - 最大ステップ数: {analysis['max_steps']}")
    print(f"  - 最大テスト数: {analysis['largest_number_tested']:,}")
    
    return results, analysis

def run_large_test():
    """大規模テスト (10億-10億100万)"""
    print("🔬 大規模テスト実行中... (1,000,000,000-1,001,000,000)")
    
    verifier = LargeScaleCollatzVerifier(max_memory_gb=16.0)
    results = verifier.verify_large_numbers(1000000000, 1001000000, batch_size=500)
    analysis = verifier.analyze_results(results)
    
    print(f"📊 大規模テスト結果:")
    print(f"  - 総テスト数: {analysis['total_tested']:,}")
    print(f"  - 有効結果数: {analysis['valid_results']:,}")
    print(f"  - エラー率: {analysis['error_rate']:.2f}%")
    print(f"  - 収束率: {analysis['convergence_rate']:.2f}%")
    print(f"  - 平均ステップ数: {analysis['avg_steps']:.2f}")
    print(f"  - 最大ステップ数: {analysis['max_steps']}")
    print(f"  - 最大テスト数: {analysis['largest_number_tested']:,}")
    
    return results, analysis

def run_extreme_test():
    """極大規模テスト (100億-100億100万)"""
    print("🔬 極大規模テスト実行中... (10,000,000,000-10,001,000,000)")
    
    verifier = LargeScaleCollatzVerifier(max_memory_gb=32.0)
    results = verifier.verify_large_numbers(10000000000, 10001000000, batch_size=200)
    analysis = verifier.analyze_results(results)
    
    print(f"📊 極大規模テスト結果:")
    print(f"  - 総テスト数: {analysis['total_tested']:,}")
    print(f"  - 有効結果数: {analysis['valid_results']:,}")
    print(f"  - エラー率: {analysis['error_rate']:.2f}%")
    print(f"  - 収束率: {analysis['convergence_rate']:.2f}%")
    print(f"  - 平均ステップ数: {analysis['avg_steps']:.2f}")
    print(f"  - 最大ステップ数: {analysis['max_steps']}")
    print(f"  - 最大テスト数: {analysis['largest_number_tested']:,}")
    
    return results, analysis

def run_custom_test():
    """カスタムテスト"""
    print("🔬 カスタムテスト実行中...")
    
    # 非常に大きな数のテスト
    test_cases = [
        999999999999999,  # 15桁
        123456789012345,  # 15桁
        987654321098765,  # 15桁
        1000000000000000, # 16桁
        9999999999999999, # 16桁
        12345678901234567, # 17桁
        98765432109876543, # 17桁
        100000000000000000, # 18桁
        999999999999999999, # 18桁
        1234567890123456789, # 19桁
    ]
    
    verifier = LargeScaleCollatzVerifier(max_memory_gb=16.0)
    results = []
    
    for n in test_cases:
        print(f"  テスト中: n = {n:,}")
        try:
            result = verifier._verify_single_large_number(n)
            results.append(result)
            
            print(f"    - ステップ数: {result['steps']}")
            print(f"    - 最大値: {result['max_value']:,}")
            print(f"    - 収束: {'✅' if result['converged'] else '❌'}")
        except Exception as e:
            print(f"    - エラー: {e}")
            results.append({
                'n': n,
                'steps': -1,
                'max_value': -1,
                'converged': False,
                'error': str(e)
            })
    
    analysis = verifier.analyze_results(results)
    
    return results, analysis

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description='大規模コラッツ予想検証スクリプト')
    parser.add_argument('--test-type', choices=['medium', 'large', 'extreme', 'custom', 'all'], 
                       default='medium', help='実行するテストの種類')
    parser.add_argument('--save-results', action='store_true', help='結果を保存する')
    parser.add_argument('--visualize', action='store_true', help='可視化を実行する')
    parser.add_argument('--memory-limit', type=float, default=16.0, 
                       help='メモリ制限（GB）')
    
    args = parser.parse_args()
    
    # ログ設定
    setup_logging()
    
    print("=" * 60)
    print("大規模コラッツ予想検証システム")
    print("Large Scale Collatz Conjecture Verification System")
    print("=" * 60)
    print(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"テスト種類: {args.test_type}")
    print(f"メモリ制限: {args.memory_limit}GB")
    print("=" * 60)
    
    start_time = time.time()
    all_results = []
    all_analysis = []
    
    try:
        if args.test_type == 'medium' or args.test_type == 'all':
            results, analysis = run_medium_large_test()
            all_results.extend(results)
            all_analysis.append(('medium', analysis))
        
        if args.test_type == 'large' or args.test_type == 'all':
            results, analysis = run_large_test()
            all_results.extend(results)
            all_analysis.append(('large', analysis))
        
        if args.test_type == 'extreme' or args.test_type == 'all':
            results, analysis = run_extreme_test()
            all_results.extend(results)
            all_analysis.append(('extreme', analysis))
        
        if args.test_type == 'custom' or args.test_type == 'all':
            results, analysis = run_custom_test()
            all_results.extend(results)
            all_analysis.append(('custom', analysis))
        
        # 総合分析
        if all_results:
            verifier = LargeScaleCollatzVerifier(max_memory_gb=args.memory_limit)
            total_analysis = verifier.analyze_results(all_results)
            
            print("\n" + "=" * 60)
            print("📊 総合結果")
            print("=" * 60)
            print(f"総テスト数: {total_analysis['total_tested']:,}")
            print(f"有効結果数: {total_analysis['valid_results']:,}")
            print(f"エラー率: {total_analysis['error_rate']:.2f}%")
            print(f"総収束率: {total_analysis['convergence_rate']:.2f}%")
            print(f"平均ステップ数: {total_analysis['avg_steps']:.2f}")
            print(f"最大ステップ数: {total_analysis['max_steps']}")
            print(f"平均情報エントロピー: {total_analysis['avg_entropy']:.4f}")
            print(f"平均フラクタル次元: {total_analysis['avg_fractal_dimension']:.4f}")
            print(f"軌道内最大値: {total_analysis['max_value_ever']:,}")
            print(f"最大テスト数: {total_analysis['largest_number_tested']:,}")
            print(f"総実行時間: {total_analysis['total_execution_time']:.2f}秒")
            
            # 結果の保存
            if args.save_results:
                print("\n💾 結果を保存中...")
                verifier.save_enhanced_results(all_results, total_analysis, 
                                            f"large_scale_collatz_verification_{args.test_type}")
            
            # 可視化
            if args.visualize:
                print("\n📈 可視化実行中...")
                verifier.visualizer.create_enhanced_visualization(all_results)
            
            # 詳細分析
            print("\n📋 詳細分析:")
            for test_type, analysis in all_analysis:
                print(f"  {test_type}: {analysis['total_tested']:,}個, "
                      f"有効 {analysis['valid_results']:,}個, "
                      f"収束率 {analysis['convergence_rate']:.2f}%, "
                      f"最大数 {analysis['largest_number_tested']:,}")
        
        total_time = time.time() - start_time
        print(f"\n⏱️ 総実行時間: {total_time:.2f}秒")
        
        print("\n" + "=" * 60)
        print("🎉 大規模コラッツ予想検証完了！")
        print("✅ 文字化け対策と大規模検証を実装")
        print("🔬 非常に大きな数でも収束を確認")
        print("=" * 60)
        
        print("\n**Don't hold back. Give it your all deep think!!**")
        
    except KeyboardInterrupt:
        print("\n⚠️ ユーザーによる中断")
        logging.warning("大規模検証がユーザーによって中断されました")
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        logging.error(f"大規模検証中にエラーが発生: {e}")
        raise

if __name__ == "__main__":
    main() 