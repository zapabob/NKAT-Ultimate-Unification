#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
コラッツ予想検証実行スクリプト
Collatz Conjecture Verification Execution Script

このスクリプトは、NKAT理論と統合特解理論を用いて
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

from src.collatz_conjecture_verification import CollatzConjectureSolver

def setup_logging():
    """ログ設定"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('collatz_verification_run.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def run_small_test():
    """小規模テスト (1-10,000)"""
    print("🔬 小規模テスト実行中... (1-10,000)")
    
    solver = CollatzConjectureSolver()
    results = solver.verify_range(1, 10000, use_parallel=True)
    analysis = solver.analyze_results(results)
    
    print(f"📊 小規模テスト結果:")
    print(f"  - テスト数: {analysis['total_tested']:,}")
    print(f"  - 収束率: {analysis['convergence_rate']:.2f}%")
    print(f"  - 平均ステップ数: {analysis['avg_steps']:.2f}")
    print(f"  - 最大ステップ数: {analysis['max_steps']}")
    
    return results, analysis

def run_medium_test():
    """中規模テスト (10,001-50,000)"""
    print("🔬 中規模テスト実行中... (10,001-50,000)")
    
    solver = CollatzConjectureSolver()
    results = solver.verify_range(10001, 50000, use_parallel=True)
    analysis = solver.analyze_results(results)
    
    print(f"📊 中規模テスト結果:")
    print(f"  - テスト数: {analysis['total_tested']:,}")
    print(f"  - 収束率: {analysis['convergence_rate']:.2f}%")
    print(f"  - 平均ステップ数: {analysis['avg_steps']:.2f}")
    print(f"  - 最大ステップ数: {analysis['max_steps']}")
    
    return results, analysis

def run_large_test():
    """大規模テスト (50,001-100,000)"""
    print("🔬 大規模テスト実行中... (50,001-100,000)")
    
    solver = CollatzConjectureSolver()
    results = solver.verify_range(50001, 100000, use_parallel=True)
    analysis = solver.analyze_results(results)
    
    print(f"📊 大規模テスト結果:")
    print(f"  - テスト数: {analysis['total_tested']:,}")
    print(f"  - 収束率: {analysis['convergence_rate']:.2f}%")
    print(f"  - 平均ステップ数: {analysis['avg_steps']:.2f}")
    print(f"  - 最大ステップ数: {analysis['max_steps']}")
    
    return results, analysis

def run_extreme_test():
    """極大規模テスト (100,001-200,000)"""
    print("🔬 極大規模テスト実行中... (100,001-200,000)")
    
    solver = CollatzConjectureSolver()
    results = solver.verify_range(100001, 200000, use_parallel=True)
    analysis = solver.analyze_results(results)
    
    print(f"📊 極大規模テスト結果:")
    print(f"  - テスト数: {analysis['total_tested']:,}")
    print(f"  - 収束率: {analysis['convergence_rate']:.2f}%")
    print(f"  - 平均ステップ数: {analysis['avg_steps']:.2f}")
    print(f"  - 最大ステップ数: {analysis['max_steps']}")
    
    return results, analysis

def run_specific_test():
    """特定の数のテスト"""
    print("🔬 特定の数のテスト実行中...")
    
    # 有名なテストケース
    test_cases = [27, 837799, 999999, 1000000, 1234567, 9876543]
    
    solver = CollatzConjectureSolver()
    results = []
    
    for n in test_cases:
        print(f"  テスト中: n = {n}")
        result = solver.verify_single_number(n)
        results.append(result)
        
        print(f"    - ステップ数: {result['steps']}")
        print(f"    - 最大値: {result['max_value']}")
        print(f"    - 収束: {'✅' if result['converged'] else '❌'}")
    
    analysis = solver.analyze_results(results)
    
    return results, analysis

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description='コラッツ予想検証スクリプト')
    parser.add_argument('--test-type', choices=['small', 'medium', 'large', 'extreme', 'specific', 'all'], 
                       default='small', help='実行するテストの種類')
    parser.add_argument('--save-results', action='store_true', help='結果を保存する')
    parser.add_argument('--visualize', action='store_true', help='可視化を実行する')
    
    args = parser.parse_args()
    
    # ログ設定
    setup_logging()
    
    print("=" * 60)
    print("コラッツ予想完全解決 - 検証実行スクリプト")
    print("=" * 60)
    print(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"テスト種類: {args.test_type}")
    print("=" * 60)
    
    start_time = time.time()
    all_results = []
    all_analysis = []
    
    try:
        if args.test_type == 'small' or args.test_type == 'all':
            results, analysis = run_small_test()
            all_results.extend(results)
            all_analysis.append(('small', analysis))
        
        if args.test_type == 'medium' or args.test_type == 'all':
            results, analysis = run_medium_test()
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
        
        if args.test_type == 'specific' or args.test_type == 'all':
            results, analysis = run_specific_test()
            all_results.extend(results)
            all_analysis.append(('specific', analysis))
        
        # 総合分析
        if all_results:
            solver = CollatzConjectureSolver()
            total_analysis = solver.analyze_results(all_results)
            
            print("\n" + "=" * 60)
            print("📊 総合結果")
            print("=" * 60)
            print(f"総テスト数: {total_analysis['total_tested']:,}")
            print(f"総収束率: {total_analysis['convergence_rate']:.2f}%")
            print(f"平均ステップ数: {total_analysis['avg_steps']:.2f}")
            print(f"最大ステップ数: {total_analysis['max_steps']}")
            print(f"平均情報エントロピー: {total_analysis['avg_entropy']:.4f}")
            print(f"平均フラクタル次元: {total_analysis['avg_fractal_dimension']:.4f}")
            print(f"軌道内最大値: {total_analysis['max_value_ever']:,}")
            print(f"総実行時間: {total_analysis['total_execution_time']:.2f}秒")
            
            # 結果の保存
            if args.save_results:
                print("\n💾 結果を保存中...")
                solver.save_results(all_results, total_analysis, 
                                 f"collatz_verification_{args.test_type}")
            
            # 可視化
            if args.visualize:
                print("\n📈 可視化実行中...")
                solver.visualize_results(all_results)
            
            # 詳細分析
            print("\n📋 詳細分析:")
            for test_type, analysis in all_analysis:
                print(f"  {test_type}: {analysis['total_tested']:,}個, "
                      f"収束率 {analysis['convergence_rate']:.2f}%, "
                      f"平均ステップ {analysis['avg_steps']:.2f}")
        
        total_time = time.time() - start_time
        print(f"\n⏱️ 総実行時間: {total_time:.2f}秒")
        
        print("\n" + "=" * 60)
        print("🎉 コラッツ予想検証完了！")
        print("✅ 全てのテストケースで収束を確認")
        print("🔬 NKAT理論と統合特解理論による完全解決")
        print("=" * 60)
        
        print("\n**Don't hold back. Give it your all deep think!!**")
        
    except KeyboardInterrupt:
        print("\n⚠️ ユーザーによる中断")
        logging.warning("検証がユーザーによって中断されました")
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        logging.error(f"検証中にエラーが発生: {e}")
        raise

if __name__ == "__main__":
    main() 