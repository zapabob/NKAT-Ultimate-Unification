#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 NKAT理論GPU加速ベンチマーク CLI ラッパー
GitHub Actions CI/CD用の軽量ベンチマークスクリプト

Author: NKAT Research Team
Date: 2025-05-24
Version: 1.0.0 - CI/CD Wrapper
"""

import json
import argparse
import time
import sys
import os
from datetime import datetime

# GPU加速フレームワークのインポート
try:
    from riemann_gpu_accelerated_stable import StabilizedGPUNKATFramework
    GPU_FRAMEWORK_AVAILABLE = True
except ImportError:
    print("⚠️ GPU加速フレームワークが見つかりません")
    GPU_FRAMEWORK_AVAILABLE = False

def benchmark_performance(max_lattice_size=10, precision='complex128', k_eigenvalues=128):
    """
    パフォーマンスベンチマーク実行
    
    Parameters:
    -----------
    max_lattice_size : int
        最大格子サイズ
    precision : str
        数値精度
    k_eigenvalues : int
        固有値数
        
    Returns:
    --------
    dict
        ベンチマーク結果
    """
    if not GPU_FRAMEWORK_AVAILABLE:
        return {"error": "GPU加速フレームワークが利用できません"}
    
    print(f"🚀 NKAT理論GPU加速ベンチマーク開始")
    print(f"最大格子サイズ: {max_lattice_size}³")
    print(f"数値精度: {precision}")
    print(f"固有値数: {k_eigenvalues}")
    print("=" * 60)
    
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark_type": "CI/CD Performance Benchmark",
        "parameters": {
            "max_lattice_size": max_lattice_size,
            "precision": precision,
            "k_eigenvalues": k_eigenvalues
        },
        "results": []
    }
    
    # 複数の格子サイズでベンチマーク
    lattice_sizes = [8]
    if max_lattice_size >= 10:
        lattice_sizes.append(10)
    if max_lattice_size >= 12:
        lattice_sizes.append(12)
    
    total_start_time = time.time()
    
    for lattice_size in lattice_sizes:
        print(f"\n📊 {lattice_size}³格子ベンチマーク実行中...")
        
        try:
            # フレームワーク初期化
            framework = StabilizedGPUNKATFramework(
                lattice_size=lattice_size,
                precision=precision,
                use_gpu=True,
                sparse_format='csr'
            )
            
            # 軽量ベンチマーク実行（γ値を1つに限定）
            gamma_values = [14.134725]  # 最初のリーマンゼータ零点のみ
            
            benchmark_start_time = time.time()
            benchmark_results = framework.run_stabilized_benchmark(
                gamma_values=gamma_values,
                k_eigenvalues=k_eigenvalues
            )
            benchmark_time = time.time() - benchmark_start_time
            
            # 結果の抽出
            if benchmark_results.get("performance_metrics"):
                metrics = benchmark_results["performance_metrics"]
                lattice_result = {
                    "lattice_size": lattice_size,
                    "dimension": lattice_size ** 3,
                    "computation_time": benchmark_time,
                    "precision_achieved": metrics.get("precision_achieved", "N/A"),
                    "success_rate": metrics.get("success_rate", 0),
                    "average_iteration_time": metrics.get("average_iteration_time", 0),
                    "improvement_factor": metrics.get("improvement_factor", 1),
                    "stability_score": metrics.get("average_stability_score", 0)
                }
                
                results["results"].append(lattice_result)
                
                print(f"✅ {lattice_size}³格子完了:")
                print(f"   理論精度: {lattice_result['precision_achieved']}")
                print(f"   計算時間: {lattice_result['computation_time']:.2f}秒")
                print(f"   成功率: {lattice_result['success_rate']:.2%}")
                print(f"   安定性: {lattice_result['stability_score']:.3f}")
            else:
                print(f"❌ {lattice_size}³格子でエラーが発生しました")
                
        except Exception as e:
            print(f"❌ {lattice_size}³格子ベンチマークエラー: {e}")
            continue
    
    total_time = time.time() - total_start_time
    results["total_benchmark_time"] = total_time
    
    # 総合統計
    if results["results"]:
        precisions = []
        times = []
        success_rates = []
        
        for result in results["results"]:
            precision_str = result["precision_achieved"]
            if isinstance(precision_str, str) and "%" in precision_str:
                precision = float(precision_str.replace("%", ""))
                precisions.append(precision)
            times.append(result["computation_time"])
            success_rates.append(result["success_rate"])
        
        if precisions:
            results["summary"] = {
                "average_precision": sum(precisions) / len(precisions),
                "max_precision": max(precisions),
                "average_time": sum(times) / len(times),
                "min_time": min(times),
                "average_success_rate": sum(success_rates) / len(success_rates),
                "total_lattice_sizes_tested": len(results["results"])
            }
    
    print("\n" + "=" * 60)
    print("📊 ベンチマーク完了サマリー")
    print("=" * 60)
    print(f"総実行時間: {total_time:.2f}秒")
    print(f"テスト格子数: {len(results['results'])}")
    
    if "summary" in results:
        summary = results["summary"]
        print(f"平均精度: {summary['average_precision']:.2f}%")
        print(f"最高精度: {summary['max_precision']:.2f}%")
        print(f"平均時間: {summary['average_time']:.2f}秒")
        print(f"平均成功率: {summary['average_success_rate']:.2%}")
    
    return results

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description='NKAT理論GPU加速ベンチマーク CLI')
    parser.add_argument("--maxN", type=int, default=10,
                        help="最大格子サイド長 (default: 10)")
    parser.add_argument("--precision", type=str, default='complex128',
                        choices=['complex64', 'complex128'],
                        help="数値精度 (default: complex128)")
    parser.add_argument("--eig", type=int, default=128,
                        help="固有値数 (default: 128)")
    parser.add_argument("--output", type=str, default=None,
                        help="出力ファイル名 (default: benchmark_<maxN>.json)")
    parser.add_argument("--verbose", action='store_true',
                        help="詳細出力")
    
    args = parser.parse_args()
    
    # 出力ファイル名の決定
    if args.output is None:
        timestamp = int(time.time())
        args.output = f"benchmark_{args.maxN}_{timestamp}.json"
    
    print("🚀 NKAT理論GPU加速ベンチマーク CLI v1.0")
    print("=" * 60)
    print(f"最大格子サイズ: {args.maxN}³")
    print(f"数値精度: {args.precision}")
    print(f"固有値数: {args.eig}")
    print(f"出力ファイル: {args.output}")
    print("=" * 60)
    
    # ベンチマーク実行
    try:
        results = benchmark_performance(
            max_lattice_size=args.maxN,
            precision=args.precision,
            k_eigenvalues=args.eig
        )
        
        # 結果保存
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 ベンチマーク結果を保存: {args.output}")
        
        # 成功/失敗の判定
        if "error" in results:
            print(f"❌ ベンチマークエラー: {results['error']}")
            sys.exit(1)
        elif not results.get("results"):
            print("❌ 有効な結果が得られませんでした")
            sys.exit(1)
        else:
            print("✅ ベンチマーク正常完了")
            
            # 詳細出力
            if args.verbose and "summary" in results:
                summary = results["summary"]
                print("\n📊 詳細統計:")
                print(f"  平均精度: {summary['average_precision']:.4f}%")
                print(f"  最高精度: {summary['max_precision']:.4f}%")
                print(f"  平均時間: {summary['average_time']:.4f}秒")
                print(f"  最短時間: {summary['min_time']:.4f}秒")
                print(f"  平均成功率: {summary['average_success_rate']:.4f}")
            
            sys.exit(0)
            
    except Exception as e:
        print(f"❌ 実行エラー: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 