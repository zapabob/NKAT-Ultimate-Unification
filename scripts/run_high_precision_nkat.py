#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔬 高精度NKAT理論計算実行スクリプト
Non-Commutative Kolmogorov-Arnold Theory - 高精度版

Author: NKAT Research Team
Date: 2025-01-24
Version: 2.0 - 高精度計算特化版

主要機能:
- 高精度スペクトル次元計算
- 理論値との詳細比較
- 収束解析
- 数値安定性評価
- RTX3080最適化
"""

import sys
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.gpu.dirac_laplacian_analysis_gpu_recovery import (
    RecoveryGPUOperatorParameters,
    RecoveryGPUDiracLaplacianAnalyzer,
    setup_logger,
    monitor_gpu_memory
)

def setup_high_precision_environment():
    """高精度計算環境のセットアップ"""
    print("🔬 高精度計算環境セットアップ中...")
    
    # PyTorchの高精度設定
    torch.set_default_dtype(torch.float64)
    
    # CUDA最適化設定
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = False  # 高精度のためTF32無効
        torch.backends.cudnn.allow_tf32 = False
        
        # メモリ効率化
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.95)
        
        print(f"✅ CUDA高精度設定完了")
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  CUDA利用不可 - CPU計算になります")
    
    # 日本語フォント設定
    try:
        plt.rcParams['font.family'] = 'MS Gothic'
    except:
        plt.rcParams['font.family'] = 'DejaVu Sans'
    
    print("✅ 高精度計算環境セットアップ完了")

def run_high_precision_analysis(dimension: int, lattice_size: int, 
                               n_eigenvalues: int = 50, 
                               n_trials: int = 3) -> dict:
    """高精度解析の実行"""
    print(f"\n🔬 {dimension}次元高精度解析開始")
    print(f"格子サイズ: {lattice_size}, 固有値数: {n_eigenvalues}, 試行回数: {n_trials}")
    
    # 高精度パラメータ設定
    params = RecoveryGPUOperatorParameters(
        dimension=dimension,
        lattice_size=lattice_size,
        theta=0.001,  # 極小非可換パラメータ
        kappa=0.001,
        mass=0.001,
        coupling=1.0,
        use_sparse=True,
        recovery_enabled=False,
        max_eigenvalues=n_eigenvalues,
        memory_limit_gb=9.0,
        use_mixed_precision=False,  # 高精度のため混合精度無効
        log_level=20  # INFO レベル
    )
    
    analyzer = RecoveryGPUDiracLaplacianAnalyzer(params)
    
    # 複数回実行して統計を取る
    results = []
    spectral_dimensions = []
    computation_times = []
    
    for trial in range(n_trials):
        print(f"\n📊 試行 {trial+1}/{n_trials}")
        
        try:
            # GPU メモリ監視
            start_memory = monitor_gpu_memory()
            
            # 計算実行
            start_time = time.time()
            
            # ディラック作用素構築
            print("   🔨 ディラック作用素構築中...")
            construction_start = time.time()
            D = analyzer.construct_discrete_dirac_operator_gpu_optimized()
            construction_time = time.time() - construction_start
            
            # スペクトル次元計算
            print("   🔍 高精度スペクトル次元計算中...")
            spectral_start = time.time()
            d_s, info = analyzer.compute_spectral_dimension_gpu_optimized(
                D, n_eigenvalues=n_eigenvalues
            )
            spectral_time = time.time() - spectral_start
            
            total_time = time.time() - start_time
            
            # GPU メモリ監視
            end_memory = monitor_gpu_memory()
            memory_used = 0.0
            if end_memory and start_memory:
                memory_used = end_memory['usage_percent'] - start_memory['usage_percent']
            
            # 結果記録
            trial_result = {
                'trial': trial + 1,
                'spectral_dimension': d_s,
                'construction_time': construction_time,
                'spectral_time': spectral_time,
                'total_time': total_time,
                'memory_used': memory_used,
                'matrix_size': D.shape[0],
                'nnz': D._nnz(),
                'eigenvalues_computed': info.get('n_eigenvalues', 0),
                'analysis_info': info
            }
            results.append(trial_result)
            spectral_dimensions.append(d_s)
            computation_times.append(total_time)
            
            print(f"   ✅ 試行 {trial+1} 完了:")
            print(f"      スペクトル次元: {d_s:.8f}")
            print(f"      計算時間: {total_time:.2f}秒")
            print(f"      メモリ使用: {memory_used:.1f}%")
            
            # メモリクリーンアップ
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"   ❌ 試行 {trial+1} でエラー: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 統計解析
    if spectral_dimensions:
        ds_mean = np.mean(spectral_dimensions)
        ds_std = np.std(spectral_dimensions)
        ds_min = np.min(spectral_dimensions)
        ds_max = np.max(spectral_dimensions)
        
        time_mean = np.mean(computation_times)
        time_std = np.std(computation_times)
        
        # 理論値との比較
        theoretical_ds = float(dimension)
        absolute_error = abs(ds_mean - theoretical_ds)
        relative_error = (absolute_error / theoretical_ds) * 100
        
        # 統計的有意性評価
        confidence_interval = 1.96 * ds_std / np.sqrt(len(spectral_dimensions))  # 95%信頼区間
        
        summary = {
            'dimension': dimension,
            'lattice_size': lattice_size,
            'n_eigenvalues': n_eigenvalues,
            'n_trials': len(spectral_dimensions),
            'spectral_dimension_mean': ds_mean,
            'spectral_dimension_std': ds_std,
            'spectral_dimension_min': ds_min,
            'spectral_dimension_max': ds_max,
            'confidence_interval': confidence_interval,
            'theoretical_dimension': theoretical_ds,
            'absolute_error': absolute_error,
            'relative_error': relative_error,
            'computation_time_mean': time_mean,
            'computation_time_std': time_std,
            'trial_results': results
        }
        
        # 結果表示
        print(f"\n📊 {dimension}次元高精度解析結果:")
        print(f"   スペクトル次元: {ds_mean:.8f} ± {ds_std:.8f}")
        print(f"   95%信頼区間: ±{confidence_interval:.8f}")
        print(f"   範囲: [{ds_min:.8f}, {ds_max:.8f}]")
        print(f"   理論値: {theoretical_ds}")
        print(f"   絶対誤差: {absolute_error:.8f}")
        print(f"   相対誤差: {relative_error:.4f}%")
        print(f"   平均計算時間: {time_mean:.2f} ± {time_std:.2f}秒")
        
        # 精度評価
        if relative_error < 0.5:
            print("   🎯 極めて高い精度!")
        elif relative_error < 1.0:
            print("   🎯 非常に高い精度!")
        elif relative_error < 2.0:
            print("   ✅ 高い精度")
        elif relative_error < 5.0:
            print("   ✅ 良好な精度")
        else:
            print("   ⚠️  精度改善が必要")
        
        return summary
    else:
        print(f"   ❌ {dimension}次元解析が実行できませんでした")
        return None

def run_convergence_study():
    """収束性研究"""
    print("\n" + "=" * 80)
    print("📈 収束性研究")
    print("=" * 80)
    
    # 固有値数による収束性の研究
    dimension = 3
    lattice_size = 8
    eigenvalue_counts = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    
    convergence_results = []
    
    for n_eigenvals in eigenvalue_counts:
        print(f"\n🔍 固有値数 {n_eigenvals} での計算...")
        
        try:
            result = run_high_precision_analysis(
                dimension=dimension,
                lattice_size=lattice_size,
                n_eigenvalues=n_eigenvals,
                n_trials=2  # 収束研究では試行回数を減らす
            )
            
            if result:
                convergence_results.append({
                    'n_eigenvalues': n_eigenvals,
                    'spectral_dimension': result['spectral_dimension_mean'],
                    'std': result['spectral_dimension_std'],
                    'relative_error': result['relative_error']
                })
                
                print(f"   → ds = {result['spectral_dimension_mean']:.6f} ± {result['spectral_dimension_std']:.6f}")
            
        except Exception as e:
            print(f"   ❌ エラー: {e}")
            continue
    
    # 収束解析
    if len(convergence_results) >= 5:
        print(f"\n📊 収束解析結果:")
        print("-" * 60)
        
        # 収束プロット用データ
        n_vals = [r['n_eigenvalues'] for r in convergence_results]
        ds_vals = [r['spectral_dimension'] for r in convergence_results]
        std_vals = [r['std'] for r in convergence_results]
        
        # 収束性の評価
        # 最後の5点の標準偏差
        recent_std = np.std(ds_vals[-5:])
        overall_std = np.std(ds_vals)
        
        print(f"全体の標準偏差: {overall_std:.6f}")
        print(f"最近5点の標準偏差: {recent_std:.6f}")
        
        # 収束判定
        if recent_std < 0.005:
            print("✅ 優秀な収束性")
        elif recent_std < 0.01:
            print("✅ 良好な収束性")
        elif recent_std < 0.02:
            print("⚠️  中程度の収束性")
        else:
            print("❌ 収束性に問題あり")
        
        # 詳細表示
        for result in convergence_results:
            print(f"n={result['n_eigenvalues']:2d}: "
                  f"ds={result['spectral_dimension']:.6f}±{result['std']:.6f} "
                  f"(誤差{result['relative_error']:.2f}%)")
        
        # 収束プロットの保存
        try:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.errorbar(n_vals, ds_vals, yerr=std_vals, marker='o', capsize=5)
            plt.axhline(y=3.0, color='r', linestyle='--', label='理論値')
            plt.xlabel('固有値数')
            plt.ylabel('スペクトル次元')
            plt.title('固有値数による収束性')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 2, 2)
            relative_errors = [r['relative_error'] for r in convergence_results]
            plt.plot(n_vals, relative_errors, 'o-')
            plt.xlabel('固有値数')
            plt.ylabel('相対誤差 (%)')
            plt.title('相対誤差の変化')
            plt.grid(True)
            
            plt.subplot(2, 2, 3)
            plt.plot(n_vals, std_vals, 's-', color='orange')
            plt.xlabel('固有値数')
            plt.ylabel('標準偏差')
            plt.title('計算精度の変化')
            plt.grid(True)
            
            plt.subplot(2, 2, 4)
            # 収束率の計算
            if len(ds_vals) > 1:
                convergence_rates = []
                for i in range(1, len(ds_vals)):
                    rate = abs(ds_vals[i] - ds_vals[i-1]) / abs(ds_vals[i-1])
                    convergence_rates.append(rate)
                
                plt.semilogy(n_vals[1:], convergence_rates, '^-', color='green')
                plt.xlabel('固有値数')
                plt.ylabel('収束率 (log scale)')
                plt.title('収束率の変化')
                plt.grid(True)
            
            plt.tight_layout()
            
            # 保存
            output_dir = Path("results/images")
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(output_dir / f"convergence_analysis_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 収束解析プロットを保存: {output_dir}/convergence_analysis_{timestamp}.png")
            
        except Exception as e:
            print(f"⚠️  プロット保存エラー: {e}")
        
        return convergence_results
    else:
        print("❌ 収束解析に十分なデータがありません")
        return []

def run_multi_dimensional_study():
    """多次元精度研究"""
    print("\n" + "=" * 80)
    print("🌐 多次元精度研究")
    print("=" * 80)
    
    # 複数次元での高精度計算
    test_cases = [
        {'dim': 3, 'lattice': 10, 'eigenvals': 50},
        {'dim': 4, 'lattice': 8, 'eigenvals': 40},
        {'dim': 5, 'lattice': 6, 'eigenvals': 30},
        {'dim': 6, 'lattice': 6, 'eigenvals': 25},
    ]
    
    multi_dim_results = []
    
    for case in test_cases:
        print(f"\n🌐 {case['dim']}次元高精度研究")
        
        try:
            result = run_high_precision_analysis(
                dimension=case['dim'],
                lattice_size=case['lattice'],
                n_eigenvalues=case['eigenvals'],
                n_trials=3
            )
            
            if result:
                multi_dim_results.append(result)
                
        except Exception as e:
            print(f"❌ {case['dim']}次元でエラー: {e}")
            continue
    
    # 多次元結果の分析
    if multi_dim_results:
        print(f"\n📊 多次元精度研究総合結果:")
        print("=" * 80)
        
        dimensions = [r['dimension'] for r in multi_dim_results]
        relative_errors = [r['relative_error'] for r in multi_dim_results]
        computation_times = [r['computation_time_mean'] for r in multi_dim_results]
        
        print(f"実行次元数: {len(multi_dim_results)}")
        print(f"平均相対誤差: {np.mean(relative_errors):.4f}%")
        print(f"平均計算時間: {np.mean(computation_times):.2f}秒")
        
        # 詳細結果
        for result in multi_dim_results:
            print(f"{result['dimension']}次元: "
                  f"ds={result['spectral_dimension_mean']:.6f}±{result['spectral_dimension_std']:.6f} "
                  f"(誤差{result['relative_error']:.2f}%, {result['computation_time_mean']:.1f}秒)")
        
        # 次元依存性プロット
        try:
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 3, 1)
            ds_means = [r['spectral_dimension_mean'] for r in multi_dim_results]
            ds_stds = [r['spectral_dimension_std'] for r in multi_dim_results]
            plt.errorbar(dimensions, ds_means, yerr=ds_stds, marker='o', capsize=5)
            plt.plot(dimensions, dimensions, 'r--', label='理論値')
            plt.xlabel('次元')
            plt.ylabel('スペクトル次元')
            plt.title('次元依存性')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 3, 2)
            plt.plot(dimensions, relative_errors, 's-', color='orange')
            plt.xlabel('次元')
            plt.ylabel('相対誤差 (%)')
            plt.title('精度の次元依存性')
            plt.grid(True)
            
            plt.subplot(2, 3, 3)
            plt.plot(dimensions, computation_times, '^-', color='green')
            plt.xlabel('次元')
            plt.ylabel('計算時間 (秒)')
            plt.title('計算時間の次元依存性')
            plt.grid(True)
            
            plt.subplot(2, 3, 4)
            matrix_sizes = [r['trial_results'][0]['matrix_size'] for r in multi_dim_results]
            plt.semilogy(dimensions, matrix_sizes, 'D-', color='purple')
            plt.xlabel('次元')
            plt.ylabel('行列サイズ (log scale)')
            plt.title('行列サイズの次元依存性')
            plt.grid(True)
            
            plt.subplot(2, 3, 5)
            confidence_intervals = [r['confidence_interval'] for r in multi_dim_results]
            plt.plot(dimensions, confidence_intervals, 'v-', color='red')
            plt.xlabel('次元')
            plt.ylabel('95%信頼区間')
            plt.title('統計的精度の次元依存性')
            plt.grid(True)
            
            plt.subplot(2, 3, 6)
            # 効率性指標（精度/計算時間）
            efficiency = [1.0/r['relative_error'] / r['computation_time_mean'] for r in multi_dim_results]
            plt.plot(dimensions, efficiency, 'h-', color='brown')
            plt.xlabel('次元')
            plt.ylabel('効率性指標')
            plt.title('計算効率性')
            plt.grid(True)
            
            plt.tight_layout()
            
            # 保存
            output_dir = Path("results/images")
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(output_dir / f"multi_dimensional_study_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 多次元研究プロットを保存: {output_dir}/multi_dimensional_study_{timestamp}.png")
            
        except Exception as e:
            print(f"⚠️  プロット保存エラー: {e}")
        
        return multi_dim_results
    else:
        print("❌ 多次元研究が実行できませんでした")
        return []

def save_comprehensive_results(convergence_results, multi_dim_results):
    """包括的結果の保存"""
    print("\n💾 包括的結果保存中...")
    
    # 結果ディレクトリの作成
    output_dir = Path("results/json")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 包括的結果の整理
    comprehensive_results = {
        'metadata': {
            'timestamp': timestamp,
            'version': '2.0',
            'description': 'NKAT高精度計算包括的結果',
            'gpu_info': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',
            'pytorch_version': torch.__version__
        },
        'convergence_study': convergence_results,
        'multi_dimensional_study': multi_dim_results,
        'summary': {
            'total_computations': len(convergence_results) + sum(len(r['trial_results']) for r in multi_dim_results),
            'dimensions_tested': list(set([r['dimension'] for r in multi_dim_results])),
            'eigenvalue_counts_tested': list(set([r['n_eigenvalues'] for r in convergence_results])),
            'average_relative_error': np.mean([r['relative_error'] for r in multi_dim_results]) if multi_dim_results else None,
            'best_precision_achieved': min([r['relative_error'] for r in multi_dim_results]) if multi_dim_results else None
        }
    }
    
    # JSON保存
    output_file = output_dir / f"nkat_high_precision_results_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"✅ 包括的結果を保存: {output_file}")
    
    # サマリーレポートの生成
    report_file = output_dir / f"nkat_precision_report_{timestamp}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("🔬 NKAT高精度計算レポート\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"実行日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n")
        f.write(f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}\n")
        f.write(f"PyTorch: {torch.__version__}\n\n")
        
        if multi_dim_results:
            f.write("📊 多次元精度研究結果:\n")
            f.write("-" * 60 + "\n")
            for result in multi_dim_results:
                f.write(f"{result['dimension']}次元: ")
                f.write(f"ds={result['spectral_dimension_mean']:.8f}±{result['spectral_dimension_std']:.8f} ")
                f.write(f"(相対誤差{result['relative_error']:.4f}%)\n")
            f.write(f"\n平均相対誤差: {np.mean([r['relative_error'] for r in multi_dim_results]):.4f}%\n")
            f.write(f"最高精度: {min([r['relative_error'] for r in multi_dim_results]):.4f}%\n\n")
        
        if convergence_results:
            f.write("📈 収束性研究結果:\n")
            f.write("-" * 60 + "\n")
            for result in convergence_results:
                f.write(f"固有値数{result['n_eigenvalues']:2d}: ")
                f.write(f"ds={result['spectral_dimension']:.6f} ")
                f.write(f"(誤差{result['relative_error']:.2f}%)\n")
            
            # 収束性評価
            ds_vals = [r['spectral_dimension'] for r in convergence_results]
            recent_std = np.std(ds_vals[-5:]) if len(ds_vals) >= 5 else np.std(ds_vals)
            f.write(f"\n収束性評価: {recent_std:.6f} (最近5点の標準偏差)\n")
    
    print(f"📄 精度レポートを保存: {report_file}")
    
    return output_file, report_file

def main():
    """メイン実行関数"""
    print("🔬 NKAT高精度計算システム v2.0")
    print("=" * 80)
    
    # 環境セットアップ
    setup_high_precision_environment()
    
    # ログ設定
    logger = setup_logger("NKAT_HighPrecision", "logs/nkat_high_precision.log")
    
    try:
        # 収束性研究
        convergence_results = run_convergence_study()
        
        # 多次元精度研究
        multi_dim_results = run_multi_dimensional_study()
        
        # 結果保存
        if convergence_results or multi_dim_results:
            output_file, report_file = save_comprehensive_results(convergence_results, multi_dim_results)
            
            print(f"\n🎉 高精度計算完了!")
            print(f"📊 結果ファイル: {output_file}")
            print(f"📄 レポート: {report_file}")
            
            # 最終サマリー
            if multi_dim_results:
                best_precision = min([r['relative_error'] for r in multi_dim_results])
                avg_precision = np.mean([r['relative_error'] for r in multi_dim_results])
                print(f"\n🎯 精度サマリー:")
                print(f"   最高精度: {best_precision:.4f}%")
                print(f"   平均精度: {avg_precision:.4f}%")
                print(f"   テスト次元数: {len(multi_dim_results)}")
        else:
            print("⚠️  計算結果が得られませんでした")
    
    except Exception as e:
        print(f"❌ 実行エラー: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 最終クリーンアップ
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("\n🏁 高精度計算システム終了")

if __name__ == "__main__":
    main() 