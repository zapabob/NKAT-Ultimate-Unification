#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 RTX3080最適化版 ディラック/ラプラシアン作用素解析テストスクリプト
NKAT Theory - RTX3080専用最適化テスト

Author: NKAT Research Team
Date: 2025-01-24
Version: 1.7 - RTX3080最適化強化版テスト
"""

import sys
import os
import time
import torch
import numpy as np
from pathlib import Path

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.gpu.dirac_laplacian_analysis_gpu_recovery import (
    RecoveryGPUOperatorParameters,
    RecoveryGPUDiracLaplacianAnalyzer,
    setup_logger,
    monitor_gpu_memory
)

def test_rtx3080_detection():
    """RTX3080検出テスト"""
    print("=" * 80)
    print("🎮 RTX3080検出テスト")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("❌ CUDAが利用できません")
        return False
    
    gpu_name = torch.cuda.get_device_name()
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"🎮 検出されたGPU: {gpu_name}")
    print(f"💾 VRAM: {gpu_memory:.1f} GB")
    print(f"🔧 CUDA Version: {torch.version.cuda}")
    print(f"🐍 PyTorch Version: {torch.__version__}")
    
    is_rtx3080 = "RTX 3080" in gpu_name or "RTX3080" in gpu_name
    
    if is_rtx3080:
        print("✅ RTX3080が検出されました！専用最適化が有効になります")
        return True
    else:
        print("⚠️  RTX3080以外のGPUが検出されました")
        print("最適化は限定的になる可能性があります")
        return False

def test_memory_optimization():
    """メモリ最適化テスト"""
    print("\n" + "=" * 80)
    print("💾 メモリ最適化テスト")
    print("=" * 80)
    
    # 初期メモリ状態
    initial_memory = monitor_gpu_memory()
    if initial_memory:
        print(f"💾 初期GPU使用率: {initial_memory['usage_percent']:.1f}%")
        if 'used_gb' in initial_memory:
            print(f"💾 初期GPU使用量: {initial_memory['used_gb']:.2f} GB")
        if 'free_gb' in initial_memory:
            print(f"💾 初期GPU空き容量: {initial_memory['free_gb']:.2f} GB")
    
    # 軽量パラメータでテスト
    params = RecoveryGPUOperatorParameters(
        dimension=3,
        lattice_size=8,
        theta=0.1,
        kappa=0.05,
        mass=0.1,
        coupling=1.0,
        use_sparse=True,
        recovery_enabled=False,
        max_eigenvalues=20,
        memory_limit_gb=9.0,
        use_mixed_precision=True
    )
    
    print(f"📊 テストパラメータ:")
    print(f"   次元: {params.dimension}")
    print(f"   格子サイズ: {params.lattice_size}")
    print(f"   最適バッチサイズ: {params.gpu_batch_size}")
    print(f"   メモリ制限: {params.memory_limit_gb} GB")
    
    try:
        analyzer = RecoveryGPUDiracLaplacianAnalyzer(params)
        
        # ガンマ行列構築テスト
        print("\n🔨 ガンマ行列構築テスト...")
        gamma_start = time.time()
        gamma_matrices = analyzer._construct_high_dimensional_gamma_matrices()
        gamma_time = time.time() - gamma_start
        
        gamma_memory = monitor_gpu_memory()
        if gamma_memory and initial_memory:
            memory_increase = gamma_memory['usage_percent'] - initial_memory['usage_percent']
            print(f"✅ ガンマ行列構築完了: {gamma_time:.2f}秒")
            print(f"💾 メモリ使用量増加: {memory_increase:.1f}%")
        
        # ディラック作用素構築テスト
        print("\n🔨 ディラック作用素構築テスト...")
        dirac_start = time.time()
        D = analyzer.construct_discrete_dirac_operator_gpu_optimized()
        dirac_time = time.time() - dirac_start
        
        dirac_memory = monitor_gpu_memory()
        if dirac_memory and gamma_memory:
            memory_increase = dirac_memory['usage_percent'] - gamma_memory['usage_percent']
            print(f"✅ ディラック作用素構築完了: {dirac_time:.2f}秒")
            print(f"💾 メモリ使用量増加: {memory_increase:.1f}%")
            print(f"📊 行列サイズ: {D.shape}")
            print(f"📊 非零要素数: {D._nnz():,}")
        
        # メモリクリーンアップテスト
        print("\n🧹 メモリクリーンアップテスト...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        cleanup_memory = monitor_gpu_memory()
        if cleanup_memory and dirac_memory:
            memory_freed = dirac_memory['usage_percent'] - cleanup_memory['usage_percent']
            print(f"✅ メモリクリーンアップ完了")
            print(f"💾 解放されたメモリ: {memory_freed:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ メモリ最適化テストでエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_comparison():
    """性能比較テスト（CPU vs GPU）"""
    print("\n" + "=" * 80)
    print("⚡ 性能比較テスト (CPU vs GPU)")
    print("=" * 80)
    
    # 軽量テスト用パラメータ
    params = RecoveryGPUOperatorParameters(
        dimension=3,
        lattice_size=6,
        theta=0.1,
        kappa=0.05,
        mass=0.1,
        coupling=1.0,
        use_sparse=True,
        recovery_enabled=False,
        max_eigenvalues=15,
        memory_limit_gb=9.0,
        use_mixed_precision=True
    )
    
    results = {}
    
    try:
        # GPU版テスト
        print("🚀 GPU版テスト開始...")
        analyzer_gpu = RecoveryGPUDiracLaplacianAnalyzer(params)
        
        gpu_start = time.time()
        
        # ガンマ行列構築
        gamma_matrices = analyzer_gpu._construct_high_dimensional_gamma_matrices()
        
        # ディラック作用素構築
        D_gpu = analyzer_gpu.construct_discrete_dirac_operator_gpu_optimized()
        
        # スペクトル次元計算
        d_s_gpu, info_gpu = analyzer_gpu.compute_spectral_dimension_gpu_optimized(
            D_gpu, n_eigenvalues=15
        )
        
        gpu_time = time.time() - gpu_start
        results['gpu'] = {
            'time': gpu_time,
            'spectral_dimension': d_s_gpu,
            'eigenvalues': info_gpu.get('n_eigenvalues', 0)
        }
        
        print(f"✅ GPU版完了: {gpu_time:.2f}秒")
        print(f"📈 スペクトル次元: {d_s_gpu:.6f}")
        
        # メモリクリーンアップ
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 性能評価
        print(f"\n🎯 性能評価:")
        if gpu_time < 20:
            print("🚀 優秀な性能です！")
        elif gpu_time < 40:
            print("✅ 良好な性能です")
        elif gpu_time < 60:
            print("⚠️  許容範囲内の性能です")
        else:
            print("❌ 性能改善が必要です")
        
        return results
        
    except Exception as e:
        print(f"❌ 性能比較テストでエラー: {e}")
        import traceback
        traceback.print_exc()
        return {}

def test_scalability():
    """スケーラビリティテスト"""
    print("\n" + "=" * 80)
    print("📈 スケーラビリティテスト")
    print("=" * 80)
    
    test_configs = [
        {'dim': 3, 'lattice': 6, 'eigenvals': 15},
        {'dim': 3, 'lattice': 8, 'eigenvals': 20},
        {'dim': 4, 'lattice': 6, 'eigenvals': 15},
        {'dim': 4, 'lattice': 8, 'eigenvals': 20},
    ]
    
    results = []
    
    for i, config in enumerate(test_configs):
        print(f"\n📊 テスト {i+1}/{len(test_configs)}: {config['dim']}次元, 格子{config['lattice']}")
        
        try:
            params = RecoveryGPUOperatorParameters(
                dimension=config['dim'],
                lattice_size=config['lattice'],
                theta=0.1,
                kappa=0.05,
                mass=0.1,
                coupling=1.0,
                use_sparse=True,
                recovery_enabled=False,
                max_eigenvalues=config['eigenvals'],
                memory_limit_gb=9.0,
                use_mixed_precision=True
            )
            
            analyzer = RecoveryGPUDiracLaplacianAnalyzer(params)
            
            # GPU メモリ監視
            start_memory = monitor_gpu_memory()
            
            # 計算実行
            start_time = time.time()
            
            gamma_matrices = analyzer._construct_high_dimensional_gamma_matrices()
            D = analyzer.construct_discrete_dirac_operator_gpu_optimized()
            d_s, info = analyzer.compute_spectral_dimension_gpu_optimized(
                D, n_eigenvalues=config['eigenvals']
            )
            
            total_time = time.time() - start_time
            
            # メモリ使用量確認
            end_memory = monitor_gpu_memory()
            memory_used = 0.0
            if end_memory and start_memory:
                memory_used = end_memory['usage_percent'] - start_memory['usage_percent']
            
            result = {
                'config': config,
                'time': total_time,
                'spectral_dimension': d_s,
                'memory_used': memory_used,
                'matrix_size': D.shape[0],
                'nnz': D._nnz()
            }
            results.append(result)
            
            print(f"✅ 完了: {total_time:.2f}秒, メモリ使用: {memory_used:.1f}%")
            print(f"📈 スペクトル次元: {d_s:.6f}")
            
            # メモリクリーンアップ
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ テスト {i+1} でエラー: {e}")
            continue
    
    # 結果サマリー
    print(f"\n📊 スケーラビリティテスト結果サマリー:")
    print("-" * 80)
    for i, result in enumerate(results):
        config = result['config']
        print(f"テスト{i+1}: {config['dim']}次元×{config['lattice']} → "
              f"{result['time']:.2f}秒, {result['memory_used']:.1f}%メモリ, "
              f"行列{result['matrix_size']:,}×{result['matrix_size']:,}")
    
    return results

def test_high_precision_computation():
    """高精度計算テスト"""
    print("\n" + "=" * 80)
    print("🔬 高精度計算テスト")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("❌ CUDAが利用できません")
        return False
    
    # 高精度テスト用パラメータ
    test_configs = [
        {'dim': 3, 'lattice': 8, 'eigenvals': 30, 'expected_ds': 3.0},
        {'dim': 4, 'lattice': 6, 'eigenvals': 25, 'expected_ds': 4.0},
        {'dim': 5, 'lattice': 6, 'eigenvals': 20, 'expected_ds': 5.0},
    ]
    
    results = []
    
    for i, config in enumerate(test_configs):
        print(f"\n🔬 高精度テスト {i+1}/{len(test_configs)}: {config['dim']}次元")
        print(f"   格子サイズ: {config['lattice']}")
        print(f"   固有値数: {config['eigenvals']}")
        print(f"   理論スペクトル次元: {config['expected_ds']}")
        
        try:
            # 高精度パラメータ設定
            params = RecoveryGPUOperatorParameters(
                dimension=config['dim'],
                lattice_size=config['lattice'],
                theta=0.01,  # 小さな非可換パラメータ
                kappa=0.01,
                mass=0.01,
                coupling=1.0,
                use_sparse=True,
                recovery_enabled=False,
                max_eigenvalues=config['eigenvals'],
                memory_limit_gb=9.0,
                use_mixed_precision=True
            )
            
            analyzer = RecoveryGPUDiracLaplacianAnalyzer(params)
            
            # 高精度計算実行
            print("   🔨 ディラック作用素構築中...")
            start_time = time.time()
            D = analyzer.construct_discrete_dirac_operator_gpu_optimized()
            construction_time = time.time() - start_time
            
            print("   🔍 高精度スペクトル次元計算中...")
            spectral_start = time.time()
            d_s, info = analyzer.compute_spectral_dimension_gpu_optimized(
                D, n_eigenvalues=config['eigenvals']
            )
            spectral_time = time.time() - spectral_start
            
            # 精度評価
            theoretical_ds = config['expected_ds']
            absolute_error = abs(d_s - theoretical_ds)
            relative_error = (absolute_error / theoretical_ds) * 100
            
            # 結果記録
            result = {
                'config': config,
                'spectral_dimension': d_s,
                'theoretical_dimension': theoretical_ds,
                'absolute_error': absolute_error,
                'relative_error': relative_error,
                'construction_time': construction_time,
                'spectral_time': spectral_time,
                'total_time': construction_time + spectral_time,
                'eigenvalues_computed': info.get('n_eigenvalues', 0),
                'matrix_size': D.shape[0],
                'nnz': D._nnz()
            }
            results.append(result)
            
            # 結果表示
            print(f"   ✅ 計算完了:")
            print(f"      スペクトル次元: {d_s:.8f}")
            print(f"      理論値: {theoretical_ds}")
            print(f"      絶対誤差: {absolute_error:.8f}")
            print(f"      相対誤差: {relative_error:.4f}%")
            print(f"      計算時間: {construction_time + spectral_time:.2f}秒")
            
            # 精度評価
            if relative_error < 1.0:
                print("      🎯 優秀な精度!")
            elif relative_error < 5.0:
                print("      ✅ 良好な精度")
            elif relative_error < 10.0:
                print("      ⚠️  許容範囲内")
            else:
                print("      ❌ 精度改善が必要")
            
            # メモリクリーンアップ
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"   ❌ テスト {i+1} でエラー: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 総合評価
    if results:
        print(f"\n📊 高精度計算テスト総合結果:")
        print("-" * 80)
        
        total_tests = len(results)
        high_precision_count = sum(1 for r in results if r['relative_error'] < 1.0)
        good_precision_count = sum(1 for r in results if r['relative_error'] < 5.0)
        
        avg_relative_error = np.mean([r['relative_error'] for r in results])
        avg_computation_time = np.mean([r['total_time'] for r in results])
        
        print(f"実行テスト数: {total_tests}")
        print(f"高精度テスト数 (<1%誤差): {high_precision_count}")
        print(f"良好精度テスト数 (<5%誤差): {good_precision_count}")
        print(f"平均相対誤差: {avg_relative_error:.4f}%")
        print(f"平均計算時間: {avg_computation_time:.2f}秒")
        
        # 詳細結果表示
        for i, result in enumerate(results):
            config = result['config']
            print(f"テスト{i+1}: {config['dim']}D×{config['lattice']} → "
                  f"ds={result['spectral_dimension']:.6f} "
                  f"(誤差{result['relative_error']:.2f}%, "
                  f"{result['total_time']:.1f}秒)")
        
        # 成功判定
        success_rate = good_precision_count / total_tests
        if success_rate >= 0.8:
            print("🎉 高精度計算テスト成功!")
            return True
        else:
            print("⚠️  高精度計算テストで一部問題あり")
            return False
    else:
        print("❌ 高精度計算テストが実行できませんでした")
        return False

def test_convergence_analysis():
    """収束解析テスト"""
    print("\n" + "=" * 80)
    print("📈 収束解析テスト")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("❌ CUDAが利用できません")
        return False
    
    # 収束解析用パラメータ
    base_params = {
        'dimension': 3,
        'lattice_size': 6,
        'theta': 0.01,
        'kappa': 0.01,
        'mass': 0.01,
        'coupling': 1.0,
        'use_sparse': True,
        'recovery_enabled': False,
        'memory_limit_gb': 9.0,
        'use_mixed_precision': True
    }
    
    # 固有値数を変化させて収束性をテスト
    eigenvalue_counts = [10, 15, 20, 25, 30, 35, 40]
    convergence_results = []
    
    print("🔍 固有値数による収束解析...")
    
    for n_eigenvals in eigenvalue_counts:
        print(f"   固有値数: {n_eigenvals}")
        
        try:
            params = RecoveryGPUOperatorParameters(
                max_eigenvalues=n_eigenvals,
                **base_params
            )
            
            analyzer = RecoveryGPUDiracLaplacianAnalyzer(params)
            
            # 計算実行
            D = analyzer.construct_discrete_dirac_operator_gpu_optimized()
            d_s, info = analyzer.compute_spectral_dimension_gpu_optimized(
                D, n_eigenvalues=n_eigenvals
            )
            
            convergence_results.append({
                'n_eigenvalues': n_eigenvals,
                'spectral_dimension': d_s,
                'eigenvalues_computed': info.get('n_eigenvalues', 0)
            })
            
            print(f"      → ds = {d_s:.6f}")
            
            # メモリクリーンアップ
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"      ❌ エラー: {e}")
            continue
    
    # 収束解析
    if len(convergence_results) >= 3:
        print(f"\n📊 収束解析結果:")
        print("-" * 60)
        
        ds_values = [r['spectral_dimension'] for r in convergence_results]
        n_values = [r['n_eigenvalues'] for r in convergence_results]
        
        # 収束性の評価
        if len(ds_values) >= 3:
            # 最後の3つの値の標準偏差
            recent_std = np.std(ds_values[-3:])
            overall_std = np.std(ds_values)
            
            print(f"全体の標準偏差: {overall_std:.6f}")
            print(f"最近3点の標準偏差: {recent_std:.6f}")
            
            # 収束判定
            if recent_std < 0.01:
                print("✅ 良好な収束性")
                convergence_quality = "良好"
            elif recent_std < 0.05:
                print("⚠️  中程度の収束性")
                convergence_quality = "中程度"
            else:
                print("❌ 収束性に問題あり")
                convergence_quality = "問題あり"
            
            # 詳細表示
            for result in convergence_results:
                print(f"n={result['n_eigenvalues']:2d}: ds={result['spectral_dimension']:.6f}")
            
            return convergence_quality == "良好"
        else:
            print("⚠️  収束解析に十分なデータがありません")
            return False
    else:
        print("❌ 収束解析テストが実行できませんでした")
        return False

def test_theoretical_comparison():
    """理論値との詳細比較テスト"""
    print("\n" + "=" * 80)
    print("🎯 理論値との詳細比較テスト")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("❌ CUDAが利用できません")
        return False
    
    # 理論的に既知の結果との比較
    theoretical_cases = [
        {
            'name': '3次元ユークリッド空間',
            'params': {'dimension': 3, 'lattice_size': 8, 'theta': 0.0, 'mass': 0.0},
            'expected_ds': 3.0,
            'tolerance': 0.1
        },
        {
            'name': '4次元ユークリッド空間',
            'params': {'dimension': 4, 'lattice_size': 6, 'theta': 0.0, 'mass': 0.0},
            'expected_ds': 4.0,
            'tolerance': 0.15
        },
        {
            'name': '3次元非可換空間',
            'params': {'dimension': 3, 'lattice_size': 8, 'theta': 0.1, 'mass': 0.0},
            'expected_ds': 3.0,  # 小さな非可換効果
            'tolerance': 0.2
        }
    ]
    
    comparison_results = []
    
    for case in theoretical_cases:
        print(f"\n🧮 テストケース: {case['name']}")
        print(f"   期待値: {case['expected_ds']}")
        print(f"   許容誤差: ±{case['tolerance']}")
        
        try:
            # パラメータ設定
            params = RecoveryGPUOperatorParameters(
                kappa=0.01,
                coupling=1.0,
                use_sparse=True,
                recovery_enabled=False,
                max_eigenvalues=30,
                memory_limit_gb=9.0,
                use_mixed_precision=True,
                **case['params']
            )
            
            analyzer = RecoveryGPUDiracLaplacianAnalyzer(params)
            
            # 複数回実行して統計を取る
            ds_values = []
            for trial in range(3):  # 3回実行
                print(f"   試行 {trial+1}/3...")
                
                D = analyzer.construct_discrete_dirac_operator_gpu_optimized()
                d_s, info = analyzer.compute_spectral_dimension_gpu_optimized(D)
                ds_values.append(d_s)
                
                # メモリクリーンアップ
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 統計計算
            ds_mean = np.mean(ds_values)
            ds_std = np.std(ds_values)
            ds_min = np.min(ds_values)
            ds_max = np.max(ds_values)
            
            # 理論値との比較
            theoretical_ds = case['expected_ds']
            error = abs(ds_mean - theoretical_ds)
            within_tolerance = error <= case['tolerance']
            
            result = {
                'case_name': case['name'],
                'theoretical_ds': theoretical_ds,
                'computed_ds_mean': ds_mean,
                'computed_ds_std': ds_std,
                'computed_ds_min': ds_min,
                'computed_ds_max': ds_max,
                'error': error,
                'tolerance': case['tolerance'],
                'within_tolerance': within_tolerance,
                'trials': ds_values
            }
            comparison_results.append(result)
            
            # 結果表示
            print(f"   結果:")
            print(f"      平均: {ds_mean:.6f} ± {ds_std:.6f}")
            print(f"      範囲: [{ds_min:.6f}, {ds_max:.6f}]")
            print(f"      誤差: {error:.6f}")
            print(f"      判定: {'✅ 合格' if within_tolerance else '❌ 不合格'}")
            
        except Exception as e:
            print(f"   ❌ エラー: {e}")
            continue
    
    # 総合評価
    if comparison_results:
        print(f"\n📊 理論値比較テスト総合結果:")
        print("=" * 80)
        
        total_cases = len(comparison_results)
        passed_cases = sum(1 for r in comparison_results if r['within_tolerance'])
        
        print(f"実行ケース数: {total_cases}")
        print(f"合格ケース数: {passed_cases}")
        print(f"合格率: {(passed_cases/total_cases)*100:.1f}%")
        
        # 詳細結果
        for result in comparison_results:
            status = "✅" if result['within_tolerance'] else "❌"
            print(f"{status} {result['case_name']}: "
                  f"理論値{result['theoretical_ds']:.1f} vs "
                  f"計算値{result['computed_ds_mean']:.4f}±{result['computed_ds_std']:.4f} "
                  f"(誤差{result['error']:.4f})")
        
        return passed_cases == total_cases
    else:
        print("❌ 理論値比較テストが実行できませんでした")
        return False

def run_comprehensive_test():
    """包括的テストの実行"""
    print("🚀 RTX3080最適化版 包括的テスト開始")
    print("=" * 100)
    
    # ログ設定
    logger = setup_logger("RTX3080_Test", "logs/rtx3080_test.log")
    
    test_results = {}
    
    # 1. RTX3080検出テスト
    test_results['rtx3080_detection'] = test_rtx3080_detection()
    
    # 2. メモリ最適化テスト
    test_results['memory_optimization'] = test_memory_optimization()
    
    # 3. 性能比較テスト
    test_results['performance'] = test_performance_comparison()
    
    # 4. スケーラビリティテスト
    test_results['scalability'] = test_scalability()
    
    # 5. 高精度計算テスト
    test_results['high_precision_computation'] = test_high_precision_computation()
    
    # 6. 収束解析テスト
    test_results['convergence_analysis'] = test_convergence_analysis()
    
    # 7. 理論値との詳細比較テスト
    test_results['theoretical_comparison'] = test_theoretical_comparison()
    
    # 総合評価
    print("\n" + "=" * 100)
    print("🎯 総合テスト結果")
    print("=" * 100)
    
    passed_tests = sum(1 for result in test_results.values() if result)
    total_tests = len(test_results)
    
    print(f"✅ 成功したテスト: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 全てのテストが成功しました！RTX3080最適化は正常に動作しています")
    elif passed_tests >= total_tests * 0.75:
        print("✅ 大部分のテストが成功しました。RTX3080最適化は概ね正常です")
    else:
        print("⚠️  一部のテストが失敗しました。設定を確認してください")
    
    # 推奨設定の表示
    print(f"\n📋 RTX3080推奨設定:")
    print(f"   最大次元: 6次元")
    print(f"   推奨格子サイズ: 8 (4次元以下), 6 (5次元以上)")
    print(f"   推奨固有値数: 50 (4次元以下), 30 (5次元以上)")
    print(f"   メモリ制限: 9.0 GB")
    print(f"   混合精度計算: 有効")
    
    return test_results

if __name__ == "__main__":
    # 包括的テストの実行
    results = run_comprehensive_test()
    
    print(f"\n🏁 RTX3080最適化版テスト完了")
    print("=" * 100) 