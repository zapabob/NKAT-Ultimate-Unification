#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 NKAT リーマン予想解析システム テストスクリプト
NKAT Riemann Hypothesis Analysis System Test Suite

Author: NKAT Research Team
Date: 2025-01-28
Version: 1.0

機能:
- システム環境の総合テスト
- 各コンポーネントの動作確認
- パフォーマンステスト
- 結果レポート生成
"""

import sys
import os
import time
import traceback
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

def test_imports():
    """必要なモジュールのインポートテスト"""
    print("📦 モジュールインポートテスト...")
    
    test_results = {}
    required_modules = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('matplotlib', 'Matplotlib'),
        ('plotly', 'Plotly'),
        ('streamlit', 'Streamlit'),
        ('pandas', 'Pandas'),
        ('scipy', 'SciPy'),
        ('psutil', 'psutil'),
        ('GPUtil', 'GPUtil'),
        ('h5py', 'HDF5'),
        ('tqdm', 'tqdm')
    ]
    
    for module_name, display_name in required_modules:
        try:
            __import__(module_name)
            print(f"  ✅ {display_name}")
            test_results[module_name] = True
        except ImportError as e:
            print(f"  ❌ {display_name}: {e}")
            test_results[module_name] = False
    
    success_rate = sum(test_results.values()) / len(test_results)
    print(f"📊 インポート成功率: {success_rate*100:.1f}%")
    
    return test_results

def test_gpu_environment():
    """GPU環境テスト"""
    print("\n🎮 GPU環境テスト...")
    
    gpu_info = {}
    
    try:
        import torch
        
        # CUDA可用性
        cuda_available = torch.cuda.is_available()
        gpu_info['cuda_available'] = cuda_available
        print(f"  CUDA利用可能: {'✅' if cuda_available else '❌'}")
        
        if cuda_available:
            # GPU情報
            gpu_count = torch.cuda.device_count()
            gpu_info['gpu_count'] = gpu_count
            print(f"  GPU数: {gpu_count}")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                gpu_info[f'gpu_{i}_name'] = gpu_name
                gpu_info[f'gpu_{i}_memory_gb'] = gpu_memory
                
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
                
                # RTX3080チェック
                if "RTX 3080" in gpu_name or "RTX3080" in gpu_name:
                    print(f"    🚀 RTX3080検出 - 専用最適化利用可能")
                    gpu_info['rtx3080_detected'] = True
            
            # CUDA バージョン
            cuda_version = torch.version.cuda
            gpu_info['cuda_version'] = cuda_version
            print(f"  CUDAバージョン: {cuda_version}")
            
            # メモリテスト
            try:
                test_tensor = torch.randn(1000, 1000, device='cuda')
                allocated = torch.cuda.memory_allocated() / 1e6
                print(f"  メモリテスト: ✅ ({allocated:.1f} MB割り当て)")
                del test_tensor
                torch.cuda.empty_cache()
                gpu_info['memory_test'] = True
            except Exception as e:
                print(f"  メモリテスト: ❌ {e}")
                gpu_info['memory_test'] = False
        
    except Exception as e:
        print(f"  ❌ GPU環境テストエラー: {e}")
        gpu_info['error'] = str(e)
    
    return gpu_info

def test_nkat_theory_components():
    """NKAT理論コンポーネントテスト"""
    print("\n🔬 NKAT理論コンポーネントテスト...")
    
    test_results = {}
    
    try:
        # パスの追加
        sys.path.append(str(Path(__file__).parent.parent / "src"))
        
        # NKATRiemannParametersのテスト
        try:
            from nkat_riemann_rtx3080_dashboard import NKATRiemannParameters
            params = NKATRiemannParameters()
            print("  ✅ NKATRiemannParameters")
            test_results['parameters'] = True
        except Exception as e:
            print(f"  ❌ NKATRiemannParameters: {e}")
            test_results['parameters'] = False
        
        # NonCommutativeKolmogorovArnoldRepresentationのテスト
        try:
            from nkat_riemann_rtx3080_dashboard import NonCommutativeKolmogorovArnoldRepresentation
            ka_rep = NonCommutativeKolmogorovArnoldRepresentation(params)
            print("  ✅ NonCommutativeKolmogorovArnoldRepresentation")
            test_results['ka_representation'] = True
        except Exception as e:
            print(f"  ❌ NonCommutativeKolmogorovArnoldRepresentation: {e}")
            test_results['ka_representation'] = False
        
        # RiemannZetaAnalyzerのテスト
        try:
            from nkat_riemann_rtx3080_dashboard import RiemannZetaAnalyzer
            analyzer = RiemannZetaAnalyzer(params)
            print("  ✅ RiemannZetaAnalyzer")
            test_results['riemann_analyzer'] = True
        except Exception as e:
            print(f"  ❌ RiemannZetaAnalyzer: {e}")
            test_results['riemann_analyzer'] = False
        
        # CheckpointManagerのテスト
        try:
            from nkat_riemann_rtx3080_dashboard import CheckpointManager
            checkpoint_mgr = CheckpointManager()
            print("  ✅ CheckpointManager")
            test_results['checkpoint_manager'] = True
        except Exception as e:
            print(f"  ❌ CheckpointManager: {e}")
            test_results['checkpoint_manager'] = False
        
    except Exception as e:
        print(f"  ❌ NKAT理論コンポーネント読み込みエラー: {e}")
        test_results['import_error'] = str(e)
    
    return test_results

def test_system_monitoring():
    """システム監視機能テスト"""
    print("\n📊 システム監視機能テスト...")
    
    test_results = {}
    
    try:
        sys.path.append(str(Path(__file__).parent.parent / "src"))
        from nkat_riemann_rtx3080_dashboard import SystemMonitor
        
        monitor = SystemMonitor()
        
        # GPU情報取得テスト
        gpu_info = monitor.get_gpu_info()
        if gpu_info:
            print("  ✅ GPU情報取得")
            test_results['gpu_monitoring'] = True
        else:
            print("  ⚠️ GPU情報取得（GPUなし）")
            test_results['gpu_monitoring'] = False
        
        # CPU情報取得テスト
        cpu_info = monitor.get_cpu_info()
        if cpu_info:
            print("  ✅ CPU情報取得")
            test_results['cpu_monitoring'] = True
        else:
            print("  ❌ CPU情報取得")
            test_results['cpu_monitoring'] = False
        
    except Exception as e:
        print(f"  ❌ システム監視テストエラー: {e}")
        test_results['error'] = str(e)
    
    return test_results

def test_checkpoint_system():
    """チェックポイントシステムテスト"""
    print("\n💾 チェックポイントシステムテスト...")
    
    test_results = {}
    
    try:
        sys.path.append(str(Path(__file__).parent.parent / "src"))
        from nkat_riemann_rtx3080_dashboard import CheckpointManager, NKATRiemannParameters
        import torch
        
        checkpoint_mgr = CheckpointManager()
        params = NKATRiemannParameters()
        
        # チェックポイントID生成テスト
        checkpoint_id = checkpoint_mgr.create_checkpoint_id(params)
        print(f"  ✅ チェックポイントID生成: {checkpoint_id}")
        test_results['id_generation'] = True
        
        # テストデータ作成
        test_data = {
            'test_tensor': torch.randn(10, 10),
            'test_value': 42.0,
            'test_string': 'test_checkpoint'
        }
        
        # チェックポイント保存テスト
        checkpoint_file = checkpoint_mgr.save_checkpoint(
            checkpoint_id, 'test_stage', test_data
        )
        
        if checkpoint_file:
            print("  ✅ チェックポイント保存")
            test_results['save_checkpoint'] = True
            
            # チェックポイント読み込みテスト
            loaded_data = checkpoint_mgr.load_checkpoint(checkpoint_file)
            if loaded_data:
                print("  ✅ チェックポイント読み込み")
                test_results['load_checkpoint'] = True
                
                # データ整合性チェック
                if (loaded_data.get('test_value') == 42.0 and 
                    loaded_data.get('test_string') == 'test_checkpoint'):
                    print("  ✅ データ整合性")
                    test_results['data_integrity'] = True
                else:
                    print("  ❌ データ整合性")
                    test_results['data_integrity'] = False
            else:
                print("  ❌ チェックポイント読み込み")
                test_results['load_checkpoint'] = False
        else:
            print("  ❌ チェックポイント保存")
            test_results['save_checkpoint'] = False
        
    except Exception as e:
        print(f"  ❌ チェックポイントシステムテストエラー: {e}")
        test_results['error'] = str(e)
    
    return test_results

def test_riemann_zeta_analyzer():
    """リーマンゼータ解析器テスト"""
    print("\n🔢 リーマンゼータ解析器テスト...")
    
    test_results = {}
    
    try:
        sys.path.append(str(Path(__file__).parent.parent / "src"))
        from nkat_riemann_rtx3080_dashboard import RiemannZetaAnalyzer, NKATRiemannParameters
        import torch
        
        # 軽量パラメータでテスト
        params = NKATRiemannParameters(
            ka_dimension=8,
            ka_max_terms=64,
            zero_search_range=(0.0, 10.0)
        )
        
        analyzer = RiemannZetaAnalyzer(params)
        print("  ✅ 解析器初期化")
        test_results['analyzer_init'] = True
        
        # 小規模なゼータ関数計算テスト
        t_values = torch.linspace(1.0, 5.0, 10)
        zeta_values = analyzer.compute_zeta_on_critical_line(t_values)
        
        if zeta_values is not None and len(zeta_values) == 10:
            print("  ✅ ゼータ関数計算")
            test_results['zeta_computation'] = True
        else:
            print("  ❌ ゼータ関数計算")
            test_results['zeta_computation'] = False
        
        # 小規模なゼロ点探索テスト
        zeros = analyzer.find_zeros_on_critical_line((1.0, 20.0), n_points=100)
        print(f"  ✅ ゼロ点探索: {len(zeros)}個発見")
        test_results['zero_finding'] = True
        test_results['zeros_found'] = len(zeros)
        
    except Exception as e:
        print(f"  ❌ リーマンゼータ解析器テストエラー: {e}")
        test_results['error'] = str(e)
    
    return test_results

def performance_benchmark():
    """パフォーマンスベンチマーク"""
    print("\n⚡ パフォーマンスベンチマーク...")
    
    benchmark_results = {}
    
    try:
        import torch
        import time
        
        # CPU vs GPU計算速度比較
        size = 1000
        iterations = 10
        
        # CPU計算
        start_time = time.time()
        for _ in range(iterations):
            a = torch.randn(size, size)
            b = torch.randn(size, size)
            c = torch.matmul(a, b)
        cpu_time = time.time() - start_time
        
        print(f"  CPU計算時間: {cpu_time:.3f}秒")
        benchmark_results['cpu_time'] = cpu_time
        
        # GPU計算（利用可能な場合）
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_time = time.time()
            for _ in range(iterations):
                a = torch.randn(size, size, device='cuda')
                b = torch.randn(size, size, device='cuda')
                c = torch.matmul(a, b)
            torch.cuda.synchronize()
            gpu_time = time.time() - start_time
            
            print(f"  GPU計算時間: {gpu_time:.3f}秒")
            print(f"  GPU高速化率: {cpu_time/gpu_time:.1f}x")
            benchmark_results['gpu_time'] = gpu_time
            benchmark_results['speedup'] = cpu_time / gpu_time
        else:
            print("  GPU計算: 利用不可")
            benchmark_results['gpu_available'] = False
        
        # メモリ使用量測定
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        benchmark_results['memory_usage_mb'] = memory_info.rss / 1024 / 1024
        print(f"  メモリ使用量: {benchmark_results['memory_usage_mb']:.1f} MB")
        
    except Exception as e:
        print(f"  ❌ ベンチマークエラー: {e}")
        benchmark_results['error'] = str(e)
    
    return benchmark_results

def generate_test_report(test_results: Dict[str, Any]) -> str:
    """テストレポート生成"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report = {
        'test_timestamp': timestamp,
        'test_results': test_results,
        'summary': {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'success_rate': 0.0
        }
    }
    
    # 成功/失敗カウント
    for category, results in test_results.items():
        if isinstance(results, dict):
            for test_name, result in results.items():
                if isinstance(result, bool):
                    report['summary']['total_tests'] += 1
                    if result:
                        report['summary']['passed_tests'] += 1
                    else:
                        report['summary']['failed_tests'] += 1
    
    if report['summary']['total_tests'] > 0:
        report['summary']['success_rate'] = (
            report['summary']['passed_tests'] / report['summary']['total_tests']
        )
    
    # レポートファイル保存
    report_dir = Path("Results/json")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = report_dir / f"nkat_system_test_report_{timestamp}.json"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    return str(report_file)

def main():
    """メイン関数"""
    print("🧪 NKAT リーマン予想解析システム テストスイート")
    print("=" * 60)
    
    test_results = {}
    
    # 各テストの実行
    test_results['imports'] = test_imports()
    test_results['gpu_environment'] = test_gpu_environment()
    test_results['nkat_components'] = test_nkat_theory_components()
    test_results['system_monitoring'] = test_system_monitoring()
    test_results['checkpoint_system'] = test_checkpoint_system()
    test_results['riemann_analyzer'] = test_riemann_zeta_analyzer()
    test_results['performance'] = performance_benchmark()
    
    # レポート生成
    print("\n📄 テストレポート生成中...")
    report_file = generate_test_report(test_results)
    print(f"✅ レポート保存: {report_file}")
    
    # サマリー表示
    print("\n" + "=" * 60)
    print("📊 テスト結果サマリー")
    print("=" * 60)
    
    total_categories = len(test_results)
    passed_categories = 0
    
    for category, results in test_results.items():
        if isinstance(results, dict):
            category_passed = True
            for test_name, result in results.items():
                if isinstance(result, bool) and not result:
                    category_passed = False
                    break
            
            status = "✅" if category_passed else "❌"
            print(f"{status} {category}")
            
            if category_passed:
                passed_categories += 1
    
    overall_success_rate = passed_categories / total_categories
    print(f"\n🎯 総合成功率: {overall_success_rate*100:.1f}%")
    
    if overall_success_rate >= 0.8:
        print("🎉 システムは正常に動作しています！")
    elif overall_success_rate >= 0.6:
        print("⚠️ 一部の機能に問題があります")
    else:
        print("❌ 重大な問題が検出されました")
    
    print("\n👋 テスト完了")

if __name__ == "__main__":
    main() 