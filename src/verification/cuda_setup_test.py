#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 CUDA環境セットアップ & テストスクリプト
NKAT超収束因子リーマン予想解析 - GPU環境検証

このスクリプトは以下を実行します:
1. CUDA環境の検出と確認
2. CuPy GPU計算テスト
3. PyTorch CUDA テスト
4. GPU メモリ情報表示
5. 簡単な性能ベンチマーク
"""

import sys
import os
import time
import numpy as np
from datetime import datetime

# Windows環境でのUnicodeエラー対策
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

def print_header():
    """ヘッダー表示"""
    print("🚀 CUDA環境セットアップ & テストスクリプト")
    print("📚 NKAT超収束因子リーマン予想解析 - GPU環境検証")
    print("🎮 Windows 11 + Python 3 + CUDA 12.x対応")
    print("=" * 80)

def test_cuda_availability():
    """CUDA環境の確認"""
    print("\n🔍 1. CUDA環境の検出と確認")
    print("-" * 60)
    
    cuda_results = {
        'cuda_available': False,
        'cupy_available': False,
        'pytorch_cuda': False,
        'gpu_count': 0,
        'gpu_devices': [],
        'errors': []
    }
    
    # 1. CuPy CUDA確認
    try:
        import cupy as cp
        cuda_results['cupy_available'] = True
        print("✅ CuPy CUDA利用可能")
        
        # GPU情報取得
        try:
            device = cp.cuda.Device()
            gpu_count = cp.cuda.runtime.getDeviceCount()
            cuda_results['gpu_count'] = gpu_count
            
            print(f"🎮 GPU数: {gpu_count}")
            
            for i in range(gpu_count):
                with cp.cuda.Device(i):
                    device_info = cp.cuda.runtime.getDeviceProperties(i)
                    memory_info = cp.cuda.runtime.memGetInfo()
                    
                    gpu_info = {
                        'id': i,
                        'name': device_info['name'].decode('utf-8'),
                        'compute_capability': f"{device_info['major']}.{device_info['minor']}",
                        'total_memory_gb': device_info['totalGlobalMem'] / 1024**3,
                        'free_memory_gb': memory_info[0] / 1024**3,
                        'used_memory_gb': (device_info['totalGlobalMem'] - memory_info[0]) / 1024**3
                    }
                    
                    cuda_results['gpu_devices'].append(gpu_info)
                    
                    print(f"   GPU {i}: {gpu_info['name']}")
                    print(f"   計算能力: {gpu_info['compute_capability']}")
                    print(f"   総メモリ: {gpu_info['total_memory_gb']:.2f} GB")
                    print(f"   利用可能: {gpu_info['free_memory_gb']:.2f} GB")
                    print(f"   使用中: {gpu_info['used_memory_gb']:.2f} GB")
            
        except Exception as e:
            error_msg = f"CuPy GPU情報取得エラー: {e}"
            print(f"⚠️ {error_msg}")
            cuda_results['errors'].append(error_msg)
            
    except ImportError as e:
        error_msg = f"CuPy未検出: {e}"
        print(f"❌ {error_msg}")
        cuda_results['errors'].append(error_msg)
        print("📦 インストール方法: pip install cupy-cuda12x")
    
    # 2. PyTorch CUDA確認
    try:
        import torch
        if torch.cuda.is_available():
            cuda_results['pytorch_cuda'] = True
            print("✅ PyTorch CUDA利用可能")
            
            pytorch_gpu_count = torch.cuda.device_count()
            print(f"🎮 PyTorch認識GPU数: {pytorch_gpu_count}")
            
            for i in range(pytorch_gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_props = torch.cuda.get_device_properties(i)
                
                print(f"   GPU {i}: {gpu_name}")
                print(f"   総メモリ: {gpu_props.total_memory / 1024**3:.2f} GB")
                print(f"   マルチプロセッサ数: {gpu_props.multi_processor_count}")
        else:
            error_msg = "PyTorch CUDA利用不可"
            print(f"❌ {error_msg}")
            cuda_results['errors'].append(error_msg)
            
    except ImportError as e:
        error_msg = f"PyTorch未検出: {e}"
        print(f"❌ {error_msg}")
        cuda_results['errors'].append(error_msg)
        print("📦 インストール方法: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    
    # 3. NVIDIA管理ライブラリ確認
    try:
        import pynvml
        pynvml.nvmlInit()
        driver_version = pynvml.nvmlSystemGetDriverVersion()
        cuda_version = pynvml.nvmlSystemGetCudaDriverVersion()
        
        print(f"🔧 NVIDIA ドライバ: {driver_version}")
        print(f"🔧 CUDA バージョン: {cuda_version}")
        
    except ImportError:
        print("⚠️ pynvml未検出（オプション）")
    except Exception as e:
        print(f"⚠️ NVIDIA管理ライブラリエラー: {e}")
    
    return cuda_results

def test_cupy_performance():
    """CuPy性能テスト"""
    print("\n🔬 2. CuPy GPU計算性能テスト")
    print("-" * 60)
    
    test_results = {}
    
    try:
        import cupy as cp
        
        # テストサイズ
        test_sizes = [1000, 5000, 10000, 50000]
        
        for size in test_sizes:
            print(f"\n📊 テストサイズ: {size} x {size} 行列")
            
            # CPU計算
            print("   💻 CPU計算中...")
            a_cpu = np.random.random((size, size)).astype(np.float32)
            b_cpu = np.random.random((size, size)).astype(np.float32)
            
            start_time = time.time()
            c_cpu = np.dot(a_cpu, b_cpu)
            cpu_time = time.time() - start_time
            
            print(f"     CPU時間: {cpu_time:.4f}秒")
            
            # GPU計算
            print("   🚀 GPU計算中...")
            try:
                a_gpu = cp.asarray(a_cpu)
                b_gpu = cp.asarray(b_cpu)
                
                # GPU計算（ウォームアップ）
                _ = cp.dot(a_gpu, b_gpu)
                cp.cuda.Stream.null.synchronize()
                
                # 実際の測定
                start_time = time.time()
                c_gpu = cp.dot(a_gpu, b_gpu)
                cp.cuda.Stream.null.synchronize()
                gpu_time = time.time() - start_time
                
                print(f"     GPU時間: {gpu_time:.4f}秒")
                
                # 高速化率計算
                speedup = cpu_time / gpu_time if gpu_time > 0 else 0
                print(f"     高速化率: {speedup:.2f}倍")
                
                # 精度検証
                c_gpu_cpu = cp.asnumpy(c_gpu)
                accuracy = np.mean(np.abs(c_cpu - c_gpu_cpu))
                print(f"     精度差: {accuracy:.2e}")
                
                test_results[size] = {
                    'cpu_time': cpu_time,
                    'gpu_time': gpu_time,
                    'speedup': speedup,
                    'accuracy': accuracy
                }
                
            except Exception as e:
                print(f"     ❌ GPU計算エラー: {e}")
                test_results[size] = {
                    'cpu_time': cpu_time,
                    'gpu_time': float('inf'),
                    'speedup': 0,
                    'accuracy': float('inf'),
                    'error': str(e)
                }
        
    except ImportError:
        print("❌ CuPy未利用: テストスキップ")
        return {}
    
    return test_results

def test_pytorch_cuda():
    """PyTorch CUDA テスト"""
    print("\n🔬 3. PyTorch CUDA計算テスト")
    print("-" * 60)
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("❌ PyTorch CUDA利用不可: テストスキップ")
            return {}
        
        # テンサー計算テスト
        print("📊 テンサー計算テスト")
        size = 5000
        
        # CPU計算
        print("   💻 CPU計算中...")
        a_cpu = torch.randn(size, size)
        b_cpu = torch.randn(size, size)
        
        start_time = time.time()
        c_cpu = torch.mm(a_cpu, b_cpu)
        cpu_time = time.time() - start_time
        
        print(f"     CPU時間: {cpu_time:.4f}秒")
        
        # GPU計算
        print("   🚀 GPU計算中...")
        device = torch.device('cuda')
        a_gpu = a_cpu.to(device)
        b_gpu = b_cpu.to(device)
        
        # ウォームアップ
        _ = torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()
        
        start_time = time.time()
        c_gpu = torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        print(f"     GPU時間: {gpu_time:.4f}秒")
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        print(f"     高速化率: {speedup:.2f}倍")
        
        # 精度検証
        c_gpu_cpu = c_gpu.cpu()
        accuracy = torch.mean(torch.abs(c_cpu - c_gpu_cpu)).item()
        print(f"     精度差: {accuracy:.2e}")
        
        # メモリ使用量確認
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        
        print(f"   💾 GPU メモリ使用量:")
        print(f"     割り当て済み: {allocated:.2f} GB")
        print(f"     キャッシュ: {cached:.2f} GB")
        
        return {
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'speedup': speedup,
            'accuracy': accuracy,
            'memory_allocated_gb': allocated,
            'memory_cached_gb': cached
        }
        
    except ImportError:
        print("❌ PyTorch未検出: テストスキップ")
        return {}
    except Exception as e:
        print(f"❌ PyTorch CUDAテストエラー: {e}")
        return {'error': str(e)}

def system_info():
    """システム情報表示"""
    print("\n🖥️ 4. システム情報")
    print("-" * 60)
    
    try:
        import psutil
        
        # CPU情報
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        print(f"💻 CPU:")
        print(f"   コア数: {cpu_count}")
        print(f"   使用率: {cpu_percent:.1f}%")
        
        # メモリ情報
        memory = psutil.virtual_memory()
        print(f"💾 メモリ:")
        print(f"   総容量: {memory.total / 1024**3:.2f} GB")
        print(f"   利用可能: {memory.available / 1024**3:.2f} GB")
        print(f"   使用率: {memory.percent:.1f}%")
        
        # Python情報
        print(f"🐍 Python:")
        print(f"   バージョン: {sys.version}")
        print(f"   プラットフォーム: {sys.platform}")
        
    except ImportError:
        print("⚠️ psutil未検出: システム情報スキップ")

def save_test_results(cuda_results, cupy_results, pytorch_results):
    """テスト結果保存"""
    print("\n💾 5. テスト結果保存")
    print("-" * 60)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'system_info': {
            'platform': sys.platform,
            'python_version': sys.version
        },
        'cuda_environment': cuda_results,
        'cupy_performance': cupy_results,
        'pytorch_cuda': pytorch_results
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cuda_setup_test_results_{timestamp}.json"
    
    try:
        import json
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"✅ テスト結果保存: {filename}")
        
    except Exception as e:
        print(f"❌ 保存エラー: {e}")
    
    return results

def main():
    """メイン実行関数"""
    print_header()
    
    # 1. CUDA環境確認
    cuda_results = test_cuda_availability()
    
    # 2. CuPy性能テスト
    cupy_results = test_cupy_performance()
    
    # 3. PyTorch CUDAテスト
    pytorch_results = test_pytorch_cuda()
    
    # 4. システム情報
    system_info()
    
    # 5. 結果保存
    save_test_results(cuda_results, cupy_results, pytorch_results)
    
    # 最終レポート
    print("\n" + "=" * 80)
    print("🏆 CUDA環境テスト 最終レポート")
    print("=" * 80)
    
    # 環境確認結果
    print("🔍 環境確認:")
    print(f"   CuPy CUDA: {'✅ 利用可能' if cuda_results['cupy_available'] else '❌ 利用不可'}")
    print(f"   PyTorch CUDA: {'✅ 利用可能' if cuda_results['pytorch_cuda'] else '❌ 利用不可'}")
    print(f"   GPU数: {cuda_results['gpu_count']}")
    
    # 性能結果
    if cupy_results:
        best_cupy_speedup = max([v.get('speedup', 0) for v in cupy_results.values()])
        print(f"🚀 CuPy最大高速化率: {best_cupy_speedup:.2f}倍")
    
    if pytorch_results and 'speedup' in pytorch_results:
        print(f"🚀 PyTorch高速化率: {pytorch_results['speedup']:.2f}倍")
    
    # エラー報告
    if cuda_results['errors']:
        print("⚠️ エラー:")
        for error in cuda_results['errors']:
            print(f"   - {error}")
    
    # 推奨事項
    print("\n📋 推奨事項:")
    if not cuda_results['cupy_available']:
        print("   - CuPyインストール: py -3 -m pip install cupy-cuda12x")
    if not cuda_results['pytorch_cuda']:
        print("   - PyTorch CUDAインストール:")
        print("     py -3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    if cuda_results['gpu_count'] == 0:
        print("   - NVIDIA GPUドライバ確認")
        print("   - CUDA Toolkitインストール確認")
    
    if cuda_results['cupy_available'] and cuda_results['pytorch_cuda']:
        print("✅ CUDA環境完全準備完了! NKAT解析を実行できます。")
        print("🚀 次のステップ: py -3 riemann_hypothesis_cuda_ultimate.py")
    
    print("\n🌟 CUDA環境セットアップテスト完了!")

if __name__ == "__main__":
    main() 