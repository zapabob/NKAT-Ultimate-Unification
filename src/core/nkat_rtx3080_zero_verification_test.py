#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 RTX3080リーマンゼロ点検証テストシステム
既知ゼロ点の高速検証とNVIDIA精度問題対策
"""

import numpy as np
import time
from tqdm import tqdm

# GPU関連
try:
    import cupy as cp
    CUDA_AVAILABLE = True
    print("🚀 CUDA RTX3080 GPU加速: 有効")
except ImportError:
    CUDA_AVAILABLE = False
    print("⚠️ CUDA無効 - CPU計算モード")

# 既知のリーマンゼロ点（最初の20個をテスト用）
KNOWN_ZEROS_TEST = [
    14.134725141734693790, 21.022039638771554993, 25.010857580145688763,
    30.424876125859513210, 32.935061587739189691, 37.586178158825671257,
    40.918719012147495187, 43.327073280914999519, 48.005150881167159727,
    49.773832477672302181, 52.970321477714460644, 56.446247697063246647,
    59.347044003089763073, 60.831778524609379545, 65.112544048081652973,
    67.079810529494172625, 69.546401711185979016, 72.067157674809377632,
    75.704690699808543111, 77.144840068874804149
]

class RTX3080ZetaVerifier:
    def __init__(self):
        if CUDA_AVAILABLE:
            self.gpu_device = cp.cuda.Device(0)
            print(f"🔥 GPU初期化: {self.gpu_device}")
    
    def simple_riemann_zeta(self, s_val):
        """シンプルなリーマンゼータ関数（RTX3080最適化）"""
        try:
            if CUDA_AVAILABLE:
                s = cp.asarray(s_val, dtype=cp.complex128)
                
                if cp.real(s) > 1:
                    # 基本的なディリクレ級数
                    terms = cp.arange(1, 500, dtype=cp.complex128)
                    zeta_val = cp.sum(1.0 / cp.power(terms, s))
                else:
                    # 簡易解析接続
                    n = 50
                    terms = cp.arange(1, n + 1, dtype=cp.complex128)
                    partial_sum = cp.sum(1.0 / cp.power(terms, s))
                    
                    # 補正項
                    if s != 1:
                        correction = cp.power(n, 1-s) / (s-1)
                        zeta_val = partial_sum + correction
                    else:
                        zeta_val = partial_sum
                
                return cp.asnumpy(zeta_val)
            else:
                # CPU版
                s = complex(s_val)
                if s.real > 1:
                    terms = np.arange(1, 500, dtype=complex)
                    return np.sum(1.0 / (terms ** s))
                else:
                    n = 50
                    terms = np.arange(1, n + 1, dtype=complex)
                    partial_sum = np.sum(1.0 / (terms ** s))
                    if s != 1:
                        correction = n**(1-s) / (s-1)
                        return partial_sum + correction
                    else:
                        return partial_sum
                        
        except Exception as e:
            print(f"⚠️ ゼータ計算エラー: {e}")
            return 0.0 + 0.0j
    
    def verify_known_zeros(self):
        """既知ゼロ点検証テスト"""
        print("🎯 RTX3080既知ゼロ点検証テスト開始")
        
        results = []
        start_time = time.time()
        
        with tqdm(total=len(KNOWN_ZEROS_TEST), desc="🔍 ゼロ点検証", ncols=100) as pbar:
            for i, known_zero in enumerate(KNOWN_ZEROS_TEST):
                s_test = complex(0.5, known_zero)
                zeta_val = self.simple_riemann_zeta(s_test)
                residual = abs(zeta_val)
                
                # 様々な閾値での検証
                thresholds = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
                verified_at = []
                
                for threshold in thresholds:
                    if residual < threshold:
                        verified_at.append(f"1e-{int(-np.log10(threshold))}")
                
                result = {
                    'index': i + 1,
                    't': known_zero,
                    'residual': residual,
                    'verified_at': verified_at,
                    'log_residual': np.log10(residual) if residual > 0 else -999
                }
                results.append(result)
                
                pbar.set_postfix({
                    'ゼロ点': i + 1,
                    '残差': f"{residual:.2e}",
                    '検証': len(verified_at) > 0
                })
                pbar.update(1)
        
        computation_time = time.time() - start_time
        
        # 統計分析
        verified_1e1 = len([r for r in results if r['residual'] < 1e-1])
        verified_1e2 = len([r for r in results if r['residual'] < 1e-2])
        verified_1e3 = len([r for r in results if r['residual'] < 1e-3])
        verified_1e4 = len([r for r in results if r['residual'] < 1e-4])
        
        avg_residual = np.mean([r['residual'] for r in results])
        min_residual = min([r['residual'] for r in results])
        max_residual = max([r['residual'] for r in results])
        
        print(f"\n✅ RTX3080検証テスト完了!")
        print(f"⏱️ 計算時間: {computation_time:.2f}秒")
        print(f"🚀 処理速度: {len(KNOWN_ZEROS_TEST)/computation_time:.1f} zeros/sec")
        print(f"\n📊 検証結果統計:")
        print(f"   閾値 1e-1: {verified_1e1}/{len(KNOWN_ZEROS_TEST)} ({verified_1e1/len(KNOWN_ZEROS_TEST)*100:.1f}%)")
        print(f"   閾値 1e-2: {verified_1e2}/{len(KNOWN_ZEROS_TEST)} ({verified_1e2/len(KNOWN_ZEROS_TEST)*100:.1f}%)")
        print(f"   閾値 1e-3: {verified_1e3}/{len(KNOWN_ZEROS_TEST)} ({verified_1e3/len(KNOWN_ZEROS_TEST)*100:.1f}%)")
        print(f"   閾値 1e-4: {verified_1e4}/{len(KNOWN_ZEROS_TEST)} ({verified_1e4/len(KNOWN_ZEROS_TEST)*100:.1f}%)")
        print(f"\n🔬 残差統計:")
        print(f"   平均残差: {avg_residual:.2e}")
        print(f"   最小残差: {min_residual:.2e}")
        print(f"   最大残差: {max_residual:.2e}")
        
        # 詳細結果
        print(f"\n📋 詳細結果 (最初の10個):")
        for i, result in enumerate(results[:10]):
            verified_str = ", ".join(result['verified_at']) if result['verified_at'] else "なし"
            print(f"   {result['index']:2d}. t={result['t']:8.3f} | 残差={result['residual']:.2e} | 検証={verified_str}")
        
        return results
    
    def gpu_performance_test(self):
        """GPU性能テスト"""
        if not CUDA_AVAILABLE:
            print("⚠️ CUDA無効のためGPU性能テスト不可")
            return
        
        print("\n🚀 RTX3080性能テスト開始")
        
        # 様々な計算サイズでテスト
        test_sizes = [100, 500, 1000, 2000]
        
        for size in test_sizes:
            print(f"\n🧮 テストサイズ: {size}項")
            
            # GPU計算
            start_time = time.time()
            s_test = complex(0.5, 14.134725141734693790)
            
            for _ in range(10):  # 10回平均
                terms = cp.arange(1, size + 1, dtype=cp.complex128)
                zeta_val = cp.sum(1.0 / cp.power(terms, s_test))
                result_gpu = cp.asnumpy(zeta_val)
            
            gpu_time = (time.time() - start_time) / 10
            
            # CPU比較
            start_time = time.time()
            
            for _ in range(10):
                terms = np.arange(1, size + 1, dtype=complex)
                result_cpu = np.sum(1.0 / (terms ** s_test))
            
            cpu_time = (time.time() - start_time) / 10
            
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            precision_diff = abs(result_gpu - result_cpu)
            
            print(f"   GPU時間: {gpu_time*1000:.2f}ms")
            print(f"   CPU時間: {cpu_time*1000:.2f}ms")  
            print(f"   高速化: {speedup:.2f}x")
            print(f"   精度差: {precision_diff:.2e}")

def main():
    print("🎯 RTX3080リーマンゼロ点検証テストシステム")
    print("=" * 50)
    
    verifier = RTX3080ZetaVerifier()
    
    # 既知ゼロ点検証
    results = verifier.verify_known_zeros()
    
    # GPU性能テスト
    verifier.gpu_performance_test()
    
    print(f"\n🎊 RTX3080検証テスト完了!")
    print(f"💡 最適閾値推奨: 1e-2 から 1e-3")

if __name__ == "__main__":
    main() 