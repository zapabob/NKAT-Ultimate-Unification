#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
θ値の迅速比較テスト
"""

import time
from nkat_riemann_hypothesis_ultimate_proof import NKATRiemannProofSystem

def quick_theta_comparison():
    """複数θ値の迅速比較"""
    theta_values = [1e-8, 1e-10, 1e-12, 1e-14]
    results = []
    
    print("🔥💎 θ値最適化迅速テスト 💎🔥")
    print("="*50)
    
    for theta in theta_values:
        print(f"\n🧪 θ={theta:.0e} テスト開始...")
        
        try:
            start_time = time.time()
            
            # 簡易システム初期化（リカバリー無効で高速化）
            prover = NKATRiemannProofSystem(
                theta=theta,
                precision_level='quantum',
                enable_recovery=False  # 高速テスト用
            )
            
            # 超短時間テスト（10点のみ）
            zeros, accuracy = prover.compute_critical_line_zeros(
                t_max=15,
                num_points=10
            )
            
            computation_time = time.time() - start_time
            
            # 非可換ゼータ関数テスト
            test_point = 0.5 + 14.134725j
            zeta_val = prover.noncommutative_zeta_function(test_point)
            
            result = {
                'theta': theta,
                'zeros_found': len(zeros),
                'accuracy': accuracy,
                'computation_time': computation_time,
                'zeta_magnitude': abs(zeta_val),
                'efficiency': len(zeros) / computation_time if computation_time > 0 else 0
            }
            
            results.append(result)
            
            print(f"   ✅ 完了: {len(zeros)}個零点, 精度={accuracy:.6f}, 時間={computation_time:.2f}秒")
            print(f"   📊 ζ_θ(0.5+14.134725i) = {abs(zeta_val):.6f}")
            
        except Exception as e:
            print(f"   ❌ θ={theta:.0e} エラー: {e}")
    
    # 結果分析
    print(f"\n📊 θ値比較結果:")
    print("-" * 50)
    print(f"{'θ値':<10} {'零点数':<8} {'精度':<12} {'時間(秒)':<10} {'効率':<10}")
    print("-" * 50)
    
    for result in results:
        print(f"{result['theta']:.0e}  {result['zeros_found']:<8} "
              f"{result['accuracy']:<12.6f} {result['computation_time']:<10.2f} "
              f"{result['efficiency']:<10.4f}")
    
    # 最適θ推薦
    if results:
        best_accuracy = max(results, key=lambda x: x['accuracy'])
        best_efficiency = max(results, key=lambda x: x['efficiency'])
        
        print(f"\n🏆 最高精度: θ={best_accuracy['theta']:.0e} (精度={best_accuracy['accuracy']:.6f})")
        print(f"⚡ 最高効率: θ={best_efficiency['theta']:.0e} (効率={best_efficiency['efficiency']:.4f})")
    
    return results

if __name__ == "__main__":
    quick_theta_comparison() 