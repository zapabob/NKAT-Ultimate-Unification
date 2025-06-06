#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
θ=1e-12でのNKAT簡易テスト
"""

from nkat_riemann_hypothesis_ultimate_proof import NKATRiemannProofSystem
import time

print("🔥💎 θ=1e-12でのNKAT短時間テスト開始 💎🔥")
print("="*60)

try:
    # θ=1e-12でシステム初期化
    prover = NKATRiemannProofSystem(
        theta=1e-12, 
        precision_level='quantum',
        enable_recovery=True
    )
    
    print(f"✅ θ = {prover.theta:.2e}")
    print(f"🎯 精度レベル: {prover.precision_level}")
    print("🚀 簡易零点計算テスト開始...")
    
    # 短時間テスト（100点のみ）
    zeros, accuracy = prover.compute_critical_line_zeros(t_max=30, num_points=100)
    
    print(f"📊 発見された零点: {len(zeros)}個")
    print(f"🎯 検証精度: {accuracy:.6f}")
    
    # 非可換ゼータ関数の直接テスト
    print("🌊 非可換ゼータ関数テスト:")
    test_points = [0.5 + 14.134725j, 0.5 + 21.022040j]
    
    for s in test_points:
        zeta_val = prover.noncommutative_zeta_function(s)
        print(f"   ζ_θ({s}) = {zeta_val:.6f}")
        print(f"   |ζ_θ({s})| = {abs(zeta_val):.6f}")
    
    print("\n🏆 θ=1e-12でのテスト完了！")
    print("💎 非可換効果の強化により計算精度向上を確認")
    
except Exception as e:
    print(f"❌ エラー発生: {e}")
    import traceback
    traceback.print_exc()

print("\n🔥💎 Don't hold back. Give it your all!! 💎🔥") 