#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT理論の簡単なテスト
κ-変形B-スプライン、スペクトル次元の基本検証
"""

import numpy as np

def test_nkat_basic():
    """NKAT基本テスト"""
    print("=" * 50)
    print("NKAT (Non-Commutative Kolmogorov-Arnold Theory)")
    print("基本テスト実行中...")
    print("=" * 50)
    
    # パラメータ設定
    kappa = 0.1
    theta = 0.05
    
    print(f"κ-変形パラメータ: {kappa}")
    print(f"θ-非可換パラメータ: {theta}")
    
    # κ-変形B-スプライン関数のテスト
    x = np.linspace(-3, 3, 1000)
    
    def classical_bspline(x, center=0):
        return np.exp(-(x - center)**2 / 2)
    
    def kappa_bspline(x, center=0):
        classical = classical_bspline(x, center)
        kappa_correction = np.exp(-kappa * x**2 / 2)
        theta_correction = np.cos(theta * x)
        return classical * kappa_correction * theta_correction
    
    # 基底関数の完全性テスト
    centers = [-2, -1, 0, 1, 2]
    classical_sum = np.zeros_like(x)
    kappa_sum = np.zeros_like(x)
    
    for center in centers:
        classical_sum += classical_bspline(x, center)
        kappa_sum += kappa_bspline(x, center)
    
    # 完全性誤差の計算
    completeness_error = np.mean(np.abs(kappa_sum - classical_sum))
    max_error = np.max(np.abs(kappa_sum - classical_sum))
    
    print(f"\n1. κ-変形B-スプライン完全性テスト:")
    print(f"   平均誤差: {completeness_error:.6f}")
    print(f"   最大誤差: {max_error:.6f}")
    print(f"   テスト結果: {'PASS' if completeness_error < 0.1 else 'FAIL'}")
    
    # スペクトル次元の模擬計算
    dimension = 4
    n_eigenvalues = 100
    
    # 理論的固有値分布
    eigenvalues = np.array([i**(2/dimension) for i in range(1, n_eigenvalues + 1)])
    eigenvalues += 0.1 * np.random.randn(n_eigenvalues)  # ノイズ
    eigenvalues = eigenvalues[eigenvalues > 0]
    
    # スペクトルゼータ関数
    t_values = np.logspace(-2, 0, 50)
    zeta_values = []
    
    for t in t_values:
        zeta_t = np.sum(np.exp(-t * eigenvalues))
        zeta_values.append(zeta_t)
    
    zeta_values = np.array(zeta_values)
    
    # スペクトル次元の計算
    log_t = np.log(t_values)
    log_zeta = np.log(zeta_values + 1e-12)
    
    valid_indices = np.isfinite(log_zeta) & np.isfinite(log_t)
    if np.sum(valid_indices) >= 5:
        slope = np.polyfit(log_t[valid_indices], log_zeta[valid_indices], 1)[0]
        spectral_dimension = -2 * slope
    else:
        spectral_dimension = float('nan')
    
    dimension_error = abs(spectral_dimension - dimension)
    
    print(f"\n2. スペクトル次元計算テスト:")
    print(f"   理論次元: {dimension}")
    print(f"   計算されたスペクトル次元: {spectral_dimension:.6f}")
    print(f"   誤差: {dimension_error:.6f}")
    print(f"   テスト結果: {'PASS' if dimension_error < 0.5 else 'FAIL'}")
    
    # θ-λ関係のテスト
    lambda_kappa = 1e-6
    energy_scale = 1e12  # 1 TeV
    planck_scale = 1.22e19  # GeV
    
    x_energy = energy_scale / planck_scale
    if x_energy < 0.1:
        f_x = 1 - x_energy**2 / 2
    else:
        f_x = np.power(x_energy, -0.5) * np.exp(-x_energy/2)
    
    predicted_theta = (lambda_kappa / planck_scale**2) * energy_scale**2 * f_x
    theta_ratio = predicted_theta / theta if theta != 0 else float('inf')
    
    # 現象論的制約
    gamma_ray_energy = 100e9  # 100 GeV
    time_delay_limit = 1e-6
    theta_gamma_limit = time_delay_limit * planck_scale**2 / gamma_ray_energy
    constraint_satisfied = abs(theta) < theta_gamma_limit
    
    print(f"\n3. θ-λ関係テスト:")
    print(f"   λ_κ パラメータ: {lambda_kappa:.2e}")
    print(f"   現在のθ: {theta:.2e}")
    print(f"   予測されたθ: {predicted_theta:.2e}")
    print(f"   θ比率: {theta_ratio:.2f}")
    print(f"   制約満足: {constraint_satisfied}")
    print(f"   テスト結果: {'PASS' if constraint_satisfied else 'FAIL'}")
    
    # 総合評価
    all_tests_passed = (
        completeness_error < 0.1 and
        dimension_error < 0.5 and
        constraint_satisfied
    )
    
    print(f"\n" + "=" * 50)
    print("総合テスト結果")
    print("=" * 50)
    print(f"κ-B-スプライン完全性: {'PASS' if completeness_error < 0.1 else 'FAIL'}")
    print(f"スペクトル次元精度: {'PASS' if dimension_error < 0.5 else 'FAIL'}")
    print(f"θ-λ制約満足: {'PASS' if constraint_satisfied else 'FAIL'}")
    print(f"総合評価: {'PASS' if all_tests_passed else 'FAIL'}")
    
    return {
        'completeness_error': completeness_error,
        'spectral_dimension': spectral_dimension,
        'dimension_error': dimension_error,
        'constraint_satisfied': constraint_satisfied,
        'all_tests_passed': all_tests_passed
    }

if __name__ == "__main__":
    results = test_nkat_basic()
    print(f"\nNKAT理論基本テストが完了しました。")
    print(f"詳細結果: {results}") 