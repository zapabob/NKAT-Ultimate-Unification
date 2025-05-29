#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT数学的基盤の簡単なテスト
κ-変形B-スプライン、スペクトル次元、θ-λ関係の基本検証

Author: NKAT Research Team
Date: 2025-01-23
"""

import numpy as np
import matplotlib.pyplot as plt
import json

def test_kappa_bspline():
    """κ-変形B-スプライン関数のテスト"""
    print("1. κ-変形B-スプライン関数のテスト")
    
    # パラメータ
    kappa = 0.1
    theta = 0.05
    x = np.linspace(-3, 3, 1000)
    
    # 古典的B-スプライン（ガウシアン近似）
    def classical_bspline(x, center=0):
        return np.exp(-(x - center)**2 / 2)
    
    # κ-変形B-スプライン
    def kappa_bspline(x, center=0):
        classical = classical_bspline(x, center)
        kappa_correction = np.exp(-kappa * x**2 / 2)
        theta_correction = np.cos(theta * x)
        return classical * kappa_correction * theta_correction
    
    # 基底関数の計算
    centers = [-2, -1, 0, 1, 2]
    classical_sum = np.zeros_like(x)
    kappa_sum = np.zeros_like(x)
    
    for center in centers:
        classical_sum += classical_bspline(x, center)
        kappa_sum += kappa_bspline(x, center)
    
    # 完全性誤差
    completeness_error = np.mean(np.abs(kappa_sum - classical_sum))
    
    print(f"   κパラメータ: {kappa}")
    print(f"   θパラメータ: {theta}")
    print(f"   完全性誤差: {completeness_error:.6f}")
    
    return {
        'kappa': kappa,
        'theta': theta,
        'completeness_error': completeness_error,
        'test_passed': completeness_error < 0.1
    }

def test_spectral_dimension():
    """スペクトル次元計算のテスト"""
    print("\n2. スペクトル次元計算のテスト")
    
    # 模擬固有値（4次元空間の期待値）
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
    
    # 線形回帰
    valid_indices = np.isfinite(log_zeta) & np.isfinite(log_t)
    if np.sum(valid_indices) >= 5:
        slope = np.polyfit(log_t[valid_indices], log_zeta[valid_indices], 1)[0]
        spectral_dimension = -2 * slope
    else:
        spectral_dimension = float('nan')
    
    dimension_error = abs(spectral_dimension - dimension)
    
    print(f"   理論次元: {dimension}")
    print(f"   計算されたスペクトル次元: {spectral_dimension:.6f}")
    print(f"   誤差: {dimension_error:.6f}")
    
    return {
        'theoretical_dimension': dimension,
        'spectral_dimension': spectral_dimension,
        'dimension_error': dimension_error,
        'test_passed': dimension_error < 0.5
    }

def test_theta_lambda_relationship():
    """θ-λ関係のテスト"""
    print("\n3. θ-λ関係のテスト")
    
    # パラメータ
    lambda_kappa = 1e-6
    theta_current = 0.05
    energy_scale = 1e12  # 1 TeV
    planck_scale = 1.22e19  # GeV
    
    # 理論的θ-λ関係
    def theoretical_theta(lambda_val, energy):
        x = energy / planck_scale
        if x < 0.1:
            f_x = 1 - x**2 / 2
        else:
            f_x = np.power(x, -0.5) * np.exp(-x/2)
        return (lambda_val / planck_scale**2) * energy**2 * f_x
    
    # 予測されたθ
    predicted_theta = theoretical_theta(lambda_kappa, energy_scale)
    theta_ratio = predicted_theta / theta_current if theta_current != 0 else float('inf')
    
    # 現象論的制約
    gamma_ray_energy = 100e9  # 100 GeV
    time_delay_limit = 1e-6
    theta_gamma_limit = time_delay_limit * planck_scale**2 / gamma_ray_energy
    
    constraint_satisfied = abs(theta_current) < theta_gamma_limit
    
    print(f"   λ_κ パラメータ: {lambda_kappa:.2e}")
    print(f"   現在のθ: {theta_current:.2e}")
    print(f"   予測されたθ: {predicted_theta:.2e}")
    print(f"   θ比率: {theta_ratio:.2f}")
    print(f"   γ線制約満足: {constraint_satisfied}")
    
    return {
        'lambda_kappa': lambda_kappa,
        'theta_current': theta_current,
        'predicted_theta': predicted_theta,
        'theta_ratio': theta_ratio,
        'constraint_satisfied': constraint_satisfied,
        'test_passed': constraint_satisfied
    }

def main():
    """メイン実行関数"""
    print("=" * 60)
    print("NKAT数学的基盤の基本テスト")
    print("=" * 60)
    
    # 各テストの実行
    results = {}
    results['kappa_bspline'] = test_kappa_bspline()
    results['spectral_dimension'] = test_spectral_dimension()
    results['theta_lambda'] = test_theta_lambda_relationship()
    
    # 総合評価
    print("\n" + "=" * 60)
    print("テスト結果サマリー")
    print("=" * 60)
    
    all_tests_passed = all([
        results['kappa_bspline']['test_passed'],
        results['spectral_dimension']['test_passed'],
        results['theta_lambda']['test_passed']
    ])
    
    print(f"κ-B-スプライン完全性誤差: {results['kappa_bspline']['completeness_error']:.6f}")
    print(f"スペクトル次元誤差: {results['spectral_dimension']['dimension_error']:.6f}")
    print(f"θ-λ制約満足: {results['theta_lambda']['constraint_satisfied']}")
    print(f"総合テスト結果: {'PASS' if all_tests_passed else 'FAIL'}")
    
    # 結果の保存
    with open('simple_nkat_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n詳細結果が 'simple_nkat_test_results.json' に保存されました")
    
    return results

if __name__ == "__main__":
    results = main() 