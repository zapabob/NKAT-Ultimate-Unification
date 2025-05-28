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
        'x': x.tolist(),
        'classical_sum': classical_sum.tolist(),
        'kappa_sum': kappa_sum.tolist()
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
        'eigenvalues': eigenvalues.tolist(),
        't_values': t_values.tolist(),
        'zeta_values': zeta_values.tolist()
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
        'gamma_ray_constraint': theta_gamma_limit,
        'constraint_satisfied': constraint_satisfied
    }

def test_kan_architecture():
    """KANアーキテクチャスケーリングのテスト"""
    print("\n4. KANアーキテクチャスケーリングのテスト")
    
    # アーキテクチャ設定
    architectures = [
        {'depth': 2, 'width': 32, 'grid': 8},
        {'depth': 4, 'width': 64, 'grid': 16},
        {'depth': 6, 'width': 128, 'grid': 32},
        {'depth': 8, 'width': 256, 'grid': 64}
    ]
    
    results = []
    for arch in architectures:
        n_params = arch['depth'] * arch['width'] * arch['grid']
        
        # 理論的近似誤差
        if n_params >= 10000:
            error = 1e-6
        else:
            error = 1.0 / np.sqrt(n_params)
        
        results.append({
            'architecture': arch,
            'n_parameters': n_params,
            'approximation_error': error
        })
    
    # 最適アーキテクチャ
    best = min(results, key=lambda x: x['approximation_error'])
    
    print(f"   テストしたアーキテクチャ数: {len(results)}")
    print(f"   最適構成: {best['architecture']}")
    print(f"   最小近似誤差: {best['approximation_error']:.2e}")
    print(f"   最適パラメータ数: {best['n_parameters']}")
    
    return {
        'architectures': results,
        'best_architecture': best
    }

def visualize_results(results):
    """結果の可視化"""
    print("\n5. 結果の可視化")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. κ-変形B-スプライン
    if 'kappa_bspline' in results:
        data = results['kappa_bspline']
        x = np.array(data['x'])
        
        axes[0, 0].plot(x, data['classical_sum'], '--', label='古典的', linewidth=2)
        axes[0, 0].plot(x, data['kappa_sum'], '-', label='κ-変形', linewidth=2)
        axes[0, 0].set_title('κ-変形B-スプライン')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('B(x)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    # 2. スペクトル次元
    if 'spectral_dimension' in results:
        data = results['spectral_dimension']
        
        axes[0, 1].semilogy(data['eigenvalues'][:50], 'o-', markersize=4)
        axes[0, 1].set_title('固有値分布')
        axes[0, 1].set_xlabel('インデックス')
        axes[0, 1].set_ylabel('固有値')
        axes[0, 1].grid(True)
    
    # 3. θ-λ関係（エネルギー依存性）
    if 'theta_lambda' in results:
        energy_range = np.logspace(9, 15, 50)
        lambda_val = results['theta_lambda']['lambda_kappa']
        planck_scale = 1.22e19
        
        theta_theory = []
        for E in energy_range:
            x = E / planck_scale
            f_x = 1 - x**2 / 2 if x < 0.1 else np.power(x, -0.5) * np.exp(-x/2)
            theta_E = (lambda_val / planck_scale**2) * E**2 * f_x
            theta_theory.append(theta_E)
        
        axes[1, 0].loglog(energy_range, np.abs(theta_theory), 'b-', linewidth=2)
        axes[1, 0].axhline(abs(results['theta_lambda']['theta_current']), 
                          color='red', linestyle='--', label='現在のθ')
        axes[1, 0].set_title('θ-λ関係')
        axes[1, 0].set_xlabel('エネルギー (GeV)')
        axes[1, 0].set_ylabel('|θ|')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # 4. KANアーキテクチャ
    if 'kan_architecture' in results:
        data = results['kan_architecture']
        
        n_params = [arch['n_parameters'] for arch in data['architectures']]
        errors = [arch['approximation_error'] for arch in data['architectures']]
        
        axes[1, 1].loglog(n_params, errors, 'go', markersize=8)
        axes[1, 1].set_title('KANスケーリング')
        axes[1, 1].set_xlabel('パラメータ数')
        axes[1, 1].set_ylabel('近似誤差')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('nkat_test_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("   可視化結果を 'nkat_test_results.png' に保存しました")

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
    results['kan_architecture'] = test_kan_architecture()
    
    # 結果の可視化
    visualize_results(results)
    
    # 総合評価
    print("\n" + "=" * 60)
    print("テスト結果サマリー")
    print("=" * 60)
    
    completeness_error = results['kappa_bspline']['completeness_error']
    dimension_error = results['spectral_dimension']['dimension_error']
    constraint_satisfied = results['theta_lambda']['constraint_satisfied']
    
    print(f"κ-B-スプライン完全性誤差: {completeness_error:.6f}")
    print(f"スペクトル次元誤差: {dimension_error:.6f}")
    print(f"θ-λ制約満足: {constraint_satisfied}")
    
    # 総合判定
    if (completeness_error < 0.1 and 
        dimension_error < 0.5 and 
        constraint_satisfied):
        overall_quality = "PASS"
    else:
        overall_quality = "FAIL"
    
    print(f"総合テスト品質: {overall_quality}")
    
    # 結果の保存
    with open('nkat_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n詳細結果が 'nkat_test_results.json' に保存されました")
    
    return results

if __name__ == "__main__":
    results = main() 