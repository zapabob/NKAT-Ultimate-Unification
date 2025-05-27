#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT (Non-Commutative Kolmogorov-Arnold Theory) 核心理論
κ-変形B-スプライン、スペクトル次元、θ-λ関係の統合実装

Author: NKAT Research Team
Date: 2025-01-23
Version: 1.0 - 統合理論版
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

# 日本語フォント設定（エラー回避）
try:
    plt.rcParams['font.family'] = 'MS Gothic'
except:
    plt.rcParams['font.family'] = 'DejaVu Sans'

@dataclass
class NKATParameters:
    """NKAT統合パラメータ"""
    kappa: float = 0.1          # κ-変形パラメータ
    theta: float = 0.05         # 非可換パラメータ
    lambda_param: float = 1e-6  # κ-ミンコフスキーパラメータ
    dimension: int = 4          # 空間次元
    lattice_size: int = 16      # 格子サイズ
    energy_scale: float = 1e12  # エネルギースケール (GeV)
    planck_scale: float = 1.22e19  # プランクスケール (GeV)

class NKATCoreTheory:
    """NKAT核心理論の統合クラス"""
    
    def __init__(self, params: NKATParameters):
        self.params = params
        self.results = {}
        
    def kappa_deformed_bspline(self, x: np.ndarray, center: float = 0, width: float = 1) -> np.ndarray:
        """
        κ-変形B-スプライン関数
        
        B_i^κ(x) = B_i(x) * exp(-κx²/2) * cos(θx)
        
        Args:
            x: 評価点
            center: 中心位置
            width: 幅パラメータ
            
        Returns:
            κ-変形B-スプライン値
        """
        # 古典的B-スプライン（ガウシアン近似）
        classical = np.exp(-(x - center)**2 / (2 * width**2))
        
        # κ-変形補正
        kappa_correction = np.exp(-self.params.kappa * x**2 / 2)
        
        # θ-変形補正（非可換性）
        theta_correction = np.cos(self.params.theta * x)
        
        return classical * kappa_correction * theta_correction
    
    def compute_spectral_dimension(self, eigenvalues: np.ndarray) -> Tuple[float, Dict]:
        """
        スペクトル次元の計算
        
        d_s = -2 * d(log Z(t))/d(log t)
        
        Args:
            eigenvalues: 作用素の固有値
            
        Returns:
            スペクトル次元と詳細情報
        """
        # 正の固有値のみ使用
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        
        if len(eigenvalues) < 10:
            return float('nan'), {'error': '有効な固有値が不足'}
        
        # スペクトルゼータ関数の計算
        t_values = np.logspace(-3, 0, 50)
        zeta_values = []
        
        for t in t_values:
            zeta_t = np.sum(np.exp(-t * eigenvalues))
            zeta_values.append(zeta_t)
        
        zeta_values = np.array(zeta_values)
        
        # 対数微分の計算
        log_t = np.log(t_values)
        log_zeta = np.log(zeta_values + 1e-12)
        
        # 線形回帰で傾きを求める
        valid_indices = np.isfinite(log_zeta) & np.isfinite(log_t)
        if np.sum(valid_indices) < 5:
            return float('nan'), {'error': '有効なデータ点が不足'}
        
        slope = np.polyfit(log_t[valid_indices], log_zeta[valid_indices], 1)[0]
        spectral_dimension = -2 * slope
        
        # R²の計算
        y_mean = np.mean(log_zeta[valid_indices])
        y_pred = slope * log_t[valid_indices] + (y_mean - slope * np.mean(log_t[valid_indices]))
        ss_res = np.sum((log_zeta[valid_indices] - y_pred) ** 2)
        ss_tot = np.sum((log_zeta[valid_indices] - y_mean) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        info = {
            'eigenvalues': eigenvalues,
            'n_eigenvalues': len(eigenvalues),
            'slope': slope,
            'r_squared': r_squared,
            't_values': t_values,
            'zeta_values': zeta_values
        }
        
        return spectral_dimension, info
    
    def theta_lambda_relationship(self, energy: float) -> Dict:
        """
        θ-λ関係の計算
        
        θ(E) = (λ_κ / M_Planck²) * E² * f(E/M_Planck)
        
        Args:
            energy: エネルギースケール
            
        Returns:
            θ-λ関係の詳細
        """
        lambda_kappa = self.params.lambda_param
        planck_scale = self.params.planck_scale
        
        # 無次元エネルギー
        x = energy / planck_scale
        
        # 現象論的関数 f(x)
        if x < 0.1:
            f_x = 1 - x**2 / 2 + x**4 / 24  # 低エネルギー展開
        else:
            f_x = np.power(x, -0.5) * np.exp(-x/2)  # 高エネルギー
        
        # 理論的θ
        theta_theoretical = (lambda_kappa / planck_scale**2) * energy**2 * f_x
        
        # 現象論的制約
        gamma_ray_energy = 100e9  # 100 GeV
        time_delay_limit = 1e-6
        theta_gamma_limit = time_delay_limit * planck_scale**2 / gamma_ray_energy
        
        # LHC制約
        lhc_energy = 13e3  # 13 TeV
        cross_section_limit = 1e-3
        lambda_lhc_limit = cross_section_limit * planck_scale**4 / lhc_energy**2
        
        return {
            'energy': energy,
            'theta_theoretical': theta_theoretical,
            'theta_current': self.params.theta,
            'theta_ratio': theta_theoretical / self.params.theta if self.params.theta != 0 else float('inf'),
            'gamma_ray_constraint': theta_gamma_limit,
            'lhc_constraint': lambda_lhc_limit,
            'constraint_satisfied': abs(self.params.theta) < theta_gamma_limit,
            'f_function_value': f_x
        }
    
    def generate_mock_eigenvalues(self, n_eigenvalues: int = 100) -> np.ndarray:
        """
        模擬固有値の生成（理論的分布に基づく）
        
        Args:
            n_eigenvalues: 固有値の数
            
        Returns:
            模擬固有値配列
        """
        dimension = self.params.dimension
        
        # 理論的固有値分布
        eigenvalues = np.array([i**(2/dimension) for i in range(1, n_eigenvalues + 1)])
        
        # ノイズの追加
        noise_level = 0.1
        eigenvalues += noise_level * np.random.randn(n_eigenvalues)
        
        # κ-変形による補正
        kappa_correction = 1 + self.params.kappa * np.arange(n_eigenvalues) / n_eigenvalues
        eigenvalues *= kappa_correction
        
        # θ-変形による補正
        theta_correction = 1 + self.params.theta * np.sin(np.arange(n_eigenvalues) * np.pi / n_eigenvalues)
        eigenvalues *= theta_correction
        
        # 正の値のみ保持
        eigenvalues = eigenvalues[eigenvalues > 0]
        
        return eigenvalues
    
    def test_completeness(self, x_range: Tuple[float, float] = (-3, 3), n_points: int = 1000) -> Dict:
        """
        κ-変形B-スプライン基底の完全性テスト
        
        Args:
            x_range: x軸の範囲
            n_points: 評価点数
            
        Returns:
            完全性テスト結果
        """
        x = np.linspace(x_range[0], x_range[1], n_points)
        
        # 複数の基底関数
        centers = [-2, -1, 0, 1, 2]
        classical_sum = np.zeros_like(x)
        kappa_sum = np.zeros_like(x)
        
        for center in centers:
            # 古典的B-スプライン
            classical = np.exp(-(x - center)**2 / 2)
            classical_sum += classical
            
            # κ-変形B-スプライン
            kappa_deformed = self.kappa_deformed_bspline(x, center)
            kappa_sum += kappa_deformed
        
        # 完全性誤差
        completeness_error = np.mean(np.abs(kappa_sum - classical_sum))
        max_error = np.max(np.abs(kappa_sum - classical_sum))
        
        return {
            'x_values': x,
            'classical_sum': classical_sum,
            'kappa_sum': kappa_sum,
            'completeness_error': completeness_error,
            'max_error': max_error,
            'test_passed': completeness_error < 0.1
        }
    
    def run_comprehensive_analysis(self) -> Dict:
        """包括的NKAT解析の実行"""
        
        print("=" * 70)
        print("NKAT (Non-Commutative Kolmogorov-Arnold Theory) 包括的解析")
        print("=" * 70)
        
        print(f"\n解析パラメータ:")
        print(f"κ = {self.params.kappa}")
        print(f"θ = {self.params.theta}")
        print(f"λ = {self.params.lambda_param:.2e}")
        print(f"次元 = {self.params.dimension}")
        print(f"エネルギースケール = {self.params.energy_scale:.2e} GeV")
        
        results = {}
        
        # 1. κ-変形B-スプライン完全性テスト
        print("\n1. κ-変形B-スプライン完全性テスト...")
        completeness_results = self.test_completeness()
        results['completeness'] = completeness_results
        
        print(f"   完全性誤差: {completeness_results['completeness_error']:.6f}")
        print(f"   最大誤差: {completeness_results['max_error']:.6f}")
        print(f"   テスト結果: {'PASS' if completeness_results['test_passed'] else 'FAIL'}")
        
        # 2. スペクトル次元計算
        print("\n2. スペクトル次元計算...")
        eigenvalues = self.generate_mock_eigenvalues()
        spectral_dim, spectral_info = self.compute_spectral_dimension(eigenvalues)
        results['spectral_dimension'] = {
            'dimension': spectral_dim,
            'theoretical': self.params.dimension,
            'error': abs(spectral_dim - self.params.dimension),
            'info': spectral_info
        }
        
        print(f"   理論次元: {self.params.dimension}")
        print(f"   計算されたスペクトル次元: {spectral_dim:.6f}")
        print(f"   誤差: {abs(spectral_dim - self.params.dimension):.6f}")
        print(f"   R²: {spectral_info.get('r_squared', 'N/A'):.6f}")
        
        # 3. θ-λ関係解析
        print("\n3. θ-λ関係解析...")
        theta_lambda_results = self.theta_lambda_relationship(self.params.energy_scale)
        results['theta_lambda'] = theta_lambda_results
        
        print(f"   現在のθ: {theta_lambda_results['theta_current']:.2e}")
        print(f"   理論的θ: {theta_lambda_results['theta_theoretical']:.2e}")
        print(f"   θ比率: {theta_lambda_results['theta_ratio']:.2f}")
        print(f"   制約満足: {theta_lambda_results['constraint_satisfied']}")
        
        # 4. 総合評価
        print("\n4. 総合評価...")
        all_tests_passed = (
            completeness_results['test_passed'] and
            abs(spectral_dim - self.params.dimension) < 0.5 and
            theta_lambda_results['constraint_satisfied']
        )
        
        results['summary'] = {
            'all_tests_passed': all_tests_passed,
            'completeness_error': completeness_results['completeness_error'],
            'spectral_dimension_error': abs(spectral_dim - self.params.dimension),
            'constraint_satisfied': theta_lambda_results['constraint_satisfied'],
            'overall_quality': 'PASS' if all_tests_passed else 'FAIL'
        }
        
        print(f"   総合テスト結果: {'PASS' if all_tests_passed else 'FAIL'}")
        
        return results
    
    def visualize_results(self, results: Dict, save_path: str = 'nkat_core_analysis.png'):
        """結果の可視化"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. κ-変形B-スプライン
        if 'completeness' in results:
            data = results['completeness']
            x = data['x_values']
            
            axes[0, 0].plot(x, data['classical_sum'], '--', label='古典的', linewidth=2)
            axes[0, 0].plot(x, data['kappa_sum'], '-', label='κ-変形', linewidth=2)
            axes[0, 0].set_title('κ-変形B-スプライン基底')
            axes[0, 0].set_xlabel('x')
            axes[0, 0].set_ylabel('B(x)')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # 2. スペクトル次元
        if 'spectral_dimension' in results:
            data = results['spectral_dimension']
            if 'info' in data and 'eigenvalues' in data['info']:
                eigenvals = data['info']['eigenvalues'][:50]
                axes[0, 1].semilogy(eigenvals, 'o-', markersize=4)
                axes[0, 1].set_title(f'固有値分布 (d_s = {data["dimension"]:.3f})')
                axes[0, 1].set_xlabel('インデックス')
                axes[0, 1].set_ylabel('固有値')
                axes[0, 1].grid(True)
        
        # 3. θ-λ関係
        if 'theta_lambda' in results:
            data = results['theta_lambda']
            
            # エネルギー依存性
            energy_range = np.logspace(9, 15, 50)
            theta_theory = []
            
            for E in energy_range:
                theta_rel = self.theta_lambda_relationship(E)
                theta_theory.append(theta_rel['theta_theoretical'])
            
            axes[1, 0].loglog(energy_range, np.abs(theta_theory), 'b-', linewidth=2)
            axes[1, 0].axhline(abs(data['theta_current']), color='red', 
                              linestyle='--', label='現在のθ')
            axes[1, 0].set_title('θ-λ関係')
            axes[1, 0].set_xlabel('エネルギー (GeV)')
            axes[1, 0].set_ylabel('|θ|')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # 4. 統計サマリー
        if 'summary' in results:
            summary = results['summary']
            
            summary_text = f"""
NKAT解析結果サマリー

完全性誤差: {summary['completeness_error']:.6f}
スペクトル次元誤差: {summary['spectral_dimension_error']:.6f}
制約満足: {summary['constraint_satisfied']}
総合品質: {summary['overall_quality']}

パラメータ:
κ = {self.params.kappa}
θ = {self.params.theta}
λ = {self.params.lambda_param:.2e}
次元 = {self.params.dimension}
            """
            
            axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes,
                           fontsize=10, verticalalignment='center',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            axes[1, 1].set_title('解析サマリー')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n可視化結果を {save_path} に保存しました")
        
        return fig

def main():
    """メイン実行関数"""
    
    # パラメータ設定
    params = NKATParameters(
        kappa=0.1,
        theta=0.05,
        lambda_param=1e-6,
        dimension=4,
        lattice_size=16,
        energy_scale=1e12,
        planck_scale=1.22e19
    )
    
    # NKAT理論解析の実行
    nkat = NKATCoreTheory(params)
    results = nkat.run_comprehensive_analysis()
    
    # 結果の可視化
    fig = nkat.visualize_results(results)
    
    # 結果の保存
    with open('nkat_core_analysis_results.json', 'w', encoding='utf-8') as f:
        # numpy配列をリストに変換
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            else:
                return obj
        
        json.dump(results, f, indent=2, ensure_ascii=False, default=convert_numpy)
    
    print(f"\n詳細結果が 'nkat_core_analysis_results.json' に保存されました")
    
    return nkat, results

if __name__ == "__main__":
    nkat, results = main() 