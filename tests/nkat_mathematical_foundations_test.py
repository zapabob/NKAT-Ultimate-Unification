"""
NKAT数学的基盤のテスト実装
κ-変形B-スプライン、ディラック/ラプラシアン作用素、κ-ミンコフスキー-θ関係の統合テスト

Author: NKAT Research Team
Date: 2025-01-23
Version: 1.0 - 統合テスト版
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
from dataclasses import dataclass

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

class NKATMathematicalFoundations:
    """NKAT数学的基盤の統合クラス"""
    
    def __init__(self, params: NKATParameters):
        self.params = params
        self.results = {}
        
    def test_kappa_deformed_bspline(self) -> Dict:
        """κ-変形B-スプライン関数のテスト"""
        print("1. κ-変形B-スプライン関数のテスト...")
        
        # 基本パラメータ
        x = np.linspace(-3, 3, 1000)
        kappa = self.params.kappa
        theta = self.params.theta
        
        # 古典的B-スプライン（簡略版）
        def classical_bspline(x, center=0, width=1):
            return np.exp(-(x - center)**2 / (2 * width**2))
        
        # κ-変形B-スプライン
        def kappa_deformed_bspline(x, center=0, width=1):
            classical = classical_bspline(x, center, width)
            kappa_correction = np.exp(-kappa * x**2 / 2)
            theta_correction = np.cos(theta * x)
            return classical * kappa_correction * theta_correction
        
        # 複数の基底関数を計算
        centers = [-2, -1, 0, 1, 2]
        classical_bases = []
        kappa_bases = []
        
        for center in centers:
            classical = classical_bspline(x, center)
            kappa_deformed = kappa_deformed_bspline(x, center)
            classical_bases.append(classical)
            kappa_bases.append(kappa_deformed)
        
        # 完全性のテスト（簡略版）
        classical_sum = np.sum(classical_bases, axis=0)
        kappa_sum = np.sum(kappa_bases, axis=0)
        
        completeness_error = np.mean(np.abs(kappa_sum - 1.0))
        
        results = {
            'x_values': x.tolist(),
            'classical_bases': [basis.tolist() for basis in classical_bases],
            'kappa_bases': [basis.tolist() for basis in kappa_bases],
            'completeness_error': completeness_error,
            'kappa_parameter': kappa,
            'theta_parameter': theta
        }
        
        print(f"   完全性誤差: {completeness_error:.6f}")
        print(f"   κパラメータ: {kappa}")
        print(f"   θパラメータ: {theta}")
        
        return results
    
    def test_spectral_dimension_calculation(self) -> Dict:
        """スペクトル次元計算のテスト"""
        print("\n2. スペクトル次元計算のテスト...")
        
        # 簡略化されたディラック作用素の固有値（模擬）
        n_eigenvalues = 100
        dimension = self.params.dimension
        
        # 理論的固有値分布（d次元での期待値）
        eigenvalues = np.array([i**(2/dimension) for i in range(1, n_eigenvalues + 1)])
        eigenvalues = eigenvalues * (1 + 0.1 * np.random.randn(n_eigenvalues))  # ノイズ追加
        eigenvalues = eigenvalues[eigenvalues > 0]  # 正の値のみ
        
        # スペクトルゼータ関数の計算
        t_values = np.logspace(-2, 0, 50)
        zeta_values = []
        
        for t in t_values:
            zeta_t = np.sum(np.exp(-t * eigenvalues))
            zeta_values.append(zeta_t)
        
        zeta_values = np.array(zeta_values)
        
        # スペクトル次元の計算
        log_t = np.log(t_values)
        log_zeta = np.log(zeta_values + 1e-12)
        
        # 線形回帰で傾きを求める
        valid_indices = np.isfinite(log_zeta) & np.isfinite(log_t)
        if np.sum(valid_indices) >= 5:
            slope = np.polyfit(log_t[valid_indices], log_zeta[valid_indices], 1)[0]
            spectral_dimension = -2 * slope
        else:
            spectral_dimension = float('nan')
        
        # 理論値との比較
        theoretical_dimension = dimension
        dimension_error = abs(spectral_dimension - theoretical_dimension)
        
        results = {
            'eigenvalues': eigenvalues.tolist(),
            'spectral_dimension': spectral_dimension,
            'theoretical_dimension': theoretical_dimension,
            'dimension_error': dimension_error,
            't_values': t_values.tolist(),
            'zeta_values': zeta_values.tolist(),
            'slope': slope if 'slope' in locals() else float('nan')
        }
        
        print(f"   計算されたスペクトル次元: {spectral_dimension:.6f}")
        print(f"   理論値: {theoretical_dimension}")
        print(f"   誤差: {dimension_error:.6f}")
        
        return results
    
    def test_theta_lambda_relationship(self) -> Dict:
        """θ-λ関係のテスト"""
        print("\n3. θ-λ関係のテスト...")
        
        lambda_kappa = self.params.lambda_param
        energy_scale = self.params.energy_scale
        planck_scale = self.params.planck_scale
        
        # 理論的θ-λ関係
        def theoretical_theta_lambda(lambda_val, energy):
            x = energy / planck_scale
            if x < 0.1:
                f_x = 1 - x**2 / 2  # 低エネルギー近似
            else:
                f_x = np.power(x, -0.5) * np.exp(-x/2)  # 高エネルギー
            return (lambda_val / planck_scale**2) * energy**2 * f_x
        
        # エネルギー範囲でのテスト
        energy_range = np.logspace(9, 15, 50)  # 1 GeV to 1 PeV
        theta_theoretical = []
        
        for E in energy_range:
            theta_E = theoretical_theta_lambda(lambda_kappa, E)
            theta_theoretical.append(theta_E)
        
        theta_theoretical = np.array(theta_theoretical)
        
        # 現在のθパラメータとの比較
        current_theta = self.params.theta
        test_energy = energy_scale
        predicted_theta = theoretical_theta_lambda(lambda_kappa, test_energy)
        
        theta_ratio = predicted_theta / current_theta if current_theta != 0 else float('inf')
        
        # 現象論的制約のテスト
        gamma_ray_energy = 100e9  # 100 GeV
        time_delay_limit = 1e-6
        theta_gamma_limit = time_delay_limit * planck_scale**2 / gamma_ray_energy
        
        constraint_satisfied = abs(current_theta) < theta_gamma_limit
        
        results = {
            'energy_range': energy_range.tolist(),
            'theta_theoretical': theta_theoretical.tolist(),
            'lambda_kappa': lambda_kappa,
            'current_theta': current_theta,
            'predicted_theta': predicted_theta,
            'theta_ratio': theta_ratio,
            'gamma_ray_constraint': theta_gamma_limit,
            'constraint_satisfied': constraint_satisfied
        }
        
        print(f"   λ_κ パラメータ: {lambda_kappa:.2e}")
        print(f"   現在のθ: {current_theta:.2e}")
        print(f"   予測されたθ: {predicted_theta:.2e}")
        print(f"   θ比率: {theta_ratio:.2f}")
        print(f"   γ線制約満足: {constraint_satisfied}")
        
        return results
    
    def test_kan_architecture_scaling(self) -> Dict:
        """KANアーキテクチャスケーリングのテスト"""
        print("\n4. KANアーキテクチャスケーリングのテスト...")
        
        # アーキテクチャパラメータ
        depth_range = [2, 4, 6, 8]
        width_range = [32, 64, 128, 256]
        grid_range = [8, 16, 32, 64]
        
        # 近似精度の理論的推定
        architecture_results = []
        
        for depth in depth_range:
            for width in width_range:
                for grid_size in grid_range:
                    # パラメータ数
                    n_params = depth * width * grid_size
                    
                    # 理論的近似誤差（簡略化）
                    if n_params >= 10000:
                        approximation_error = 1e-6
                    else:
                        approximation_error = 1.0 / np.sqrt(n_params)
                    
                    # 物理的精度の推定
                    spectral_precision = min(approximation_error * 100, 1e-3)
                    
                    architecture_results.append({
                        'depth': depth,
                        'width': width,
                        'grid_size': grid_size,
                        'n_parameters': n_params,
                        'approximation_error': approximation_error,
                        'spectral_precision': spectral_precision
                    })
        
        # 最適アーキテクチャの特定
        best_config = min(architecture_results, key=lambda x: x['approximation_error'])
        
        results = {
            'architecture_configs': architecture_results,
            'best_configuration': best_config,
            'scaling_analysis': {
                'min_error': min(config['approximation_error'] for config in architecture_results),
                'max_error': max(config['approximation_error'] for config in architecture_results),
                'optimal_params': best_config['n_parameters']
            }
        }
        
        print(f"   テストしたアーキテクチャ数: {len(architecture_results)}")
        print(f"   最適構成: 深度={best_config['depth']}, 幅={best_config['width']}, グリッド={best_config['grid_size']}")
        print(f"   最小近似誤差: {best_config['approximation_error']:.2e}")
        print(f"   最適パラメータ数: {best_config['n_parameters']}")
        
        return results
    
    def visualize_results(self, save_path: str = 'nkat_mathematical_foundations_test.png'):
        """結果の可視化"""
        print("\n5. 結果の可視化...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. κ-変形B-スプライン
        if 'kappa_bspline' in self.results:
            data = self.results['kappa_bspline']
            x = np.array(data['x_values'])
            
            for i, basis in enumerate(data['classical_bases'][:3]):
                axes[0, 0].plot(x, basis, '--', alpha=0.7, label=f'古典 {i}')
            
            for i, basis in enumerate(data['kappa_bases'][:3]):
                axes[0, 0].plot(x, basis, '-', linewidth=2, label=f'κ-変形 {i}')
            
            axes[0, 0].set_title('κ-変形B-スプライン基底関数')
            axes[0, 0].set_xlabel('x')
            axes[0, 0].set_ylabel('B(x)')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # 2. スペクトル次元
        if 'spectral_dimension' in self.results:
            data = self.results['spectral_dimension']
            
            axes[0, 1].semilogy(data['eigenvalues'][:50], 'o-', markersize=4)
            axes[0, 1].set_title('固有値分布')
            axes[0, 1].set_xlabel('固有値インデックス')
            axes[0, 1].set_ylabel('固有値')
            axes[0, 1].grid(True)
        
        # 3. θ-λ関係
        if 'theta_lambda' in self.results:
            data = self.results['theta_lambda']
            
            axes[1, 0].loglog(data['energy_range'], np.abs(data['theta_theoretical']), 'b-', linewidth=2)
            axes[1, 0].axhline(abs(data['current_theta']), color='red', linestyle='--', 
                              label=f'現在のθ = {data["current_theta"]:.2e}')
            axes[1, 0].set_title('θ-λ関係')
            axes[1, 0].set_xlabel('エネルギー (GeV)')
            axes[1, 0].set_ylabel('|θ|')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # 4. KANアーキテクチャスケーリング
        if 'kan_scaling' in self.results:
            data = self.results['kan_scaling']
            configs = data['architecture_configs']
            
            n_params = [config['n_parameters'] for config in configs]
            errors = [config['approximation_error'] for config in configs]
            
            axes[1, 1].loglog(n_params, errors, 'go', markersize=6, alpha=0.7)
            axes[1, 1].set_title('KANアーキテクチャスケーリング')
            axes[1, 1].set_xlabel('パラメータ数')
            axes[1, 1].set_ylabel('近似誤差')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   可視化結果を {save_path} に保存しました")
        
        return fig
    
    def run_comprehensive_test(self) -> Dict:
        """包括的テストの実行"""
        print("=" * 70)
        print("NKAT数学的基盤の包括的テスト")
        print("=" * 70)
        
        print(f"\nテストパラメータ:")
        print(f"κ = {self.params.kappa}")
        print(f"θ = {self.params.theta}")
        print(f"λ = {self.params.lambda_param:.2e}")
        print(f"次元 = {self.params.dimension}")
        print(f"格子サイズ = {self.params.lattice_size}")
        
        # 各テストの実行
        self.results['kappa_bspline'] = self.test_kappa_deformed_bspline()
        self.results['spectral_dimension'] = self.test_spectral_dimension_calculation()
        self.results['theta_lambda'] = self.test_theta_lambda_relationship()
        self.results['kan_scaling'] = self.test_kan_architecture_scaling()
        
        # 結果の可視化
        fig = self.visualize_results()
        
        # 統合結果の計算
        summary = {
            'test_parameters': self.params.__dict__,
            'kappa_bspline_completeness_error': self.results['kappa_bspline']['completeness_error'],
            'spectral_dimension_error': self.results['spectral_dimension']['dimension_error'],
            'theta_lambda_ratio': self.results['theta_lambda']['theta_ratio'],
            'optimal_kan_parameters': self.results['kan_scaling']['best_configuration']['n_parameters'],
            'all_constraints_satisfied': self.results['theta_lambda']['constraint_satisfied'],
            'overall_test_quality': 'PASS' if (
                self.results['kappa_bspline']['completeness_error'] < 0.1 and
                self.results['spectral_dimension']['dimension_error'] < 0.5 and
                self.results['theta_lambda']['constraint_satisfied']
            ) else 'FAIL'
        }
        
        print(f"\n" + "=" * 70)
        print("テスト結果サマリー")
        print("=" * 70)
        print(f"κ-B-スプライン完全性誤差: {summary['kappa_bspline_completeness_error']:.6f}")
        print(f"スペクトル次元誤差: {summary['spectral_dimension_error']:.6f}")
        print(f"θ-λ比率: {summary['theta_lambda_ratio']:.2f}")
        print(f"最適KANパラメータ数: {summary['optimal_kan_parameters']}")
        print(f"制約満足: {summary['all_constraints_satisfied']}")
        print(f"総合テスト品質: {summary['overall_test_quality']}")
        
        # 結果の保存
        with open('nkat_mathematical_foundations_test_results.json', 'w', encoding='utf-8') as f:
            json.dump({
                'summary': summary,
                'detailed_results': self.results
            }, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n詳細結果が 'nkat_mathematical_foundations_test_results.json' に保存されました")
        
        return summary

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
    
    # テスト実行
    tester = NKATMathematicalFoundations(params)
    summary = tester.run_comprehensive_test()
    
    return tester, summary

if __name__ == "__main__":
    tester, summary = main() 