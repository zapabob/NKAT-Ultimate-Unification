"""
κ-ミンコフスキーのλパラメータと創発的なθパラメータとの関係の詳細解析
Non-Commutative Kolmogorov-Arnold Theory (NKAT) における変形パラメータ理論

Author: NKAT Research Team
Date: 2025-01-23
Version: 1.0 - 変形パラメータ関係論
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict, Callable
import sympy as sp
from sympy import symbols, exp, cos, sin, diff, integrate, simplify, Matrix, I, pi
from dataclasses import dataclass
import warnings
from pathlib import Path
import json
from scipy.optimize import minimize, curve_fit
from scipy.special import gamma, factorial
import seaborn as sns

# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'
sns.set_style('whitegrid')

@dataclass
class DeformationParameters:
    """変形パラメータの定義"""
    lambda_kappa: float  # κ-ミンコフスキー変形パラメータ
    theta_nc: float  # 非可換パラメータ
    energy_scale: float  # エネルギースケール (GeV)
    planck_scale: float = 1.22e19  # プランクスケール (GeV)
    
    def __post_init__(self):
        """パラメータの妥当性検証"""
        if self.lambda_kappa <= 0:
            raise ValueError("λ_κパラメータは正の値である必要があります")
        if abs(self.theta_nc) > 1.0:
            warnings.warn("θパラメータが大きすぎる可能性があります")
        if self.energy_scale <= 0:
            raise ValueError("エネルギースケールは正の値である必要があります")

class KappaMinkowskiTheta:
    """
    κ-ミンコフスキー変形とθ-変形の関係解析クラス
    
    主要な理論的関係：
    1. λ_κ と θ の現象論的関係
    2. エネルギースケール依存性
    3. 量子重力効果の創発
    4. 実験的観測可能性
    """
    
    def __init__(self, params: DeformationParameters):
        self.params = params
        self.lambda_kappa = params.lambda_kappa
        self.theta_nc = params.theta_nc
        self.E_scale = params.energy_scale
        self.M_planck = params.planck_scale
        
        # シンボリック変数の定義
        self.x, self.p, self.E, self.t = symbols('x p E t', real=True)
        self.lambda_sym = symbols('lambda', positive=True)
        self.theta_sym = symbols('theta', real=True)
        self.kappa_sym = symbols('kappa', positive=True)
        
        # 物理定数（自然単位系）
        self.hbar = 1.0  # ℏ = 1
        self.c = 1.0     # c = 1
        
    def theoretical_theta_lambda_relation(self, lambda_kappa: float, 
                                        energy_scale: float) -> float:
        """
        理論的なθ-λ関係の計算
        
        基本関係式：
        θ(E) = (λ_κ / M_Planck²) * E² * f(E/M_Planck)
        
        ここで、f(x)は無次元関数で、以下の性質を持つ：
        - f(0) = 1 (低エネルギー極限)
        - f(x) ~ x^α (高エネルギー極限、α < 0)
        """
        # 無次元エネルギー
        x = energy_scale / self.M_planck
        
        # 現象論的関数 f(x) の定義
        # 低エネルギー: f(x) ≈ 1 - x²/2
        # 高エネルギー: f(x) ≈ x^(-1/2)
        if x < 0.1:
            f_x = 1 - x**2 / 2 + x**4 / 24  # テイラー展開
        else:
            f_x = np.power(x, -0.5) * np.exp(-x/2)  # 指数的カットオフ
        
        # θパラメータの計算
        theta_theoretical = (lambda_kappa / self.M_planck**2) * energy_scale**2 * f_x
        
        return theta_theoretical
    
    def emergent_theta_from_kappa_minkowski(self, lambda_kappa: float,
                                          momentum_scale: float) -> Dict:
        """
        κ-ミンコフスキー代数からの創発的θパラメータ
        
        κ-ミンコフスキー代数：
        [x^μ, x^ν] = iλ_κ (η^μν x² - x^μ x^ν)
        [x^μ, p^ν] = iℏ η^μν (1 + λ_κ p²)
        [p^μ, p^ν] = 0
        
        から導出される有効的な非可換性
        """
        # 運動量スケールでの有効的非可換性
        p_scale = momentum_scale
        
        # κ-ミンコフスキー代数からの有効θ
        # 1次近似: θ_eff ≈ λ_κ * p²
        theta_eff_1st = lambda_kappa * p_scale**2
        
        # 2次補正: 相対論的効果
        gamma_factor = 1 / np.sqrt(1 - (p_scale / self.M_planck)**2)
        theta_eff_2nd = theta_eff_1st * (1 + lambda_kappa * p_scale**2 / 2)
        
        # 量子補正: ループ効果
        alpha_em = 1/137  # 微細構造定数
        quantum_correction = 1 + alpha_em * np.log(p_scale / 1e-3) / (4 * pi)
        theta_eff_quantum = theta_eff_2nd * quantum_correction
        
        # 非摂動的効果: インスタントン寄与
        instanton_factor = np.exp(-2 * pi / (alpha_em * lambda_kappa * p_scale**2))
        theta_eff_nonpert = theta_eff_quantum + instanton_factor * lambda_kappa
        
        return {
            'theta_1st_order': theta_eff_1st,
            'theta_2nd_order': theta_eff_2nd,
            'theta_quantum': theta_eff_quantum,
            'theta_nonperturbative': theta_eff_nonpert,
            'gamma_factor': gamma_factor,
            'quantum_correction': quantum_correction,
            'instanton_factor': instanton_factor
        }
    
    def renormalization_group_flow(self, lambda_initial: float,
                                 theta_initial: float,
                                 energy_range: np.ndarray) -> Dict:
        """
        繰り込み群流れによるパラメータの走り
        
        β関数：
        β_λ = dλ/d(log μ) = -γ_λ λ + δ_λ λ²
        β_θ = dθ/d(log μ) = -γ_θ θ + δ_θ θ² + ε_λθ λθ
        """
        # 繰り込み群係数（現象論的）
        gamma_lambda = 0.1  # λの異常次元
        delta_lambda = 0.05  # λの2ループ係数
        gamma_theta = 0.15   # θの異常次元
        delta_theta = 0.03   # θの2ループ係数
        epsilon_lambda_theta = 0.02  # λ-θ混合項
        
        # エネルギースケールの対数
        log_mu = np.log(energy_range / energy_range[0])
        
        # 微分方程式の数値解
        lambda_running = np.zeros_like(log_mu)
        theta_running = np.zeros_like(log_mu)
        
        lambda_running[0] = lambda_initial
        theta_running[0] = theta_initial
        
        for i in range(1, len(log_mu)):
            dt = log_mu[i] - log_mu[i-1]
            
            # β関数の評価
            beta_lambda = (-gamma_lambda * lambda_running[i-1] + 
                          delta_lambda * lambda_running[i-1]**2)
            beta_theta = (-gamma_theta * theta_running[i-1] + 
                         delta_theta * theta_running[i-1]**2 +
                         epsilon_lambda_theta * lambda_running[i-1] * theta_running[i-1])
            
            # オイラー法による積分
            lambda_running[i] = lambda_running[i-1] + beta_lambda * dt
            theta_running[i] = theta_running[i-1] + beta_theta * dt
        
        return {
            'energy_scales': energy_range,
            'lambda_running': lambda_running,
            'theta_running': theta_running,
            'log_mu': log_mu,
            'beta_functions': {
                'gamma_lambda': gamma_lambda,
                'delta_lambda': delta_lambda,
                'gamma_theta': gamma_theta,
                'delta_theta': delta_theta,
                'epsilon_lambda_theta': epsilon_lambda_theta
            }
        }
    
    def phenomenological_constraints(self) -> Dict:
        """
        現象論的制約からのパラメータ関係
        
        実験的制約：
        1. γ線天文学からのθ制約
        2. 粒子物理学からのλ制約
        3. 重力波観測からの制約
        4. 宇宙論的制約
        """
        constraints = {}
        
        # 1. γ線天文学制約
        # 時間遅延: Δt/t ≈ θ E / M_Planck²
        gamma_ray_energy = 100e9  # 100 GeV
        time_delay_limit = 1e-6   # 観測限界
        
        theta_gamma_limit = (time_delay_limit * self.M_planck**2 / gamma_ray_energy)
        constraints['theta_gamma_ray'] = theta_gamma_limit
        
        # 2. 粒子物理学制約
        # 散乱断面積修正: δσ/σ ≈ λ s / M_Planck⁴
        lhc_energy = 13e3  # 13 TeV
        cross_section_limit = 1e-3  # 3σ制限
        
        lambda_lhc_limit = (cross_section_limit * self.M_planck**4 / lhc_energy**2)
        constraints['lambda_lhc'] = lambda_lhc_limit
        
        # 3. 重力波制約
        # 波形修正: δh/h ≈ θ f² / M_Planck²
        gw_frequency = 100  # Hz
        strain_sensitivity = 1e-23
        
        theta_gw_limit = (strain_sensitivity * self.M_planck**2 / gw_frequency**2)
        constraints['theta_gravitational_wave'] = theta_gw_limit
        
        # 4. 宇宙論的制約
        # ダークエネルギー: ρ_Λ ≈ λ M_Planck⁴
        dark_energy_density = 2.8e-47  # GeV⁴
        
        lambda_cosmological = dark_energy_density / self.M_planck**4
        constraints['lambda_cosmological'] = lambda_cosmological
        
        # 5. 統合制約
        # θとλの関係から導出される制約
        energy_scales = np.logspace(0, 19, 100)  # 1 GeV to Planck scale
        
        theta_theoretical = []
        for E in energy_scales:
            theta_E = self.theoretical_theta_lambda_relation(self.lambda_kappa, E)
            theta_theoretical.append(theta_E)
        
        constraints['energy_scales'] = energy_scales
        constraints['theta_theoretical'] = np.array(theta_theoretical)
        
        return constraints
    
    def experimental_predictions(self) -> Dict:
        """
        実験的予測の計算
        
        観測可能な効果：
        1. 修正された分散関係
        2. 真空複屈折
        3. 粒子生成閾値の変化
        4. 重力波の伝播速度変化
        """
        predictions = {}
        
        # 1. 修正された分散関係
        # E² = p²c² + m²c⁴ + θ p⁴/M_Planck²
        def modified_dispersion(momentum, mass, theta):
            classical = np.sqrt(momentum**2 + mass**2)
            correction = theta * momentum**4 / (2 * self.M_planck**2 * classical)
            return classical + correction
        
        # 高エネルギー光子の場合
        photon_energies = np.logspace(9, 15, 100)  # 1 GeV to 1 PeV
        dispersion_corrections = []
        
        for E in photon_energies:
            p = E  # 光子の場合 E = pc
            correction = self.theta_nc * p**4 / (2 * self.M_planck**2 * E)
            relative_correction = correction / E
            dispersion_corrections.append(relative_correction)
        
        predictions['photon_energies'] = photon_energies
        predictions['dispersion_corrections'] = np.array(dispersion_corrections)
        
        # 2. 真空複屈折
        # 偏光回転角: φ = θ B² L / M_Planck²
        magnetic_fields = np.logspace(12, 15, 50)  # 10¹² to 10¹⁵ Gauss
        path_length = 1e3  # 1 kpc in natural units
        
        birefringence_angles = []
        for B in magnetic_fields:
            phi = self.theta_nc * B**2 * path_length / self.M_planck**2
            birefringence_angles.append(phi)
        
        predictions['magnetic_fields'] = magnetic_fields
        predictions['birefringence_angles'] = np.array(birefringence_angles)
        
        # 3. 粒子生成閾値
        # 修正された閾値: E_th = m (1 + λ m²/M_Planck²)
        particle_masses = np.array([0.511e-3, 0.106, 1.777, 0.938, 4.18])  # e, μ, τ, p, b
        particle_names = ['electron', 'muon', 'tau', 'proton', 'bottom']
        
        threshold_corrections = []
        for m in particle_masses:
            correction = self.lambda_kappa * m**2 / self.M_planck**2
            threshold_corrections.append(correction)
        
        predictions['particle_masses'] = particle_masses
        predictions['particle_names'] = particle_names
        predictions['threshold_corrections'] = np.array(threshold_corrections)
        
        # 4. 重力波伝播
        # 速度修正: δc/c = θ f²/M_Planck²
        gw_frequencies = np.logspace(-3, 4, 100)  # mHz to 10 kHz
        
        gw_speed_corrections = []
        for f in gw_frequencies:
            delta_c = self.theta_nc * f**2 / self.M_planck**2
            gw_speed_corrections.append(delta_c)
        
        predictions['gw_frequencies'] = gw_frequencies
        predictions['gw_speed_corrections'] = np.array(gw_speed_corrections)
        
        return predictions
    
    def fit_experimental_data(self, mock_data: Dict) -> Dict:
        """
        模擬実験データへのフィッティング
        
        Args:
            mock_data: 模擬実験データ
            
        Returns:
            フィッティング結果
        """
        # 模擬データの生成（実際の実験データの代わり）
        if not mock_data:
            mock_data = self._generate_mock_data()
        
        # フィッティング関数の定義
        def theta_lambda_model(energy, lambda_param, alpha, beta):
            """θ-λ関係のパラメトリックモデル"""
            x = energy / self.M_planck
            return (lambda_param / self.M_planck**2) * energy**2 * np.power(x, alpha) * np.exp(-beta * x)
        
        # 実験データの準備
        energies = mock_data['energies']
        theta_observed = mock_data['theta_values']
        theta_errors = mock_data['theta_errors']
        
        # 非線形最小二乗フィッティング
        try:
            popt, pcov = curve_fit(
                theta_lambda_model, 
                energies, 
                theta_observed,
                sigma=theta_errors,
                p0=[self.lambda_kappa, -0.5, 0.1],  # 初期推定値
                bounds=([1e-10, -2.0, 0.0], [1e-5, 2.0, 10.0])  # パラメータ範囲
            )
            
            lambda_fit, alpha_fit, beta_fit = popt
            param_errors = np.sqrt(np.diag(pcov))
            
            # フィッティング品質の評価
            theta_fit = theta_lambda_model(energies, *popt)
            chi_squared = np.sum(((theta_observed - theta_fit) / theta_errors)**2)
            dof = len(energies) - len(popt)
            reduced_chi_squared = chi_squared / dof
            
            fit_results = {
                'lambda_fitted': lambda_fit,
                'alpha_fitted': alpha_fit,
                'beta_fitted': beta_fit,
                'parameter_errors': param_errors,
                'chi_squared': chi_squared,
                'reduced_chi_squared': reduced_chi_squared,
                'degrees_of_freedom': dof,
                'theta_fitted': theta_fit,
                'fit_quality': 'good' if reduced_chi_squared < 2.0 else 'poor'
            }
            
        except Exception as e:
            fit_results = {
                'error': str(e),
                'fit_quality': 'failed'
            }
        
        return fit_results
    
    def _generate_mock_data(self) -> Dict:
        """模擬実験データの生成"""
        # エネルギー範囲
        energies = np.logspace(9, 15, 20)  # 1 GeV to 1 PeV
        
        # 理論値の計算
        theta_theory = []
        for E in energies:
            theta_E = self.theoretical_theta_lambda_relation(self.lambda_kappa, E)
            theta_theory.append(theta_E)
        
        theta_theory = np.array(theta_theory)
        
        # 実験誤差の追加（対数正規分布）
        relative_errors = 0.1 + 0.05 * np.random.randn(len(energies))
        theta_errors = np.abs(theta_theory * relative_errors)
        theta_observed = theta_theory + theta_errors * np.random.randn(len(energies))
        
        return {
            'energies': energies,
            'theta_values': theta_observed,
            'theta_errors': theta_errors,
            'theta_theory': theta_theory
        }
    
    def visualize_theta_lambda_relationship(self, save_path: Optional[str] = None):
        """θ-λ関係の可視化"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 基本的なθ-λ関係
        energy_range = np.logspace(0, 19, 1000)
        theta_values = []
        
        for E in energy_range:
            theta_E = self.theoretical_theta_lambda_relation(self.lambda_kappa, E)
            theta_values.append(theta_E)
        
        axes[0, 0].loglog(energy_range, theta_values, 'b-', linewidth=2, 
                         label=f'λ_κ = {self.lambda_kappa:.2e}')
        axes[0, 0].axhline(self.theta_nc, color='red', linestyle='--', 
                          label=f'θ_NC = {self.theta_nc:.2e}')
        axes[0, 0].set_xlabel('エネルギー (GeV)')
        axes[0, 0].set_ylabel('θ パラメータ')
        axes[0, 0].set_title('θ-λ基本関係')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. 繰り込み群流れ
        rg_results = self.renormalization_group_flow(
            self.lambda_kappa, self.theta_nc, energy_range[::100]
        )
        
        axes[0, 1].semilogx(rg_results['energy_scales'], rg_results['lambda_running'], 
                           'g-', label='λ(μ)')
        axes[0, 1].semilogx(rg_results['energy_scales'], rg_results['theta_running'], 
                           'r-', label='θ(μ)')
        axes[0, 1].set_xlabel('エネルギースケール μ (GeV)')
        axes[0, 1].set_ylabel('走るパラメータ')
        axes[0, 1].set_title('繰り込み群流れ')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. 現象論的制約
        constraints = self.phenomenological_constraints()
        
        axes[0, 2].loglog(constraints['energy_scales'], 
                         np.abs(constraints['theta_theoretical']), 
                         'b-', label='理論予測')
        axes[0, 2].axhline(constraints['theta_gamma_ray'], color='purple', 
                          linestyle='--', label='γ線制約')
        axes[0, 2].axhline(constraints['theta_gravitational_wave'], color='orange', 
                          linestyle='--', label='重力波制約')
        axes[0, 2].set_xlabel('エネルギー (GeV)')
        axes[0, 2].set_ylabel('|θ| 制約')
        axes[0, 2].set_title('現象論的制約')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # 4. 実験的予測
        predictions = self.experimental_predictions()
        
        axes[1, 0].loglog(predictions['photon_energies'], 
                         np.abs(predictions['dispersion_corrections']), 
                         'c-', linewidth=2)
        axes[1, 0].set_xlabel('光子エネルギー (GeV)')
        axes[1, 0].set_ylabel('相対的分散補正')
        axes[1, 0].set_title('修正された分散関係')
        axes[1, 0].grid(True)
        
        # 5. 真空複屈折
        axes[1, 1].loglog(predictions['magnetic_fields'], 
                         np.abs(predictions['birefringence_angles']), 
                         'm-', linewidth=2)
        axes[1, 1].set_xlabel('磁場強度 (Gauss)')
        axes[1, 1].set_ylabel('偏光回転角 (rad)')
        axes[1, 1].set_title('真空複屈折')
        axes[1, 1].grid(True)
        
        # 6. フィッティング結果
        mock_data = self._generate_mock_data()
        fit_results = self.fit_experimental_data(mock_data)
        
        if 'theta_fitted' in fit_results:
            axes[1, 2].errorbar(mock_data['energies'], mock_data['theta_values'], 
                               yerr=mock_data['theta_errors'], fmt='ro', 
                               label='模擬データ')
            axes[1, 2].loglog(mock_data['energies'], fit_results['theta_fitted'], 
                             'b-', label='フィッティング')
            axes[1, 2].loglog(mock_data['energies'], mock_data['theta_theory'], 
                             'g--', label='理論値')
            axes[1, 2].set_xlabel('エネルギー (GeV)')
            axes[1, 2].set_ylabel('θ パラメータ')
            axes[1, 2].set_title(f'データフィッティング (χ²/dof = {fit_results["reduced_chi_squared"]:.2f})')
            axes[1, 2].legend()
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        return fig

def demonstrate_kappa_minkowski_theta():
    """κ-ミンコフスキー-θ関係のデモンストレーション"""
    
    print("=" * 80)
    print("κ-ミンコフスキーのλパラメータと創発的なθパラメータとの関係解析")
    print("=" * 80)
    
    # パラメータ設定
    params = DeformationParameters(
        lambda_kappa=1e-6,  # κ-ミンコフスキーパラメータ
        theta_nc=1e-8,      # 非可換パラメータ
        energy_scale=1e12,  # 1 TeV
        planck_scale=1.22e19
    )
    
    analyzer = KappaMinkowskiTheta(params)
    
    print(f"\n解析パラメータ:")
    print(f"λ_κ パラメータ: {params.lambda_kappa:.2e}")
    print(f"θ_NC パラメータ: {params.theta_nc:.2e}")
    print(f"エネルギースケール: {params.energy_scale:.2e} GeV")
    print(f"プランクスケール: {params.planck_scale:.2e} GeV")
    
    # 1. 理論的θ-λ関係
    print("\n1. 理論的θ-λ関係の計算...")
    energy_scales = [1e9, 1e12, 1e15, 1e18]  # GeV
    
    for E in energy_scales:
        theta_theory = analyzer.theoretical_theta_lambda_relation(params.lambda_kappa, E)
        print(f"E = {E:.0e} GeV: θ = {theta_theory:.2e}")
    
    # 2. 創発的θパラメータ
    print("\n2. κ-ミンコフスキー代数からの創発的θ...")
    momentum_scales = [1e3, 1e6, 1e9, 1e12]  # GeV
    
    for p in momentum_scales:
        emergent_results = analyzer.emergent_theta_from_kappa_minkowski(
            params.lambda_kappa, p
        )
        print(f"p = {p:.0e} GeV:")
        print(f"  1次: θ = {emergent_results['theta_1st_order']:.2e}")
        print(f"  2次: θ = {emergent_results['theta_2nd_order']:.2e}")
        print(f"  量子: θ = {emergent_results['theta_quantum']:.2e}")
        print(f"  非摂動: θ = {emergent_results['theta_nonperturbative']:.2e}")
    
    # 3. 繰り込み群流れ
    print("\n3. 繰り込み群流れの解析...")
    energy_range = np.logspace(3, 18, 100)
    rg_results = analyzer.renormalization_group_flow(
        params.lambda_kappa, params.theta_nc, energy_range
    )
    
    print(f"初期値: λ = {params.lambda_kappa:.2e}, θ = {params.theta_nc:.2e}")
    print(f"最終値: λ = {rg_results['lambda_running'][-1]:.2e}, θ = {rg_results['theta_running'][-1]:.2e}")
    print(f"λの変化率: {(rg_results['lambda_running'][-1] / params.lambda_kappa - 1) * 100:.1f}%")
    print(f"θの変化率: {(rg_results['theta_running'][-1] / params.theta_nc - 1) * 100:.1f}%")
    
    # 4. 現象論的制約
    print("\n4. 現象論的制約の評価...")
    constraints = analyzer.phenomenological_constraints()
    
    print(f"γ線天文学制約: |θ| < {constraints['theta_gamma_ray']:.2e}")
    print(f"重力波制約: |θ| < {constraints['theta_gravitational_wave']:.2e}")
    print(f"LHC制約: λ < {constraints['lambda_lhc']:.2e}")
    print(f"宇宙論的制約: λ ~ {constraints['lambda_cosmological']:.2e}")
    
    # 制約の整合性チェック
    if params.theta_nc < constraints['theta_gamma_ray']:
        print("✓ θパラメータはγ線制約を満たしています")
    else:
        print("✗ θパラメータがγ線制約に違反しています")
    
    if params.lambda_kappa < constraints['lambda_lhc']:
        print("✓ λパラメータはLHC制約を満たしています")
    else:
        print("✗ λパラメータがLHC制約に違反しています")
    
    # 5. 実験的予測
    print("\n5. 実験的予測の計算...")
    predictions = analyzer.experimental_predictions()
    
    # 代表的な値での予測
    test_energy = 1e12  # 1 TeV
    idx = np.argmin(np.abs(predictions['photon_energies'] - test_energy))
    dispersion_correction = predictions['dispersion_corrections'][idx]
    print(f"1 TeV光子の分散補正: δE/E = {dispersion_correction:.2e}")
    
    test_field = 1e14  # 10¹⁴ Gauss
    idx = np.argmin(np.abs(predictions['magnetic_fields'] - test_field))
    birefringence = predictions['birefringence_angles'][idx]
    print(f"10¹⁴ Gauss磁場での偏光回転: φ = {birefringence:.2e} rad")
    
    # 粒子生成閾値
    for i, name in enumerate(predictions['particle_names']):
        correction = predictions['threshold_corrections'][i]
        print(f"{name}生成閾値補正: δE/E = {correction:.2e}")
    
    # 6. データフィッティング
    print("\n6. 模擬データフィッティング...")
    mock_data = analyzer._generate_mock_data()
    fit_results = analyzer.fit_experimental_data(mock_data)
    
    if 'lambda_fitted' in fit_results:
        print(f"フィッティング結果:")
        print(f"  λ_fitted = {fit_results['lambda_fitted']:.2e} ± {fit_results['parameter_errors'][0]:.2e}")
        print(f"  α_fitted = {fit_results['alpha_fitted']:.3f} ± {fit_results['parameter_errors'][1]:.3f}")
        print(f"  β_fitted = {fit_results['beta_fitted']:.3f} ± {fit_results['parameter_errors'][2]:.3f}")
        print(f"  χ²/dof = {fit_results['reduced_chi_squared']:.2f}")
        print(f"  フィッティング品質: {fit_results['fit_quality']}")
    else:
        print(f"フィッティングエラー: {fit_results.get('error', 'Unknown error')}")
    
    # 7. 結果の可視化
    print("\n7. 結果の可視化...")
    fig = analyzer.visualize_theta_lambda_relationship(
        save_path='kappa_minkowski_theta_relationship.png'
    )
    
    # 8. 結果の保存
    results_summary = {
        'parameters': params.__dict__,
        'theoretical_relations': {
            'energy_scales': energy_scales,
            'theta_values': [analyzer.theoretical_theta_lambda_relation(params.lambda_kappa, E) 
                           for E in energy_scales]
        },
        'rg_flow_summary': {
            'initial_lambda': params.lambda_kappa,
            'final_lambda': rg_results['lambda_running'][-1],
            'initial_theta': params.theta_nc,
            'final_theta': rg_results['theta_running'][-1]
        },
        'constraints': {k: v for k, v in constraints.items() 
                       if not isinstance(v, np.ndarray)},
        'fit_results': fit_results,
        'analysis_timestamp': str(np.datetime64('now'))
    }
    
    with open('kappa_minkowski_theta_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False, default=str)
    
    print("\n解析結果が 'kappa_minkowski_theta_analysis.json' に保存されました。")
    
    return analyzer, results_summary

if __name__ == "__main__":
    # 解析のデモンストレーション
    analyzer, results = demonstrate_kappa_minkowski_theta() 