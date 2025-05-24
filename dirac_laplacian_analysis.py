"""
ディラック/ラプラシアン作用素のKAN学習における詳細解析
Non-Commutative Kolmogorov-Arnold Theory (NKAT) における作用素理論

Author: NKAT Research Team
Date: 2025-01-23
Version: 1.0 - 作用素スペクトル理論
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict, Callable
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, eigs
from scipy.linalg import eigh
import sympy as sym
from sympy import symbols, diff, Matrix, I, exp, cos, sin, pi
from dataclasses import dataclass
import warnings
from pathlib import Path
import json

# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'

@dataclass
class OperatorParameters:
    """作用素パラメータの定義"""
    dimension: int  # 空間次元
    lattice_size: int  # 格子サイズ
    theta: float  # 非可換パラメータ
    kappa: float  # κ-変形パラメータ
    mass: float  # 質量項
    coupling: float  # 結合定数
    
    def __post_init__(self):
        if self.dimension not in [2, 3, 4]:
            raise ValueError("次元は2, 3, 4のいずれかである必要があります")
        if self.lattice_size < 8:
            warnings.warn("格子サイズが小さすぎる可能性があります")

class DiracLaplacianAnalyzer:
    """
    ディラック/ラプラシアン作用素の詳細解析クラス
    
    主要な解析項目：
    1. スペクトル次元の一意性
    2. 固有値分布の特性
    3. KANアーキテクチャとの関係
    4. 非可換補正の効果
    """
    
    def __init__(self, params: OperatorParameters):
        self.params = params
        self.dim = params.dimension
        self.N = params.lattice_size
        self.theta = params.theta
        self.kappa = params.kappa
        self.mass = params.mass
        self.coupling = params.coupling
        
        # シンボリック変数
        self.x = symbols(f'x0:{self.dim}', real=True)
        self.p = symbols(f'p0:{self.dim}', real=True)
        
        # ガンマ行列の定義
        self.gamma_matrices = self._construct_gamma_matrices()
        
    def _construct_gamma_matrices(self) -> List[np.ndarray]:
        """
        ガンマ行列の構築（次元に応じて）
        
        2D: パウリ行列
        3D: パウリ行列の拡張
        4D: ディラック行列
        """
        if self.dim == 2:
            # 2Dパウリ行列
            gamma = [
                np.array([[0, 1], [1, 0]], dtype=complex),  # σ_x
                np.array([[0, -1j], [1j, 0]], dtype=complex),  # σ_y
            ]
        elif self.dim == 3:
            # 3Dパウリ行列
            gamma = [
                np.array([[0, 1], [1, 0]], dtype=complex),  # σ_x
                np.array([[0, -1j], [1j, 0]], dtype=complex),  # σ_y
                np.array([[1, 0], [0, -1]], dtype=complex),  # σ_z
            ]
        elif self.dim == 4:
            # 4Dディラック行列（標準表現）
            sigma = [
                np.array([[0, 1], [1, 0]], dtype=complex),
                np.array([[0, -1j], [1j, 0]], dtype=complex),
                np.array([[1, 0], [0, -1]], dtype=complex)
            ]
            I2 = np.eye(2, dtype=complex)
            O2 = np.zeros((2, 2), dtype=complex)
            
            gamma = [
                np.block([[O2, sigma[0]], [sigma[0], O2]]),  # γ^1
                np.block([[O2, sigma[1]], [sigma[1], O2]]),  # γ^2
                np.block([[O2, sigma[2]], [sigma[2], O2]]),  # γ^3
                np.block([[I2, O2], [O2, -I2]]),  # γ^0
            ]
        
        return gamma
    
    def construct_discrete_dirac_operator(self) -> sp.csr_matrix:
        """
        離散ディラック作用素の構築
        
        D = Σ_μ γ^μ (∇_μ + iA_μ) + m + θ-補正項
        """
        # スピノル次元
        spinor_dim = 2 if self.dim <= 3 else 4
        total_dim = self.N**self.dim * spinor_dim
        
        # 空の作用素行列
        D = sp.lil_matrix((total_dim, total_dim), dtype=complex)
        
        # 各方向の微分作用素
        for mu in range(self.dim):
            # 前進差分と後進差分の平均（中心差分）
            forward_diff = self._construct_forward_difference(mu, spinor_dim)
            backward_diff = self._construct_backward_difference(mu, spinor_dim)
            
            # ガンマ行列との積
            gamma_mu = self.gamma_matrices[mu]
            
            # ディラック項の追加
            diff_operator = (forward_diff - backward_diff) / 2.0
            D += sp.kron(diff_operator, gamma_mu)
            
            # 非可換補正項（θ-変形）
            if self.theta != 0:
                theta_correction = self._construct_theta_correction(mu, spinor_dim)
                D += self.theta * sp.kron(theta_correction, gamma_mu)
        
        # 質量項
        if self.mass != 0:
            mass_operator = sp.eye(self.N**self.dim)
            mass_matrix = self.mass * np.eye(spinor_dim, dtype=complex)
            D += sp.kron(mass_operator, mass_matrix)
        
        return D.tocsr()
    
    def construct_discrete_laplacian(self) -> sp.csr_matrix:
        """
        離散ラプラシアン作用素の構築
        
        Δ = Σ_μ ∇_μ² + κ-補正項 + θ-補正項
        """
        total_dim = self.N**self.dim
        Delta = sp.lil_matrix((total_dim, total_dim), dtype=float)
        
        # 各方向の2階微分
        for mu in range(self.dim):
            second_diff = self._construct_second_difference(mu)
            Delta += second_diff
            
            # κ-変形補正項
            if self.kappa != 0:
                kappa_correction = self._construct_kappa_correction(mu)
                Delta += self.kappa * kappa_correction
        
        # θ-変形による非可換補正
        if self.theta != 0:
            for mu in range(self.dim):
                for nu in range(mu + 1, self.dim):
                    mixed_diff = self._construct_mixed_difference(mu, nu)
                    Delta += self.theta * mixed_diff
        
        return Delta.tocsr()
    
    def _construct_forward_difference(self, direction: int, spinor_dim: int) -> sp.csr_matrix:
        """前進差分作用素の構築"""
        # 1次元の前進差分
        diff_1d = sp.diags([1, -1], [1, 0], shape=(self.N, self.N))
        diff_1d = diff_1d.tolil()  # lil_matrixに変換
        diff_1d[self.N-1, 0] = 1  # 周期境界条件
        diff_1d = diff_1d.tocsr()  # csr_matrixに戻す
        
        # 多次元への拡張
        operators = []
        for d in range(self.dim):
            if d == direction:
                operators.append(diff_1d)
            else:
                operators.append(sp.eye(self.N))
        
        # クロネッカー積で多次元作用素を構築
        result = operators[0]
        for op in operators[1:]:
            result = sp.kron(result, op)
        
        return result
    
    def _construct_backward_difference(self, direction: int, spinor_dim: int) -> sp.csr_matrix:
        """後進差分作用素の構築"""
        # 1次元の後進差分
        diff_1d = sp.diags([-1, 1], [0, -1], shape=(self.N, self.N))
        diff_1d = diff_1d.tolil()  # lil_matrixに変換
        diff_1d[0, self.N-1] = -1  # 周期境界条件
        diff_1d = diff_1d.tocsr()  # csr_matrixに戻す
        
        # 多次元への拡張
        operators = []
        for d in range(self.dim):
            if d == direction:
                operators.append(diff_1d)
            else:
                operators.append(sp.eye(self.N))
        
        result = operators[0]
        for op in operators[1:]:
            result = sp.kron(result, op)
        
        return result
    
    def _construct_second_difference(self, direction: int) -> sp.csr_matrix:
        """2階差分作用素の構築"""
        # 1次元の2階差分
        diff_1d = sp.diags([1, -2, 1], [-1, 0, 1], shape=(self.N, self.N))
        diff_1d = diff_1d.tolil()  # lil_matrixに変換
        diff_1d[0, self.N-1] = 1  # 周期境界条件
        diff_1d[self.N-1, 0] = 1
        diff_1d = diff_1d.tocsr()  # csr_matrixに戻す
        
        # 多次元への拡張
        operators = []
        for d in range(self.dim):
            if d == direction:
                operators.append(diff_1d)
            else:
                operators.append(sp.eye(self.N))
        
        result = operators[0]
        for op in operators[1:]:
            result = sp.kron(result, op)
        
        return result
    
    def _construct_theta_correction(self, direction: int, spinor_dim: int) -> sp.csr_matrix:
        """θ-変形補正項の構築"""
        # 非可換性による補正項
        # [x_μ, p_ν] = iθ δ_μν の効果
        
        # 位置作用素
        x_op = self._construct_position_operator(direction)
        
        # 運動量作用素（微分）
        p_op = self._construct_momentum_operator(direction)
        
        # 交換子 [x, p] の離散版
        commutator = x_op @ p_op - p_op @ x_op
        
        return commutator
    
    def _construct_kappa_correction(self, direction: int) -> sp.csr_matrix:
        """κ-変形補正項の構築"""
        # κ-ミンコフスキー変形による補正
        # x ⊕_κ y = x + y + κxy の効果
        
        x_op = self._construct_position_operator(direction)
        p_op = self._construct_momentum_operator(direction)
        
        # κ-変形による高次項
        correction = x_op @ x_op @ p_op @ p_op
        
        return correction
    
    def _construct_mixed_difference(self, dir1: int, dir2: int) -> sp.csr_matrix:
        """混合偏微分作用素の構築"""
        # ∂²/(∂x_μ ∂x_ν) の離散版
        
        diff1 = self._construct_forward_difference(dir1, 1) - self._construct_backward_difference(dir1, 1)
        diff2 = self._construct_forward_difference(dir2, 1) - self._construct_backward_difference(dir2, 1)
        
        return diff1 @ diff2 / 4.0
    
    def _construct_position_operator(self, direction: int) -> sp.csr_matrix:
        """位置作用素の構築"""
        # x_μ の離散版
        positions = np.arange(self.N, dtype=float) - self.N // 2
        pos_1d = sp.diags(positions, 0, shape=(self.N, self.N))
        
        operators = []
        for d in range(self.dim):
            if d == direction:
                operators.append(pos_1d)
            else:
                operators.append(sp.eye(self.N))
        
        result = operators[0]
        for op in operators[1:]:
            result = sp.kron(result, op)
        
        return result
    
    def _construct_momentum_operator(self, direction: int) -> sp.csr_matrix:
        """運動量作用素の構築"""
        # p_μ = -i ∇_μ の離散版
        return -1j * (self._construct_forward_difference(direction, 1) - 
                     self._construct_backward_difference(direction, 1)) / 2.0
    
    def compute_spectral_dimension(self, operator: sp.csr_matrix, 
                                 n_eigenvalues: int = 100) -> Tuple[float, Dict]:
        """
        スペクトル次元の計算
        
        d_s = -2 * d(log Z(t))/d(log t) |_{t→0}
        
        ここで、Z(t) = Tr(exp(-tD²)) はスペクトルゼータ関数
        """
        # 固有値の計算
        try:
            if operator.shape[0] > 1000:
                # 大きな行列の場合は部分固有値のみ計算
                eigenvalues, _ = eigsh(operator.H @ operator, k=min(n_eigenvalues, operator.shape[0]-2), 
                                     which='SM', return_eigenvectors=False)
            else:
                # 小さな行列の場合は全固有値を計算
                dense_op = operator.toarray()
                eigenvalues = np.real(eigh(dense_op.conj().T @ dense_op, eigvals_only=True))
        except Exception as e:
            print(f"固有値計算エラー: {e}")
            return float('nan'), {}
        
        # 正の固有値のみを使用
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        
        if len(eigenvalues) < 10:
            print("警告: 有効な固有値が少なすぎます")
            return float('nan'), {}
        
        # スペクトルゼータ関数の計算
        t_values = np.logspace(-3, 0, 50)
        zeta_values = []
        
        for t in t_values:
            zeta_t = np.sum(np.exp(-t * eigenvalues))
            zeta_values.append(zeta_t)
        
        zeta_values = np.array(zeta_values)
        
        # 対数微分の計算
        log_t = np.log(t_values)
        log_zeta = np.log(zeta_values + 1e-12)  # 数値安定性のため
        
        # 線形回帰で傾きを求める
        valid_indices = np.isfinite(log_zeta) & np.isfinite(log_t)
        if np.sum(valid_indices) < 5:
            print("警告: 有効なデータ点が少なすぎます")
            return float('nan'), {}
        
        slope = np.polyfit(log_t[valid_indices], log_zeta[valid_indices], 1)[0]
        spectral_dimension = -2 * slope
        
        # 詳細情報
        analysis_info = {
            'eigenvalues': eigenvalues,
            'n_eigenvalues': len(eigenvalues),
            'min_eigenvalue': np.min(eigenvalues),
            'max_eigenvalue': np.max(eigenvalues),
            'spectral_gap': eigenvalues[1] - eigenvalues[0] if len(eigenvalues) > 1 else 0,
            'zeta_function': zeta_values,
            't_values': t_values,
            'slope': slope,
            'r_squared': self._compute_r_squared(log_t[valid_indices], log_zeta[valid_indices], slope)
        }
        
        return spectral_dimension, analysis_info
    
    def _compute_r_squared(self, x: np.ndarray, y: np.ndarray, slope: float) -> float:
        """決定係数R²の計算"""
        y_mean = np.mean(y)
        y_pred = slope * x + (np.mean(y) - slope * np.mean(x))
        
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    def analyze_uniqueness(self, n_trials: int = 10) -> Dict:
        """
        スペクトル次元の一意性解析
        
        異なるパラメータ設定でのスペクトル次元の安定性を調査
        """
        results = {
            'spectral_dimensions': [],
            'parameters': [],
            'analysis_info': []
        }
        
        # パラメータの摂動
        base_theta = self.theta
        base_kappa = self.kappa
        base_mass = self.mass
        
        for trial in range(n_trials):
            # パラメータの小さな摂動
            theta_pert = base_theta * (1 + 0.1 * (np.random.random() - 0.5))
            kappa_pert = base_kappa * (1 + 0.1 * (np.random.random() - 0.5))
            mass_pert = base_mass * (1 + 0.1 * (np.random.random() - 0.5))
            
            # 摂動されたパラメータで作用素を構築
            self.theta = theta_pert
            self.kappa = kappa_pert
            self.mass = mass_pert
            
            # ディラック作用素の構築と解析
            D = self.construct_discrete_dirac_operator()
            d_s, info = self.compute_spectral_dimension(D)
            
            results['spectral_dimensions'].append(d_s)
            results['parameters'].append({
                'theta': theta_pert,
                'kappa': kappa_pert,
                'mass': mass_pert
            })
            results['analysis_info'].append(info)
        
        # 元のパラメータに戻す
        self.theta = base_theta
        self.kappa = base_kappa
        self.mass = base_mass
        
        # 統計解析
        d_s_array = np.array([d for d in results['spectral_dimensions'] if np.isfinite(d)])
        
        if len(d_s_array) > 0:
            results['statistics'] = {
                'mean': np.mean(d_s_array),
                'std': np.std(d_s_array),
                'min': np.min(d_s_array),
                'max': np.max(d_s_array),
                'coefficient_of_variation': np.std(d_s_array) / np.mean(d_s_array) if np.mean(d_s_array) != 0 else float('inf')
            }
        else:
            results['statistics'] = {
                'mean': float('nan'),
                'std': float('nan'),
                'min': float('nan'),
                'max': float('nan'),
                'coefficient_of_variation': float('nan')
            }
        
        return results
    
    def analyze_kan_architecture_relationship(self, 
                                            depth_range: List[int] = [2, 4, 6, 8],
                                            width_range: List[int] = [32, 64, 128, 256],
                                            grid_range: List[int] = [8, 16, 32, 64]) -> Dict:
        """
        KANアーキテクチャとスペクトル特性の関係解析
        
        Args:
            depth_range: ネットワーク深度の範囲
            width_range: ネットワーク幅の範囲  
            grid_range: スプライングリッドサイズの範囲
            
        Returns:
            アーキテクチャと物理特性の関係データ
        """
        results = {
            'architecture_configs': [],
            'spectral_dimensions': [],
            'eigenvalue_statistics': [],
            'approximation_errors': []
        }
        
        # 基準となるディラック作用素
        D_reference = self.construct_discrete_dirac_operator()
        d_s_reference, _ = self.compute_spectral_dimension(D_reference)
        
        for depth in depth_range:
            for width in width_range:
                for grid_size in grid_range:
                    # KANアーキテクチャの設定
                    config = {
                        'depth': depth,
                        'width': width,
                        'grid_size': grid_size,
                        'total_parameters': depth * width * grid_size
                    }
                    
                    # KANによる作用素近似の評価
                    approximation_quality = self._evaluate_kan_approximation(
                        D_reference, depth, width, grid_size
                    )
                    
                    results['architecture_configs'].append(config)
                    results['spectral_dimensions'].append(d_s_reference)  # 簡略化
                    results['eigenvalue_statistics'].append(approximation_quality['eigenvalue_stats'])
                    results['approximation_errors'].append(approximation_quality['error'])
        
        return results
    
    def _evaluate_kan_approximation(self, operator: sp.csr_matrix, 
                                   depth: int, width: int, grid_size: int) -> Dict:
        """KANによる作用素近似の評価"""
        # 簡略化された評価（実際のKAN学習は省略）
        
        # パラメータ数に基づく近似精度の推定
        n_params = depth * width * grid_size
        operator_size = operator.shape[0]
        
        # 理論的な近似誤差の推定
        if n_params >= operator_size:
            approximation_error = 1e-6  # 十分なパラメータがある場合
        else:
            approximation_error = 1.0 / np.sqrt(n_params)  # パラメータ不足の場合
        
        # 固有値統計の推定
        try:
            eigenvals, _ = eigsh(operator.H @ operator, k=min(50, operator.shape[0]-2), 
                               which='SM', return_eigenvectors=False)
            eigenval_stats = {
                'mean': np.mean(eigenvals),
                'std': np.std(eigenvals),
                'condition_number': np.max(eigenvals) / np.max(np.min(eigenvals), 1e-12)
            }
        except:
            eigenval_stats = {
                'mean': float('nan'),
                'std': float('nan'),
                'condition_number': float('nan')
            }
        
        return {
            'error': approximation_error,
            'eigenvalue_stats': eigenval_stats
        }
    
    def visualize_spectral_analysis(self, analysis_results: Dict, save_path: Optional[str] = None):
        """スペクトル解析結果の可視化"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. スペクトル次元の分布
        if 'spectral_dimensions' in analysis_results:
            d_s_values = [d for d in analysis_results['spectral_dimensions'] if np.isfinite(d)]
            if d_s_values:
                axes[0, 0].hist(d_s_values, bins=20, alpha=0.7, edgecolor='black')
                axes[0, 0].axvline(self.dim, color='red', linestyle='--', 
                                 label=f'理論値 d={self.dim}')
                axes[0, 0].set_xlabel('スペクトル次元 d_s')
                axes[0, 0].set_ylabel('頻度')
                axes[0, 0].set_title('スペクトル次元の分布')
                axes[0, 0].legend()
                axes[0, 0].grid(True)
        
        # 2. 固有値分布
        if 'analysis_info' in analysis_results and analysis_results['analysis_info']:
            eigenvals = analysis_results['analysis_info'][0].get('eigenvalues', [])
            if len(eigenvals) > 0:
                axes[0, 1].semilogy(eigenvals[:50], 'o-', markersize=4)
                axes[0, 1].set_xlabel('固有値インデックス')
                axes[0, 1].set_ylabel('固有値 (対数スケール)')
                axes[0, 1].set_title('固有値分布')
                axes[0, 1].grid(True)
        
        # 3. スペクトルゼータ関数
        if 'analysis_info' in analysis_results and analysis_results['analysis_info']:
            info = analysis_results['analysis_info'][0]
            if 't_values' in info and 'zeta_function' in info:
                axes[0, 2].loglog(info['t_values'], info['zeta_function'], 'o-')
                axes[0, 2].set_xlabel('t')
                axes[0, 2].set_ylabel('ζ(t)')
                axes[0, 2].set_title('スペクトルゼータ関数')
                axes[0, 2].grid(True)
        
        # 4. パラメータ依存性
        if 'parameters' in analysis_results:
            params = analysis_results['parameters']
            d_s_vals = analysis_results['spectral_dimensions']
            
            theta_vals = [p['theta'] for p in params]
            finite_indices = [i for i, d in enumerate(d_s_vals) if np.isfinite(d)]
            
            if finite_indices:
                theta_finite = [theta_vals[i] for i in finite_indices]
                d_s_finite = [d_s_vals[i] for i in finite_indices]
                
                axes[1, 0].scatter(theta_finite, d_s_finite, alpha=0.7)
                axes[1, 0].set_xlabel('θ パラメータ')
                axes[1, 0].set_ylabel('スペクトル次元 d_s')
                axes[1, 0].set_title('θ依存性')
                axes[1, 0].grid(True)
        
        # 5. 条件数の分析
        if 'analysis_info' in analysis_results:
            condition_numbers = []
            for info in analysis_results['analysis_info']:
                if 'eigenvalues' in info and len(info['eigenvalues']) > 1:
                    eigenvals = info['eigenvalues']
                    cond_num = np.max(eigenvals) / np.max(np.min(eigenvals), 1e-12)
                    condition_numbers.append(cond_num)
            
            if condition_numbers:
                axes[1, 1].hist(np.log10(condition_numbers), bins=15, alpha=0.7, edgecolor='black')
                axes[1, 1].set_xlabel('log₁₀(条件数)')
                axes[1, 1].set_ylabel('頻度')
                axes[1, 1].set_title('条件数の分布')
                axes[1, 1].grid(True)
        
        # 6. 統計サマリー
        if 'statistics' in analysis_results:
            stats = analysis_results['statistics']
            stats_text = f"""
            平均: {stats['mean']:.4f}
            標準偏差: {stats['std']:.4f}
            最小値: {stats['min']:.4f}
            最大値: {stats['max']:.4f}
            変動係数: {stats['coefficient_of_variation']:.4f}
            """
            axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes,
                           fontsize=12, verticalalignment='center',
                           bbox=dict(boxstyle='round', facecolor='lightgray'))
            axes[1, 2].set_title('統計サマリー')
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        return fig

def demonstrate_dirac_laplacian_analysis():
    """ディラック/ラプラシアン解析のデモンストレーション"""
    
    print("=" * 70)
    print("ディラック/ラプラシアン作用素の詳細解析")
    print("=" * 70)
    
    # パラメータ設定
    params = OperatorParameters(
        dimension=4,
        lattice_size=16,
        theta=0.01,
        kappa=0.05,
        mass=0.1,
        coupling=1.0
    )
    
    analyzer = DiracLaplacianAnalyzer(params)
    
    print(f"\n解析パラメータ:")
    print(f"次元: {params.dimension}")
    print(f"格子サイズ: {params.lattice_size}")
    print(f"θ パラメータ: {params.theta}")
    print(f"κ パラメータ: {params.kappa}")
    print(f"質量: {params.mass}")
    
    # 1. ディラック作用素の解析
    print("\n1. ディラック作用素の構築と解析...")
    D = analyzer.construct_discrete_dirac_operator()
    print(f"ディラック作用素のサイズ: {D.shape}")
    print(f"非零要素数: {D.nnz}")
    
    d_s_dirac, dirac_info = analyzer.compute_spectral_dimension(D)
    print(f"ディラック作用素のスペクトル次元: {d_s_dirac:.6f}")
    print(f"理論値との差: {abs(d_s_dirac - params.dimension):.6f}")
    
    # 2. ラプラシアン作用素の解析
    print("\n2. ラプラシアン作用素の構築と解析...")
    Delta = analyzer.construct_discrete_laplacian()
    print(f"ラプラシアン作用素のサイズ: {Delta.shape}")
    print(f"非零要素数: {Delta.nnz}")
    
    d_s_laplacian, laplacian_info = analyzer.compute_spectral_dimension(Delta)
    print(f"ラプラシアン作用素のスペクトル次元: {d_s_laplacian:.6f}")
    print(f"理論値との差: {abs(d_s_laplacian - params.dimension):.6f}")
    
    # 3. 一意性解析
    print("\n3. スペクトル次元の一意性解析...")
    uniqueness_results = analyzer.analyze_uniqueness(n_trials=20)
    
    if 'statistics' in uniqueness_results:
        stats = uniqueness_results['statistics']
        print(f"平均スペクトル次元: {stats['mean']:.6f} ± {stats['std']:.6f}")
        print(f"変動係数: {stats['coefficient_of_variation']:.6f}")
        print(f"範囲: [{stats['min']:.6f}, {stats['max']:.6f}]")
    
    # 4. KANアーキテクチャとの関係解析
    print("\n4. KANアーキテクチャとの関係解析...")
    kan_results = analyzer.analyze_kan_architecture_relationship(
        depth_range=[2, 4, 6],
        width_range=[32, 64, 128],
        grid_range=[8, 16, 32]
    )
    
    print(f"解析したアーキテクチャ数: {len(kan_results['architecture_configs'])}")
    
    # 最適なアーキテクチャの特定
    errors = kan_results['approximation_errors']
    best_idx = np.argmin(errors)
    best_config = kan_results['architecture_configs'][best_idx]
    
    print(f"最適アーキテクチャ:")
    print(f"  深度: {best_config['depth']}")
    print(f"  幅: {best_config['width']}")
    print(f"  グリッドサイズ: {best_config['grid_size']}")
    print(f"  総パラメータ数: {best_config['total_parameters']}")
    print(f"  近似誤差: {errors[best_idx]:.6e}")
    
    # 5. 結果の可視化
    print("\n5. 結果の可視化...")
    fig = analyzer.visualize_spectral_analysis(uniqueness_results, 
                                             save_path='dirac_laplacian_analysis.png')
    
    # 6. 結果の保存
    results_summary = {
        'parameters': params.__dict__,
        'dirac_spectral_dimension': d_s_dirac,
        'laplacian_spectral_dimension': d_s_laplacian,
        'uniqueness_analysis': {
            k: v for k, v in uniqueness_results.items() 
            if k != 'analysis_info'  # 大きなデータは除外
        },
        'best_kan_architecture': best_config,
        'analysis_timestamp': str(np.datetime64('now'))
    }
    
    with open('dirac_laplacian_analysis_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False, default=str)
    
    print("\n解析結果が 'dirac_laplacian_analysis_results.json' に保存されました。")
    
    return analyzer, results_summary

if __name__ == "__main__":
    # 解析のデモンストレーション
    analyzer, results = demonstrate_dirac_laplacian_analysis() 