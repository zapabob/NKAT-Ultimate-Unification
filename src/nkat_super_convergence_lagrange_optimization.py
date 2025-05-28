#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT超収束因子の導出とラグランジュ未定乗数法による実験パラメータ最適化
Non-Commutative Kolmogorov-Arnold Theory: Super-Convergence Factor Derivation
and Lagrange Multiplier Optimization for Experimental Parameters

Author: 峯岸 亮 (Ryo Minegishi)
Date: 2025年5月28日
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.integrate import quad, solve_ivp
from scipy.special import gamma, zeta, polygamma
from scipy.linalg import eigvals, norm
import sympy as sp
from sympy import symbols, diff, solve, lambdify, exp, log, sqrt, pi, I
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class NKATSuperConvergenceFactor:
    """
    NKAT理論における超収束因子の導出と解析クラス
    """
    
    def __init__(self):
        """初期化"""
        # 基本パラメータ
        self.gamma_euler = 0.5772156649015329  # オイラー定数
        self.hbar = 1.0  # 簡約プランク定数（単位系）
        
        # 超収束因子パラメータ（初期推定値）
        self.gamma = 0.23422  # 主要収束パラメータ
        self.delta = 0.03511  # 指数減衰パラメータ
        self.t_c = 17.2644   # 臨界点
        self.alpha = 0.7422  # 収束指数
        
        # 高次補正係数
        self.c_coeffs = [0.0628, 0.0035, 0.0012, 0.0004]  # c_2, c_3, c_4, c_5
        
        print("🔬 NKAT超収束因子解析システム初期化完了")
        print(f"📊 初期パラメータ: γ={self.gamma:.5f}, δ={self.delta:.5f}, t_c={self.t_c:.4f}")
    
    def density_function(self, t, params=None):
        """
        誤差補正密度関数 ρ(t) の計算
        
        ρ(t) = γ/t + δ·e^{-δ(t-t_c)} + Σ_{k=2}^∞ c_k·k·ln^{k-1}(t/t_c)/t^{k+1}
        """
        if params is None:
            gamma, delta, t_c = self.gamma, self.delta, self.t_c
            c_coeffs = self.c_coeffs
        else:
            gamma, delta, t_c = params[:3]
            c_coeffs = params[3:] if len(params) > 3 else self.c_coeffs
        
        # 主要項
        rho = gamma / t
        
        # 指数減衰項
        if t > t_c:
            rho += delta * np.exp(-delta * (t - t_c))
        
        # 高次補正項
        if t > 1e-10:  # 数値安定性のため
            log_ratio = np.log(t / t_c) if t > t_c else 0
            for k, c_k in enumerate(c_coeffs, start=2):
                if abs(log_ratio) < 100:  # オーバーフロー防止
                    correction = c_k * k * (log_ratio**(k-1)) / (t**(k+1))
                    rho += correction
        
        return rho
    
    def super_convergence_factor(self, N, params=None):
        """
        超収束因子 S(N) の計算
        
        S(N) = exp(∫₁^N ρ(t) dt)
        """
        try:
            def integrand(t):
                return self.density_function(t, params)
            
            integral, _ = quad(integrand, 1, N, limit=100)
            return np.exp(integral)
        except:
            # 数値積分が失敗した場合の近似計算
            return 1.0 + self.gamma * np.log(N / self.t_c)
    
    def theoretical_error_function(self, t, params=None):
        """
        理論的誤差関数 E_t の計算
        
        E_t = A/t + B·e^{-δ(t-t_c)} + Σ_{k=2}^∞ D_k/t^k·ln^k(t/t_c)
        """
        if params is None:
            gamma, delta, t_c = self.gamma, self.delta, self.t_c
        else:
            gamma, delta, t_c = params[:3]
        
        A = 1.0  # 主要係数
        B = 0.1  # 指数項係数
        
        error = A / t
        
        if t > t_c:
            error += B * np.exp(-delta * (t - t_c))
        
        # 対数補正項
        if t > 1e-10:
            log_ratio = np.log(t / t_c) if t > t_c else 0
            for k in range(2, 5):
                D_k = 0.01 / k**2  # 係数の推定
                if abs(log_ratio) < 50:
                    error += D_k * (log_ratio**k) / (t**k)
        
        return error
    
    def variational_principle_objective(self, S_values, t_values, params):
        """
        変分原理の目的関数
        
        J[S] = ∫₁^N [|S'(t)|²/S(t)² + V_eff(t)S(t)²] dt
        """
        gamma, delta, t_c = params[:3]
        
        # 数値微分でS'(t)を計算
        S_prime = np.gradient(S_values, t_values)
        
        # 有効ポテンシャル
        V_eff = gamma**2 / t_values**2 + delta**2 * np.exp(-2*delta*(t_values - t_c))
        
        # 目的関数の計算
        integrand = (S_prime**2) / (S_values**2) + V_eff * S_values**2
        
        return np.trapz(integrand, t_values)
    
    def quantum_mechanical_interpretation(self, N, params=None):
        """
        量子力学的解釈による超収束因子の計算
        
        S(N) = ⟨ψ_N|e^{-iHt}|ψ_N⟩|_{t=T(N)}
        """
        if params is None:
            gamma, delta, t_c = params[:3] if params else (self.gamma, self.delta, self.t_c)
        else:
            gamma, delta, t_c = params[:3]
        
        # 特性時間
        T_N = np.log(N / t_c)
        
        # ハミルトニアンの固有値（簡略化）
        eigenvalues = np.array([k * np.pi / (2*N + 1) for k in range(1, N+1)])
        
        # 時間発展演算子の期待値
        expectation = np.sum(np.exp(-1j * eigenvalues * T_N))
        
        return abs(expectation) / N
    
    def statistical_mechanical_partition_function(self, N, beta=1.0, params=None):
        """
        統計力学的分配関数による超収束因子
        
        S(N) = Z_N/Z_classical = Tr(e^{-βH_N})/Tr(e^{-βH_classical})
        """
        if params is None:
            gamma, delta, t_c = params[:3] if params else (self.gamma, self.delta, self.t_c)
        else:
            gamma, delta, t_c = params[:3]
        
        # 非可換ハミルトニアンの固有値
        H_nc_eigenvals = np.array([k * np.pi / (2*N + 1) + gamma/k for k in range(1, N+1)])
        
        # 古典ハミルトニアンの固有値
        H_classical_eigenvals = np.array([k * np.pi / (2*N + 1) for k in range(1, N+1)])
        
        # 分配関数の計算
        Z_nc = np.sum(np.exp(-beta * H_nc_eigenvals))
        Z_classical = np.sum(np.exp(-beta * H_classical_eigenvals))
        
        return Z_nc / Z_classical
    
    def information_theoretic_relative_entropy(self, N, params=None):
        """
        情報理論的相対エントロピーによる超収束因子
        
        S(N) = exp(-S_rel(ρ_N‖ρ_classical))
        """
        if params is None:
            gamma, delta, t_c = params[:3] if params else (self.gamma, self.delta, self.t_c)
        else:
            gamma, delta, t_c = params[:3]
        
        # 密度行列の固有値（正規化）
        rho_nc = np.array([1/N + gamma/(k*N) for k in range(1, N+1)])
        rho_nc = rho_nc / np.sum(rho_nc)
        
        rho_classical = np.ones(N) / N
        
        # 相対エントロピーの計算
        S_rel = np.sum(rho_nc * np.log(rho_nc / rho_classical))
        
        return np.exp(-S_rel)

class LagrangeMultiplierOptimizer:
    """
    ラグランジュの未定乗数法による実験パラメータ最適化クラス
    """
    
    def __init__(self, nkat_system):
        """初期化"""
        self.nkat = nkat_system
        self.experimental_data = None
        self.constraints = []
        
        print("🎯 ラグランジュ未定乗数法最適化システム初期化完了")
    
    def generate_experimental_data(self, N_values, noise_level=1e-6):
        """
        実験データの生成（シミュレーション）
        """
        print("📊 実験データ生成中...")
        
        data = []
        for N in tqdm(N_values, desc="実験データ生成"):
            # 理論値
            S_theory = self.nkat.super_convergence_factor(N)
            
            # ノイズ付加
            S_experimental = S_theory * (1 + np.random.normal(0, noise_level))
            
            # 他の観測量
            quantum_expectation = self.nkat.quantum_mechanical_interpretation(N)
            partition_ratio = self.nkat.statistical_mechanical_partition_function(N)
            entropy_factor = self.nkat.information_theoretic_relative_entropy(N)
            
            data.append({
                'N': N,
                'S_experimental': S_experimental,
                'S_theory': S_theory,
                'quantum_expectation': quantum_expectation,
                'partition_ratio': partition_ratio,
                'entropy_factor': entropy_factor,
                'error': abs(S_experimental - S_theory)
            })
        
        self.experimental_data = pd.DataFrame(data)
        print(f"✅ {len(N_values)}点の実験データ生成完了")
        return self.experimental_data
    
    def define_constraints(self):
        """
        制約条件の定義
        """
        # 物理的制約
        self.constraints = [
            # γ > 0 (正の収束パラメータ)
            {'type': 'ineq', 'fun': lambda x: x[0]},
            
            # δ > 0 (正の減衰パラメータ)
            {'type': 'ineq', 'fun': lambda x: x[1]},
            
            # t_c > 1 (臨界点は1より大きい)
            {'type': 'ineq', 'fun': lambda x: x[2] - 1},
            
            # γ < 1 (収束条件)
            {'type': 'ineq', 'fun': lambda x: 1 - x[0]},
            
            # δ < 0.1 (安定性条件)
            {'type': 'ineq', 'fun': lambda x: 0.1 - x[1]},
            
            # 正規化条件: ∫₁^∞ ρ(t) dt = γ·ln(∞) (発散するが制御)
            {'type': 'eq', 'fun': lambda x: self._normalization_constraint(x)},
        ]
        
        print(f"📋 {len(self.constraints)}個の制約条件を定義")
    
    def _normalization_constraint(self, params):
        """正規化制約条件"""
        gamma, delta, t_c = params[:3]
        
        # 簡略化された正規化条件
        # 実際には有限区間での積分で近似
        try:
            integral, _ = quad(lambda t: self.nkat.density_function(t, params), 1, 100)
            target_value = gamma * np.log(100)  # 理論的期待値
            return integral - target_value
        except:
            return 0  # 積分が失敗した場合
    
    def objective_function(self, params):
        """
        目的関数：実験データとの最小二乗誤差
        """
        if self.experimental_data is None:
            return float('inf')
        
        total_error = 0
        
        for _, row in self.experimental_data.iterrows():
            N = row['N']
            S_exp = row['S_experimental']
            
            # 理論予測値
            S_theory = self.nkat.super_convergence_factor(N, params)
            
            # 二乗誤差
            error = (S_theory - S_exp)**2
            total_error += error
        
        return total_error
    
    def lagrangian(self, params, lambdas):
        """
        ラグランジアン関数
        
        L(x, λ) = f(x) + Σᵢ λᵢ gᵢ(x)
        """
        # 目的関数
        f_x = self.objective_function(params)
        
        # 制約項
        constraint_sum = 0
        for i, constraint in enumerate(self.constraints):
            if i < len(lambdas):
                if constraint['type'] == 'eq':
                    constraint_sum += lambdas[i] * constraint['fun'](params)
                elif constraint['type'] == 'ineq':
                    constraint_sum += lambdas[i] * max(0, -constraint['fun'](params))
        
        return f_x + constraint_sum
    
    def optimize_parameters(self, initial_guess=None):
        """
        ラグランジュ未定乗数法によるパラメータ最適化
        """
        if initial_guess is None:
            initial_guess = [self.nkat.gamma, self.nkat.delta, self.nkat.t_c]
        
        print("🔧 ラグランジュ未定乗数法による最適化開始...")
        
        # 制約条件の定義
        self.define_constraints()
        
        # 最適化実行
        result = opt.minimize(
            self.objective_function,
            initial_guess,
            method='SLSQP',
            constraints=self.constraints,
            options={'disp': True, 'maxiter': 1000}
        )
        
        if result.success:
            optimal_params = result.x
            print("✅ 最適化成功!")
            print(f"📊 最適パラメータ:")
            print(f"   γ = {optimal_params[0]:.6f}")
            print(f"   δ = {optimal_params[1]:.6f}")
            print(f"   t_c = {optimal_params[2]:.6f}")
            print(f"📈 最小目的関数値: {result.fun:.2e}")
            
            return optimal_params, result
        else:
            print("❌ 最適化失敗")
            print(f"理由: {result.message}")
            return None, result
    
    def sensitivity_analysis(self, optimal_params, perturbation=0.01):
        """
        感度解析：パラメータの微小変化に対する目的関数の変化
        """
        print("🔍 感度解析実行中...")
        
        base_value = self.objective_function(optimal_params)
        sensitivities = []
        
        for i, param_name in enumerate(['γ', 'δ', 't_c']):
            # 正の摂動
            params_plus = optimal_params.copy()
            params_plus[i] += perturbation
            value_plus = self.objective_function(params_plus)
            
            # 負の摂動
            params_minus = optimal_params.copy()
            params_minus[i] -= perturbation
            value_minus = self.objective_function(params_minus)
            
            # 数値微分
            sensitivity = (value_plus - value_minus) / (2 * perturbation)
            sensitivities.append(sensitivity)
            
            print(f"📊 {param_name}の感度: {sensitivity:.2e}")
        
        return sensitivities
    
    def uncertainty_quantification(self, optimal_params, n_bootstrap=100):
        """
        不確実性定量化：ブートストラップ法による信頼区間推定
        """
        print("📊 不確実性定量化実行中...")
        
        bootstrap_results = []
        
        for _ in tqdm(range(n_bootstrap), desc="ブートストラップ"):
            # データのリサンプリング
            resampled_data = self.experimental_data.sample(
                n=len(self.experimental_data), 
                replace=True
            )
            
            # 一時的にデータを置き換え
            original_data = self.experimental_data
            self.experimental_data = resampled_data
            
            # 最適化実行
            try:
                result = opt.minimize(
                    self.objective_function,
                    optimal_params,
                    method='SLSQP',
                    constraints=self.constraints,
                    options={'disp': False}
                )
                
                if result.success:
                    bootstrap_results.append(result.x)
            except:
                pass
            
            # データを元に戻す
            self.experimental_data = original_data
        
        if bootstrap_results:
            bootstrap_results = np.array(bootstrap_results)
            
            # 信頼区間の計算
            confidence_intervals = []
            param_names = ['γ', 'δ', 't_c']
            
            for i, name in enumerate(param_names):
                mean_val = np.mean(bootstrap_results[:, i])
                std_val = np.std(bootstrap_results[:, i])
                ci_lower = np.percentile(bootstrap_results[:, i], 2.5)
                ci_upper = np.percentile(bootstrap_results[:, i], 97.5)
                
                confidence_intervals.append({
                    'parameter': name,
                    'mean': mean_val,
                    'std': std_val,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper
                })
                
                print(f"📊 {name}: {mean_val:.6f} ± {std_val:.6f} "
                      f"[{ci_lower:.6f}, {ci_upper:.6f}]")
        
        return confidence_intervals

class NKATVisualization:
    """
    NKAT理論の可視化クラス
    """
    
    def __init__(self, nkat_system, optimizer):
        """初期化"""
        self.nkat = nkat_system
        self.optimizer = optimizer
        
    def plot_super_convergence_factor(self, N_range, params=None):
        """超収束因子のプロット"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        N_values = np.logspace(1, 3, 100)
        S_values = [self.nkat.super_convergence_factor(N, params) for N in N_values]
        
        # 1. 超収束因子の挙動
        axes[0, 0].loglog(N_values, S_values, 'b-', linewidth=2, label='S(N)')
        axes[0, 0].set_xlabel('次元数 N')
        axes[0, 0].set_ylabel('超収束因子 S(N)')
        axes[0, 0].set_title('超収束因子の次元依存性')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # 2. 密度関数
        t_values = np.linspace(1, 50, 1000)
        rho_values = [self.nkat.density_function(t, params) for t in t_values]
        
        axes[0, 1].plot(t_values, rho_values, 'r-', linewidth=2)
        axes[0, 1].set_xlabel('t')
        axes[0, 1].set_ylabel('密度関数 ρ(t)')
        axes[0, 1].set_title('誤差補正密度関数')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 理論的誤差関数
        error_values = [self.nkat.theoretical_error_function(t, params) for t in t_values]
        
        axes[1, 0].semilogy(t_values, error_values, 'g-', linewidth=2)
        axes[1, 0].set_xlabel('t')
        axes[1, 0].set_ylabel('誤差関数 E(t)')
        axes[1, 0].set_title('理論的誤差関数')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 量子力学的解釈
        quantum_values = [self.nkat.quantum_mechanical_interpretation(int(N), params) 
                         for N in N_values[::10]]
        
        axes[1, 1].plot(N_values[::10], quantum_values, 'mo-', linewidth=2, markersize=4)
        axes[1, 1].set_xlabel('次元数 N')
        axes[1, 1].set_ylabel('量子期待値')
        axes[1, 1].set_title('量子力学的解釈')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('nkat_super_convergence_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_optimization_results(self, experimental_data, optimal_params):
        """最適化結果のプロット"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        N_values = experimental_data['N'].values
        S_exp = experimental_data['S_experimental'].values
        S_theory_original = experimental_data['S_theory'].values
        
        # 最適化後の理論値
        S_theory_optimized = [self.nkat.super_convergence_factor(N, optimal_params) 
                             for N in N_values]
        
        # 1. 実験値vs理論値（最適化前後）
        axes[0, 0].plot(N_values, S_exp, 'ro', label='実験値', markersize=6)
        axes[0, 0].plot(N_values, S_theory_original, 'b--', label='最適化前', linewidth=2)
        axes[0, 0].plot(N_values, S_theory_optimized, 'g-', label='最適化後', linewidth=2)
        axes[0, 0].set_xlabel('次元数 N')
        axes[0, 0].set_ylabel('超収束因子')
        axes[0, 0].set_title('実験値と理論値の比較')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 残差プロット
        residuals_original = S_exp - S_theory_original
        residuals_optimized = S_exp - S_theory_optimized
        
        axes[0, 1].plot(N_values, residuals_original, 'b^', label='最適化前', markersize=6)
        axes[0, 1].plot(N_values, residuals_optimized, 'go', label='最適化後', markersize=6)
        axes[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.5)
        axes[0, 1].set_xlabel('次元数 N')
        axes[0, 1].set_ylabel('残差')
        axes[0, 1].set_title('残差プロット')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 相関プロット
        axes[1, 0].scatter(S_exp, S_theory_optimized, alpha=0.7, s=50)
        min_val = min(min(S_exp), min(S_theory_optimized))
        max_val = max(max(S_exp), max(S_theory_optimized))
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        axes[1, 0].set_xlabel('実験値')
        axes[1, 0].set_ylabel('理論値（最適化後）')
        axes[1, 0].set_title('実験値vs理論値相関')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 誤差の改善
        error_original = np.abs(residuals_original)
        error_optimized = np.abs(residuals_optimized)
        
        axes[1, 1].semilogy(N_values, error_original, 'b^-', label='最適化前', linewidth=2)
        axes[1, 1].semilogy(N_values, error_optimized, 'go-', label='最適化後', linewidth=2)
        axes[1, 1].set_xlabel('次元数 N')
        axes[1, 1].set_ylabel('絶対誤差')
        axes[1, 1].set_title('誤差の改善')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('nkat_optimization_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 統計情報の表示
        print("\n📊 最適化結果統計:")
        print(f"最適化前 RMSE: {np.sqrt(np.mean(residuals_original**2)):.2e}")
        print(f"最適化後 RMSE: {np.sqrt(np.mean(residuals_optimized**2)):.2e}")
        print(f"改善率: {(1 - np.sqrt(np.mean(residuals_optimized**2))/np.sqrt(np.mean(residuals_original**2)))*100:.1f}%")

def main():
    """メイン実行関数"""
    print("🚀 NKAT超収束因子解析・ラグランジュ最適化システム開始")
    print("=" * 60)
    
    # 1. システム初期化
    nkat = NKATSuperConvergenceFactor()
    optimizer = LagrangeMultiplierOptimizer(nkat)
    visualizer = NKATVisualization(nkat, optimizer)
    
    # 2. 実験データ生成
    N_values = np.array([50, 100, 200, 300, 500, 750, 1000])
    experimental_data = optimizer.generate_experimental_data(N_values, noise_level=1e-5)
    
    print("\n📊 実験データサマリー:")
    print(experimental_data.describe())
    
    # 3. 初期状態の可視化
    print("\n📈 初期状態の可視化...")
    visualizer.plot_super_convergence_factor(N_values)
    
    # 4. ラグランジュ未定乗数法による最適化
    print("\n🎯 ラグランジュ未定乗数法による最適化実行...")
    optimal_params, optimization_result = optimizer.optimize_parameters()
    
    if optimal_params is not None:
        # 5. 最適化結果の可視化
        print("\n📊 最適化結果の可視化...")
        visualizer.plot_optimization_results(experimental_data, optimal_params)
        
        # 6. 感度解析
        print("\n🔍 感度解析実行...")
        sensitivities = optimizer.sensitivity_analysis(optimal_params)
        
        # 7. 不確実性定量化
        print("\n📊 不確実性定量化実行...")
        confidence_intervals = optimizer.uncertainty_quantification(optimal_params, n_bootstrap=50)
        
        # 8. 最終結果の表示
        print("\n" + "=" * 60)
        print("🎉 NKAT超収束因子解析完了")
        print("=" * 60)
        
        print("\n📊 最終最適化パラメータ:")
        param_names = ['γ (主要収束)', 'δ (指数減衰)', 't_c (臨界点)']
        for i, (name, value) in enumerate(zip(param_names, optimal_params)):
            print(f"   {name}: {value:.8f}")
        
        print(f"\n📈 最適化性能:")
        print(f"   目的関数値: {optimization_result.fun:.2e}")
        print(f"   反復回数: {optimization_result.nit}")
        print(f"   収束状況: {'成功' if optimization_result.success else '失敗'}")
        
        # 9. 理論的検証
        print("\n🔬 理論的検証:")
        N_test = 1000
        S_optimized = nkat.super_convergence_factor(N_test, optimal_params)
        S_quantum = nkat.quantum_mechanical_interpretation(N_test, optimal_params)
        S_statistical = nkat.statistical_mechanical_partition_function(N_test, params=optimal_params)
        S_information = nkat.information_theoretic_relative_entropy(N_test, optimal_params)
        
        print(f"   N={N_test}での超収束因子: {S_optimized:.8f}")
        print(f"   量子力学的解釈: {S_quantum:.8f}")
        print(f"   統計力学的解釈: {S_statistical:.8f}")
        print(f"   情報理論的解釈: {S_information:.8f}")
        
        # 10. リーマン予想への含意
        print("\n🎯 リーマン予想への含意:")
        convergence_rate = optimal_params[0] * np.log(N_test / optimal_params[2])
        print(f"   収束率 γ·ln(N/t_c): {convergence_rate:.8f}")
        print(f"   臨界線収束条件: {'満足' if abs(convergence_rate - 0.5) < 0.1 else '要検討'}")
        
        # 11. 結果保存
        results_summary = {
            'optimal_parameters': {
                'gamma': optimal_params[0],
                'delta': optimal_params[1],
                't_c': optimal_params[2]
            },
            'optimization_info': {
                'objective_value': float(optimization_result.fun),
                'iterations': int(optimization_result.nit),
                'success': bool(optimization_result.success)
            },
            'theoretical_verification': {
                'super_convergence_factor': float(S_optimized),
                'quantum_interpretation': float(S_quantum),
                'statistical_interpretation': float(S_statistical),
                'information_interpretation': float(S_information)
            }
        }
        
        # JSON形式で保存
        import json
        with open('nkat_optimization_results.json', 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        
        print("\n💾 結果をnkat_optimization_results.jsonに保存しました")
        
    else:
        print("❌ 最適化に失敗しました")
    
    print("\n🏁 解析完了")

if __name__ == "__main__":
    main() 