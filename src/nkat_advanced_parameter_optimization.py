#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT高度パラメータ最適化・自動フィッティング・ベイズ最適化システム
Advanced Parameter Optimization, Auto-Fitting, and Bayesian Optimization System

著者: NKAT研究チーム
日付: 2025-01-19
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize, differential_evolution, basinhopping
from scipy.stats import multivariate_normal, norm, uniform
from scipy.special import gamma, zeta
import optuna
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
import json
import os
from datetime import datetime
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
import pickle
import hashlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

class NKATAdvancedOptimizer:
    """NKAT高度パラメータ最適化システム"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.setup_logging()
        self.optimization_history = []
        self.best_params = {}
        self.gp_model = None
        self.scaler = StandardScaler()
        
    def _default_config(self) -> Dict:
        """デフォルト設定"""
        return {
            'optimization_methods': ['bayesian', 'differential_evolution', 'basin_hopping', 'particle_swarm'],
            'n_trials': 1000,
            'n_jobs': -1,  # 全CPU使用
            'random_state': 42,
            'parameter_bounds': {
                'theta': (1e-60, 1e-30),
                'kappa': (1e-60, 1e-30),
                'lambda_star_real': (0.1, 1.0),
                'lambda_star_imag': (10, 20),
                'A_coeffs': [(0.1, 10.0)] * 10,
                'B_coeffs': [(0.1, 10.0)] * 5,
                'field_coeffs': [(0.1, 10.0)] * 10,
                'interaction_coeffs': [(0.1, 10.0)] * 10
            },
            'objective_functions': ['mse', 'mae', 'r2', 'correlation', 'spectral_distance'],
            'constraints': {
                'unitarity': True,
                'causality': True,
                'energy_conservation': True
            },
            'convergence_criteria': {
                'tolerance': 1e-6,
                'max_iterations': 1000,
                'patience': 50
            }
        }
    
    def setup_logging(self):
        """ログ設定"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('parameter_optimization.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def objective_function(self, params: np.ndarray, experimental_data: np.ndarray, 
                          theory_type: str = 'both', metric: str = 'mse') -> float:
        """目的関数の計算"""
        try:
            # パラメータの復元
            param_dict = self._params_to_dict(params, theory_type)
            
            x_data = np.linspace(0, 10, len(experimental_data))
            
            if theory_type in ['unified', 'both']:
                unified_result = self._unified_theory(x_data, param_dict)
            else:
                unified_result = np.zeros_like(experimental_data)
            
            if theory_type in ['nkat', 'both']:
                nkat_result = self._nkat_theory(x_data, param_dict)
            else:
                nkat_result = np.zeros_like(experimental_data)
            
            # 理論結果の統合
            if theory_type == 'both':
                theory_result = (unified_result + nkat_result) / 2
            elif theory_type == 'unified':
                theory_result = unified_result
            else:
                theory_result = nkat_result
            
            # メトリック計算
            if metric == 'mse':
                return mean_squared_error(np.abs(experimental_data), np.abs(theory_result))
            elif metric == 'mae':
                return mean_absolute_error(np.abs(experimental_data), np.abs(theory_result))
            elif metric == 'r2':
                return -r2_score(np.abs(experimental_data), np.abs(theory_result))  # 最小化のため負値
            elif metric == 'correlation':
                corr = np.corrcoef(np.abs(experimental_data), np.abs(theory_result))[0, 1]
                return -abs(corr) if not np.isnan(corr) else 1e6
            elif metric == 'spectral_distance':
                exp_spectrum = np.fft.fft(experimental_data)
                theory_spectrum = np.fft.fft(theory_result)
                return np.linalg.norm(np.abs(exp_spectrum) - np.abs(theory_spectrum))
            else:
                return mean_squared_error(np.abs(experimental_data), np.abs(theory_result))
                
        except Exception as e:
            self.logger.warning(f"目的関数計算エラー: {e}")
            return 1e6
    
    def _params_to_dict(self, params: np.ndarray, theory_type: str) -> Dict:
        """パラメータ配列を辞書に変換"""
        param_dict = {}
        idx = 0
        
        if theory_type in ['unified', 'both']:
            # 統合特解理論パラメータ
            param_dict['lambda_star'] = params[idx] + 1j * params[idx + 1]
            idx += 2
            param_dict['A_coeffs'] = params[idx:idx + 10]
            idx += 10
            param_dict['B_coeffs'] = params[idx:idx + 5]
            idx += 5
        
        if theory_type in ['nkat', 'both']:
            # NKAT理論パラメータ
            param_dict['theta'] = params[idx]
            idx += 1
            param_dict['kappa'] = params[idx]
            idx += 1
            param_dict['field_coeffs'] = params[idx:idx + 10]
            idx += 10
            param_dict['interaction_coeffs'] = params[idx:idx + 10]
            idx += 10
        
        return param_dict
    
    def _unified_theory(self, x: np.ndarray, params: Dict) -> np.ndarray:
        """統合特解理論の計算"""
        lambda_star = params.get('lambda_star', 0.5 + 1j * 14.134725)
        A_coeffs = params.get('A_coeffs', np.ones(10))
        B_coeffs = params.get('B_coeffs', np.ones(5))
        
        result = np.zeros_like(x, dtype=complex)
        
        for q in range(len(A_coeffs)):
            exp_term = np.exp(1j * lambda_star * x)
            
            internal_sum = 0
            for p in range(len(A_coeffs)):
                for k in range(1, 6):
                    psi_term = np.sin(k * np.pi * x) * np.exp(-k * x**2)
                    internal_sum += A_coeffs[q] * psi_term
            
            external_prod = 1
            for ell in range(len(B_coeffs)):
                phi_term = np.cos(ell * np.pi * x) * np.exp(-ell * x**2 / 2)
                external_prod *= B_coeffs[ell] * phi_term
            
            result += exp_term * internal_sum * external_prod
        
        return result
    
    def _nkat_theory(self, x: np.ndarray, params: Dict) -> np.ndarray:
        """NKAT理論の計算"""
        theta = params.get('theta', 1e-45)
        kappa = params.get('kappa', 1e-40)
        field_coeffs = params.get('field_coeffs', np.ones(10))
        interaction_coeffs = params.get('interaction_coeffs', np.ones(10))
        
        result = np.zeros_like(x, dtype=complex)
        
        for i in range(len(field_coeffs)):
            phi_field = field_coeffs[i] * np.exp(-i * x**2 / 2)
            
            interaction_sum = 0
            for j in range(len(interaction_coeffs)):
                psi_interaction = interaction_coeffs[j] * np.sin(j * np.pi * x)
                x_term = x * np.exp(-j * x**2 / 4)
                interaction_sum += psi_interaction * x_term
            
            moyal_correction = 1 + 1j * theta * x * np.gradient(interaction_sum, x)
            kappa_correction = 1 + kappa * x**2 / 2
            
            result += phi_field * interaction_sum * moyal_correction * kappa_correction
        
        return result
    
    def bayesian_optimization(self, experimental_data: np.ndarray, 
                             theory_type: str = 'both', n_trials: int = None) -> Dict:
        """ベイズ最適化によるパラメータ探索"""
        self.logger.info("ベイズ最適化を開始...")
        
        n_trials = n_trials or self.config['n_trials']
        
        def objective(trial):
            # パラメータの提案
            params = []
            
            if theory_type in ['unified', 'both']:
                # 統合特解理論パラメータ
                lambda_real = trial.suggest_float('lambda_real', 0.1, 1.0)
                lambda_imag = trial.suggest_float('lambda_imag', 10, 20)
                params.extend([lambda_real, lambda_imag])
                
                for i in range(10):
                    A_coeff = trial.suggest_float(f'A_coeff_{i}', 0.1, 10.0)
                    params.append(A_coeff)
                
                for i in range(5):
                    B_coeff = trial.suggest_float(f'B_coeff_{i}', 0.1, 10.0)
                    params.append(B_coeff)
            
            if theory_type in ['nkat', 'both']:
                # NKAT理論パラメータ
                theta = trial.suggest_float('theta', 1e-60, 1e-30, log=True)
                kappa = trial.suggest_float('kappa', 1e-60, 1e-30, log=True)
                params.extend([theta, kappa])
                
                for i in range(10):
                    field_coeff = trial.suggest_float(f'field_coeff_{i}', 0.1, 10.0)
                    params.append(field_coeff)
                
                for i in range(10):
                    interaction_coeff = trial.suggest_float(f'interaction_coeff_{i}', 0.1, 10.0)
                    params.append(interaction_coeff)
            
            # 目的関数の評価
            return self.objective_function(np.array(params), experimental_data, theory_type)
        
        # Optunaによる最適化
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, n_jobs=self.config['n_jobs'])
        
        # 結果の整理
        best_params = study.best_params
        best_value = study.best_value
        
        # パラメータ配列への変換
        param_array = self._optuna_params_to_array(best_params, theory_type)
        
        return {
            'method': 'bayesian_optimization',
            'best_params': best_params,
            'best_param_array': param_array,
            'best_value': best_value,
            'n_trials': n_trials,
            'optimization_history': study.trials_dataframe()
        }
    
    def _optuna_params_to_array(self, optuna_params: Dict, theory_type: str) -> np.ndarray:
        """Optunaパラメータを配列に変換"""
        params = []
        
        if theory_type in ['unified', 'both']:
            params.extend([optuna_params['lambda_real'], optuna_params['lambda_imag']])
            
            for i in range(10):
                params.append(optuna_params[f'A_coeff_{i}'])
            
            for i in range(5):
                params.append(optuna_params[f'B_coeff_{i}'])
        
        if theory_type in ['nkat', 'both']:
            params.extend([optuna_params['theta'], optuna_params['kappa']])
            
            for i in range(10):
                params.append(optuna_params[f'field_coeff_{i}'])
            
            for i in range(10):
                params.append(optuna_params[f'interaction_coeff_{i}'])
        
        return np.array(params)
    
    def differential_evolution_optimization(self, experimental_data: np.ndarray,
                                          theory_type: str = 'both') -> Dict:
        """差分進化による最適化"""
        self.logger.info("差分進化最適化を開始...")
        
        # パラメータ境界の設定
        bounds = self._get_parameter_bounds(theory_type)
        
        def objective(params):
            return self.objective_function(params, experimental_data, theory_type)
        
        # 差分進化による最適化
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=self.config['convergence_criteria']['max_iterations'],
            tol=self.config['convergence_criteria']['tolerance'],
            workers=self.config['n_jobs'],
            updating='deferred'
        )
        
        return {
            'method': 'differential_evolution',
            'success': result.success,
            'best_params': result.x,
            'best_value': result.fun,
            'n_iterations': result.nit,
            'n_evaluations': result.nfev
        }
    
    def _get_parameter_bounds(self, theory_type: str) -> List[Tuple]:
        """パラメータ境界の取得"""
        bounds = []
        
        if theory_type in ['unified', 'both']:
            bounds.extend([
                (0.1, 1.0), (10, 20)  # lambda_star
            ])
            bounds.extend([(0.1, 10.0)] * 10)  # A_coeffs
            bounds.extend([(0.1, 10.0)] * 5)   # B_coeffs
        
        if theory_type in ['nkat', 'both']:
            bounds.extend([
                (1e-60, 1e-30), (1e-60, 1e-30)  # theta, kappa
            ])
            bounds.extend([(0.1, 10.0)] * 10)  # field_coeffs
            bounds.extend([(0.1, 10.0)] * 10)  # interaction_coeffs
        
        return bounds
    
    def basin_hopping_optimization(self, experimental_data: np.ndarray,
                                  theory_type: str = 'both') -> Dict:
        """Basin Hoppingによる最適化"""
        self.logger.info("Basin Hopping最適化を開始...")
        
        bounds = self._get_parameter_bounds(theory_type)
        
        def objective(params):
            return self.objective_function(params, experimental_data, theory_type)
        
        # 初期値の設定
        x0 = np.array([(b[0] + b[1]) / 2 for b in bounds])
        
        # Basin Hoppingによる最適化
        result = basinhopping(
            objective,
            x0,
            niter=self.config['convergence_criteria']['max_iterations'],
            T=1.0,
            stepsize=0.1,
            minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': bounds}
        )
        
        return {
            'method': 'basin_hopping',
            'success': result.success,
            'best_params': result.x,
            'best_value': result.fun,
            'n_iterations': result.nit,
            'n_evaluations': result.nfev
        }
    
    def particle_swarm_optimization(self, experimental_data: np.ndarray,
                                   theory_type: str = 'both', n_particles: int = 50) -> Dict:
        """粒子群最適化"""
        self.logger.info("粒子群最適化を開始...")
        
        bounds = self._get_parameter_bounds(theory_type)
        n_dimensions = len(bounds)
        
        # 粒子群の初期化
        particles = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(n_particles, n_dimensions)
        )
        
        velocities = np.random.uniform(-0.1, 0.1, (n_particles, n_dimensions))
        
        # 最適化パラメータ
        w = 0.7  # 慣性重み
        c1 = 2.0  # 個体学習係数
        c2 = 2.0  # 社会学習係数
        
        # 最適化履歴
        best_positions = particles.copy()
        best_values = np.array([self.objective_function(p, experimental_data, theory_type) 
                               for p in particles])
        
        global_best_idx = np.argmin(best_values)
        global_best_position = best_positions[global_best_idx].copy()
        global_best_value = best_values[global_best_idx]
        
        # 最適化ループ
        for iteration in range(self.config['convergence_criteria']['max_iterations']):
            # 速度更新
            r1 = np.random.rand(n_particles, n_dimensions)
            r2 = np.random.rand(n_particles, n_dimensions)
            
            velocities = (w * velocities + 
                         c1 * r1 * (best_positions - particles) +
                         c2 * r2 * (global_best_position - particles))
            
            # 位置更新
            particles += velocities
            
            # 境界制約
            for i, (low, high) in enumerate(bounds):
                particles[:, i] = np.clip(particles[:, i], low, high)
            
            # 評価値更新
            for i in range(n_particles):
                value = self.objective_function(particles[i], experimental_data, theory_type)
                if value < best_values[i]:
                    best_values[i] = value
                    best_positions[i] = particles[i].copy()
                    
                    if value < global_best_value:
                        global_best_value = value
                        global_best_position = particles[i].copy()
            
            # 収束判定
            if iteration % 10 == 0:
                self.logger.info(f"PSO iteration {iteration}: best_value = {global_best_value:.6f}")
        
        return {
            'method': 'particle_swarm_optimization',
            'success': True,
            'best_params': global_best_position,
            'best_value': global_best_value,
            'n_iterations': self.config['convergence_criteria']['max_iterations'],
            'n_particles': n_particles
        }
    
    def multi_objective_optimization(self, experimental_data: np.ndarray,
                                    theory_type: str = 'both') -> Dict:
        """多目的最適化"""
        self.logger.info("多目的最適化を開始...")
        
        objectives = self.config['objective_functions']
        
        def multi_objective(params):
            results = {}
            for obj in objectives:
                results[obj] = self.objective_function(params, experimental_data, theory_type, obj)
            return results
        
        # 重み付き目的関数
        weights = {'mse': 0.4, 'mae': 0.3, 'correlation': 0.2, 'spectral_distance': 0.1}
        
        def weighted_objective(params):
            obj_values = multi_objective(params)
            weighted_sum = sum(weights[obj] * obj_values[obj] for obj in weights.keys())
            return weighted_sum
        
        # 最適化実行
        bounds = self._get_parameter_bounds(theory_type)
        
        result = differential_evolution(
            weighted_objective,
            bounds,
            maxiter=self.config['convergence_criteria']['max_iterations'],
            tol=self.config['convergence_criteria']['tolerance']
        )
        
        # 詳細な結果評価
        best_params = result.x
        detailed_objectives = multi_objective(best_params)
        
        return {
            'method': 'multi_objective_optimization',
            'success': result.success,
            'best_params': best_params,
            'best_value': result.fun,
            'detailed_objectives': detailed_objectives,
            'weights': weights,
            'n_iterations': result.nit
        }
    
    def comprehensive_optimization(self, experimental_data: np.ndarray,
                                 theory_type: str = 'both') -> Dict:
        """包括的最適化（複数手法の組み合わせ）"""
        self.logger.info("包括的最適化を開始...")
        
        results = {}
        
        # 各最適化手法の実行
        optimization_methods = {
            'bayesian': self.bayesian_optimization,
            'differential_evolution': self.differential_evolution_optimization,
            'basin_hopping': self.basin_hopping_optimization,
            'particle_swarm': self.particle_swarm_optimization,
            'multi_objective': self.multi_objective_optimization
        }
        
        for method_name, method_func in optimization_methods.items():
            try:
                self.logger.info(f"{method_name}最適化を実行中...")
                result = method_func(experimental_data, theory_type)
                results[method_name] = result
                
                self.logger.info(f"{method_name}完了: best_value = {result['best_value']:.6f}")
                
            except Exception as e:
                self.logger.error(f"{method_name}最適化でエラー: {e}")
                results[method_name] = {'error': str(e)}
        
        # 最良結果の選択
        best_method = None
        best_value = float('inf')
        
        for method_name, result in results.items():
            if 'error' not in result and result['best_value'] < best_value:
                best_value = result['best_value']
                best_method = method_name
        
        # 統合結果
        comprehensive_result = {
            'all_results': results,
            'best_method': best_method,
            'best_overall_value': best_value,
            'best_overall_params': results[best_method]['best_params'] if best_method else None,
            'theory_type': theory_type,
            'timestamp': datetime.now().isoformat()
        }
        
        self.optimization_history.append(comprehensive_result)
        return comprehensive_result
    
    def visualize_optimization_results(self, results: Dict, save_path: str = None):
        """最適化結果の可視化"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('NKATパラメータ最適化結果', fontsize=16, fontweight='bold')
        
        # 1. 各手法の性能比較
        methods = []
        values = []
        
        for method_name, result in results['all_results'].items():
            if 'error' not in result:
                methods.append(method_name)
                values.append(result['best_value'])
        
        bars = axes[0, 0].bar(methods, values, color=['blue', 'red', 'green', 'orange', 'purple'])
        axes[0, 0].set_ylabel('目的関数値')
        axes[0, 0].set_title('最適化手法比較')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 値の表示
        for bar, value in zip(bars, values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                           f'{value:.4f}', ha='center', va='bottom', fontsize=8)
        
        # 2. 最良パラメータの分布
        if results['best_overall_params'] is not None:
            best_params = results['best_overall_params']
            axes[0, 1].hist(best_params, bins=20, alpha=0.7, color='lightblue')
            axes[0, 1].set_xlabel('パラメータ値')
            axes[0, 1].set_ylabel('頻度')
            axes[0, 1].set_title('最良パラメータ分布')
        
        # 3. 多目的最適化の詳細（該当する場合）
        if 'multi_objective' in results['all_results'] and 'error' not in results['all_results']['multi_objective']:
            mo_result = results['all_results']['multi_objective']
            if 'detailed_objectives' in mo_result:
                obj_names = list(mo_result['detailed_objectives'].keys())
                obj_values = list(mo_result['detailed_objectives'].values())
                
                axes[0, 2].bar(obj_names, obj_values, color=['red', 'blue', 'green', 'orange'])
                axes[0, 2].set_ylabel('目的関数値')
                axes[0, 2].set_title('多目的最適化詳細')
                axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. ベイズ最適化履歴（該当する場合）
        if 'bayesian' in results['all_results'] and 'error' not in results['all_results']['bayesian']:
            bayes_result = results['all_results']['bayesian']
            if 'optimization_history' in bayes_result:
                history = bayes_result['optimization_history']
                if 'value' in history.columns:
                    axes[1, 0].plot(history['value'], 'b-', linewidth=2)
                    axes[1, 0].set_xlabel('試行回数')
                    axes[1, 0].set_ylabel('目的関数値')
                    axes[1, 0].set_title('ベイズ最適化履歴')
                    axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 理論比較（実験データ vs 最適化結果）
        if results['best_overall_params'] is not None:
            x_test = np.linspace(0, 10, 1000)
            experimental_data = np.random.rand(1000) + 1j * np.random.rand(1000)  # 仮の実験データ
            
            # 最適化されたパラメータでの理論計算
            theory_result = self._calculate_theory_result(x_test, results['best_overall_params'], results['theory_type'])
            
            axes[1, 1].plot(x_test, np.abs(experimental_data[:len(x_test)]), 'b-', label='実験データ', alpha=0.7)
            axes[1, 1].plot(x_test, np.abs(theory_result), 'r-', label='最適化理論', alpha=0.7)
            axes[1, 1].set_xlabel('x')
            axes[1, 1].set_ylabel('|Ψ(x)|')
            axes[1, 1].set_title('実験データ vs 最適化理論')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 収束性解析
        convergence_data = []
        for method_name, result in results['all_results'].items():
            if 'error' not in result and 'n_iterations' in result:
                convergence_data.append((method_name, result['n_iterations'], result['best_value']))
        
        if convergence_data:
            methods, iterations, values = zip(*convergence_data)
            scatter = axes[1, 2].scatter(iterations, values, c=range(len(methods)), cmap='viridis', s=100)
            axes[1, 2].set_xlabel('反復回数')
            axes[1, 2].set_ylabel('最良値')
            axes[1, 2].set_title('収束性比較')
            axes[1, 2].grid(True, alpha=0.3)
            
            # 凡例
            for i, method in enumerate(methods):
                axes[1, 2].annotate(method, (iterations[i], values[i]), 
                                   xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"可視化結果を保存: {save_path}")
        
        plt.show()
    
    def _calculate_theory_result(self, x: np.ndarray, params: np.ndarray, theory_type: str) -> np.ndarray:
        """理論結果の計算"""
        param_dict = self._params_to_dict(params, theory_type)
        
        if theory_type == 'unified':
            return self._unified_theory(x, param_dict)
        elif theory_type == 'nkat':
            return self._nkat_theory(x, param_dict)
        else:  # both
            unified_result = self._unified_theory(x, param_dict)
            nkat_result = self._nkat_theory(x, param_dict)
            return (unified_result + nkat_result) / 2
    
    def save_optimization_results(self, results: Dict, filename: str = None):
        """最適化結果の保存"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_results_{timestamp}.json"
        
        # 数値データの変換
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.complex128, complex)):
                return {'real': float(obj.real), 'imag': float(obj.imag)}
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            else:
                return obj
        
        json_data = convert_for_json(results)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"最適化結果を保存: {filename}")
    
    def generate_optimization_report(self, results: Dict, output_path: str = None):
        """最適化レポートの生成"""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"optimization_report_{timestamp}.md"
        
        report = f"""# NKATパラメータ最適化レポート

## 最適化概要
- **最適化日時**: {results['timestamp']}
- **理論タイプ**: {results['theory_type']}
- **最良手法**: {results['best_method']}
- **最良値**: {results['best_overall_value']:.6f}

## 各手法の結果

"""
        
        for method_name, result in results['all_results'].items():
            if 'error' in result:
                report += f"### {method_name}\n- **ステータス**: エラー\n- **エラー内容**: {result['error']}\n\n"
            else:
                report += f"### {method_name}\n"
                report += f"- **成功**: {result.get('success', 'N/A')}\n"
                report += f"- **最良値**: {result['best_value']:.6f}\n"
                report += f"- **反復回数**: {result.get('n_iterations', 'N/A')}\n"
                report += f"- **評価回数**: {result.get('n_evaluations', 'N/A')}\n\n"
        
        report += f"""
## 最適化パラメータ
最良パラメータ（{results['best_method']}）:
```
{results['best_overall_params']}
```

## 結論
{results['best_method']}が最も効果的な最適化手法として選択されました。
最良値 {results['best_overall_value']:.6f} を達成しています。

**Don't hold back. Give it your all deep think!!**
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"最適化レポートを生成: {output_path}")

def main():
    """メイン実行関数"""
    print("NKAT高度パラメータ最適化・自動フィッティング・ベイズ最適化システム")
    print("=" * 70)
    
    # システム初期化
    optimizer = NKATAdvancedOptimizer()
    
    # 仮の実験データ生成
    print("実験データを生成中...")
    experimental_data = np.random.rand(1000) + 1j * np.random.rand(1000)
    
    # 包括的最適化実行
    print("包括的最適化を実行中...")
    results = optimizer.comprehensive_optimization(experimental_data, 'both')
    
    # 結果の可視化
    print("結果の可視化中...")
    optimizer.visualize_optimization_results(results)
    
    # 結果保存
    print("結果を保存中...")
    optimizer.save_optimization_results(results)
    optimizer.generate_optimization_report(results)
    
    print("最適化完了！")
    print(f"最良手法: {results['best_method']}")
    print(f"最良値: {results['best_overall_value']:.6f}")

if __name__ == "__main__":
    main() 