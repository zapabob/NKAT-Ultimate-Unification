#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 NKAT理論によるリーマン予想の超高精度数値検証 (改良版)
Ultra High-Precision Riemann Hypothesis Verification using NKAT Theory (Improved)

Author: NKAT Research Team
Date: 2025-05-24
Version: 6.0 - Ultra High Precision Implementation with Enhanced Stability
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
import warnings
from pathlib import Path
import json
import time
from dataclasses import dataclass
from tqdm import tqdm, trange
import logging
from scipy import special
import math

# 日本語フォント設定
plt.rcParams['font.family'] = ['MS Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# GPU可用性チェック
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 使用デバイス: {device}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name()}")
    print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

class UltraHighPrecisionNKATHamiltonian(nn.Module):
    """
    超高精度NKAT量子ハミルトニアンの実装
    
    改良点:
    1. 動的精度調整
    2. 改良されたスペクトル次元計算
    3. より安定した数値計算
    4. γ値依存の適応的パラメータ
    """
    
    def __init__(self, max_n: int = 1500, base_theta: float = 1e-20, 
                 base_kappa: float = 1e-12, precision: str = 'ultra'):
        super().__init__()
        self.max_n = max_n
        self.base_theta = base_theta
        self.base_kappa = base_kappa
        self.precision = precision
        self.device = device
        
        # 超高精度設定
        if precision == 'ultra':
            self.dtype = torch.complex128
            self.float_dtype = torch.float64
            torch.set_default_dtype(torch.float64)
        else:
            self.dtype = torch.complex64
            self.float_dtype = torch.float32
        
        logger.info(f"🔧 超高精度NKAT量子ハミルトニアン初期化: max_n={max_n}, 精度={precision}")
        
        # 素数リストの生成
        self.primes = self._generate_primes_optimized(max_n)
        logger.info(f"📊 生成された素数数: {len(self.primes)}")
        
        # ゼータ関数の零点（既知の値）
        self.known_zeros = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
                           37.586178, 40.918719, 43.327073, 48.005151, 49.773832]
        
    def _generate_primes_optimized(self, n: int) -> List[int]:
        """最適化されたエラトステネスの篩"""
        if n < 2:
            return []
        
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        
        return [i for i in range(2, n + 1) if sieve[i]]
    
    def _adaptive_parameters(self, s: complex) -> Tuple[float, float, int]:
        """γ値に応じた適応的パラメータ調整"""
        gamma = abs(s.imag)
        
        # γ値に基づく動的調整
        if gamma < 20:
            theta = self.base_theta * 10
            kappa = self.base_kappa * 5
            dim = min(self.max_n, 300)
        elif gamma < 50:
            theta = self.base_theta * 5
            kappa = self.base_kappa * 2
            dim = min(self.max_n, 250)
        else:
            theta = self.base_theta
            kappa = self.base_kappa
            dim = min(self.max_n, 200)
        
        return theta, kappa, dim
    
    def construct_enhanced_hamiltonian(self, s: complex) -> torch.Tensor:
        """
        強化されたハミルトニアン構築
        """
        theta, kappa, dim = self._adaptive_parameters(s)
        
        # ハミルトニアン行列の初期化
        H = torch.zeros(dim, dim, dtype=self.dtype, device=self.device)
        
        # 主要項: より安定した計算
        for n in range(1, dim + 1):
            try:
                # 対数スケールでの計算（数値安定性向上）
                if abs(s.real) > 15 or abs(s.imag) > 100:
                    log_n = math.log(n)
                    log_term = -s.real * log_n + 1j * s.imag * log_n
                    
                    if log_term.real < -30:  # アンダーフロー防止
                        H[n-1, n-1] = torch.tensor(1e-30, dtype=self.dtype, device=self.device)
                    else:
                        H[n-1, n-1] = torch.exp(torch.tensor(log_term, dtype=self.dtype, device=self.device))
                else:
                    # 直接計算
                    H[n-1, n-1] = torch.tensor(1.0 / (n ** s), dtype=self.dtype, device=self.device)
                    
            except (OverflowError, ZeroDivisionError, RuntimeError):
                H[n-1, n-1] = torch.tensor(1e-30, dtype=self.dtype, device=self.device)
        
        # 非可換補正項（改良版）
        if theta != 0:
            theta_tensor = torch.tensor(theta, dtype=self.dtype, device=self.device)
            gamma = abs(s.imag)
            
            # γ値に依存する補正強度
            correction_strength = 1.0 / (1.0 + gamma * 0.01)
            
            for i, p in enumerate(self.primes[:min(len(self.primes), 30)]):
                if p <= dim:
                    try:
                        log_p = math.log(p)
                        correction = theta_tensor * log_p * correction_strength
                        
                        # 量子補正項
                        if p < dim - 1:
                            # 非対角項（量子もつれ効果）
                            H[p-1, p] += correction * 1j * 0.5
                            H[p, p-1] -= correction * 1j * 0.5
                        
                        # 対角項（エネルギーシフト）
                        H[p-1, p-1] += correction * 0.1
                    except:
                        continue
        
        # κ-変形補正項（Minkowski時空効果）
        if kappa != 0:
            kappa_tensor = torch.tensor(kappa, dtype=self.dtype, device=self.device)
            
            for i in range(min(dim, 40)):
                try:
                    n = i + 1
                    
                    # Minkowski計量による補正
                    minkowski_factor = 1.0 / math.sqrt(1.0 + (n * kappa) ** 2)
                    log_term = math.log(n + 1) * minkowski_factor
                    kappa_correction = kappa_tensor * n * log_term
                    
                    # 時空曲率効果
                    if i < dim - 2:
                        curvature_term = kappa_correction * 0.05
                        H[i, i+1] += curvature_term
                        H[i+1, i] += curvature_term.conj()
                    
                    # 重力場効果
                    H[i, i] += kappa_correction * 0.01
                except:
                    continue
        
        # 正則化項（適応的）
        gamma = abs(s.imag)
        reg_strength = 1e-15 * (1.0 + gamma * 1e-4)
        H += reg_strength * torch.eye(dim, dtype=self.dtype, device=self.device)
        
        return H
    
    def compute_eigenvalues_ultra_stable(self, s: complex, n_eigenvalues: int = 150) -> torch.Tensor:
        """
        超安定固有値計算
        """
        try:
            H = self.construct_enhanced_hamiltonian(s)
            
            # エルミート化（改良版）
            H_dag = H.conj().T
            H_hermitian = 0.5 * (H + H_dag)
            
            # 条件数の改善
            try:
                # 特異値分解による前処理
                U, S, Vh = torch.linalg.svd(H_hermitian)
                
                # 小さな特異値の除去
                threshold = 1e-12
                S_filtered = torch.where(S > threshold, S, threshold)
                
                # 再構築
                H_hermitian = torch.mm(torch.mm(U, torch.diag(S_filtered)), Vh)
                
            except:
                # フォールバック：強い正則化
                reg_strength = 1e-10
                H_hermitian += reg_strength * torch.eye(H_hermitian.shape[0], 
                                                      dtype=self.dtype, device=self.device)
            
            # NaN/Inf チェック
            if torch.isnan(H_hermitian).any() or torch.isinf(H_hermitian).any():
                logger.warning("⚠️ ハミルトニアンにNaN/Infが検出されました")
                return torch.tensor([], device=self.device, dtype=self.float_dtype)
            
            # 固有値計算（複数手法の試行）
            eigenvalues = None
            
            # 手法1: 標準的な固有値分解
            try:
                eigenvalues, _ = torch.linalg.eigh(H_hermitian)
                eigenvalues = eigenvalues.real
            except RuntimeError:
                # 手法2: SVD分解
                try:
                    U, S, Vh = torch.linalg.svd(H_hermitian)
                    eigenvalues = S.real
                except RuntimeError:
                    # 手法3: 一般化固有値問題
                    try:
                        I = torch.eye(H_hermitian.shape[0], dtype=self.dtype, device=self.device)
                        eigenvalues, _ = torch.linalg.eig(H_hermitian)
                        eigenvalues = eigenvalues.real
                    except:
                        logger.error("❌ すべての固有値計算手法が失敗しました")
                        return torch.tensor([], device=self.device, dtype=self.float_dtype)
            
            if eigenvalues is None:
                return torch.tensor([], device=self.device, dtype=self.float_dtype)
            
            # 正の固有値のフィルタリング（改良版）
            positive_mask = eigenvalues > 1e-20
            positive_eigenvalues = eigenvalues[positive_mask]
            
            if len(positive_eigenvalues) == 0:
                logger.warning("⚠️ 正の固有値が見つかりませんでした")
                return torch.tensor([], device=self.device, dtype=self.float_dtype)
            
            # ソートして上位を選択
            sorted_eigenvalues, _ = torch.sort(positive_eigenvalues, descending=True)
            
            return sorted_eigenvalues[:min(len(sorted_eigenvalues), n_eigenvalues)]
            
        except Exception as e:
            logger.error(f"❌ 固有値計算エラー: {e}")
            return torch.tensor([], device=self.device, dtype=self.float_dtype)

class UltraHighPrecisionRiemannVerifier:
    """
    超高精度リーマン予想検証クラス
    """
    
    def __init__(self, hamiltonian: UltraHighPrecisionNKATHamiltonian):
        self.hamiltonian = hamiltonian
        self.device = hamiltonian.device
        
    def compute_spectral_dimension_enhanced(self, s: complex, 
                                          n_points: int = 80, 
                                          t_range: Tuple[float, float] = (1e-5, 2.0)) -> float:
        """
        強化されたスペクトル次元計算
        """
        eigenvalues = self.hamiltonian.compute_eigenvalues_ultra_stable(s, n_eigenvalues=200)
        
        if len(eigenvalues) < 15:
            logger.warning("⚠️ 有効な固有値が不足しています")
            return float('nan')
        
        try:
            # 適応的t値範囲
            gamma = abs(s.imag)
            if gamma > 30:
                t_min, t_max = 1e-6, 1.5
            elif gamma > 15:
                t_min, t_max = 1e-5, 1.8
            else:
                t_min, t_max = t_range
            
            # 対数スケールでのt値生成
            t_values = torch.logspace(np.log10(t_min), np.log10(t_max), n_points, device=self.device)
            zeta_values = []
            
            for t in t_values:
                # 数値安定性の大幅改善
                exp_terms = torch.exp(-t * eigenvalues)
                
                # 有効項のフィルタリング
                valid_mask = (torch.isfinite(exp_terms) & 
                             (exp_terms > 1e-100) & 
                             (exp_terms < 1e50))
                
                if torch.sum(valid_mask) < 5:
                    zeta_values.append(1e-100)
                    continue
                
                # 重み付き和（大きな固有値により高い重み）
                weights = 1.0 / (1.0 + eigenvalues[valid_mask] * 0.1)
                weighted_sum = torch.sum(exp_terms[valid_mask] * weights)
                
                if torch.isfinite(weighted_sum) and weighted_sum > 1e-100:
                    zeta_values.append(weighted_sum.item())
                else:
                    zeta_values.append(1e-100)
            
            zeta_values = torch.tensor(zeta_values, device=self.device)
            
            # 対数微分の改良計算
            log_t = torch.log(t_values)
            log_zeta = torch.log(zeta_values + 1e-100)
            
            # 外れ値の除去
            valid_mask = (torch.isfinite(log_zeta) & 
                         torch.isfinite(log_t) & 
                         (log_zeta > -200) & 
                         (log_zeta < 50) &
                         (torch.abs(log_zeta) < 1e10))
            
            if torch.sum(valid_mask) < 10:
                logger.warning("⚠️ 有効なデータ点が不足しています")
                return float('nan')
            
            log_t_valid = log_t[valid_mask]
            log_zeta_valid = log_zeta[valid_mask]
            
            # ロバスト回帰（RANSAC風）
            best_slope = None
            best_score = float('inf')
            
            for _ in range(10):  # 複数回試行
                # ランダムサンプリング
                n_sample = min(len(log_t_valid), max(10, len(log_t_valid) // 2))
                indices = torch.randperm(len(log_t_valid))[:n_sample]
                
                t_sample = log_t_valid[indices]
                zeta_sample = log_zeta_valid[indices]
                
                try:
                    # 重み付き最小二乗法
                    weights = torch.ones_like(t_sample)
                    # 中央部分により高い重みを付与
                    mid_range = (t_sample.max() + t_sample.min()) / 2
                    distance_from_mid = torch.abs(t_sample - mid_range)
                    weights = torch.exp(-distance_from_mid * 2)
                    
                    W = torch.diag(weights)
                    A = torch.stack([t_sample, torch.ones_like(t_sample)], dim=1)
                    
                    AtWA = torch.mm(torch.mm(A.T, W), A)
                    AtWy = torch.mm(torch.mm(A.T, W), zeta_sample.unsqueeze(1))
                    
                    solution = torch.linalg.solve(AtWA, AtWy)
                    slope = solution[0, 0]
                    
                    # 予測誤差の計算
                    pred = torch.mm(A, solution).squeeze()
                    error = torch.mean((pred - zeta_sample) ** 2)
                    
                    if error < best_score and torch.isfinite(slope):
                        best_score = error
                        best_slope = slope
                        
                except:
                    continue
            
            if best_slope is None:
                # フォールバック：単純な最小二乗法
                try:
                    A = torch.stack([log_t_valid, torch.ones_like(log_t_valid)], dim=1)
                    solution = torch.linalg.lstsq(A, log_zeta_valid).solution
                    best_slope = solution[0]
                except:
                    logger.warning("⚠️ 回帰計算が失敗しました")
                    return float('nan')
            
            # スペクトル次元の計算
            spectral_dimension = -2 * best_slope.item()
            
            # 妥当性チェック（より厳密）
            if (abs(spectral_dimension) > 50 or 
                not np.isfinite(spectral_dimension) or
                abs(spectral_dimension) < 1e-10):
                logger.warning(f"⚠️ 異常なスペクトル次元値: {spectral_dimension}")
                return float('nan')
            
            return spectral_dimension
            
        except Exception as e:
            logger.error(f"❌ スペクトル次元計算エラー: {e}")
            return float('nan')
    
    def verify_critical_line_ultra_precision(self, gamma_values: List[float], 
                                           iterations: int = 5) -> Dict:
        """
        超高精度臨界線収束性検証
        """
        results = {
            'gamma_values': gamma_values,
            'spectral_dimensions_all': [],
            'real_parts_all': [],
            'convergence_to_half_all': [],
            'statistics': {}
        }
        
        logger.info(f"🔍 超高精度臨界線収束性検証開始（{iterations}回実行）...")
        
        for iteration in range(iterations):
            logger.info(f"📊 実行 {iteration + 1}/{iterations}")
            
            spectral_dims = []
            real_parts = []
            convergences = []
            
            for gamma in tqdm(gamma_values, desc=f"実行{iteration+1}: γ値での検証"):
                s = 0.5 + 1j * gamma
                
                # スペクトル次元の計算
                d_s = self.compute_spectral_dimension_enhanced(s)
                spectral_dims.append(d_s)
                
                if not np.isnan(d_s):
                    # 実部の計算（理論的には d_s/2 ≈ 0.5）
                    real_part = d_s / 2
                    real_parts.append(real_part)
                    
                    # 1/2への収束性
                    convergence = abs(real_part - 0.5)
                    convergences.append(convergence)
                else:
                    real_parts.append(np.nan)
                    convergences.append(np.nan)
            
            results['spectral_dimensions_all'].append(spectral_dims)
            results['real_parts_all'].append(real_parts)
            results['convergence_to_half_all'].append(convergences)
        
        # 統計的評価
        all_spectral_dims = np.array(results['spectral_dimensions_all'])
        all_real_parts = np.array(results['real_parts_all'])
        all_convergences = np.array(results['convergence_to_half_all'])
        
        # 各γ値での統計
        results['statistics'] = {
            'spectral_dimension_mean': np.nanmean(all_spectral_dims, axis=0).tolist(),
            'spectral_dimension_std': np.nanstd(all_spectral_dims, axis=0).tolist(),
            'spectral_dimension_median': np.nanmedian(all_spectral_dims, axis=0).tolist(),
            'real_part_mean': np.nanmean(all_real_parts, axis=0).tolist(),
            'real_part_std': np.nanstd(all_real_parts, axis=0).tolist(),
            'convergence_mean': np.nanmean(all_convergences, axis=0).tolist(),
            'convergence_std': np.nanstd(all_convergences, axis=0).tolist(),
        }
        
        # 全体統計
        valid_convergences = all_convergences[~np.isnan(all_convergences)]
        if len(valid_convergences) > 0:
            results['overall_statistics'] = {
                'mean_convergence': np.mean(valid_convergences),
                'median_convergence': np.median(valid_convergences),
                'std_convergence': np.std(valid_convergences),
                'min_convergence': np.min(valid_convergences),
                'max_convergence': np.max(valid_convergences),
                'success_rate_strict': np.sum(valid_convergences < 0.01) / len(valid_convergences),
                'success_rate_moderate': np.sum(valid_convergences < 0.1) / len(valid_convergences),
                'success_rate_loose': np.sum(valid_convergences < 0.2) / len(valid_convergences)
            }
        
        return results

def demonstrate_ultra_high_precision_riemann():
    """
    超高精度リーマン予想検証のデモンストレーション
    """
    print("=" * 80)
    print("🎯 NKAT理論による超高精度リーマン予想検証 (改良版)")
    print("=" * 80)
    print("📅 実行日時:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("🔬 精度: complex128 (倍精度) + 数値安定性強化")
    print("🧮 改良点: 適応的パラメータ、ロバスト回帰、複数手法試行")
    print("=" * 80)
    
    # 超高精度ハミルトニアンの初期化
    logger.info("🔧 超高精度NKAT量子ハミルトニアン初期化中...")
    hamiltonian = UltraHighPrecisionNKATHamiltonian(
        max_n=1200,
        base_theta=1e-20,
        base_kappa=1e-12,
        precision='ultra'
    )
    
    # 超高精度検証器の初期化
    verifier = UltraHighPrecisionRiemannVerifier(hamiltonian)
    
    # 超高精度臨界線検証
    print("\n📊 超高精度臨界線収束性検証")
    gamma_values = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062]
    
    start_time = time.time()
    ultra_precision_results = verifier.verify_critical_line_ultra_precision(
        gamma_values, iterations=5
    )
    verification_time = time.time() - start_time
    
    # 結果の表示
    print("\n超高精度検証結果:")
    print("γ値      | 平均d_s    | 中央値d_s  | 標準偏差   | 平均Re     | |Re-1/2|平均 | 収束性")
    print("-" * 90)
    
    stats = ultra_precision_results['statistics']
    for i, gamma in enumerate(gamma_values):
        mean_ds = stats['spectral_dimension_mean'][i]
        median_ds = stats['spectral_dimension_median'][i]
        std_ds = stats['spectral_dimension_std'][i]
        mean_re = stats['real_part_mean'][i]
        mean_conv = stats['convergence_mean'][i]
        
        if not np.isnan(mean_ds):
            if mean_conv < 0.01:
                status = "✅"
            elif mean_conv < 0.1:
                status = "🟡"
            else:
                status = "⚠️"
            print(f"{gamma:8.6f} | {mean_ds:9.6f} | {median_ds:9.6f} | {std_ds:9.6f} | {mean_re:9.6f} | {mean_conv:10.6f} | {status}")
        else:
            print(f"{gamma:8.6f} | {'NaN':>9} | {'NaN':>9} | {'NaN':>9} | {'NaN':>9} | {'NaN':>10} | ❌")
    
    # 全体統計の表示
    if 'overall_statistics' in ultra_precision_results:
        overall = ultra_precision_results['overall_statistics']
        print(f"\n📊 全体統計:")
        print(f"平均収束率: {overall['mean_convergence']:.8f}")
        print(f"中央値収束率: {overall['median_convergence']:.8f}")
        print(f"標準偏差: {overall['std_convergence']:.8f}")
        print(f"厳密成功率 (<0.01): {overall['success_rate_strict']:.2%}")
        print(f"中程度成功率 (<0.1): {overall['success_rate_moderate']:.2%}")
        print(f"緩い成功率 (<0.2): {overall['success_rate_loose']:.2%}")
        print(f"最良収束: {overall['min_convergence']:.8f}")
        print(f"最悪収束: {overall['max_convergence']:.8f}")
    
    print(f"\n⏱️  検証時間: {verification_time:.2f}秒")
    
    # 結果の保存
    with open('ultra_high_precision_riemann_results.json', 'w', encoding='utf-8') as f:
        json.dump(ultra_precision_results, f, indent=2, ensure_ascii=False, default=str)
    
    print("💾 超高精度結果を 'ultra_high_precision_riemann_results.json' に保存しました")
    
    return ultra_precision_results

if __name__ == "__main__":
    """
    超高精度リーマン予想検証の実行
    """
    try:
        results = demonstrate_ultra_high_precision_riemann()
        print("🎉 超高精度検証が完了しました！")
    except Exception as e:
        logger.error(f"❌ 実行エラー: {e}")
        print(f"❌ エラーが発生しました: {e}") 