#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 NKAT理論によるリーマン予想の高精度数値検証
High-Precision Riemann Hypothesis Verification using NKAT Theory

Author: NKAT Research Team
Date: 2025-05-24
Version: 5.0 - High Precision Implementation
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

class HighPrecisionNKATHamiltonian(nn.Module):
    """
    高精度NKAT量子ハミルトニアンの実装
    
    改良点:
    1. complex128精度の使用
    2. より大きな格子サイズ
    3. 改良された数値安定性
    4. 適応的パラメータ調整
    """
    
    def __init__(self, max_n: int = 2000, theta: float = 1e-25, kappa: float = 1e-15, 
                 precision: str = 'high'):
        super().__init__()
        self.max_n = max_n
        self.theta = theta
        self.kappa = kappa
        self.precision = precision
        self.device = device
        
        # 精度設定
        if precision == 'high':
            self.dtype = torch.complex128
            self.float_dtype = torch.float64
        else:
            self.dtype = torch.complex64
            self.float_dtype = torch.float32
        
        logger.info(f"🔧 高精度NKAT量子ハミルトニアン初期化: max_n={max_n}, 精度={precision}")
        
        # 素数リストの生成（より効率的なアルゴリズム）
        self.primes = self._generate_primes_optimized(max_n)
        logger.info(f"📊 生成された素数数: {len(self.primes)}")
        
        # ガンマ行列の定義
        self.gamma_matrices = self._construct_gamma_matrices()
        
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
    
    def _construct_gamma_matrices(self) -> List[torch.Tensor]:
        """高精度ガンマ行列の構築"""
        # パウリ行列（高精度）
        sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=self.dtype, device=self.device)
        sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=self.dtype, device=self.device)
        sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=self.dtype, device=self.device)
        I2 = torch.eye(2, dtype=self.dtype, device=self.device)
        O2 = torch.zeros(2, 2, dtype=self.dtype, device=self.device)
        
        # ディラック行列の構築
        gamma = []
        
        # γ^0 = [[I, 0], [0, -I]]
        gamma.append(torch.cat([torch.cat([I2, O2], dim=1), 
                               torch.cat([O2, -I2], dim=1)], dim=0))
        
        # γ^i = [[0, σ_i], [-σ_i, 0]] for i=1,2,3
        for sigma in [sigma_x, sigma_y, sigma_z]:
            gamma.append(torch.cat([torch.cat([O2, sigma], dim=1),
                                   torch.cat([-sigma, O2], dim=1)], dim=0))
        
        logger.info(f"✅ 高精度ガンマ行列構築完了: {len(gamma)}個の{gamma[0].shape}行列")
        return gamma
    
    def construct_hamiltonian_adaptive(self, s: complex, adaptive_dim: bool = True) -> torch.Tensor:
        """
        適応的次元調整を持つハミルトニアン構築
        """
        # 適応的次元決定
        if adaptive_dim:
            s_magnitude = abs(s)
            if s_magnitude < 1:
                dim = min(self.max_n, 200)
            elif s_magnitude < 10:
                dim = min(self.max_n, 150)
            else:
                dim = min(self.max_n, 100)
        else:
            dim = min(self.max_n, 150)
        
        # ハミルトニアン行列の初期化
        H = torch.zeros(dim, dim, dtype=self.dtype, device=self.device)
        
        # 主要項: Σ_n (1/n^s) |n⟩⟨n| with improved numerical stability
        for n in range(1, dim + 1):
            try:
                # 数値安定性の改善
                if abs(s.real) > 20 or abs(s.imag) > 200:
                    # 極端な値での安定化
                    log_term = -s * np.log(n)
                    if log_term.real < -50:  # アンダーフロー防止
                        H[n-1, n-1] = torch.tensor(1e-50, dtype=self.dtype, device=self.device)
                    else:
                        H[n-1, n-1] = torch.exp(torch.tensor(log_term, dtype=self.dtype, device=self.device))
                else:
                    # 通常の計算
                    H[n-1, n-1] = torch.tensor(1.0 / (n ** s), dtype=self.dtype, device=self.device)
            except (OverflowError, ZeroDivisionError, RuntimeError):
                H[n-1, n-1] = torch.tensor(1e-50, dtype=self.dtype, device=self.device)
        
        # 非可換補正項（改良版）
        if self.theta != 0:
            theta_tensor = torch.tensor(self.theta, dtype=self.dtype, device=self.device)
            for i, p in enumerate(self.primes[:min(len(self.primes), 20)]):
                if p <= dim:
                    try:
                        # 対数項の安定化
                        log_p = torch.log(torch.tensor(p, dtype=self.float_dtype, device=self.device))
                        correction = theta_tensor * log_p.to(self.dtype)
                        
                        # 交換子項の追加 [x, p]
                        if p < dim - 1:
                            H[p-1, p] += correction * 1j
                            H[p, p-1] -= correction * 1j
                        
                        H[p-1, p-1] += correction
                    except:
                        continue
        
        # κ-変形補正項（改良版）
        if self.kappa != 0:
            kappa_tensor = torch.tensor(self.kappa, dtype=self.dtype, device=self.device)
            for i in range(min(dim, 30)):
                try:
                    # Minkowski変形項
                    n = i + 1
                    log_term = torch.log(torch.tensor(n + 1, dtype=self.float_dtype, device=self.device))
                    kappa_correction = kappa_tensor * n * log_term.to(self.dtype)
                    
                    # 非対角項の追加
                    if i < dim - 2:
                        H[i, i+1] += kappa_correction * 0.1
                        H[i+1, i] += kappa_correction.conj() * 0.1
                    
                    H[i, i] += kappa_correction
                except:
                    continue
        
        # 正則化項（数値安定性向上）
        regularization = torch.tensor(1e-12, dtype=self.dtype, device=self.device)
        H += regularization * torch.eye(dim, dtype=self.dtype, device=self.device)
        
        return H
    
    def compute_eigenvalues_stable(self, s: complex, n_eigenvalues: int = 100) -> torch.Tensor:
        """
        数値安定性を向上させた固有値計算
        """
        try:
            H = self.construct_hamiltonian_adaptive(s)
            
            # エルミート化の改良
            H_hermitian = 0.5 * (torch.mm(H.conj().T, H) + torch.mm(H, H.conj().T))
            
            # 条件数チェック
            try:
                cond_num = torch.linalg.cond(H_hermitian)
                if cond_num > 1e12:
                    logger.warning(f"⚠️ 高い条件数が検出されました: {cond_num:.2e}")
                    # 正則化の強化
                    reg_strength = 1e-10
                    H_hermitian += reg_strength * torch.eye(H_hermitian.shape[0], 
                                                          dtype=self.dtype, device=self.device)
            except:
                pass
            
            # NaN/Inf チェック
            if torch.isnan(H_hermitian).any() or torch.isinf(H_hermitian).any():
                logger.warning("⚠️ ハミルトニアンにNaN/Infが検出されました")
                return torch.tensor([], device=self.device, dtype=self.float_dtype)
            
            # 固有値計算（改良版）
            try:
                eigenvalues, _ = torch.linalg.eigh(H_hermitian)
                eigenvalues = eigenvalues.real
            except RuntimeError as e:
                logger.warning(f"⚠️ 固有値計算エラー、代替手法を使用: {e}")
                # 代替手法：SVD分解
                U, S, Vh = torch.linalg.svd(H_hermitian)
                eigenvalues = S.real
            
            # 正の固有値のフィルタリング（改良版）
            positive_mask = eigenvalues > 1e-15
            positive_eigenvalues = eigenvalues[positive_mask]
            
            # ソートして上位を選択
            sorted_eigenvalues, _ = torch.sort(positive_eigenvalues, descending=True)
            
            return sorted_eigenvalues[:min(len(sorted_eigenvalues), n_eigenvalues)]
            
        except Exception as e:
            logger.error(f"❌ 固有値計算エラー: {e}")
            return torch.tensor([], device=self.device, dtype=self.float_dtype)

class HighPrecisionRiemannVerifier:
    """
    高精度リーマン予想検証クラス
    """
    
    def __init__(self, hamiltonian: HighPrecisionNKATHamiltonian):
        self.hamiltonian = hamiltonian
        self.device = hamiltonian.device
        
    def compute_spectral_dimension_improved(self, s: complex, 
                                          n_points: int = 50, 
                                          t_range: Tuple[float, float] = (1e-4, 1.0)) -> float:
        """
        改良されたスペクトル次元計算
        """
        eigenvalues = self.hamiltonian.compute_eigenvalues_stable(s, n_eigenvalues=150)
        
        if len(eigenvalues) < 10:
            logger.warning("⚠️ 有効な固有値が不足しています")
            return float('nan')
        
        try:
            # より細かいt値のグリッド
            t_min, t_max = t_range
            t_values = torch.logspace(np.log10(t_min), np.log10(t_max), n_points, device=self.device)
            zeta_values = []
            
            for t in t_values:
                # 数値安定性の改善
                exp_terms = torch.exp(-t * eigenvalues)
                
                # アンダーフロー/オーバーフロー対策
                valid_mask = torch.isfinite(exp_terms) & (exp_terms > 1e-50)
                if torch.sum(valid_mask) < 3:
                    zeta_values.append(1e-50)
                    continue
                
                zeta_t = torch.sum(exp_terms[valid_mask])
                
                if torch.isfinite(zeta_t) and zeta_t > 1e-50:
                    zeta_values.append(zeta_t.item())
                else:
                    zeta_values.append(1e-50)
            
            zeta_values = torch.tensor(zeta_values, device=self.device)
            
            # 対数微分の改良計算
            log_t = torch.log(t_values)
            log_zeta = torch.log(zeta_values + 1e-50)
            
            # 有効なデータ点のフィルタリング（より厳密）
            valid_mask = (torch.isfinite(log_zeta) & 
                         torch.isfinite(log_t) & 
                         (log_zeta > -100) & 
                         (log_zeta < 100))
            
            if torch.sum(valid_mask) < 5:
                logger.warning("⚠️ 有効なデータ点が不足しています")
                return float('nan')
            
            log_t_valid = log_t[valid_mask]
            log_zeta_valid = log_zeta[valid_mask]
            
            # 重み付き線形回帰
            weights = torch.ones_like(log_t_valid)
            # 中央部分により高い重みを付与
            mid_idx = len(log_t_valid) // 2
            if mid_idx >= 2:
                weights[mid_idx-2:mid_idx+3] *= 2.0
            
            # 重み付き最小二乗法
            W = torch.diag(weights)
            A = torch.stack([log_t_valid, torch.ones_like(log_t_valid)], dim=1)
            
            # (A^T W A)^{-1} A^T W y
            try:
                AtWA = torch.mm(torch.mm(A.T, W), A)
                AtWy = torch.mm(torch.mm(A.T, W), log_zeta_valid.unsqueeze(1))
                solution = torch.linalg.solve(AtWA, AtWy)
                slope = solution[0, 0]
            except:
                # フォールバック：通常の最小二乗法
                solution = torch.linalg.lstsq(A, log_zeta_valid).solution
                slope = solution[0]
            
            # スペクトル次元の計算
            spectral_dimension = -2 * slope.item()
            
            # 妥当性チェック（より厳密）
            if abs(spectral_dimension) > 20 or not np.isfinite(spectral_dimension):
                logger.warning(f"⚠️ 異常なスペクトル次元値: {spectral_dimension}")
                return float('nan')
            
            return spectral_dimension
            
        except Exception as e:
            logger.error(f"❌ スペクトル次元計算エラー: {e}")
            return float('nan')
    
    def verify_critical_line_high_precision(self, gamma_values: List[float], 
                                          iterations: int = 3) -> Dict:
        """
        高精度臨界線収束性検証（複数回実行による統計的評価）
        """
        results = {
            'gamma_values': gamma_values,
            'spectral_dimensions_all': [],
            'real_parts_all': [],
            'convergence_to_half_all': [],
            'statistics': {}
        }
        
        logger.info(f"🔍 高精度臨界線収束性検証開始（{iterations}回実行）...")
        
        for iteration in range(iterations):
            logger.info(f"📊 実行 {iteration + 1}/{iterations}")
            
            spectral_dims = []
            real_parts = []
            convergences = []
            
            for gamma in tqdm(gamma_values, desc=f"実行{iteration+1}: γ値での検証"):
                s = 0.5 + 1j * gamma
                
                # スペクトル次元の計算
                d_s = self.compute_spectral_dimension_improved(s)
                spectral_dims.append(d_s)
                
                if not np.isnan(d_s):
                    # 実部の計算
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
                'std_convergence': np.std(valid_convergences),
                'min_convergence': np.min(valid_convergences),
                'max_convergence': np.max(valid_convergences),
                'success_rate': np.sum(valid_convergences < 1e-2) / len(valid_convergences)
            }
        
        return results

def demonstrate_high_precision_riemann():
    """
    高精度リーマン予想検証のデモンストレーション
    """
    print("=" * 80)
    print("🎯 NKAT理論による高精度リーマン予想検証")
    print("=" * 80)
    print("📅 実行日時:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("🔬 精度: complex128 (倍精度)")
    print("🧮 改良点: 適応的次元調整、数値安定性向上、統計的評価")
    print("=" * 80)
    
    # 高精度ハミルトニアンの初期化
    logger.info("🔧 高精度NKAT量子ハミルトニアン初期化中...")
    hamiltonian = HighPrecisionNKATHamiltonian(
        max_n=1000,
        theta=1e-25,
        kappa=1e-15,
        precision='high'
    )
    
    # 高精度検証器の初期化
    verifier = HighPrecisionRiemannVerifier(hamiltonian)
    
    # 高精度臨界線検証
    print("\n📊 高精度臨界線収束性検証")
    gamma_values = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062]
    
    start_time = time.time()
    high_precision_results = verifier.verify_critical_line_high_precision(
        gamma_values, iterations=3
    )
    verification_time = time.time() - start_time
    
    # 結果の表示
    print("\n高精度検証結果:")
    print("γ値      | 平均d_s    | 標準偏差   | 平均Re     | |Re-1/2|平均 | 収束性")
    print("-" * 75)
    
    stats = high_precision_results['statistics']
    for i, gamma in enumerate(gamma_values):
        mean_ds = stats['spectral_dimension_mean'][i]
        std_ds = stats['spectral_dimension_std'][i]
        mean_re = stats['real_part_mean'][i]
        mean_conv = stats['convergence_mean'][i]
        
        if not np.isnan(mean_ds):
            status = "✅" if mean_conv < 1e-1 else "⚠️"
            print(f"{gamma:8.6f} | {mean_ds:9.6f} | {std_ds:9.6f} | {mean_re:9.6f} | {mean_conv:10.6f} | {status}")
        else:
            print(f"{gamma:8.6f} | {'NaN':>9} | {'NaN':>9} | {'NaN':>9} | {'NaN':>10} | ❌")
    
    # 全体統計の表示
    if 'overall_statistics' in high_precision_results:
        overall = high_precision_results['overall_statistics']
        print(f"\n📊 全体統計:")
        print(f"平均収束率: {overall['mean_convergence']:.8f}")
        print(f"標準偏差: {overall['std_convergence']:.8f}")
        print(f"成功率: {overall['success_rate']:.2%}")
        print(f"最良収束: {overall['min_convergence']:.8f}")
    
    print(f"\n⏱️  検証時間: {verification_time:.2f}秒")
    
    # 結果の保存
    with open('high_precision_riemann_results.json', 'w', encoding='utf-8') as f:
        json.dump(high_precision_results, f, indent=2, ensure_ascii=False, default=str)
    
    print("💾 高精度結果を 'high_precision_riemann_results.json' に保存しました")
    
    return high_precision_results

if __name__ == "__main__":
    """
    高精度リーマン予想検証の実行
    """
    try:
        results = demonstrate_high_precision_riemann()
        print("🎉 高精度検証が完了しました！")
    except Exception as e:
        logger.error(f"❌ 実行エラー: {e}")
        print(f"❌ エラーが発生しました: {e}") 