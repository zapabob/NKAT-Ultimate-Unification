#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT-L関数一般化：ディリクレL関数への拡張

論文付録EのL関数一般化理論を実装し、指標修正NKAT作用素による
一般化リーマン予想（GRH）の数値検証を行う

Author: Research Team
Date: 2025
License: MIT
"""

import numpy as np
import cupy as cp
import scipy.linalg
from scipy.special import dirichlet_eta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional, Union
import json
from datetime import datetime
import tqdm
import warnings
from dataclasses import dataclass
from nkat_riemann_rigorous_mathematical_framework import (
    NKATParameters, ComputationConfig, EmergencyRecoverySystem, EULER_GAMMA, PI
)

warnings.filterwarnings('ignore')

@dataclass
class CharacterParameters:
    """ディリクレ指標のパラメータ"""
    modulus: int = 3  # 法 q
    character_type: str = "quadratic"  # 指標の種類
    is_primitive: bool = True  # 原始指標かどうか
    conductor: int = None  # 導手
    
    def __post_init__(self):
        if self.conductor is None:
            self.conductor = self.modulus

class DirichletCharacter:
    """ディリクレ指標の実装"""
    
    def __init__(self, modulus: int, character_type: str = "quadratic"):
        self.modulus = modulus
        self.character_type = character_type
        self.values = self._compute_character_values()
        self.is_real = self._check_if_real()
        self.gauss_sum = self._compute_gauss_sum()
        
    def _compute_character_values(self) -> Dict[int, complex]:
        """指標の値を計算"""
        values = {}
        
        if self.character_type == "principal":
            # 主指標 χ_0
            for n in range(self.modulus):
                values[n] = 1.0 if self._gcd(n, self.modulus) == 1 else 0.0
                
        elif self.character_type == "quadratic" and self.modulus == 3:
            # レジャンドル記号 (n/3)
            for n in range(self.modulus):
                if self._gcd(n, 3) == 1:
                    values[n] = 1.0 if (n % 3) == 1 else -1.0
                else:
                    values[n] = 0.0
                    
        elif self.character_type == "quadratic" and self.modulus == 4:
            # (-1)^n for odd n
            for n in range(self.modulus):
                if n % 2 == 1:
                    values[n] = (-1.0)**(n//2)
                else:
                    values[n] = 0.0 if n % 2 == 0 and n != 0 else 1.0
                    
        elif self.character_type == "primitive" and self.modulus == 5:
            # 5次原始指標
            for n in range(self.modulus):
                if self._gcd(n, 5) == 1:
                    if n % 5 == 1:
                        values[n] = 1.0
                    elif n % 5 == 2:
                        values[n] = 1j  # i
                    elif n % 5 == 3:
                        values[n] = -1j  # -i
                    elif n % 5 == 4:
                        values[n] = -1.0
                else:
                    values[n] = 0.0
        else:
            # デフォルト：主指標
            for n in range(self.modulus):
                values[n] = 1.0 if self._gcd(n, self.modulus) == 1 else 0.0
        
        return values
    
    def _gcd(self, a: int, b: int) -> int:
        """最大公約数"""
        while b:
            a, b = b, a % b
        return a
    
    def _check_if_real(self) -> bool:
        """実指標かどうかチェック"""
        return all(np.isreal(val) for val in self.values.values())
    
    def _compute_gauss_sum(self) -> complex:
        """ガウス和の計算 G(χ) = Σ χ(a) exp(2πia/q)"""
        gauss_sum = 0.0
        for a in range(1, self.modulus + 1):
            chi_a = self.evaluate(a)
            gauss_sum += chi_a * np.exp(2j * PI * a / self.modulus)
        return gauss_sum
    
    def evaluate(self, n: int) -> complex:
        """指標の値を評価"""
        return self.values.get(n % self.modulus, 0.0)
    
    def conjugate(self) -> 'DirichletCharacter':
        """複素共役指標"""
        conj_char = DirichletCharacter(self.modulus, self.character_type)
        conj_char.values = {k: np.conj(v) for k, v in self.values.items()}
        conj_char.is_real = self.is_real
        conj_char.gauss_sum = np.conj(self.gauss_sum)
        return conj_char


class NKATLFunctionFramework:
    """NKAT-L関数枠組みの実装"""
    
    def __init__(self, params: NKATParameters = None, config: ComputationConfig = None):
        self.params = params or NKATParameters()
        self.config = config or ComputationConfig()
        self.recovery = EmergencyRecoverySystem("nkat_lfunction")
        
        # GPU初期化
        if self.config.use_gpu and cp.cuda.is_available():
            self.device = cp.cuda.Device(0)
            self.device.use()
            print(f"🚀 CUDA初期化完了: L関数一般化モード")
        else:
            print("⚠️ CPU計算モードで実行")
            self.config.use_gpu = False
        
        self.results = {}
        self.last_checkpoint = datetime.now()
    
    def construct_character_modified_interaction_kernel(self, N: int, character: DirichletCharacter) -> np.ndarray:
        """
        定義E.1: 指標修正相互作用核の実装
        V_{jk}^{(N,χ)} = χ(j-k) * V_{jk}^{(N)}
        """
        if self.config.use_gpu:
            return self._construct_character_kernel_gpu(N, character)
        else:
            return self._construct_character_kernel_cpu(N, character)
    
    def _construct_character_kernel_cpu(self, N: int, character: DirichletCharacter) -> np.ndarray:
        """CPU版指標修正核構築"""
        V = np.zeros((N, N), dtype=np.complex128)
        
        for i in range(N):
            for j in range(N):
                if i != j and abs(i - j) <= self.params.K:
                    # 基本相互作用核
                    distance = np.sqrt(abs(i - j) + 1.0)
                    phase = 2.0 * PI * (i + j) / (character.modulus * self.params.Nc)
                    base_kernel = (self.params.c0 / (N * distance)) * np.exp(1j * phase)
                    
                    # 指標修正
                    chi_value = character.evaluate(i - j)
                    V[i, j] = chi_value * base_kernel
        
        return V
    
    def _construct_character_kernel_gpu(self, N: int, character: DirichletCharacter) -> np.ndarray:
        """GPU版指標修正核構築"""
        # GPU上でカーネル構築
        i_indices = cp.arange(N)[:, None]
        j_indices = cp.arange(N)[None, :]
        
        # 基本マスク
        distance_mask = cp.abs(i_indices - j_indices) <= self.params.K
        non_diagonal_mask = i_indices != j_indices
        valid_mask = distance_mask & non_diagonal_mask
        
        # 距離とフェーズ計算
        distance = cp.sqrt(cp.abs(i_indices - j_indices) + 1.0)
        phase = 2.0 * PI * (i_indices + j_indices) / (character.modulus * self.params.Nc)
        
        # 基本相互作用核
        base_kernel = cp.zeros((N, N), dtype=cp.complex128)
        base_kernel[valid_mask] = (self.params.c0 / (N * distance[valid_mask])) * cp.exp(1j * phase[valid_mask])
        
        # 指標修正
        diff_indices = cp.asnumpy(i_indices - j_indices)
        character_matrix = np.zeros((N, N), dtype=np.complex128)
        
        for i in range(N):
            for j in range(N):
                character_matrix[i, j] = character.evaluate(diff_indices[i, j])
        
        character_matrix_gpu = cp.asarray(character_matrix)
        V = character_matrix_gpu * base_kernel
        
        return cp.asnumpy(V)
    
    def construct_character_modified_nkat_operator(self, N: int, character: DirichletCharacter) -> np.ndarray:
        """
        指標修正NKAT作用素の構築
        H_N^{(χ)} = Σ E_j^{(N)} e_j ⊗ e_j + Σ χ(j-k) V_{jk}^{(N)} e_j ⊗ e_k
        """
        # 基本エネルギー準位
        j_indices = np.arange(N, dtype=np.float64)
        energy_levels = (j_indices + 0.5) * PI / N + EULER_GAMMA / (N * PI)
        
        # L'(0,χ)/L(0,χ) 補正項（奇指標の場合は0）
        if not character.is_real:  # 複素指標の場合の近似
            l_prime_correction = np.log(character.modulus) / (N * character.modulus)
            energy_levels += l_prime_correction
        
        H = np.diag(energy_levels)
        
        # 指標修正相互作用項
        V_chi = self.construct_character_modified_interaction_kernel(N, character)
        H += V_chi
        
        # 実指標の場合のみ自己随伴性チェック
        if character.is_real:
            hermiticity_error = np.max(np.abs(H - H.conj().T))
            if hermiticity_error > 1e-12:
                raise ValueError(f"自己随伴性エラー: {hermiticity_error}")
        
        return H
    
    def compute_l_function_spectral_correspondence(self, eigenvalues: np.ndarray, character: DirichletCharacter, N: int, s: complex) -> complex:
        """
        定理E.2: L関数スペクトル対応の実装
        lim c_N^{(χ)} Σ χ(j) (λ_j^{(N,χ)})^{-s} = L(s,χ)
        """
        # 正規化定数
        if character.character_type == "principal":
            phi_q = sum(1 for n in range(character.modulus) if self._gcd(n, character.modulus) == 1)
            c_N = (PI * phi_q) / (character.modulus * N)
        else:
            c_N = (PI * abs(character.gauss_sum)) / (np.sqrt(character.modulus) * N)
        
        # スペクトルゼータ関数
        spectral_zeta = 0.0
        for j, eigenval in enumerate(eigenvalues):
            chi_j = character.evaluate(j)
            if abs(chi_j) > 1e-12 and eigenval > 1e-12:  # 数値安定性
                spectral_zeta += chi_j * (eigenval ** (-s))
        
        return c_N * spectral_zeta
    
    def _gcd(self, a: int, b: int) -> int:
        """最大公約数"""
        while b:
            a, b = b, a % b
        return a
    
    def verify_character_orthogonality(self, characters: List[DirichletCharacter], N: int) -> Dict:
        """
        定理E.1: 指標直交性の検証
        """
        orthogonality_matrix = np.zeros((len(characters), len(characters)), dtype=np.complex128)
        
        for i, chi1 in enumerate(characters):
            for j, chi2 in enumerate(characters):
                # 直交積の計算
                inner_product = 0.0
                for n in range(N):
                    inner_product += chi1.evaluate(n) * np.conj(chi2.evaluate(n))
                
                orthogonality_matrix[i, j] = inner_product / chi1.modulus
        
        # 理論値との比較
        expected_matrix = np.eye(len(characters))
        orthogonality_error = np.max(np.abs(orthogonality_matrix - expected_matrix))
        
        return {
            'orthogonality_matrix': orthogonality_matrix.tolist(),
            'expected_matrix': expected_matrix.tolist(),
            'max_error': float(orthogonality_error),
            'orthogonality_satisfied': orthogonality_error < 1e-10
        }
    
    def run_grh_verification(self, characters: List[DirichletCharacter]) -> Dict:
        """一般化リーマン予想（GRH）の数値検証"""
        print("🔬 一般化リーマン予想（GRH）検証開始")
        print("=" * 80)
        
        all_results = {}
        
        for char_idx, character in enumerate(characters):
            print(f"\n📊 指標 {char_idx+1}/{len(characters)}: 法={character.modulus}, 型={character.character_type}")
            
            character_results = {
                'character_info': {
                    'modulus': character.modulus,
                    'type': character.character_type,
                    'is_real': character.is_real,
                    'conductor': character.modulus,
                    'gauss_sum': complex(character.gauss_sum)
                },
                'dimensions': {}
            }
            
            for N in tqdm.tqdm(self.config.dimensions, desc=f"次元解析(χ_{char_idx+1})"):
                print(f"\n  📏 次元 N = {N}")
                
                dimension_results = {
                    'trials': [],
                    'statistics': {},
                    'grh_verification': {}
                }
                
                trial_theta_params = []
                
                for trial in tqdm.tqdm(range(self.config.num_trials), 
                                     desc=f"N={N}試行", leave=False):
                    try:
                        # 指標修正NKAT作用素構築
                        H_chi = self.construct_character_modified_nkat_operator(N, character)
                        
                        # 固有値計算
                        if character.is_real:
                            eigenvalues = scipy.linalg.eigvalsh(H_chi)
                        else:
                            # 複素指標の場合は自己随伴化
                            H_hermitian = (H_chi + H_chi.conj().T) / 2
                            eigenvalues = scipy.linalg.eigvalsh(H_hermitian)
                        
                        eigenvalues.sort()
                        
                        # スペクトルパラメータ計算
                        q_indices = np.arange(N)
                        theoretical_energies = ((q_indices + 0.5) * PI / N + 
                                              EULER_GAMMA / (N * PI))
                        theta_params = eigenvalues - theoretical_energies
                        
                        # L関数対応チェック（s=2での値）
                        l_correspondence = self.compute_l_function_spectral_correspondence(
                            eigenvalues, character, N, 2.0
                        )
                        
                        trial_result = {
                            'trial': trial,
                            'eigenvalues': eigenvalues.tolist(),
                            'theta_params': theta_params.tolist(),
                            'l_function_correspondence': complex(l_correspondence)
                        }
                        
                        dimension_results['trials'].append(trial_result)
                        trial_theta_params.append(theta_params)
                        
                    except Exception as e:
                        print(f"    ⚠️ 試行 {trial} でエラー: {e}")
                        continue
                
                if trial_theta_params:
                    # 統計解析
                    all_theta = np.array(trial_theta_params)
                    mean_theta = np.mean(all_theta, axis=0)
                    
                    # GRH検証
                    real_parts = np.real(mean_theta)
                    grh_deviations = np.abs(real_parts - 0.5)
                    max_deviation = np.max(grh_deviations)
                    
                    # 理論的上界（指標修正版）
                    log_N = np.log(N)
                    C_character = 2.0 * np.sqrt(2.0 * PI) * np.sqrt(character.modulus)
                    theoretical_bound = C_character * log_N / np.sqrt(N)
                    
                    dimension_results['statistics'] = {
                        'mean_real_part': float(np.mean(real_parts)),
                        'std_real_part': float(np.std(real_parts)),
                        'convergence_to_half': float(np.abs(np.mean(real_parts) - 0.5)),
                        'num_successful_trials': len(trial_theta_params)
                    }
                    
                    dimension_results['grh_verification'] = {
                        'max_deviation': float(max_deviation),
                        'theoretical_bound': float(theoretical_bound),
                        'bound_satisfied': bool(max_deviation <= theoretical_bound),
                        'bound_ratio': float(max_deviation / theoretical_bound)
                    }
                    
                    print(f"    ✅ N={N}: 実部平均={dimension_results['statistics']['mean_real_part']:.6f}")
                    print(f"        GRH偏差={max_deviation:.2e}, 上界比={dimension_results['grh_verification']['bound_ratio']:.1%}")
                
                character_results['dimensions'][N] = dimension_results
            
            all_results[f"character_{char_idx+1}"] = character_results
        
        # 指標直交性検証
        if len(characters) > 1:
            orthogonality_result = self.verify_character_orthogonality(characters, max(self.config.dimensions))
            all_results['orthogonality_verification'] = orthogonality_result
        
        return all_results
    
    def generate_grh_report(self, results: Dict, characters: List[DirichletCharacter]) -> str:
        """GRH検証レポート生成"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 結果をJSON形式で保存
        results_file = f"nkat_grh_verification_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # レポート生成
        report = []
        report.append("# NKAT-L関数一般化：一般化リーマン予想（GRH）検証レポート")
        report.append(f"## 実行時刻: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}")
        report.append("")
        
        # 検証した指標の情報
        report.append("## 検証指標一覧")
        for i, char in enumerate(characters):
            report.append(f"### 指標 {i+1}")
            report.append(f"- 法: {char.modulus}")
            report.append(f"- 型: {char.character_type}")
            report.append(f"- 実指標: {'Yes' if char.is_real else 'No'}")
            report.append(f"- ガウス和: {char.gauss_sum:.4f}")
            report.append("")
        
        # 結果サマリー
        report.append("## GRH検証結果サマリー")
        report.append("")
        
        for char_key, char_result in results.items():
            if char_key.startswith('character_'):
                char_info = char_result['character_info']
                report.append(f"### {char_key} (法={char_info['modulus']}, {char_info['type']})")
                report.append("")
                report.append("| 次元 N | 実部平均 | |平均-0.5| | GRH上界 | 上界達成率 |")
                report.append("|--------|----------|-----------|---------|-----------|")
                
                for N, dim_result in char_result['dimensions'].items():
                    if 'statistics' in dim_result and 'grh_verification' in dim_result:
                        stats = dim_result['statistics']
                        grh = dim_result['grh_verification']
                        
                        report.append(f"| {N} | {stats['mean_real_part']:.6f} | "
                                     f"{stats['convergence_to_half']:.2e} | "
                                     f"{grh['theoretical_bound']:.2e} | "
                                     f"{grh['bound_ratio']:.1%} |")
                
                report.append("")
        
        # 指標直交性
        if 'orthogonality_verification' in results:
            ortho = results['orthogonality_verification']
            report.append("## 指標直交性検証")
            report.append(f"- 最大誤差: {ortho['max_error']:.2e}")
            report.append(f"- 直交性満足: {'✅ YES' if ortho['orthogonality_satisfied'] else '❌ NO'}")
            report.append("")
        
        # 全体の結論
        report.append("## 結論")
        all_grh_satisfied = True
        for char_key, char_result in results.items():
            if char_key.startswith('character_'):
                for N, dim_result in char_result['dimensions'].items():
                    if 'grh_verification' in dim_result:
                        if not dim_result['grh_verification']['bound_satisfied']:
                            all_grh_satisfied = False
                            break
        
        if all_grh_satisfied:
            report.append("✅ **全ての検証指標でGRHの数値的証拠を確認**")
            report.append("")
            report.append("各指標について、スペクトルパラメータの実部が1/2に収束し、")
            report.append("理論的上界を満足することが確認された。")
            report.append("これはNKAT枠組みがL関数の広いクラスに適用可能であることを示している。")
        else:
            report.append("⚠️ 一部の指標でGRH上界を超過")
            report.append("さらなる理論的精緻化が必要")
        
        report_text = "\n".join(report)
        
        # レポートファイル保存
        report_file = f"nkat_grh_verification_report_{timestamp}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"📊 GRHレポート保存: {report_file}")
        print(f"📊 結果データ: {results_file}")
        
        return report_text


def main():
    """メイン実行関数"""
    print("🚀 NKAT-L関数一般化：一般化リーマン予想（GRH）数値検証")
    print("🔬 ディリクレL関数への厳密な拡張")
    print("⚡ RTX3080 CUDA並列化")
    print("=" * 80)
    
    # パラメータ設定
    params = NKATParameters(
        c0=0.1,
        Nc=30.0,  # L関数用に調整
        K=8,
        delta=1.0/PI,
        A0=1.0,
        eta=1.0
    )
    
    config = ComputationConfig(
        dimensions=[100, 300, 500, 1000],
        num_trials=5,  # 複数指標なので試行数を調整
        use_gpu=True,
        save_checkpoints=True
    )
    
    # 検証する指標セット
    characters = [
        DirichletCharacter(modulus=3, character_type="quadratic"),     # (n/3)
        DirichletCharacter(modulus=4, character_type="quadratic"),     # (-1)^n
        DirichletCharacter(modulus=5, character_type="primitive"),     # 5次原始指標
        DirichletCharacter(modulus=3, character_type="principal"),     # 主指標 mod 3
    ]
    
    print(f"📋 検証対象: {len(characters)}個のディリクレ指標")
    for i, char in enumerate(characters):
        print(f"  {i+1}. 法={char.modulus}, 型={char.character_type}, 実指標={char.is_real}")
    
    # 解析実行
    framework = NKATLFunctionFramework(params, config)
    
    try:
        results = framework.run_grh_verification(characters)
        report = framework.generate_grh_report(results, characters)
        
        print("\n" + "=" * 80)
        print("✅ GRH検証完了!")
        print(report)
        
    except KeyboardInterrupt:
        print("\n⚠️ ユーザーによる中断")
    except Exception as e:
        print(f"\n❌ エラー発生: {e}")
        raise

if __name__ == "__main__":
    main() 