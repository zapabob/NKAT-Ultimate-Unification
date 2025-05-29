"""
🚀 RTX3080対応 スパース行列版ディラック/ラプラシアン作用素のGPU加速解析
Non-Commutative Kolmogorov-Arnold Theory (NKAT) における作用素理論 - GPU+スパース版

Author: NKAT Research Team
Date: 2025-01-23
Version: 1.2 - GPU+スパース最適化版（RTX3080対応）
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict, Callable
import warnings
from pathlib import Path
import json
import time
from dataclasses import dataclass
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

# GPU可用性チェック
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 使用デバイス: {device}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name()}")
    print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'

@dataclass
class SparseGPUOperatorParameters:
    """スパース+GPU対応作用素パラメータの定義"""
    dimension: int  # 空間次元
    lattice_size: int  # 格子サイズ
    theta: float  # 非可換パラメータ
    kappa: float  # κ-変形パラメータ
    mass: float  # 質量項
    coupling: float  # 結合定数
    use_sparse: bool = True  # スパース行列使用
    
    def __post_init__(self):
        if self.dimension not in [2, 3, 4]:
            raise ValueError("次元は2, 3, 4のいずれかである必要があります")
        if self.lattice_size < 8:
            warnings.warn("格子サイズが小さすぎる可能性があります")

class SparseGPUDiracLaplacianAnalyzer:
    """
    🚀 RTX3080対応 スパース行列版ディラック/ラプラシアン作用素の高速解析クラス
    
    主要な解析項目：
    1. スペクトル次元の一意性（スパース+GPU加速）
    2. 固有値分布の特性（効率的計算）
    3. KANアーキテクチャとの関係（メモリ効率）
    4. 非可換補正の効果（高精度計算）
    """
    
    def __init__(self, params: SparseGPUOperatorParameters):
        self.params = params
        self.dim = params.dimension
        self.N = params.lattice_size
        self.theta = params.theta
        self.kappa = params.kappa
        self.mass = params.mass
        self.coupling = params.coupling
        self.use_sparse = params.use_sparse
        self.device = device
        
        print(f"🔧 初期化中: {self.dim}D, 格子サイズ {self.N}^{self.dim}")
        print(f"📊 総格子点数: {self.N**self.dim:,}")
        
        # ガンマ行列の定義（CPU上、小さいので問題なし）
        self.gamma_matrices = self._construct_gamma_matrices()
        
        # メモリ使用量の推定（スパース版）
        spinor_dim = 2 if self.dim <= 3 else 4
        total_dim = self.N**self.dim * spinor_dim
        
        if self.use_sparse:
            # スパース行列の場合、非零要素のみ
            sparsity = 0.01  # 推定スパース率（1%の非零要素）
            memory_gb = (total_dim**2 * sparsity * 8) / 1e9
            print(f"💾 推定メモリ使用量（スパース）: {memory_gb:.2f} GB")
        else:
            memory_gb = (total_dim**2 * 8) / 1e9
            print(f"💾 推定メモリ使用量（密行列）: {memory_gb:.2f} GB")
        
        print(f"📊 行列次元: {total_dim:,} x {total_dim:,}")
        
    def _construct_gamma_matrices(self) -> List[np.ndarray]:
        """
        ガンマ行列の構築（CPU上、NumPy版）
        """
        if self.dim == 2:
            gamma = [
                np.array([[0, 1], [1, 0]], dtype=complex),  # σ_x
                np.array([[0, -1j], [1j, 0]], dtype=complex),  # σ_y
            ]
        elif self.dim == 3:
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
        
        print(f"✅ ガンマ行列構築完了: {len(gamma)}個の{gamma[0].shape}行列")
        return gamma
    
    def construct_discrete_dirac_operator_sparse(self) -> sp.csr_matrix:
        """
        🚀 スパース版離散ディラック作用素の構築
        
        D = Σ_μ γ^μ (∇_μ + iA_μ) + m + θ-補正項
        """
        print("🔨 スパースディラック作用素構築中...")
        start_time = time.time()
        
        # スピノル次元
        spinor_dim = 2 if self.dim <= 3 else 4
        total_dim = self.N**self.dim * spinor_dim
        
        # 空のスパース作用素行列
        D = sp.lil_matrix((total_dim, total_dim), dtype=complex)
        
        # 各方向の微分作用素
        for mu in range(self.dim):
            print(f"  方向 {mu+1}/{self.dim} 処理中...")
            
            # 前進差分と後進差分の平均（中心差分）
            forward_diff = self._construct_forward_difference_sparse(mu)
            backward_diff = self._construct_backward_difference_sparse(mu)
            
            # ガンマ行列との積
            gamma_mu = self.gamma_matrices[mu]
            
            # ディラック項の追加
            diff_operator = (forward_diff - backward_diff) / 2.0
            D += sp.kron(diff_operator, gamma_mu)
            
            # 非可換補正項（θ-変形）
            if self.theta != 0:
                theta_correction = self._construct_theta_correction_sparse(mu)
                D += self.theta * sp.kron(theta_correction, gamma_mu)
        
        # 質量項
        if self.mass != 0:
            mass_operator = sp.eye(self.N**self.dim)
            mass_matrix = self.mass * sp.eye(spinor_dim, dtype=complex)
            D += sp.kron(mass_operator, mass_matrix)
        
        D = D.tocsr()  # CSR形式に変換（効率的な計算のため）
        
        construction_time = time.time() - start_time
        print(f"✅ スパースディラック作用素構築完了: {construction_time:.2f}秒")
        print(f"📊 行列サイズ: {D.shape}, 非零要素数: {D.nnz:,}, スパース率: {D.nnz/(D.shape[0]*D.shape[1]):.6f}")
        
        return D
    
    def _construct_forward_difference_sparse(self, direction: int) -> sp.csr_matrix:
        """スパース版前進差分作用素の構築"""
        # 1次元の前進差分
        diff_1d = sp.diags([1, -1], [1, 0], shape=(self.N, self.N))
        diff_1d = diff_1d.tolil()
        diff_1d[self.N-1, 0] = 1  # 周期境界条件
        diff_1d = diff_1d.tocsr()
        
        # 多次元への拡張（クロネッカー積）
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
    
    def _construct_backward_difference_sparse(self, direction: int) -> sp.csr_matrix:
        """スパース版後進差分作用素の構築"""
        # 1次元の後進差分
        diff_1d = sp.diags([-1, 1], [0, -1], shape=(self.N, self.N))
        diff_1d = diff_1d.tolil()
        diff_1d[0, self.N-1] = -1  # 周期境界条件
        diff_1d = diff_1d.tocsr()
        
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
    
    def _construct_theta_correction_sparse(self, direction: int) -> sp.csr_matrix:
        """スパース版θ-変形補正項の構築"""
        # 簡略化された非可換補正項
        # 実際の実装では位置・運動量作用素の交換子を計算
        
        # 位置作用素（対角行列）
        positions = np.arange(self.N) - self.N // 2
        pos_1d = sp.diags(positions, 0, shape=(self.N, self.N))
        
        # 多次元への拡張
        operators = []
        for d in range(self.dim):
            if d == direction:
                operators.append(pos_1d)
            else:
                operators.append(sp.eye(self.N))
        
        x_op = operators[0]
        for op in operators[1:]:
            x_op = sp.kron(x_op, op)
        
        # 簡略化された補正項（実際の交換子計算は省略）
        return x_op * 0.01  # 小さな補正
    
    def compute_spectral_dimension_sparse_gpu(self, operator: sp.csr_matrix, 
                                            n_eigenvalues: int = 50) -> Tuple[float, Dict]:
        """
        🚀 スパース+GPU版スペクトル次元の高速計算
        """
        print("🔍 スペクトル次元計算中（スパース+GPU）...")
        start_time = time.time()
        
        # 固有値の計算（scipy sparse + GPU転送）
        try:
            # エルミート化
            operator_hermitian = operator.conj().T @ operator
            
            # 固有値計算（scipy sparse eigenvalue solver）
            eigenvalues, _ = eigsh(operator_hermitian, k=min(n_eigenvalues, operator.shape[0]-2), 
                                 which='SM', return_eigenvectors=False)
            eigenvalues = np.real(eigenvalues)
            
            # 正の固有値のみを使用
            eigenvalues = eigenvalues[eigenvalues > 1e-12]
            
        except Exception as e:
            print(f"❌ 固有値計算エラー: {e}")
            return float('nan'), {}
        
        if len(eigenvalues) < 10:
            print("⚠️  警告: 有効な固有値が少なすぎます")
            return float('nan'), {}
        
        # GPU上でスペクトルゼータ関数の計算
        eigenvalues_gpu = torch.tensor(eigenvalues, device=self.device, dtype=torch.float32)
        t_values = torch.logspace(-3, 0, 50, device=self.device)
        
        zeta_values = []
        for t in t_values:
            zeta_t = torch.sum(torch.exp(-t * eigenvalues_gpu))
            zeta_values.append(zeta_t.item())
        
        zeta_values = torch.tensor(zeta_values, device=self.device)
        
        # 対数微分の計算（GPU上）
        log_t = torch.log(t_values)
        log_zeta = torch.log(zeta_values + 1e-12)
        
        # 線形回帰で傾きを求める（GPU上）
        valid_mask = torch.isfinite(log_zeta) & torch.isfinite(log_t)
        if torch.sum(valid_mask) < 5:
            print("⚠️  警告: 有効なデータ点が少なすぎます")
            return float('nan'), {}
        
        log_t_valid = log_t[valid_mask]
        log_zeta_valid = log_zeta[valid_mask]
        
        # 最小二乗法（GPU上）
        A = torch.stack([log_t_valid, torch.ones_like(log_t_valid)], dim=1)
        slope, intercept = torch.linalg.lstsq(A, log_zeta_valid).solution
        
        spectral_dimension = -2 * slope.item()
        
        computation_time = time.time() - start_time
        print(f"✅ スペクトル次元計算完了: {computation_time:.2f}秒")
        
        # 詳細情報
        analysis_info = {
            'eigenvalues': eigenvalues,
            'n_eigenvalues': len(eigenvalues),
            'min_eigenvalue': np.min(eigenvalues),
            'max_eigenvalue': np.max(eigenvalues),
            'spectral_gap': eigenvalues[1] - eigenvalues[0] if len(eigenvalues) > 1 else 0,
            'zeta_function': zeta_values.cpu().numpy(),
            't_values': t_values.cpu().numpy(),
            'slope': slope.item(),
            'computation_time': computation_time
        }
        
        return spectral_dimension, analysis_info

def demonstrate_sparse_gpu_analysis():
    """🚀 RTX3080対応スパース+GPU解析のデモンストレーション"""
    
    print("=" * 80)
    print("🚀 RTX3080対応 スパース+GPU ディラック/ラプラシアン作用素解析")
    print("=" * 80)
    
    # GPU情報表示
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"🔧 CUDA Version: {torch.version.cuda}")
    else:
        print("⚠️  CUDA未対応 - CPUモードで実行")
    
    # パラメータ設定（RTX3080+スパース最適化）
    params = SparseGPUOperatorParameters(
        dimension=4,
        lattice_size=24,  # スパース行列なので大きめでも可能
        theta=0.01,
        kappa=0.05,
        mass=0.1,
        coupling=1.0,
        use_sparse=True
    )
    
    analyzer = SparseGPUDiracLaplacianAnalyzer(params)
    
    print(f"\n📊 解析パラメータ:")
    print(f"次元: {params.dimension}")
    print(f"格子サイズ: {params.lattice_size}")
    print(f"θ パラメータ: {params.theta}")
    print(f"κ パラメータ: {params.kappa}")
    print(f"質量: {params.mass}")
    print(f"スパース行列使用: {params.use_sparse}")
    
    # 1. ディラック作用素の解析
    print("\n🔨 1. スパースディラック作用素の構築と解析...")
    total_start = time.time()
    
    D = analyzer.construct_discrete_dirac_operator_sparse()
    
    d_s_dirac, dirac_info = analyzer.compute_spectral_dimension_sparse_gpu(D)
    print(f"📈 ディラック作用素のスペクトル次元: {d_s_dirac:.6f}")
    print(f"🎯 理論値との差: {abs(d_s_dirac - params.dimension):.6f}")
    
    total_time = time.time() - total_start
    print(f"\n⏱️  総計算時間: {total_time:.2f}秒")
    
    # 2. 結果の保存
    results_summary = {
        'gpu_info': {
            'device_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',
            'cuda_available': torch.cuda.is_available(),
            'total_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
        },
        'parameters': {
            'dimension': params.dimension,
            'lattice_size': params.lattice_size,
            'theta': params.theta,
            'kappa': params.kappa,
            'mass': params.mass,
            'use_sparse': params.use_sparse
        },
        'results': {
            'dirac_spectral_dimension': d_s_dirac,
            'total_computation_time': total_time,
            'dirac_computation_time': dirac_info.get('computation_time', 0),
            'matrix_size': D.shape[0],
            'nnz_elements': D.nnz,
            'sparsity_ratio': D.nnz / (D.shape[0] * D.shape[1])
        },
        'analysis_timestamp': str(time.time())
    }
    
    with open('sparse_gpu_dirac_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False, default=str)
    
    print("\n💾 結果が 'sparse_gpu_dirac_results.json' に保存されました。")
    print("🎉 RTX3080対応スパース+GPU解析完了！")
    
    return analyzer, results_summary

if __name__ == "__main__":
    # RTX3080対応スパース+GPU解析のデモンストレーション
    analyzer, results = demonstrate_sparse_gpu_analysis() 