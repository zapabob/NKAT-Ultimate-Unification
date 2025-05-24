"""
🚀 RTX3080対応 ディラック/ラプラシアン作用素のGPU加速解析
Non-Commutative Kolmogorov-Arnold Theory (NKAT) における作用素理論 - GPU版

Author: NKAT Research Team
Date: 2025-01-23
Version: 1.1 - GPU加速版（RTX3080最適化）
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

# GPU可用性チェック
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 使用デバイス: {device}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name()}")
    print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'

@dataclass
class GPUOperatorParameters:
    """GPU対応作用素パラメータの定義"""
    dimension: int  # 空間次元
    lattice_size: int  # 格子サイズ
    theta: float  # 非可換パラメータ
    kappa: float  # κ-変形パラメータ
    mass: float  # 質量項
    coupling: float  # 結合定数
    dtype: torch.dtype = torch.complex64  # GPU最適化のためcomplex64使用
    
    def __post_init__(self):
        if self.dimension not in [2, 3, 4]:
            raise ValueError("次元は2, 3, 4のいずれかである必要があります")
        if self.lattice_size < 8:
            warnings.warn("格子サイズが小さすぎる可能性があります")
        if self.lattice_size > 64 and device.type == 'cpu':
            warnings.warn("CPUモードで大きな格子サイズは遅くなります")

class GPUDiracLaplacianAnalyzer:
    """
    🚀 RTX3080対応 ディラック/ラプラシアン作用素の高速GPU解析クラス
    
    主要な解析項目：
    1. スペクトル次元の一意性（GPU加速）
    2. 固有値分布の特性（バッチ処理）
    3. KANアーキテクチャとの関係（並列計算）
    4. 非可換補正の効果（高精度計算）
    """
    
    def __init__(self, params: GPUOperatorParameters):
        self.params = params
        self.dim = params.dimension
        self.N = params.lattice_size
        self.theta = params.theta
        self.kappa = params.kappa
        self.mass = params.mass
        self.coupling = params.coupling
        self.dtype = params.dtype
        self.device = device
        
        print(f"🔧 初期化中: {self.dim}D, 格子サイズ {self.N}x{self.N}x{self.N}x{self.N}")
        print(f"📊 総格子点数: {self.N**self.dim:,}")
        
        # ガンマ行列の定義（GPU上）
        self.gamma_matrices = self._construct_gamma_matrices_gpu()
        
        # メモリ使用量の推定
        spinor_dim = 2 if self.dim <= 3 else 4
        total_dim = self.N**self.dim * spinor_dim
        # complex64 = 8 bytes per element
        memory_gb = (total_dim**2 * 8) / 1e9  # 正確な計算
        print(f"💾 推定メモリ使用量: {memory_gb:.2f} GB")
        print(f"📊 行列次元: {total_dim:,} x {total_dim:,}")
        
        if memory_gb > 8:  # RTX3080は10GB VRAMだが、安全マージン
            warnings.warn(f"⚠️  大きなメモリ使用量 ({memory_gb:.1f} GB) - バッチ処理を推奨")
        
    def _construct_gamma_matrices_gpu(self) -> List[torch.Tensor]:
        """
        🚀 GPU上でガンマ行列の構築（次元に応じて）
        
        2D: パウリ行列
        3D: パウリ行列の拡張  
        4D: ディラック行列
        """
        if self.dim == 2:
            # 2Dパウリ行列
            gamma = [
                torch.tensor([[0, 1], [1, 0]], dtype=self.dtype, device=self.device),  # σ_x
                torch.tensor([[0, -1j], [1j, 0]], dtype=self.dtype, device=self.device),  # σ_y
            ]
        elif self.dim == 3:
            # 3Dパウリ行列
            gamma = [
                torch.tensor([[0, 1], [1, 0]], dtype=self.dtype, device=self.device),  # σ_x
                torch.tensor([[0, -1j], [1j, 0]], dtype=self.dtype, device=self.device),  # σ_y
                torch.tensor([[1, 0], [0, -1]], dtype=self.dtype, device=self.device),  # σ_z
            ]
        elif self.dim == 4:
            # 4Dディラック行列（標準表現）
            sigma = [
                torch.tensor([[0, 1], [1, 0]], dtype=self.dtype, device=self.device),
                torch.tensor([[0, -1j], [1j, 0]], dtype=self.dtype, device=self.device),
                torch.tensor([[1, 0], [0, -1]], dtype=self.dtype, device=self.device)
            ]
            I2 = torch.eye(2, dtype=self.dtype, device=self.device)
            O2 = torch.zeros(2, 2, dtype=self.dtype, device=self.device)
            
            # 正しいディラック行列の構築
            gamma = [
                # γ^1 = [[0, σ_x], [σ_x, 0]]
                torch.cat([torch.cat([O2, sigma[0]], dim=1), 
                          torch.cat([sigma[0], O2], dim=1)], dim=0),
                # γ^2 = [[0, σ_y], [σ_y, 0]]  
                torch.cat([torch.cat([O2, sigma[1]], dim=1),
                          torch.cat([sigma[1], O2], dim=1)], dim=0),
                # γ^3 = [[0, σ_z], [σ_z, 0]]
                torch.cat([torch.cat([O2, sigma[2]], dim=1),
                          torch.cat([sigma[2], O2], dim=1)], dim=0),
                # γ^0 = [[I, 0], [0, -I]]
                torch.cat([torch.cat([I2, O2], dim=1),
                          torch.cat([O2, -I2], dim=1)], dim=0),
            ]
        
        print(f"✅ ガンマ行列構築完了: {len(gamma)}個の{gamma[0].shape}行列")
        return gamma
    
    def construct_discrete_dirac_operator_gpu(self) -> torch.Tensor:
        """
        🚀 GPU上で離散ディラック作用素の構築
        
        D = Σ_μ γ^μ (∇_μ + iA_μ) + m + θ-補正項
        """
        print("🔨 ディラック作用素構築中...")
        start_time = time.time()
        
        # スピノル次元
        spinor_dim = 2 if self.dim <= 3 else 4
        total_dim = self.N**self.dim * spinor_dim
        
        # 空の作用素行列（GPU上）
        D = torch.zeros(total_dim, total_dim, dtype=self.dtype, device=self.device)
        
        # 各方向の微分作用素
        for mu in range(self.dim):
            print(f"  方向 {mu+1}/{self.dim} 処理中...")
            
            # 前進差分と後進差分の平均（中心差分）
            forward_diff = self._construct_forward_difference_gpu(mu)
            backward_diff = self._construct_backward_difference_gpu(mu)
            
            # ガンマ行列との積
            gamma_mu = self.gamma_matrices[mu]
            
            # ディラック項の追加
            diff_operator = (forward_diff - backward_diff) / 2.0
            D += torch.kron(diff_operator, gamma_mu)
            
            # 非可換補正項（θ-変形）
            if self.theta != 0:
                theta_correction = self._construct_theta_correction_gpu(mu)
                D += self.theta * torch.kron(theta_correction, gamma_mu)
        
        # 質量項
        if self.mass != 0:
            mass_operator = torch.eye(self.N**self.dim, dtype=self.dtype, device=self.device)
            mass_matrix = self.mass * torch.eye(spinor_dim, dtype=self.dtype, device=self.device)
            D += torch.kron(mass_operator, mass_matrix)
        
        construction_time = time.time() - start_time
        print(f"✅ ディラック作用素構築完了: {construction_time:.2f}秒")
        print(f"📊 行列サイズ: {D.shape}, 非零要素率: {(D != 0).float().mean():.4f}")
        
        return D
    
    def construct_discrete_laplacian_gpu(self) -> torch.Tensor:
        """
        🚀 GPU上で離散ラプラシアン作用素の構築
        
        Δ = Σ_μ ∇_μ² + κ-補正項 + θ-補正項
        """
        print("🔨 ラプラシアン作用素構築中...")
        start_time = time.time()
        
        total_dim = self.N**self.dim
        Delta = torch.zeros(total_dim, total_dim, dtype=torch.float32, device=self.device)
        
        # 各方向の2階微分
        for mu in range(self.dim):
            print(f"  方向 {mu+1}/{self.dim} 処理中...")
            second_diff = self._construct_second_difference_gpu(mu)
            Delta += second_diff
            
            # κ-変形補正項
            if self.kappa != 0:
                kappa_correction = self._construct_kappa_correction_gpu(mu)
                Delta += self.kappa * kappa_correction
        
        # θ-変形による非可換補正
        if self.theta != 0:
            for mu in range(self.dim):
                for nu in range(mu + 1, self.dim):
                    mixed_diff = self._construct_mixed_difference_gpu(mu, nu)
                    Delta += self.theta * mixed_diff
        
        construction_time = time.time() - start_time
        print(f"✅ ラプラシアン作用素構築完了: {construction_time:.2f}秒")
        
        return Delta
    
    def _construct_forward_difference_gpu(self, direction: int) -> torch.Tensor:
        """🚀 GPU上で前進差分作用素の構築"""
        # 1次元の前進差分
        diag_main = -torch.ones(self.N, device=self.device)
        diag_upper = torch.ones(self.N-1, device=self.device)
        
        diff_1d = torch.diag(diag_main) + torch.diag(diag_upper, diagonal=1)
        diff_1d[-1, 0] = 1  # 周期境界条件
        
        # 多次元への拡張（クロネッカー積）
        result = torch.eye(1, device=self.device)
        for d in range(self.dim):
            if d == direction:
                result = torch.kron(result, diff_1d)
            else:
                result = torch.kron(result, torch.eye(self.N, device=self.device))
        
        return result
    
    def _construct_backward_difference_gpu(self, direction: int) -> torch.Tensor:
        """🚀 GPU上で後進差分作用素の構築"""
        # 1次元の後進差分
        diag_main = torch.ones(self.N, device=self.device)
        diag_lower = -torch.ones(self.N-1, device=self.device)
        
        diff_1d = torch.diag(diag_main) + torch.diag(diag_lower, diagonal=-1)
        diff_1d[0, -1] = -1  # 周期境界条件
        
        # 多次元への拡張
        result = torch.eye(1, device=self.device)
        for d in range(self.dim):
            if d == direction:
                result = torch.kron(result, diff_1d)
            else:
                result = torch.kron(result, torch.eye(self.N, device=self.device))
        
        return result
    
    def _construct_second_difference_gpu(self, direction: int) -> torch.Tensor:
        """🚀 GPU上で2階差分作用素の構築"""
        # 1次元の2階差分
        diag_main = -2 * torch.ones(self.N, device=self.device)
        diag_off = torch.ones(self.N-1, device=self.device)
        
        diff_1d = torch.diag(diag_main) + torch.diag(diag_off, diagonal=1) + torch.diag(diag_off, diagonal=-1)
        diff_1d[0, -1] = 1  # 周期境界条件
        diff_1d[-1, 0] = 1
        
        # 多次元への拡張
        result = torch.eye(1, device=self.device)
        for d in range(self.dim):
            if d == direction:
                result = torch.kron(result, diff_1d)
            else:
                result = torch.kron(result, torch.eye(self.N, device=self.device))
        
        return result
    
    def _construct_theta_correction_gpu(self, direction: int) -> torch.Tensor:
        """🚀 GPU上でθ-変形補正項の構築"""
        # 非可換性による補正項 [x_μ, p_ν] = iθ δ_μν の効果
        
        # 位置作用素
        x_op = self._construct_position_operator_gpu(direction)
        
        # 運動量作用素（微分）
        p_op = self._construct_momentum_operator_gpu(direction)
        
        # 交換子 [x, p] の離散版
        commutator = torch.mm(x_op, p_op) - torch.mm(p_op, x_op)
        
        return commutator
    
    def _construct_kappa_correction_gpu(self, direction: int) -> torch.Tensor:
        """🚀 GPU上でκ-変形補正項の構築"""
        # κ-ミンコフスキー変形による補正 x ⊕_κ y = x + y + κxy の効果
        
        x_op = self._construct_position_operator_gpu(direction)
        p_op = self._construct_momentum_operator_gpu(direction)
        
        # κ-変形による高次項
        correction = torch.mm(torch.mm(x_op, x_op), torch.mm(p_op, p_op))
        
        return correction
    
    def _construct_mixed_difference_gpu(self, dir1: int, dir2: int) -> torch.Tensor:
        """🚀 GPU上で混合偏微分作用素の構築"""
        # ∂²/(∂x_μ ∂x_ν) の離散版
        
        diff1 = self._construct_forward_difference_gpu(dir1) - self._construct_backward_difference_gpu(dir1)
        diff2 = self._construct_forward_difference_gpu(dir2) - self._construct_backward_difference_gpu(dir2)
        
        return torch.mm(diff1, diff2) / 4.0
    
    def _construct_position_operator_gpu(self, direction: int) -> torch.Tensor:
        """🚀 GPU上で位置作用素の構築"""
        # x_μ の離散版
        positions = torch.arange(self.N, dtype=torch.float32, device=self.device) - self.N // 2
        pos_1d = torch.diag(positions)
        
        # 多次元への拡張
        result = torch.eye(1, device=self.device)
        for d in range(self.dim):
            if d == direction:
                result = torch.kron(result, pos_1d)
            else:
                result = torch.kron(result, torch.eye(self.N, device=self.device))
        
        return result
    
    def _construct_momentum_operator_gpu(self, direction: int) -> torch.Tensor:
        """🚀 GPU上で運動量作用素の構築"""
        # p_μ = -i ∇_μ の離散版
        forward = self._construct_forward_difference_gpu(direction)
        backward = self._construct_backward_difference_gpu(direction)
        
        return -1j * (forward - backward) / 2.0
    
    def compute_spectral_dimension_gpu(self, operator: torch.Tensor, 
                                     n_eigenvalues: int = 100) -> Tuple[float, Dict]:
        """
        🚀 GPU上でスペクトル次元の高速計算
        
        d_s = -2 * d(log Z(t))/d(log t) |_{t→0}
        
        ここで、Z(t) = Tr(exp(-tD²)) はスペクトルゼータ関数
        """
        print("🔍 スペクトル次元計算中...")
        start_time = time.time()
        
        # 固有値の計算（GPU上）
        try:
            # エルミート化
            if operator.dtype.is_complex:
                operator_hermitian = torch.mm(operator.conj().T, operator)
            else:
                operator_hermitian = torch.mm(operator.T, operator)
            
            # 固有値計算（PyTorchのeigh使用）
            eigenvalues, _ = torch.linalg.eigh(operator_hermitian)
            eigenvalues = eigenvalues.real
            
            # 正の固有値のみを使用
            eigenvalues = eigenvalues[eigenvalues > 1e-12]
            eigenvalues = eigenvalues[:n_eigenvalues]  # 上位n_eigenvalues個
            
        except Exception as e:
            print(f"❌ 固有値計算エラー: {e}")
            return float('nan'), {}
        
        if len(eigenvalues) < 10:
            print("⚠️  警告: 有効な固有値が少なすぎます")
            return float('nan'), {}
        
        # スペクトルゼータ関数の計算（GPU上）
        t_values = torch.logspace(-3, 0, 50, device=self.device)
        zeta_values = []
        
        for t in t_values:
            zeta_t = torch.sum(torch.exp(-t * eigenvalues))
            zeta_values.append(zeta_t.item())
        
        zeta_values = torch.tensor(zeta_values, device=self.device)
        
        # 対数微分の計算
        log_t = torch.log(t_values)
        log_zeta = torch.log(zeta_values + 1e-12)  # 数値安定性のため
        
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
            'eigenvalues': eigenvalues.cpu().numpy(),
            'n_eigenvalues': len(eigenvalues),
            'min_eigenvalue': torch.min(eigenvalues).item(),
            'max_eigenvalue': torch.max(eigenvalues).item(),
            'spectral_gap': (eigenvalues[1] - eigenvalues[0]).item() if len(eigenvalues) > 1 else 0,
            'zeta_function': zeta_values.cpu().numpy(),
            't_values': t_values.cpu().numpy(),
            'slope': slope.item(),
            'computation_time': computation_time
        }
        
        return spectral_dimension, analysis_info

def demonstrate_gpu_dirac_laplacian_analysis():
    """🚀 RTX3080対応ディラック/ラプラシアン解析のデモンストレーション"""
    
    print("=" * 80)
    print("🚀 RTX3080対応 ディラック/ラプラシアン作用素の高速GPU解析")
    print("=" * 80)
    
    # GPU情報表示
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"🔧 CUDA Version: {torch.version.cuda}")
    else:
        print("⚠️  CUDA未対応 - CPUモードで実行")
    
    # パラメータ設定（RTX3080最適化）
    params = GPUOperatorParameters(
        dimension=4,
        lattice_size=16,  # RTX3080で安全な大きさ（16^4 = 65,536格子点）
        theta=0.01,
        kappa=0.05,
        mass=0.1,
        coupling=1.0,
        dtype=torch.complex64  # メモリ効率のため
    )
    
    analyzer = GPUDiracLaplacianAnalyzer(params)
    
    print(f"\n📊 解析パラメータ:")
    print(f"次元: {params.dimension}")
    print(f"格子サイズ: {params.lattice_size}")
    print(f"θ パラメータ: {params.theta}")
    print(f"κ パラメータ: {params.kappa}")
    print(f"質量: {params.mass}")
    print(f"データ型: {params.dtype}")
    
    # 1. ディラック作用素の解析
    print("\n🔨 1. ディラック作用素の構築と解析...")
    total_start = time.time()
    
    D = analyzer.construct_discrete_dirac_operator_gpu()
    print(f"📊 ディラック作用素サイズ: {D.shape}")
    print(f"💾 メモリ使用量: {D.element_size() * D.numel() / 1e9:.2f} GB")
    
    d_s_dirac, dirac_info = analyzer.compute_spectral_dimension_gpu(D)
    print(f"📈 ディラック作用素のスペクトル次元: {d_s_dirac:.6f}")
    print(f"🎯 理論値との差: {abs(d_s_dirac - params.dimension):.6f}")
    
    # 2. ラプラシアン作用素の解析
    print("\n🔨 2. ラプラシアン作用素の構築と解析...")
    Delta = analyzer.construct_discrete_laplacian_gpu()
    print(f"📊 ラプラシアン作用素サイズ: {Delta.shape}")
    
    d_s_laplacian, laplacian_info = analyzer.compute_spectral_dimension_gpu(Delta.to(analyzer.dtype))
    print(f"📈 ラプラシアン作用素のスペクトル次元: {d_s_laplacian:.6f}")
    print(f"🎯 理論値との差: {abs(d_s_laplacian - params.dimension):.6f}")
    
    total_time = time.time() - total_start
    print(f"\n⏱️  総計算時間: {total_time:.2f}秒")
    
    # 3. 結果の保存
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
            'dtype': str(params.dtype)
        },
        'results': {
            'dirac_spectral_dimension': d_s_dirac,
            'laplacian_spectral_dimension': d_s_laplacian,
            'total_computation_time': total_time,
            'dirac_computation_time': dirac_info.get('computation_time', 0),
            'laplacian_computation_time': laplacian_info.get('computation_time', 0)
        },
        'analysis_timestamp': str(torch.tensor(time.time()).item())
    }
    
    with open('gpu_dirac_laplacian_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False, default=str)
    
    print("\n💾 結果が 'gpu_dirac_laplacian_results.json' に保存されました。")
    print("🎉 RTX3080対応GPU解析完了！")
    
    return analyzer, results_summary

if __name__ == "__main__":
    # RTX3080対応解析のデモンストレーション
    analyzer, results = demonstrate_gpu_dirac_laplacian_analysis() 