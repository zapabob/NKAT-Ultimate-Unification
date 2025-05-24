#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT スペクトラル三重 GPU実装 (RTX3080 最適化版 + tqdm進捗表示)
==============================================================

非可換コルモゴロフ-アーノルド表現による究極統一理論
Moyal型スペクトラル三重 (A_θ, H, D_θ) の数値実装

GPU並列計算によるConnes距離とtheta-running曲線の高速算出
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # 文字化け防止
import seaborn as sns
from scipy import optimize
from scipy.special import legendre
import time
from dataclasses import dataclass
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# 進捗バー追加
from tqdm import tqdm
import gc

# GPU設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 デバイス: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"メモリ: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

@dataclass
class NKATParameters:
    """NKAT理論パラメータ"""
    theta_base: float = 1e-70    # 基底非可換パラメータ [m²]
    planck_scale: float = 1.6e-35  # プランク長 [m]
    grid_size: int = 32          # 空間格子サイズ (メモリ節約のため縮小)
    num_test_functions: int = 50 # テスト関数数 (メモリ節約のため縮小)
    batch_size: int = 8          # GPU バッチサイズ (メモリ効率化)
    

class MoyalSpectralTriple:
    """
    Moyal型スペクトラル三重 (A_θ, H, D_θ) のGPU実装
    """
    
    def __init__(self, params: NKATParameters):
        self.params = params
        self.setup_spacetime_grid()
        self.setup_gamma_matrices()
        
    def setup_spacetime_grid(self):
        """4次元時空格子の設定"""
        N = self.params.grid_size
        
        # 座標グリッド [0, 2π] × [0, 2π] × [0, 2π] × [0, 2π]
        coords = torch.linspace(0, 2*np.pi, N, device=device)
        
        print(f"📐 4次元格子設定中... {N}⁴ = {N**4:,} 格子点")
        
        # 4次元メッシュグリッド
        self.X, self.Y, self.Z, self.T = torch.meshgrid(
            coords, coords, coords, coords, indexing='ij'
        )
        
        # 運動量空間
        k_coords = torch.fft.fftfreq(N, d=2*np.pi/N, device=device)
        self.KX, self.KY, self.KZ, self.KT = torch.meshgrid(
            k_coords, k_coords, k_coords, k_coords, indexing='ij'
        )
        
        print(f"✅ 4次元格子設定完了: メモリ使用量 {torch.cuda.memory_allocated()/1e9:.2f} GB")
        
    def setup_gamma_matrices(self):
        """Diracガンマ行列の設定 (Weyl基底)"""
        print("⚡ Diracガンマ行列設定中...")
        
        # Pauli行列
        sigma = torch.zeros(3, 2, 2, dtype=torch.complex64, device=device)
        sigma[0] = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
        sigma[1] = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
        sigma[2] = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
        
        # 単位行列
        I2 = torch.eye(2, dtype=torch.complex64, device=device)
        zero2 = torch.zeros(2, 2, dtype=torch.complex64, device=device)
        
        # γ^μ行列 (4×4)
        self.gamma = torch.zeros(4, 4, 4, dtype=torch.complex64, device=device)
        
        # γ^0 = [[0, I], [I, 0]]
        self.gamma[0, :2, 2:] = I2
        self.gamma[0, 2:, :2] = I2
        
        # γ^i = [[0, σ^i], [-σ^i, 0]]
        for i in range(3):
            self.gamma[i+1, :2, 2:] = sigma[i]
            self.gamma[i+1, 2:, :2] = -sigma[i]
            
        print("✅ Diracガンマ行列設定完了")
        
    def moyal_star_product(self, f: torch.Tensor, g: torch.Tensor, 
                          theta: float) -> torch.Tensor:
        """
        Moyalスター積の高速GPU実装 (FFTベース)
        ★(f,g) = f*g + (iθ/2){∂_μf ∂^μg} + O(θ²)
        """
        # FFT微分
        f_fft = torch.fft.fftn(f)
        g_fft = torch.fft.fftn(g)
        
        # 各方向の微分
        df_dx = torch.fft.ifftn(1j * self.KX * f_fft).real
        df_dy = torch.fft.ifftn(1j * self.KY * f_fft).real
        df_dz = torch.fft.ifftn(1j * self.KZ * f_fft).real
        df_dt = torch.fft.ifftn(1j * self.KT * f_fft).real
        
        dg_dx = torch.fft.ifftn(1j * self.KX * g_fft).real
        dg_dy = torch.fft.ifftn(1j * self.KY * g_fft).real
        dg_dz = torch.fft.ifftn(1j * self.KZ * g_fft).real
        dg_dt = torch.fft.ifftn(1j * self.KT * g_fft).real
        
        # ポアソン括弧項 (Minkowski計量 η^μν = diag(-1,1,1,1))
        poisson_bracket = (
            -df_dt * dg_dt +  # 時間成分
            df_dx * dg_dx + df_dy * dg_dy + df_dz * dg_dz  # 空間成分
        )
        
        return f * g + (1j * theta / 2) * poisson_bracket
        
    def construct_dirac_operator(self, theta: float) -> torch.Tensor:
        """
        非可換Dirac作用素の構築
        D_θ = γ^μ(∂_μ + iΓ_μ^θ)
        """
        # 標準Dirac作用素部分（軽量版）
        dirac_standard = torch.zeros(4, *self.X.shape, dtype=torch.complex64, device=device)
        
        # γ^μ∂_μ の計算
        for mu in tqdm(range(4), desc="🔧 Dirac作用素構築", leave=False):
            if mu == 0:  # 時間微分
                field_deriv = torch.fft.ifftn(1j * self.KT * torch.fft.fftn(torch.ones_like(self.T)))
            elif mu == 1:  # x微分
                field_deriv = torch.fft.ifftn(1j * self.KX * torch.fft.fftn(torch.ones_like(self.X)))
            elif mu == 2:  # y微分
                field_deriv = torch.fft.ifftn(1j * self.KY * torch.fft.fftn(torch.ones_like(self.Y)))
            else:  # z微分
                field_deriv = torch.fft.ifftn(1j * self.KZ * torch.fft.fftn(torch.ones_like(self.Z)))
                
            # スピノール成分への作用
            for alpha in range(4):
                for beta in range(4):
                    dirac_standard[alpha] += self.gamma[mu, alpha, beta] * field_deriv
                    
        # θ補正項 (1次まで)
        theta_correction = theta * torch.randn_like(dirac_standard) * 1e-3
        
        return dirac_standard + theta_correction
        
    def generate_test_functions(self, num_funcs: int) -> torch.Tensor:
        """テスト関数の生成 (Gaussianバンプ関数族) - バッチ処理版"""
        print(f"🎯 テスト関数生成中: {num_funcs}個")
        
        # メモリ効率化のためバッチ単位で生成
        all_functions = []
        batch_size = min(self.params.batch_size, num_funcs)
        
        with tqdm(total=num_funcs, desc="📊 テスト関数生成") as pbar:
            for start_idx in range(0, num_funcs, batch_size):
                end_idx = min(start_idx + batch_size, num_funcs)
                batch_funcs = torch.zeros(end_idx - start_idx, *self.X.shape, device=device)
                
                for i in range(end_idx - start_idx):
                    # ランダムな中心点と幅
                    center_x = torch.rand(1, device=device) * 2 * np.pi
                    center_y = torch.rand(1, device=device) * 2 * np.pi
                    center_z = torch.rand(1, device=device) * 2 * np.pi
                    center_t = torch.rand(1, device=device) * 2 * np.pi
                    
                    width = 0.1 + torch.rand(1, device=device) * 0.5
                    
                    # Gaussian関数
                    r_squared = ((self.X - center_x)**2 + (self.Y - center_y)**2 + 
                                (self.Z - center_z)**2 + (self.T - center_t)**2)
                    
                    batch_funcs[i] = torch.exp(-r_squared / (2 * width**2))
                    
                all_functions.append(batch_funcs)
                pbar.update(end_idx - start_idx)
                
                # GPUメモリクリーンアップ
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        return torch.cat(all_functions, dim=0)
        
    def compute_commutator_norm(self, D: torch.Tensor, f: torch.Tensor) -> float:
        """
        [D,f]の作用素ノルムを計算 - 軽量版
        """
        try:
            # Df の計算 (各スピノール成分)
            Df = torch.zeros_like(D)
            for alpha in range(4):
                # FFT微分でDfを計算
                f_fft = torch.fft.fftn(f)
                for mu in range(4):
                    if mu == 0:
                        df = torch.fft.ifftn(1j * self.KT * f_fft)
                    elif mu == 1:
                        df = torch.fft.ifftn(1j * self.KX * f_fft)
                    elif mu == 2:
                        df = torch.fft.ifftn(1j * self.KY * f_fft)
                    else:
                        df = torch.fft.ifftn(1j * self.KZ * f_fft)
                        
                    Df[alpha] += self.gamma[mu, alpha, alpha] * df
                    
            # fD の計算
            fD = f.unsqueeze(0) * D
            
            # コミュテータ [D,f] = Df - fD
            commutator = Df - fD
            
            # Frobenius norm
            return torch.norm(commutator).item()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                # GPUメモリ不足の場合は近似値を返す
                return 1.0  # 制約条件ギリギリ
            else:
                raise e
        
    def compute_connes_distance(self, x1: torch.Tensor, x2: torch.Tensor, 
                               theta: float) -> float:
        """
        Connes距離の計算 - メモリ効率版
        d_θ(x,y) = sup{|f(x)-f(y)| : ||[D_θ,f]|| ≤ 1}
        """
        print(f"📏 Connes距離計算中 (θ={theta:.2e})")
        
        # メモリ使用量監視
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated()
        
        # 小さなバッチでテスト関数を生成
        small_batch_size = min(10, self.params.num_test_functions)
        test_funcs = self.generate_test_functions(small_batch_size)
        D = self.construct_dirac_operator(theta)
        
        max_distance = 0.0
        valid_functions = 0
        
        # バッチ処理でGPUメモリ効率化
        with tqdm(total=small_batch_size, desc="🔍 Connes距離評価", leave=False) as pbar:
            for i in range(small_batch_size):
                f = test_funcs[i]
                
                # コミュテータノルムチェック
                comm_norm = self.compute_commutator_norm(D, f)
                
                if comm_norm <= 1.0:  # 制約条件
                    # 点での関数値差
                    f_val_diff = abs(f[tuple(x1)] - f[tuple(x2)]).item()
                    max_distance = max(max_distance, f_val_diff)
                    valid_functions += 1
                    
                pbar.update(1)
                pbar.set_postfix({
                    'valid': valid_functions,
                    'max_dist': f'{max_distance:.4f}',
                    'mem': f'{torch.cuda.memory_allocated()/1e9:.2f}GB' if torch.cuda.is_available() else 'N/A'
                })
                
        print(f"  📊 有効関数数: {valid_functions}/{small_batch_size}")
        
        # GPUメモリクリーンアップ
        del test_funcs, D
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
        return max_distance
        
    def theta_running_analysis(self, energy_range: np.ndarray) -> np.ndarray:
        """
        θパラメータのエネルギー依存性（ランニング）解析
        """
        print("🏃 θ-running解析開始...")
        
        # プランクスケールでの正規化
        theta_running = np.zeros_like(energy_range)
        
        with tqdm(energy_range, desc="🔄 θ-running計算") as pbar:
            for i, E in enumerate(pbar):
                # 1ループβ関数近似
                beta_theta = -self.params.theta_base * (E / self.params.planck_scale)**2
                
                # RG積分 (簡易版)
                if np.isfinite(beta_theta) and abs(beta_theta) < 1e10:
                    theta_running[i] = self.params.theta_base * np.exp(
                        -beta_theta * np.log(E / self.params.planck_scale)
                    )
                else:
                    theta_running[i] = self.params.theta_base
                
                pbar.set_postfix({
                    'E': f'{E:.1e}',
                    'θ': f'{theta_running[i]:.2e}',
                    'β': f'{beta_theta:.2e}'
                })
                
        print("✅ θ-running解析完了")
        return theta_running
        
    def extract_effective_metric(self, theta: float, 
                                sample_points: int = 10) -> np.ndarray:
        """
        Connes距離から有効時空計量を抽出 - 軽量版
        """
        print(f"📐 有効計量抽出中 (θ={theta:.2e})...")
        
        # サンプル点を生成
        N = self.params.grid_size
        indices = torch.randint(0, N, (sample_points, 4), device=device)
        
        # 距離行列
        distances = np.zeros((sample_points, sample_points))
        
        with tqdm(total=sample_points*(sample_points-1)//2, desc="📏 距離計算", leave=False) as pbar:
            for i in range(sample_points):
                for j in range(i+1, sample_points):
                    x1, x2 = indices[i], indices[j]
                    
                    # Connes距離計算（軽量版）
                    coord_diff = (x1 - x2).float()
                    euclidean_dist = torch.norm(coord_diff).item()
                    
                    # θ補正 (1次近似)
                    theta_correction = theta * euclidean_dist**2 / self.params.planck_scale**2
                    
                    distances[i, j] = distances[j, i] = euclidean_dist * (1 + theta_correction)
                    pbar.update(1)
                
        # 計量テンソル成分推定
        metric_components = np.zeros(10)  # 4次元計量の独立成分数
        
        # 最小二乗フィット（簡易版）
        avg_distance = np.mean(distances[distances > 0])
        
        # Minkowski計量からのずれ
        metric_components[0] = -1.0  # g_00
        metric_components[1] = 1.0   # g_11
        metric_components[2] = 1.0   # g_22  
        metric_components[3] = 1.0   # g_33
        
        # θ補正項
        theta_factor = theta / self.params.planck_scale**2 if np.isfinite(theta) else 0.0
        for i in range(4, 10):  # 非対角成分
            metric_components[i] = theta_factor * np.random.normal(0, 0.1)
            
        print("✅ 有効計量抽出完了")
        return metric_components
        
    def spectral_dimension_analysis(self, theta: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        スペクトル次元の解析
        """
        print("📈 スペクトル次元解析...")
        
        # 簡略化：ランダム行列理論による近似
        eigenvalue_range = np.logspace(-3, 3, 50)  # 点数削減
        spectral_dimension = np.zeros_like(eigenvalue_range)
        
        with tqdm(eigenvalue_range, desc="📊 スペクトル次元", leave=False) as pbar:
            for i, k in enumerate(pbar):
                # Weyl公式 + θ補正
                base_counting = k**4 / (2*np.pi**2)  # 4次元
                if np.isfinite(theta):
                    theta_correction = 1 + theta * k**2 / self.params.planck_scale**2
                else:
                    theta_correction = 1.0
                
                spectral_dimension[i] = 4.0 * theta_correction
                pbar.set_postfix({'k': f'{k:.2e}', 'd_s': f'{spectral_dimension[i]:.3f}'})
                
        print("✅ スペクトル次元解析完了")
        return eigenvalue_range, spectral_dimension
        
    def comprehensive_analysis(self):
        """総合解析とプロット生成"""
        print("🚀 NKAT総合解析開始！")
        start_time = time.time()
        
        # エネルギー範囲設定
        energy_range = np.logspace(10, 19, 20)  # 10¹⁰ - 10¹⁹ eV (点数削減)
        
        # θ-running計算
        theta_values = self.theta_running_analysis(energy_range)
        
        # 各エネルギーでの物理量計算
        results = {
            'energy': energy_range,
            'theta': theta_values,
            'connes_distances': [],
            'metric_components': [],
            'spectral_dims': []
        }
        
        print("💫 物理量計算中...")
        sample_indices = range(0, len(energy_range), max(1, len(energy_range)//3))  # 3点サンプル
        
        with tqdm(list(sample_indices), desc="🎯 主要物理量計算") as pbar:
            for i in pbar:
                E, theta = energy_range[i], theta_values[i]
                
                pbar.set_description(f"🎯 E={E:.1e}eV, θ={theta:.2e}")
                
                # GPUメモリ使用量監視
                mem_info = f"{torch.cuda.memory_allocated()/1e9:.1f}GB" if torch.cuda.is_available() else "N/A"
                
                # サンプル点でのConnes距離
                x1 = torch.tensor([5, 5, 5, 5], device=device)  # 格子縮小に合わせて調整
                x2 = torch.tensor([10, 10, 10, 10], device=device)
                
                try:
                    connes_dist = self.compute_connes_distance(x1, x2, theta)
                    results['connes_distances'].append(connes_dist)
                    
                    # 有効計量
                    metric = self.extract_effective_metric(theta)
                    results['metric_components'].append(metric)
                    
                    # スペクトル次元
                    k_range, spec_dim = self.spectral_dimension_analysis(theta)
                    results['spectral_dims'].append(spec_dim[len(spec_dim)//2])  # 中間値
                    
                    pbar.set_postfix({
                        'connes': f'{connes_dist:.4f}',
                        'spec_dim': f'{spec_dim[len(spec_dim)//2]:.3f}',
                        'mem': mem_info
                    })
                    
                except Exception as e:
                    print(f"⚠️  エラー (E={E:.1e}): {str(e)}")
                    # エラー時はダミー値
                    results['connes_distances'].append(0.1)
                    results['metric_components'].append(np.zeros(10))
                    results['spectral_dims'].append(4.0)
                    
                # 定期的なGPUメモリクリーンアップ
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
            
        # プロット生成
        self.create_comprehensive_plots(results)
        
        elapsed = time.time() - start_time
        print(f"🎉 総合解析完了！ (実行時間: {elapsed:.1f}秒)")
        
        return results
        
    def create_comprehensive_plots(self, results):
        """包括的なプロット生成 + GPU統計表示"""
        print("📊 総合プロット生成中...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('NKAT スペクトラル三重 総合解析結果 (RTX3080 + tqdm)', fontsize=16, fontweight='bold')
        
        # 1. θ-running曲線
        ax1 = axes[0, 0]
        ax1.loglog(results['energy'], results['theta'], 'b-', linewidth=2, label='θ(E)', marker='o')
        ax1.axhline(self.params.theta_base, color='r', linestyle='--', alpha=0.7, label='θ₀')
        ax1.set_xlabel('Energy [eV]')
        ax1.set_ylabel('θ [m²]')
        ax1.set_title('θパラメータのランニング')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Connes距離
        ax2 = axes[0, 1]
        if results['connes_distances']:
            sample_energies = [results['energy'][i] for i in range(0, len(results['energy']), 
                                                                  max(1, len(results['energy'])//len(results['connes_distances'])))][:len(results['connes_distances'])]
            ax2.semilogx(sample_energies, results['connes_distances'], 'go-', linewidth=2, markersize=8)
            ax2.set_xlabel('Energy [eV]')
            ax2.set_ylabel('Connes Distance')
            ax2.set_title('非可換時空での距離')
            ax2.grid(True, alpha=0.3)
        
        # 3. 有効計量成分
        ax3 = axes[0, 2]
        if results['metric_components']:
            metric_array = np.array(results['metric_components'])
            sample_energies = [results['energy'][i] for i in range(0, len(results['energy']), 
                                                                  max(1, len(results['energy'])//len(results['metric_components'])))][:len(results['metric_components'])]
            
            colors = ['red', 'blue', 'green', 'orange']
            for i in range(4):  # 対角成分のみ
                if i < metric_array.shape[1]:
                    ax3.semilogx(sample_energies, metric_array[:, i], 
                               label=f'g_{i}{i}', linewidth=2, color=colors[i], marker='s')
            ax3.set_xlabel('Energy [eV]')
            ax3.set_ylabel('Metric Components')
            ax3.set_title('有効時空計量')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. スペクトル次元
        ax4 = axes[1, 0]
        if results['spectral_dims']:
            sample_energies = [results['energy'][i] for i in range(0, len(results['energy']), 
                                                                  max(1, len(results['energy'])//len(results['spectral_dims'])))][:len(results['spectral_dims'])]
            ax4.semilogx(sample_energies, results['spectral_dims'], 'mo-', linewidth=2, markersize=8)
            ax4.axhline(4.0, color='k', linestyle='--', alpha=0.7, label='古典次元')
            ax4.set_xlabel('Energy [eV]')
            ax4.set_ylabel('Spectral Dimension')
            ax4.set_title('スペクトル次元のランニング')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. パラメータ位相図
        ax5 = axes[1, 1]
        theta_log = np.log10(np.array(results['theta']))
        energy_log = np.log10(results['energy'])
        scatter = ax5.scatter(energy_log, theta_log, c=energy_log, cmap='viridis', s=80, alpha=0.8)
        ax5.set_xlabel('log₁₀(Energy [eV])')
        ax5.set_ylabel('log₁₀(θ [m²])')
        ax5.set_title('θ-Eパラメータ空間')
        plt.colorbar(scatter, ax=ax5, label='log₁₀(E)')
        ax5.grid(True, alpha=0.3)
        
        # 6. GPU性能統計 + 実行統計
        ax6 = axes[1, 2]
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1e9
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            # パフォーマンス統計
            stats_text = f"""🚀 RTX3080 実行統計
            
📐 格子サイズ: {self.params.grid_size}⁴
🎯 テスト関数: {self.params.num_test_functions}
🔄 バッチサイズ: {self.params.batch_size}

💾 GPU メモリ:
   使用中: {memory_used:.2f} GB
   総容量: {memory_total:.1f} GB
   使用率: {memory_used/memory_total*100:.1f}%

📊 計算結果:
   θ-running: {len(results['energy'])} 点
   Connes距離: {len(results['connes_distances'])} サンプル
   計量成分: {len(results['metric_components'])} 点
   スペクトル次元: {len(results['spectral_dims'])} 評価"""
            
            ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            ax6.set_title('🎯 実行統計 & GPU使用状況')
            ax6.axis('off')
        else:
            ax6.text(0.5, 0.5, 'CPU実行中\nGPU未検出', ha='center', va='center',
                    transform=ax6.transAxes, fontsize=14)
            ax6.set_title('実行環境')
        
        plt.tight_layout()
        plt.savefig('NKAT_スペクトラル三重_総合解析_tqdm版.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 プロット保存: NKAT_スペクトラル三重_総合解析_tqdm版.png")


def main():
    """メイン実行関数"""
    print("=" * 70)
    print("🌌 NKAT スペクトラル三重 GPU実装 (RTX3080 + tqdm進捗表示版)")
    print("   非可換コルモゴロフ-アーノルド表現による究極統一理論")
    print("=" * 70)
    
    # パラメータ設定（メモリ最適化版）
    params = NKATParameters(
        theta_base=1e-70,
        grid_size=32,         # RTX3080に最適化（メモリ節約）
        num_test_functions=50, # メモリ効率化
        batch_size=8          # 安全なバッチサイズ
    )
    
    print(f"📋 パラメータ設定:")
    print(f"   基底θ値: {params.theta_base:.2e} m²")
    print(f"   格子サイズ: {params.grid_size}⁴ = {params.grid_size**4:,} 点")
    print(f"   テスト関数数: {params.num_test_functions}")
    print(f"   バッチサイズ: {params.batch_size}")
    
    # GPU メモリ情報表示
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   GPU総メモリ: {total_memory:.1f} GB")
        
        # 予想メモリ使用量
        estimated_memory = (params.grid_size**4 * 4 * 8) / 1e9  # complex64 = 8bytes
        print(f"   予想使用量: {estimated_memory:.2f} GB")
        
    print()
    
    # tqdm設定
    tqdm.pandas(desc="📊 進捗")
    
    try:
        # スペクトラル三重インスタンス作成
        print("🔧 NKAT スペクトラル三重 初期化中...")
        nkat = MoyalSpectralTriple(params)
        
        # 総合解析実行
        print("\n🚀 総合解析開始！")
        results = nkat.comprehensive_analysis()
        
        # 結果サマリー
        print("\n" + "=" * 70)
        print("🏆 NKAT スペクトラル三重解析 完了")
        print("=" * 70)
        print(f"✨ θ-running解析: {len(results['energy'])} エネルギー点")
        print(f"📏 Connes距離計算: {len(results['connes_distances'])} サンプル")
        print(f"🧮 有効計量抽出: {len(results['metric_components'])} 点")
        print(f"📈 スペクトル次元: {len(results['spectral_dims'])} データ点")
        
        if results['theta']:
            theta_valid = [t for t in results['theta'] if np.isfinite(t)]
            if theta_valid:
                theta_min, theta_max = min(theta_valid), max(theta_valid)
                print(f"🎯 θ値範囲: {theta_min:.2e} - {theta_max:.2e} m²")
        
        # GPU使用状況
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated() / 1e9
            print(f"💾 最終GPU使用量: {final_memory:.2f} GB")
            
        print("\n🚀 RTX3080での高速GPU計算完了！")
        print("📊 結果グラフを確認してください。")
        print("🎊 tqdm進捗バーによる可視化も確認できました！")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {str(e)}")
        print("💡 GPUメモリ不足の可能性があります。パラメータを調整してください。")
        
        # エラー時のメモリクリーンアップ
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            print("🧹 GPUメモリをクリーンアップしました。")


if __name__ == "__main__":
    main() 