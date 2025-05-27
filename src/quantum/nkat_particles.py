# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'MS Gothic'

# 物理定数
Gev_to_eV = 1e9
Planck_mass = 1.22e19  # GeV
c = 3e8  # m/s
hbar = 6.582e-25  # GeV⋅s

class QuantumInformon(nn.Module):
    def __init__(self, dim: int = 4, mass: float = 1e15, spin: float = 1.5):
        super().__init__()
        self.dim = dim
        self.mass = mass  # GeV単位
        self.spin = spin
        
        # 量子状態
        self.state = nn.Parameter(torch.randn(dim))
        # 情報電荷
        self.info_charge = nn.Parameter(torch.tensor(1.0))
        # 量子数
        self.quantum_numbers = nn.Parameter(torch.tensor([0.5, -0.5, 1.5, -1.5]))
    
    def compute_information_capacity(self, energy: torch.Tensor) -> torch.Tensor:
        """量子ビット容量の計算（理論式に基づく）"""
        scaled_energy = energy / (self.mass * Gev_to_eV)
        return torch.log2(1 + scaled_energy)
    
    def compute_transmission_velocity(self, energy: torch.Tensor) -> torch.Tensor:
        """情報伝達速度の計算（理論式に基づく）"""
        # v_I = c(1 - m_I^2 c^4/E^2)^(1/2)
        if energy > self.mass:
            return c * torch.sqrt(1 - (self.mass / energy)**2)
        return torch.tensor(0.0)
    
    def interact_with_field(self, field_strength: torch.Tensor) -> torch.Tensor:
        """場との相互作用（理論式に基づく）"""
        # 非可換相互作用
        return self.info_charge * torch.einsum('i,i', self.state, field_strength)

class NoncommutativeGaugeBoson(nn.Module):
    def __init__(self, dim: int = 4, mass: float = 1e18, spin: float = 2.0):
        super().__init__()
        self.dim = dim
        self.mass = mass  # GeV単位
        self.spin = spin
        
        # 場の強さテンソル
        self.field_strength = nn.Parameter(torch.randn(dim, dim))
        # 結合定数
        self.coupling = nn.Parameter(torch.tensor(0.1))
    
    def compute_decay_rate(self, mode: str = 'photon') -> torch.Tensor:
        """崩壊率の計算（理論式に基づく）"""
        if mode == 'photon':
            # Γ ≈ g_NQG^2 m_NQG^3 / M_Pl^2
            return self.coupling**2 * (self.mass / Planck_mass)**3
        elif mode == 'electron':
            # Γ ≈ g_NQG^2 m_NQG / (16π)
            return self.coupling**2 * self.mass / (16 * np.pi)
        elif mode == 'amaterasu':
            # アマテラス粒子への崩壊率
            if self.mass > 1e17:  # アマテラス粒子の質量より大きい場合
                return self.coupling**2 * (self.mass/1e17)**2 * torch.log(torch.tensor(self.mass/1e17))
            else:
                return torch.tensor(0.0)  # エネルギー的に不可能
        else:
            raise ValueError(f"Unknown decay mode: {mode}")
    
    def interact_with_matter(self, matter_tensor: torch.Tensor) -> torch.Tensor:
        """物質場との相互作用（理論式に基づく）"""
        # 非可換ゲージ相互作用
        return self.coupling * torch.einsum('ij,jk->ik', self.field_strength, matter_tensor)

    def nqg_to_amaterasu_branching_ratio(self, energy: torch.Tensor) -> torch.Tensor:
        """NQG粒子からアマテラス粒子への崩壊分岐比"""
        # 最小エネルギーしきい値（アマテラス粒子の質量）
        threshold = torch.tensor(1e17)
        
        if energy < threshold:
            return torch.tensor(0.0)  # エネルギー不足
        
        # エネルギー依存の分岐比（最大50%まで）
        ratio = 0.5 * (1.0 - torch.exp(-(energy - threshold)/1e18))
        return torch.clamp(ratio, 0.0, 0.5)

class AmaterasuParticle(nn.Module):
    """アマテラス粒子クラス - 光と情報を操作する高次元粒子"""
    def __init__(self, dim: int = 5, mass: float = 1e17, spin: float = 2.5):
        super().__init__()
        self.dim = dim
        self.mass = mass  # GeV単位
        self.spin = spin
        
        # 特殊場テンソル - 5次元構造
        self.field_tensor = nn.Parameter(torch.randn(dim, dim, dim))
        # 光学的結合定数
        self.optical_coupling = nn.Parameter(torch.tensor(0.2))
        # 情報結合定数
        self.info_coupling = nn.Parameter(torch.tensor(0.3))
        # 高次元量子数
        self.quantum_signature = nn.Parameter(torch.tensor([1.0, -1.0, 0.5, -0.5, 0.0]))
        # 相互作用用の変換行列を追加
        self.dim_converter = nn.Parameter(torch.randn(dim, 4))  # 5次元から4次元への変換
    
    def compute_light_bending(self, energy: torch.Tensor) -> torch.Tensor:
        """光の屈折度の計算"""
        try:
            refraction = 1.0 + (self.optical_coupling * energy**2) / (self.mass**2)
            # 無限大チェックを追加
            if torch.isinf(refraction) or torch.isnan(refraction):
                return torch.tensor(1e30)  # 非常に大きな値で代用
            if torch.isinf(refraction) or refraction > 1e20:
                # 無限大の屈折率は完全遮蔽に近い
                refraction_factor = 0.99
            else:
                # 有限の屈折率では屈折率に応じて遮蔽係数を計算
                refraction_factor = 1.0 - 1.0/torch.clamp(refraction, min=1.0001)
                refraction_factor = min(refraction_factor.item(), 0.98)  # 上限を設定
            return torch.tensor(refraction_factor)  # Tensorとして返す
        except Exception as e:
            print(f"光屈折度計算エラー: {e}")
            return torch.tensor(1e30)  # エラー時も大きな値を返す
    
    def compute_information_amplification(self, info_density: torch.Tensor) -> torch.Tensor:
        """情報増幅率の計算"""
        # A_info = β_A * log(1 + ρ_info/ρ_0)
        base_density = torch.tensor(1e10)  # 基準情報密度
        return self.info_coupling * torch.log(1 + info_density/base_density)
    
    def interact_with_nqg(self, nqg_boson: NoncommutativeGaugeBoson) -> torch.Tensor:
        """NQG粒子との相互作用"""
        # 次元を変換してから相互作用
        converted_tensor = torch.einsum('ijk,kl->ijl', self.field_tensor, self.dim_converter)
        interaction_strength = torch.einsum('ijl,lm->ijm', converted_tensor, nqg_boson.field_strength)
        return self.optical_coupling * nqg_boson.coupling * torch.norm(interaction_strength)

    def compute_electromagnetic_shielding(self, field_strength: torch.Tensor, distance: torch.Tensor, radiation_type: str = 'electromagnetic') -> torch.Tensor:
        """電磁遮蔽効果の計算
        
        Args:
            field_strength: 電磁場の強さ
            distance: アマテラス場の厚み
            radiation_type: 放射線の種類 ('electromagnetic', 'gamma', 'cosmic', 'neutron')
            
        Returns:
            shield_factor: 遮蔽係数（0: 完全透過、1: 完全遮蔽）
        """
        # 放射線タイプによるエネルギー係数
        radiation_energy_factor = {
            'electromagnetic': 1.0,      # 標準電磁波
            'gamma': 5.0,                # ガンマ線
            'cosmic': 10.0,              # 宇宙線
            'neutron': 2.0,              # 中性子線
            'x-ray': 3.0                 # X線
        }.get(radiation_type, 1.0)
        
        # 屈折度に基づく遮蔽係数（屈折度が高いほど遮蔽も強い）
        energy = torch.norm(field_strength) * 1e18 * radiation_energy_factor  # 電場強度からエネルギーを概算
        refraction = self.compute_light_bending(energy)
        
        # 無限大チェックを追加
        if torch.isinf(refraction) or refraction > 1e20:
            # 無限大の屈折率は完全遮蔽に近い
            refraction_factor = 0.99
        else:
            # 有限の屈折率では屈折率に応じて遮蔽係数を計算
            refraction_factor = 1.0 - 1.0/torch.clamp(refraction, min=1.0001)
            refraction_factor = min(refraction_factor.item(), 0.98)  # 上限を設定
        
        # 位相制御による干渉性遮蔽
        phase_efficiency = {
            'electromagnetic': 1.0,      # 電磁波に対して最も効果的
            'gamma': 0.7,                # ガンマ線に対してやや効果的
            'cosmic': 0.4,               # 宇宙線に対しては効果が限定的
            'neutron': 0.2,              # 中性子線に対してはあまり効果がない
            'x-ray': 0.8                 # X線に対して効果的
        }.get(radiation_type, 1.0)
        
        phase_control = torch.cos(torch.einsum('ijk,j->ik', self.field_tensor[:3,:3,:3], 
                                               field_strength[:3])**2) ** 2
        phase_control = torch.mean(phase_control) * phase_efficiency
        
        # 情報エントロピー変換による吸収（距離依存）
        entropy_efficiency = {
            'electromagnetic': 1.0,      # 電磁波に対して最も効果的
            'gamma': 0.8,                # ガンマ線に対して効果的
            'cosmic': 0.6,               # 宇宙線に対してもかなり効果的
            'neutron': 0.4,              # 中性子線に対しても一定の効果
            'x-ray': 0.9                 # X線に対して非常に効果的
        }.get(radiation_type, 1.0)
        
        info_absorption = (1.0 - torch.exp(-self.info_coupling * distance / 1e-15)) * entropy_efficiency
        
        # 屈折度、位相制御、情報吸収の組み合わせ
        shield_factor = refraction_factor * 0.6 + phase_control * 0.2 + info_absorption * 0.2
        
        return torch.clamp(torch.tensor(shield_factor), 0.0, 1.0)  # 0～1の範囲に正規化

class ParticleSimulation:
    def __init__(self, dim: int = 4):
        self.informon = QuantumInformon(dim)
        self.nqg = NoncommutativeGaugeBoson(dim)
        self.history = {
            'energy': [],  # エネルギー値も保存
            'info_capacity': [],
            'velocity': [],
            'decay_rate': [],
            'interaction': [],
            'nqg_to_amaterasu_ratio': []  # NQGからアマテラスへの崩壊分岐比を追加
        }
        # アマテラス粒子の追加
        self.amaterasu = AmaterasuParticle(dim=dim+1)
        # アマテラス関連の履歴項目を追加
        self.history['light_bending'] = []
        self.history['info_amplification'] = []
        self.history['amaterasu_nqg_interaction'] = []
        # 電磁遮蔽関連の履歴項目を追加
        self.history['em_shielding'] = []
        self.history['gamma_shielding'] = []
        self.history['cosmic_shielding'] = []
        self.history['neutron_shielding'] = []
        self.history['xray_shielding'] = []
    
    def run_simulation(self, steps: int = 1000, energy_range: torch.Tensor = None, 
                      field_strength: torch.Tensor = None, distance: torch.Tensor = None) -> None:
        """粒子シミュレーションの実行"""
        if energy_range is None:
            energy_range = torch.logspace(15, 20, steps)  # 10^15 to 10^20 GeV
        
        if field_strength is None:
            # 標準的な電磁場の強さを設定（単位：V/m）
            field_strength = torch.tensor([1.0, 0.0, 0.0, 0.0]) * 1e-9  # 弱い電場
        
        if distance is None:
            # アマテラス場の標準的な厚みを設定（単位：m）
            distance = torch.tensor(1e-3)  # 1 mm
        
        for energy in energy_range:
            # エネルギー値を保存
            self.history['energy'].append(energy.item())
            
            # 情報子の計算
            capacity = self.informon.compute_information_capacity(energy)
            velocity = self.informon.compute_transmission_velocity(energy)
            self.history['info_capacity'].append(capacity.item())
            self.history['velocity'].append(velocity.item())
            
            # NQGボソンの計算
            decay_rate = self.nqg.compute_decay_rate('photon')
            self.history['decay_rate'].append(decay_rate.item())
            
            # NQGからアマテラスへの崩壊分岐比
            nqg_to_amaterasu = self.nqg.nqg_to_amaterasu_branching_ratio(energy)
            self.history['nqg_to_amaterasu_ratio'].append(nqg_to_amaterasu.item())
            
            # 相互作用の計算
            interaction = self.informon.interact_with_field(self.nqg.field_strength[0])
            self.history['interaction'].append(interaction.item())
            
            # アマテラス粒子の計算
            light_bending = self.amaterasu.compute_light_bending(energy)
            self.history['light_bending'].append(light_bending.item())
            
            info_density = torch.tensor(1e12 + energy.item() * 1e-5)  # エネルギーに比例した情報密度
            info_amplification = self.amaterasu.compute_information_amplification(info_density)
            self.history['info_amplification'].append(info_amplification.item())
            
            # アマテラス粒子とNQGボソンの相互作用
            amaterasu_nqg_interaction = self.amaterasu.interact_with_nqg(self.nqg)
            self.history['amaterasu_nqg_interaction'].append(amaterasu_nqg_interaction.item())
            
            # エネルギーに応じて電磁場の強さを調整
            adjusted_field = field_strength * (energy / 1e15)
            
            # 様々な種類の放射線に対する遮蔽効果を計算
            em_shield = self.amaterasu.compute_electromagnetic_shielding(
                adjusted_field, distance, 'electromagnetic')
            gamma_shield = self.amaterasu.compute_electromagnetic_shielding(
                adjusted_field, distance, 'gamma')
            cosmic_shield = self.amaterasu.compute_electromagnetic_shielding(
                adjusted_field, distance, 'cosmic')
            neutron_shield = self.amaterasu.compute_electromagnetic_shielding(
                adjusted_field, distance, 'neutron')
            xray_shield = self.amaterasu.compute_electromagnetic_shielding(
                adjusted_field, distance, 'x-ray')
            
            # 遮蔽効果の履歴を記録
            self.history['em_shielding'].append(em_shield.item())
            self.history['gamma_shielding'].append(gamma_shield.item())
            self.history['cosmic_shielding'].append(cosmic_shield.item())
            self.history['neutron_shielding'].append(neutron_shield.item())
            self.history['xray_shielding'].append(xray_shield.item())
            
            if len(self.history['info_capacity']) % 100 == 0:
                print(f"エネルギー: {energy:.2e} GeV")
                print(f"情報容量: {capacity:.2f} qubits")
                print(f"伝達速度: {velocity/3e8:.4f}c")
                print(f"崩壊率: {decay_rate:.2e}")
                print(f"NQG→アマテラス分岐比: {nqg_to_amaterasu:.4f}")
                print(f"相互作用強度: {interaction:.2e}")
                print(f"光屈折度: {light_bending:.4f}")
                print(f"情報増幅率: {info_amplification:.4f}")
                print(f"アマテラス-NQG相互作用: {amaterasu_nqg_interaction:.2e}")
                print(f"電磁波遮蔽効率: {em_shield:.4f} (99%=ほぼ完全遮蔽)")
                print(f"ガンマ線遮蔽効率: {gamma_shield:.4f}")
                print(f"宇宙線遮蔽効率: {cosmic_shield:.4f}")
                print(f"中性子線遮蔽効率: {neutron_shield:.4f}")
                print(f"X線遮蔽効率: {xray_shield:.4f}\n")
    
    def find_threshold_energy(self) -> float:
        """光屈折度が無限大に発散するエネルギー閾値を特定"""
        threshold = 0.0
        for i in range(len(self.history['energy'])):
            energy = self.history['energy'][i]
            bending = self.history['light_bending'][i]
            
            if torch.isinf(torch.tensor(bending)) or bending > 1e20:
                threshold = energy
                break
                
        return threshold
    
    def plot_results(self) -> None:
        """結果のプロット"""
        # 基本的な粒子特性のプロット
        fig1 = plt.figure(figsize=(15, 12))
        
        # 情報容量
        ax1 = fig1.add_subplot(321)
        ax1.plot(self.history['energy'], self.history['info_capacity'])
        ax1.set_title('量子情報容量の変化')
        ax1.set_xlabel('エネルギー (GeV)')
        ax1.set_ylabel('容量 (qubits)')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.grid(True)
        
        # 伝達速度
        ax2 = fig1.add_subplot(322)
        ax2.plot(self.history['energy'], [v/3e8 for v in self.history['velocity']])
        ax2.set_title('情報伝達速度 (光速比)')
        ax2.set_xlabel('エネルギー (GeV)')
        ax2.set_ylabel('速度 (c)')
        ax2.set_xscale('log')
        ax2.grid(True)
        
        # 崩壊率とNQG→アマテラス分岐比
        ax3 = fig1.add_subplot(323)
        ax3.plot(self.history['energy'], self.history['decay_rate'], 'b-', label='光子崩壊率')
        ax3.set_xlabel('エネルギー (GeV)')
        ax3.set_ylabel('崩壊率', color='b')
        ax3.tick_params(axis='y', labelcolor='b')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        
        ax3_twin = ax3.twinx()
        ax3_twin.plot(self.history['energy'], self.history['nqg_to_amaterasu_ratio'], 'r-', label='アマテラス分岐比')
        ax3_twin.set_ylabel('アマテラス分岐比', color='r')
        ax3_twin.tick_params(axis='y', labelcolor='r')
        
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        ax3.grid(True)
        
        # 相互作用
        ax4 = fig1.add_subplot(324)
        ax4.plot(self.history['energy'], self.history['interaction'])
        ax4.set_title('粒子間相互作用強度')
        ax4.set_xlabel('エネルギー (GeV)')
        ax4.set_ylabel('相互作用強度')
        ax4.set_xscale('log')
        if min(self.history['interaction']) > 0:
            ax4.set_yscale('log')
        ax4.grid(True)
            
        # アマテラス光屈折度
        ax5 = fig1.add_subplot(325)
        # 無限大値を除外して描画
        valid_indices = [i for i, v in enumerate(self.history['light_bending']) if not np.isinf(v) and v < 1e20]
        valid_energies = [self.history['energy'][i] for i in valid_indices]
        valid_bending = [self.history['light_bending'][i] for i in valid_indices]
        
        ax5.plot(valid_energies, valid_bending)
        ax5.set_title('アマテラス光屈折度 (臨界値以下)')
        ax5.set_xlabel('エネルギー (GeV)')
        ax5.set_ylabel('屈折度')
        ax5.set_xscale('log')
        ax5.set_yscale('log')
        
        # 臨界エネルギー値の表示
        threshold = self.find_threshold_energy()
        if threshold > 0:
            ax5.axvline(x=threshold, color='r', linestyle='--')
            ax5.text(threshold*0.8, ax5.get_ylim()[0]*2, f'臨界エネルギー: {threshold:.2e} GeV', 
                    rotation=90, verticalalignment='bottom')
        ax5.grid(True)
        
        # アマテラスNQG相互作用
        ax6 = fig1.add_subplot(326)
        ax6.plot(self.history['energy'], self.history['amaterasu_nqg_interaction'])
        ax6.set_title('アマテラス-NQG相互作用')
        ax6.set_xlabel('エネルギー (GeV)')
        ax6.set_ylabel('相互作用強度')
        ax6.set_xscale('log')
        ax6.set_yscale('log')
        ax6.grid(True)
        
        plt.tight_layout()
        plt.savefig('nkat_particles_basic.png', dpi=300, bbox_inches='tight')
        
        # 遮蔽効果に特化したプロット
        fig2 = plt.figure(figsize=(15, 10))
        
        # すべての遮蔽効果を1つのグラフに
        ax1 = fig2.add_subplot(211)
        ax1.plot(self.history['energy'], self.history['em_shielding'], label='電磁波')
        ax1.plot(self.history['energy'], self.history['gamma_shielding'], label='ガンマ線')
        ax1.plot(self.history['energy'], self.history['cosmic_shielding'], label='宇宙線')
        ax1.plot(self.history['energy'], self.history['neutron_shielding'], label='中性子線')
        ax1.plot(self.history['energy'], self.history['xray_shielding'], label='X線')
        ax1.set_title('アマテラス場による各種放射線遮蔽効果')
        ax1.set_xlabel('エネルギー (GeV)')
        ax1.set_ylabel('遮蔽効率 (0-1)')
        ax1.set_xscale('log')
        ax1.grid(True)
        ax1.legend()
        
        # 遮蔽効果と光屈折度の関係
        ax2 = fig2.add_subplot(212)
        
        # 左軸: 遮蔽効率
        ax2.plot(self.history['energy'], self.history['em_shielding'], 'b-', label='電磁波遮蔽効率')
        ax2.set_xlabel('エネルギー (GeV)')
        ax2.set_ylabel('遮蔽効率 (0-1)', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        ax2.set_xscale('log')
        
        # 右軸: 光屈折度 (対数スケール)
        ax2_twin = ax2.twinx()
        
        # 有効な値のみプロット（無限大を除外）
        ax2_twin.plot(valid_energies, valid_bending, 'r-', label='光屈折度')
        ax2_twin.set_ylabel('光屈折度 (対数スケール)', color='r')
        ax2_twin.tick_params(axis='y', labelcolor='r')
        ax2_twin.set_yscale('log')
        
        # 臨界エネルギー値の表示
        if threshold > 0:
            ax2.axvline(x=threshold, color='g', linestyle='--')
            ax2.text(threshold*0.8, 0.5, f'臨界エネルギー: {threshold:.2e} GeV', 
                    rotation=90, verticalalignment='center')
        
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('nkat_amaterasu_shielding.png', dpi=300, bbox_inches='tight')
        
        # 遮蔽効率と屈折度の相関グラフ
        plt.figure(figsize=(10, 6))
        plt.scatter(valid_bending, [self.history['em_shielding'][i] for i in valid_indices], alpha=0.7)
        plt.xscale('log')
        plt.xlabel('光屈折度 (対数スケール)')
        plt.ylabel('電磁波遮蔽効率')
        plt.title('光屈折度と電磁波遮蔽効率の相関')
        plt.grid(True)
        plt.savefig('nkat_refraction_shielding_correlation.png', dpi=300, bbox_inches='tight')
        
        plt.close('all')

def main():
    print("NKAT粒子シミュレーションを開始します...")
    
    # シミュレーションの設定
    dim = 4
    steps = 1000
    energy_range = torch.logspace(15, 20, steps)
    
    # 電磁場の強さを設定
    field_strength = torch.tensor([1.0, 0.0, 0.0, 0.0]) * 1e-9  # 弱い電場
    
    # アマテラス場の厚みを設定
    distance = torch.tensor(1e-3)  # 1 mm
    
    # シミュレーションの実行
    sim = ParticleSimulation(dim=dim)
    print("\nシミュレーションを実行中...")
    sim.run_simulation(steps=steps, energy_range=energy_range, 
                       field_strength=field_strength, distance=distance)
    
    # 結果のプロット
    print("\n結果をプロット中...")
    sim.plot_results()
    
    # 臨界エネルギーの特定
    threshold = sim.find_threshold_energy()
    if threshold > 0:
        print(f"\n光屈折度が無限大に発散する臨界エネルギー: {threshold:.4e} GeV")
        
        # 臨界エネルギー付近での詳細解析
        if threshold > 1e16:  # 閾値が十分に高い場合のみ
            print("\n臨界エネルギー付近での詳細解析を開始...")
            critical_energy_analysis(threshold, steps=200)
    
    print("\nシミュレーションが完了しました。")
    print("結果は'nkat_particles_basic.png'と'nkat_amaterasu_shielding.png'に保存されました。")
    
    # 追加の実験: 様々な場の厚さでの遮蔽効果
    print("\n様々な厚さでの遮蔽効果シミュレーションを開始...")
    thickness_experiments(steps=100)
    
    # 追加の実験: 様々な放射線タイプに対する遮蔽効果
    print("\n様々な放射線タイプに対する遮蔽効果の比較...")
    radiation_comparison(steps=50)

def thickness_experiments(steps: int = 100):
    """様々な厚さでのアマテラス場遮蔽効果を調査"""
    # 固定エネルギーでの実験（中間エネルギー値）
    energy = torch.tensor(1e17)  # 10^17 GeV
    field_strength = torch.tensor([1.0, 0.0, 0.0, 0.0]) * 1e-9
    
    # 様々な厚さでの遮蔽効果
    thicknesses = torch.logspace(-6, 0, steps)  # 1μm～1m
    
    em_shields = []
    gamma_shields = []
    cosmic_shields = []
    
    # アマテラス粒子のインスタンス化
    amaterasu = AmaterasuParticle(dim=5)
    
    for thickness in thicknesses:
        # 各種放射線に対する遮蔽効果
        em_shield = amaterasu.compute_electromagnetic_shielding(
            field_strength, thickness, 'electromagnetic')
        gamma_shield = amaterasu.compute_electromagnetic_shielding(
            field_strength, thickness, 'gamma')
        cosmic_shield = amaterasu.compute_electromagnetic_shielding(
            field_strength, thickness, 'cosmic')
        
        em_shields.append(em_shield.item())
        gamma_shields.append(gamma_shield.item())
        cosmic_shields.append(cosmic_shield.item())
    
    # 結果のプロット
    plt.figure(figsize=(10, 6))
    plt.plot(thicknesses.numpy(), em_shields, 'b-', label='電磁波')
    plt.plot(thicknesses.numpy(), gamma_shields, 'r-', label='ガンマ線')
    plt.plot(thicknesses.numpy(), cosmic_shields, 'g-', label='宇宙線')
    
    plt.title(f'アマテラス場の厚さと放射線遮蔽効果の関係 (固定エネルギー: {energy:.1e} GeV)')
    plt.xlabel('アマテラス場の厚さ (m)')
    plt.ylabel('遮蔽効率 (0-1)')
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('nkat_thickness_shielding.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("厚さ実験が完了しました。結果は'nkat_thickness_shielding.png'に保存されました。")

def radiation_comparison(steps: int = 50):
    """様々な放射線タイプに対する遮蔽効果の比較"""
    # エネルギー範囲
    energies = torch.logspace(15, 20, steps)
    field_strength = torch.tensor([1.0, 0.0, 0.0, 0.0]) * 1e-9
    distance = torch.tensor(1e-3)  # 1 mm
    
    # 放射線タイプ
    radiation_types = ['electromagnetic', 'gamma', 'cosmic', 'neutron', 'x-ray']
    
    # 結果保存用
    results = {rad_type: [] for rad_type in radiation_types}
    
    # アマテラス粒子のインスタンス化
    amaterasu = AmaterasuParticle(dim=5)
    
    for energy in energies:
        # エネルギーに応じて電磁場の強さを調整
        adjusted_field = field_strength * (energy / 1e15)
        
        # 各放射線タイプに対する遮蔽効果を計算
        for rad_type in radiation_types:
            shield = amaterasu.compute_electromagnetic_shielding(
                adjusted_field, distance, rad_type)
            results[rad_type].append(shield.item())
    
    # 結果のプロット
    plt.figure(figsize=(12, 7))
    
    colors = ['b', 'r', 'g', 'purple', 'orange']
    
    for i, rad_type in enumerate(radiation_types):
        plt.plot(energies.numpy(), results[rad_type], color=colors[i], 
                 label=f'{rad_type}')
    
    plt.title('様々な放射線タイプに対するアマテラス場の遮蔽効果比較')
    plt.xlabel('エネルギー (GeV)')
    plt.ylabel('遮蔽効率 (0-1)')
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('nkat_radiation_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("放射線比較実験が完了しました。結果は'nkat_radiation_comparison.png'に保存されました。")

def critical_energy_analysis(threshold_energy: float, steps: int = 200):
    """臨界エネルギー付近での詳細解析"""
    # 臨界エネルギー付近のエネルギー範囲（前後10%）
    energy_min = threshold_energy * 0.9
    energy_max = threshold_energy * 1.1
    energies = torch.linspace(energy_min, energy_max, steps)
    
    # 初期設定
    field_strength = torch.tensor([1.0, 0.0, 0.0, 0.0]) * 1e-9
    distance = torch.tensor(1e-3)  # 1 mm
    
    # 結果保存用
    light_bendings = []
    em_shields = []
    
    # アマテラス粒子のインスタンス化
    amaterasu = AmaterasuParticle(dim=5)
    
    for energy in energies:
        # 光屈折度の計算
        bending = amaterasu.compute_light_bending(energy)
        light_bendings.append(bending.item() if not torch.isinf(bending) and bending < 1e20 else 1e20)
        
        # 電磁遮蔽効果の計算
        adjusted_field = field_strength * (energy / 1e15)
        shield = amaterasu.compute_electromagnetic_shielding(
            adjusted_field, distance, 'electromagnetic')
        em_shields.append(shield.item())
    
    # 結果のプロット
    plt.figure(figsize=(12, 7))
    
    # 左軸: 遮蔽効率
    ax1 = plt.gca()
    ax1.plot(energies.numpy(), em_shields, 'b-', label='電磁波遮蔽効率')
    ax1.set_xlabel('エネルギー (GeV)')
    ax1.set_ylabel('遮蔽効率 (0-1)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # 右軸: 光屈折度
    ax2 = ax1.twinx()
    ax2.plot(energies.numpy(), light_bendings, 'r-', label='光屈折度')
    ax2.set_ylabel('光屈折度', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_yscale('log')
    
    # 臨界エネルギーの表示
    plt.axvline(x=threshold_energy, color='g', linestyle='--')
    plt.text(threshold_energy*0.99, min(light_bendings)*2, 
             f'臨界エネルギー: {threshold_energy:.4e} GeV', 
             rotation=90, verticalalignment='bottom')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title('臨界エネルギー付近でのアマテラス粒子の光学特性と遮蔽効果')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('nkat_critical_energy_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("臨界エネルギー分析が完了しました。結果は'nkat_critical_energy_analysis.png'に保存されました。")

if __name__ == "__main__":
    main() 