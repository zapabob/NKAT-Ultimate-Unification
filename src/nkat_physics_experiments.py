#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 NKAT理論の物理学的実験検証設計
NKAT Theory Physical Experiment Design and Verification

Author: NKAT Research Team
Date: 2025-05-24
Version: 1.0 - Physics Experiment Framework
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict, Any, Union
import warnings
from pathlib import Path
import json
import time
from dataclasses import dataclass
from tqdm import tqdm, trange
import logging
from collections import defaultdict
import scipy.constants as const
from scipy import signal, optimize
from astropy import units as u
from astropy.coordinates import SkyCoord
import pandas as pd

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

@dataclass
class NKATPhysicsConfig:
    """NKAT物理実験設定"""
    experiment_type: str  # 'gamma_ray', 'gravitational_wave', 'cosmology'
    theta_parameter: float = 1e-25  # 非可換パラメータ
    kappa_parameter: float = 1e-15  # Minkowski変形パラメータ
    energy_range: Tuple[float, float] = (1e12, 1e20)  # eV
    frequency_range: Tuple[float, float] = (10, 1000)  # Hz
    observation_time: float = 86400  # 秒
    detector_sensitivity: float = 1e-23  # 検出器感度
    precision: str = 'high'

class NKATGammaRayPredictor(nn.Module):
    """
    NKAT理論によるγ線天文学予測
    
    非可換時空効果によるγ線スペクトルの変調を予測
    """
    
    def __init__(self, config: NKATPhysicsConfig):
        super().__init__()
        self.config = config
        self.device = device
        
        # 物理定数
        self.c = const.c  # 光速
        self.h = const.h  # プランク定数
        self.hbar = const.hbar  # 換算プランク定数
        
        # NKAT理論パラメータ
        self.theta = config.theta_parameter
        self.kappa = config.kappa_parameter
        
        logger.info(f"🔬 NKATγ線予測器初期化: θ={self.theta:.2e}, κ={self.kappa:.2e}")
        
    def compute_nkat_dispersion_relation(self, energy: torch.Tensor, 
                                       momentum: torch.Tensor) -> torch.Tensor:
        """
        NKAT理論による分散関係の計算
        
        E² = p²c² + m²c⁴ + θ-補正項 + κ-補正項
        """
        # 標準的な分散関係
        E_standard = torch.sqrt(momentum**2 * self.c**2)
        
        # θ-変形による補正（非可換性）
        theta_correction = torch.tensor(self.theta, device=self.device) * energy * momentum
        
        # κ-変形による補正（Minkowski変形）
        kappa_correction = torch.tensor(self.kappa, device=self.device) * energy**2 / self.c**2
        
        # 修正された分散関係
        E_modified = E_standard + theta_correction + kappa_correction
        
        return E_modified
    
    def predict_gamma_ray_spectrum_modification(self, 
                                              energies: torch.Tensor,
                                              source_distance: float = 1e26) -> Dict:
        """
        γ線スペクトルの修正予測
        
        Args:
            energies: γ線エネルギー [eV]
            source_distance: 天体までの距離 [m]
        """
        # 運動量の計算
        momenta = energies / self.c
        
        # NKAT分散関係による修正
        modified_energies = self.compute_nkat_dispersion_relation(energies, momenta)
        
        # 伝播時間の修正
        travel_time_standard = source_distance / self.c
        
        # NKAT効果による時間遅延
        theta_delay = torch.tensor(self.theta, device=self.device) * energies * source_distance / (self.c**3)
        kappa_delay = torch.tensor(self.kappa, device=self.device) * energies**2 * source_distance / (self.c**5)
        
        total_delay = theta_delay + kappa_delay
        
        # スペクトル修正の計算
        spectral_modification = (modified_energies - energies) / energies
        
        results = {
            'original_energies': energies.cpu().numpy(),
            'modified_energies': modified_energies.cpu().numpy(),
            'spectral_modification': spectral_modification.cpu().numpy(),
            'time_delay': total_delay.cpu().numpy(),
            'relative_delay': (total_delay / travel_time_standard).cpu().numpy()
        }
        
        return results
    
    def design_gamma_ray_experiment(self, target_sources: List[str]) -> Dict:
        """
        γ線天文学実験の設計
        """
        logger.info("🔭 γ線天文学実験設計開始...")
        
        # 代表的なγ線源の設定
        gamma_sources = {
            'Crab_Nebula': {'distance': 2e19, 'flux': 1e-6, 'energy_cutoff': 1e13},
            'Vela_Pulsar': {'distance': 3e19, 'flux': 5e-7, 'energy_cutoff': 5e12},
            'Markarian_421': {'distance': 4e25, 'flux': 1e-7, 'energy_cutoff': 1e14},
            'PKS_2155-304': {'distance': 3e25, 'flux': 2e-7, 'energy_cutoff': 2e13}
        }
        
        experiment_design = {
            'target_sources': target_sources,
            'observation_strategy': {},
            'predicted_signals': {},
            'detector_requirements': {},
            'data_analysis_plan': {}
        }
        
        # エネルギー範囲の設定
        energies = torch.logspace(
            np.log10(self.config.energy_range[0]), 
            np.log10(self.config.energy_range[1]), 
            100, device=self.device
        )
        
        for source_name in target_sources:
            if source_name in gamma_sources:
                source_data = gamma_sources[source_name]
                
                # NKAT効果の予測
                predictions = self.predict_gamma_ray_spectrum_modification(
                    energies, source_data['distance']
                )
                
                # 観測戦略の設計
                observation_time = self.config.observation_time
                required_sensitivity = source_data['flux'] * 0.01  # 1%の精度
                
                experiment_design['observation_strategy'][source_name] = {
                    'observation_time': observation_time,
                    'required_sensitivity': required_sensitivity,
                    'energy_resolution': 0.1,  # 10%エネルギー分解能
                    'angular_resolution': 0.1  # 0.1度角度分解能
                }
                
                experiment_design['predicted_signals'][source_name] = predictions
                
                # 検出可能性の評価
                max_modification = np.max(np.abs(predictions['spectral_modification']))
                detectability = max_modification / required_sensitivity
                
                experiment_design['predicted_signals'][source_name]['detectability'] = detectability
                experiment_design['predicted_signals'][source_name]['feasible'] = detectability > 1.0
        
        # 検出器要件の設定
        experiment_design['detector_requirements'] = {
            'energy_range': self.config.energy_range,
            'effective_area': 1e4,  # m²
            'energy_resolution': 0.1,
            'angular_resolution': 0.1,
            'background_rejection': 1e-6,
            'observation_time': self.config.observation_time
        }
        
        # データ解析計画
        experiment_design['data_analysis_plan'] = {
            'spectral_analysis': 'エネルギースペクトルの詳細解析',
            'timing_analysis': '到着時間の精密測定',
            'correlation_analysis': '複数源での相関解析',
            'statistical_methods': 'ベイズ統計による信号抽出',
            'systematic_uncertainties': '系統誤差の評価と補正'
        }
        
        return experiment_design

class NKATGravitationalWavePredictor(nn.Module):
    """
    NKAT理論による重力波検出予測
    
    非可換時空効果による重力波の伝播修正を予測
    """
    
    def __init__(self, config: NKATPhysicsConfig):
        super().__init__()
        self.config = config
        self.device = device
        
        # 重力定数
        self.G = const.G
        self.c = const.c
        
        # NKAT理論パラメータ
        self.theta = config.theta_parameter
        self.kappa = config.kappa_parameter
        
        logger.info(f"🌊 NKAT重力波予測器初期化: θ={self.theta:.2e}, κ={self.kappa:.2e}")
    
    def compute_nkat_gravitational_wave_modification(self, 
                                                   frequencies: torch.Tensor,
                                                   source_distance: float = 1e25) -> Dict:
        """
        重力波の NKAT 修正の計算
        """
        # 重力波の波長
        wavelengths = self.c / frequencies
        
        # NKAT効果による位相修正
        theta_phase = torch.tensor(self.theta, device=self.device) * frequencies * source_distance / self.c**2
        kappa_phase = torch.tensor(self.kappa, device=self.device) * frequencies**2 * source_distance / self.c**3
        
        total_phase_shift = theta_phase + kappa_phase
        
        # 振幅修正
        theta_amplitude = torch.tensor(self.theta, device=self.device) * frequencies / self.c
        kappa_amplitude = torch.tensor(self.kappa, device=self.device) * frequencies**2 / self.c**2
        
        amplitude_modification = 1 + theta_amplitude + kappa_amplitude
        
        # 群速度の修正
        group_velocity_modification = 1 - 2 * torch.tensor(self.kappa, device=self.device) * frequencies / self.c
        
        results = {
            'frequencies': frequencies.cpu().numpy(),
            'phase_shift': total_phase_shift.cpu().numpy(),
            'amplitude_modification': amplitude_modification.cpu().numpy(),
            'group_velocity_modification': group_velocity_modification.cpu().numpy(),
            'arrival_time_delay': (total_phase_shift / (2 * np.pi * frequencies)).cpu().numpy()
        }
        
        return results
    
    def design_gravitational_wave_experiment(self) -> Dict:
        """
        重力波検出実験の設計
        """
        logger.info("🌊 重力波検出実験設計開始...")
        
        # 周波数範囲の設定
        frequencies = torch.logspace(
            np.log10(self.config.frequency_range[0]),
            np.log10(self.config.frequency_range[1]),
            100, device=self.device
        )
        
        # 代表的な重力波源
        gw_sources = {
            'BH_merger_10_10': {'distance': 1e25, 'chirp_mass': 30, 'duration': 0.1},
            'NS_merger': {'distance': 5e24, 'chirp_mass': 1.2, 'duration': 10},
            'BH_merger_100_100': {'distance': 3e26, 'chirp_mass': 70, 'duration': 0.01}
        }
        
        experiment_design = {
            'detector_network': {},
            'predicted_modifications': {},
            'sensitivity_requirements': {},
            'data_analysis_strategy': {}
        }
        
        # 各重力波源に対する予測
        for source_name, source_data in gw_sources.items():
            predictions = self.compute_nkat_gravitational_wave_modification(
                frequencies, source_data['distance']
            )
            
            experiment_design['predicted_modifications'][source_name] = predictions
            
            # 検出可能性の評価
            max_phase_shift = np.max(np.abs(predictions['phase_shift']))
            max_amplitude_mod = np.max(np.abs(predictions['amplitude_modification'] - 1))
            
            experiment_design['predicted_modifications'][source_name]['max_phase_shift'] = max_phase_shift
            experiment_design['predicted_modifications'][source_name]['max_amplitude_modification'] = max_amplitude_mod
            experiment_design['predicted_modifications'][source_name]['detectability_phase'] = max_phase_shift > 0.1
            experiment_design['predicted_modifications'][source_name]['detectability_amplitude'] = max_amplitude_mod > 0.01
        
        # 検出器ネットワークの設計
        experiment_design['detector_network'] = {
            'LIGO_Hanford': {'sensitivity': 1e-23, 'arm_length': 4000, 'location': 'USA'},
            'LIGO_Livingston': {'sensitivity': 1e-23, 'arm_length': 4000, 'location': 'USA'},
            'Virgo': {'sensitivity': 1e-23, 'arm_length': 3000, 'location': 'Italy'},
            'KAGRA': {'sensitivity': 1e-24, 'arm_length': 3000, 'location': 'Japan'},
            'Einstein_Telescope': {'sensitivity': 1e-25, 'arm_length': 10000, 'location': 'Europe_future'}
        }
        
        # 感度要件
        experiment_design['sensitivity_requirements'] = {
            'strain_sensitivity': 1e-25,
            'frequency_range': self.config.frequency_range,
            'phase_accuracy': 0.01,  # ラジアン
            'amplitude_accuracy': 0.001,
            'timing_accuracy': 1e-6  # 秒
        }
        
        # データ解析戦略
        experiment_design['data_analysis_strategy'] = {
            'matched_filtering': 'NKAT修正テンプレートとのマッチドフィルタリング',
            'parameter_estimation': 'ベイズ推定によるNKATパラメータ抽出',
            'multi_detector_analysis': '複数検出器での相関解析',
            'systematic_error_control': '系統誤差の同定と除去',
            'background_characterization': 'ノイズ特性の詳細解析'
        }
        
        return experiment_design

class NKATCosmologyPredictor(nn.Module):
    """
    NKAT理論による宇宙論的観測予測
    
    宇宙マイクロ波背景放射(CMB)や大規模構造への影響を予測
    """
    
    def __init__(self, config: NKATPhysicsConfig):
        super().__init__()
        self.config = config
        self.device = device
        
        # 宇宙論パラメータ
        self.H0 = 70  # km/s/Mpc
        self.Omega_m = 0.3
        self.Omega_Lambda = 0.7
        
        # NKAT理論パラメータ
        self.theta = config.theta_parameter
        self.kappa = config.kappa_parameter
        
        logger.info(f"🌌 NKAT宇宙論予測器初期化: θ={self.theta:.2e}, κ={self.kappa:.2e}")
    
    def compute_cmb_power_spectrum_modification(self, l_values: torch.Tensor) -> Dict:
        """
        CMBパワースペクトラムのNKAT修正
        """
        # 標準的なCMBパワースペクトラム（簡単なモデル）
        l_peak = 220  # 第一ピークの位置
        C_l_standard = torch.exp(-(l_values - l_peak)**2 / (2 * 50**2))
        
        # NKAT効果による修正
        theta_correction = torch.tensor(self.theta, device=self.device) * l_values**2 / 1e10
        kappa_correction = torch.tensor(self.kappa, device=self.device) * l_values / 1e5
        
        C_l_modified = C_l_standard * (1 + theta_correction + kappa_correction)
        
        # ピーク位置のシフト
        peak_shift = torch.tensor(self.theta, device=self.device) * 1e15 + torch.tensor(self.kappa, device=self.device) * 1e10
        
        results = {
            'l_values': l_values.cpu().numpy(),
            'C_l_standard': C_l_standard.cpu().numpy(),
            'C_l_modified': C_l_modified.cpu().numpy(),
            'relative_modification': ((C_l_modified - C_l_standard) / C_l_standard).cpu().numpy(),
            'peak_shift': peak_shift.cpu().numpy()
        }
        
        return results
    
    def design_cosmology_experiment(self) -> Dict:
        """
        宇宙論的観測実験の設計
        """
        logger.info("🌌 宇宙論的観測実験設計開始...")
        
        # 多重極モーメントの範囲
        l_values = torch.arange(2, 3000, device=self.device, dtype=torch.float32)
        
        # CMB修正の計算
        cmb_predictions = self.compute_cmb_power_spectrum_modification(l_values)
        
        experiment_design = {
            'cmb_observations': {},
            'large_scale_structure': {},
            'predicted_signatures': {},
            'observational_requirements': {}
        }
        
        # CMB観測の設計
        experiment_design['cmb_observations'] = {
            'temperature_sensitivity': 1e-6,  # K
            'polarization_sensitivity': 1e-7,  # K
            'angular_resolution': 5,  # arcmin
            'frequency_channels': [30, 44, 70, 100, 143, 217, 353, 545, 857],  # GHz
            'sky_coverage': 0.8,  # 全天の80%
            'observation_time': 4 * 365 * 24 * 3600  # 4年間
        }
        
        # 大規模構造観測
        experiment_design['large_scale_structure'] = {
            'galaxy_survey_area': 14000,  # deg²
            'redshift_range': (0.1, 2.0),
            'galaxy_density': 1e-3,  # arcmin⁻²
            'photometric_accuracy': 0.02,
            'spectroscopic_sample': 1e6
        }
        
        # 予測される観測シグネチャ
        max_cmb_modification = np.max(np.abs(cmb_predictions['relative_modification']))
        
        experiment_design['predicted_signatures'] = {
            'cmb_power_spectrum': cmb_predictions,
            'max_modification': max_cmb_modification,
            'detectability': max_cmb_modification > 1e-5,
            'peak_shift_detectability': abs(cmb_predictions['peak_shift']) > 1.0,
            'polarization_effects': 'E-mode/B-modeパターンの修正',
            'lensing_modifications': '重力レンズ効果の変更'
        }
        
        # 観測要件
        experiment_design['observational_requirements'] = {
            'temperature_map_noise': 1e-6,  # K⋅arcmin
            'polarization_map_noise': 1e-7,  # K⋅arcmin
            'systematic_error_control': 1e-7,
            'calibration_accuracy': 1e-4,
            'foreground_removal': 'マルチ周波数成分分離'
        }
        
        return experiment_design

def demonstrate_nkat_physics_experiments():
    """
    NKAT理論の物理学的実験検証デモンストレーション
    """
    print("=" * 80)
    print("🎯 NKAT理論の物理学的実験検証設計")
    print("=" * 80)
    print("📅 実行日時:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("🔬 実験分野: γ線天文学、重力波検出、宇宙論的観測")
    print("=" * 80)
    
    all_experiments = {}
    
    # 1. γ線天文学実験
    print("\n🔭 1. γ線天文学実験設計")
    print("目的：非可換時空効果によるγ線スペクトル変調の検出")
    
    gamma_config = NKATPhysicsConfig(
        experiment_type='gamma_ray',
        theta_parameter=1e-25,
        kappa_parameter=1e-15,
        energy_range=(1e12, 1e20),
        observation_time=86400 * 365  # 1年間
    )
    
    gamma_predictor = NKATGammaRayPredictor(gamma_config)
    gamma_experiment = gamma_predictor.design_gamma_ray_experiment([
        'Crab_Nebula', 'Markarian_421', 'PKS_2155-304'
    ])
    
    print(f"✅ 対象天体数: {len(gamma_experiment['target_sources'])}")
    
    feasible_sources = []
    for source, predictions in gamma_experiment['predicted_signals'].items():
        if predictions.get('feasible', False):
            feasible_sources.append(source)
            max_mod = predictions.get('detectability', 0)
            print(f"📊 {source}: 検出可能性 = {max_mod:.3f}")
    
    print(f"🎯 検出可能な天体数: {len(feasible_sources)}")
    all_experiments['gamma_ray'] = gamma_experiment
    
    # 2. 重力波検出実験
    print("\n🌊 2. 重力波検出実験設計")
    print("目的：NKAT効果による重力波伝播の修正検出")
    
    gw_config = NKATPhysicsConfig(
        experiment_type='gravitational_wave',
        theta_parameter=1e-25,
        kappa_parameter=1e-15,
        frequency_range=(10, 1000),
        detector_sensitivity=1e-23
    )
    
    gw_predictor = NKATGravitationalWavePredictor(gw_config)
    gw_experiment = gw_predictor.design_gravitational_wave_experiment()
    
    print(f"✅ 検出器ネットワーク: {len(gw_experiment['detector_network'])}台")
    
    detectable_sources = []
    for source, predictions in gw_experiment['predicted_modifications'].items():
        phase_detectable = predictions.get('detectability_phase', False)
        amplitude_detectable = predictions.get('detectability_amplitude', False)
        
        if phase_detectable or amplitude_detectable:
            detectable_sources.append(source)
            print(f"📊 {source}: 位相検出={phase_detectable}, 振幅検出={amplitude_detectable}")
    
    print(f"🎯 検出可能な重力波源: {len(detectable_sources)}")
    all_experiments['gravitational_wave'] = gw_experiment
    
    # 3. 宇宙論的観測実験
    print("\n🌌 3. 宇宙論的観測実験設計")
    print("目的：CMBや大規模構造におけるNKAT効果の検出")
    
    cosmo_config = NKATPhysicsConfig(
        experiment_type='cosmology',
        theta_parameter=1e-25,
        kappa_parameter=1e-15
    )
    
    cosmo_predictor = NKATCosmologyPredictor(cosmo_config)
    cosmo_experiment = cosmo_predictor.design_cosmology_experiment()
    
    cmb_detectable = cosmo_experiment['predicted_signatures']['detectability']
    peak_shift_detectable = cosmo_experiment['predicted_signatures']['peak_shift_detectability']
    max_modification = cosmo_experiment['predicted_signatures']['max_modification']
    
    print(f"✅ CMB修正検出可能性: {cmb_detectable}")
    print(f"📊 最大修正: {max_modification:.2e}")
    print(f"📊 ピークシフト検出可能性: {peak_shift_detectable}")
    
    all_experiments['cosmology'] = cosmo_experiment
    
    # 4. 統合実験戦略
    print("\n📊 4. 統合実験戦略")
    print("=" * 50)
    
    total_feasible = len(feasible_sources) + len(detectable_sources) + int(cmb_detectable)
    print(f"✅ 検出可能な実験: {total_feasible}/3分野")
    
    # 実験の優先順位
    priorities = []
    if feasible_sources:
        priorities.append("γ線天文学")
    if detectable_sources:
        priorities.append("重力波検出")
    if cmb_detectable:
        priorities.append("宇宙論的観測")
    
    print(f"📋 推奨実験順序: {' → '.join(priorities)}")
    
    # 5. 実験スケジュール
    experiment_schedule = {
        'Phase_1_Preparation': {
            'duration': '2年',
            'activities': ['検出器較正', 'バックグラウンド測定', 'システム最適化']
        },
        'Phase_2_Observation': {
            'duration': '3年',
            'activities': ['データ取得', 'リアルタイム解析', '品質管理']
        },
        'Phase_3_Analysis': {
            'duration': '2年',
            'activities': ['詳細解析', 'NKAT信号抽出', '結果検証']
        }
    }
    
    print("\n📅 実験スケジュール:")
    for phase, details in experiment_schedule.items():
        print(f"  {phase}: {details['duration']} - {', '.join(details['activities'])}")
    
    # 6. 結果の保存
    all_experiments['experiment_schedule'] = experiment_schedule
    all_experiments['summary'] = {
        'total_experiments': 3,
        'feasible_experiments': total_feasible,
        'recommended_priorities': priorities,
        'estimated_duration': '7年',
        'required_funding': '推定10億ドル'
    }
    
    with open('nkat_physics_experiments.json', 'w', encoding='utf-8') as f:
        json.dump(all_experiments, f, indent=2, ensure_ascii=False, default=str)
    
    print("\n💾 実験設計を 'nkat_physics_experiments.json' に保存しました")
    
    # 7. 可視化
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # γ線スペクトル修正
        if 'Crab_Nebula' in gamma_experiment['predicted_signals']:
            crab_data = gamma_experiment['predicted_signals']['Crab_Nebula']
            energies = crab_data['original_energies']
            modifications = crab_data['spectral_modification']
            
            ax1.loglog(energies, np.abs(modifications), 'b-', linewidth=2)
            ax1.set_xlabel('エネルギー [eV]')
            ax1.set_ylabel('相対的スペクトル修正')
            ax1.set_title('γ線スペクトル修正（Crab Nebula）')
            ax1.grid(True, alpha=0.3)
        
        # 重力波位相修正
        if 'BH_merger_10_10' in gw_experiment['predicted_modifications']:
            bh_data = gw_experiment['predicted_modifications']['BH_merger_10_10']
            frequencies = bh_data['frequencies']
            phase_shifts = bh_data['phase_shift']
            
            ax2.semilogx(frequencies, phase_shifts, 'r-', linewidth=2)
            ax2.set_xlabel('周波数 [Hz]')
            ax2.set_ylabel('位相シフト [rad]')
            ax2.set_title('重力波位相修正（BH合体）')
            ax2.grid(True, alpha=0.3)
        
        # CMBパワースペクトラム修正
        cmb_data = cosmo_experiment['predicted_signatures']['cmb_power_spectrum']
        l_values = cmb_data['l_values']
        relative_mod = cmb_data['relative_modification']
        
        ax3.semilogx(l_values, relative_mod, 'g-', linewidth=2)
        ax3.set_xlabel('多重極モーメント l')
        ax3.set_ylabel('相対的修正')
        ax3.set_title('CMBパワースペクトラム修正')
        ax3.grid(True, alpha=0.3)
        
        # 実験検出可能性
        experiments = ['γ線天文学', '重力波検出', '宇宙論観測']
        detectability = [
            len(feasible_sources) / len(gamma_experiment['target_sources']),
            len(detectable_sources) / len(gw_experiment['predicted_modifications']),
            float(cmb_detectable)
        ]
        
        colors = ['blue', 'red', 'green']
        bars = ax4.bar(experiments, detectability, color=colors, alpha=0.7)
        ax4.set_ylabel('検出可能性')
        ax4.set_title('NKAT効果検出可能性')
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3)
        
        # 値をバーの上に表示
        for bar, value in zip(bars, detectability):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('nkat_physics_experiments.png', dpi=300, bbox_inches='tight')
        print("📊 グラフを 'nkat_physics_experiments.png' に保存しました")
        plt.show()
        
    except Exception as e:
        logger.warning(f"⚠️ 可視化エラー: {e}")
    
    return all_experiments

if __name__ == "__main__":
    """
    NKAT理論の物理学的実験検証実行
    """
    try:
        experiments = demonstrate_nkat_physics_experiments()
        print("🎉 NKAT理論の物理学的実験設計が完了しました！")
    except Exception as e:
        logger.error(f"❌ 実行エラー: {e}")
        print(f"❌ エラーが発生しました: {e}") 