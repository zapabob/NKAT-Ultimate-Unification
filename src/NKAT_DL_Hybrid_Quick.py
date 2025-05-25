#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌌 NKAT Deep Learning - Quick Optuna Version
RTX3080最適化 + 電源断対応 + 軽量Optuna版

轻量化Optuna設定:
- n_trials: 20 (75→20)  
- timeout: 300秒 (3600→300)
- quick_epochs: 5 (20→5)
- データ: 400サンプル (800→400)

使用方法:
py -3 NKAT_DL_Hybrid_Quick.py           # 基本実行
py -3 NKAT_DL_Hybrid_Quick.py --resume  # 電源断から復旧
py -3 NKAT_DL_Hybrid_Quick.py --no-optuna --epochs 50  # Optuna無し高速テスト
"""

# ===================================================================
# 📦 Import and Setup
# ===================================================================

import os
import sys
import time
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

# Progress monitoring  
from tqdm import tqdm
import psutil

# Optimization
import optuna

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

# ===================================================================
# 🔧 Environment Setup
# ===================================================================

# CUDA setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 使用デバイス: {device}")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"🔥 GPU: {gpu_name}")
    print(f"💾 VRAM: {gpu_memory:.1f} GB")
    
    # RTX3080最適化
    if "RTX 3080" in gpu_name or "RTX3080" in gpu_name:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("⚡ RTX3080最適化有効")

# Mixed precision
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

# Working directory
work_dir = os.getcwd()

@dataclass
class HybridNKATConfig:
    """Quick Optuna版 NKAT最適化設定"""
    # 物理パラメータ
    theta_base: float = 1e-70
    planck_scale: float = 1.6e-35
    target_spectral_dim: float = 4.0
    spectral_dim_tolerance: float = 0.1
    
    # RTX3080 VRAM最大活用設定
    grid_size: int = 64
    batch_size: int = 24
    num_test_functions: int = 256
    
    # KAN DL設定
    kan_layers: List[int] = field(default_factory=lambda: [4, 512, 256, 128, 4])
    learning_rate: float = 3e-4
    num_epochs: int = 150
    
    # 🔧 Quick Optuna設定
    n_trials: int = 20  # 軽量化（75→20）
    optuna_quick_epochs: int = 5  # 大幅短縮（20→5）
    optuna_samples: int = 400  # 軽量化（800→400）
    optuna_timeout: int = 300  # 5分制限（3600→300）
    study_name: str = "NKAT_Quick_Optimization"
    
    # 物理制約重み
    weight_spectral_dim: float = 20.0
    weight_jacobi: float = 2.0
    weight_connes: float = 2.0
    weight_theta_reg: float = 0.2
    weight_running: float = 4.0
    
    # 監視設定
    progress_update_freq: int = 3
    gpu_monitoring: bool = True
    
    # 電源断対応設定
    checkpoint_freq: int = 5
    auto_backup: bool = True
    resume_from_checkpoint: bool = False
    checkpoint_dir: str = "./nkat_checkpoints_quick"
    max_checkpoints: int = 10
    emergency_save_interval: int = 30

# ===================================================================
# 🧠 Advanced KAN Layer
# ===================================================================

class AdvancedKANLayer(nn.Module):
    """Advanced KAN with learnable spline knots"""
    def __init__(self, input_dim: int, output_dim: int, 
                 grid_size: int = 8, spline_order: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        
        # Learnable spline coefficients
        self.spline_coeffs = nn.Parameter(
            torch.randn(input_dim, output_dim, grid_size) * 0.1
        )
        
        # Learnable knot positions
        self.knot_positions = nn.Parameter(
            torch.linspace(-2, 2, grid_size).unsqueeze(0).unsqueeze(0).repeat(input_dim, output_dim, 1)
        )
        
        # Scaling and bias
        self.scale = nn.Parameter(torch.ones(input_dim, output_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
    def forward(self, x):
        batch_size = x.shape[0]
        x_norm = torch.tanh(x)
        
        output = torch.zeros(batch_size, self.output_dim, device=x.device)
        
        for i in range(self.input_dim):
            for j in range(self.output_dim):
                knots = self.knot_positions[i, j]
                coeffs = self.spline_coeffs[i, j]
                
                distances = (x_norm[:, i:i+1] - knots.unsqueeze(0))**2
                basis_values = torch.exp(-2.0 * distances)
                spline_output = torch.sum(basis_values * coeffs.unsqueeze(0), dim=1)
                
                output[:, j] += self.scale[i, j] * spline_output
        
        return output + self.bias

# ===================================================================
# 🎯 Hybrid NKAT Model
# ===================================================================

class HybridNKATModel(nn.Module):
    """RTX3080最適化 KAN NKAT Model"""
    def __init__(self, config: HybridNKATConfig):
        super().__init__()
        self.config = config
        
        # KAN stack for Dirac operator
        self.kan_layers = nn.ModuleList()
        for i in range(len(config.kan_layers) - 1):
            self.kan_layers.append(
                AdvancedKANLayer(config.kan_layers[i], config.kan_layers[i+1], 
                               grid_size=16)
            )
        
        # θ parameter learning
        self.theta_base_log = nn.Parameter(torch.log(torch.tensor(config.theta_base)))
        self.theta_running_net = nn.Sequential(
            nn.Linear(1, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
        # Gamma matrices
        self.register_buffer('gamma_matrices', self._create_gamma_matrices())
        
    def _create_gamma_matrices(self):
        """Create Dirac gamma matrices"""
        gamma = torch.zeros(4, 4, 4, dtype=torch.complex64)
        
        # γ⁰ (time-like)
        gamma[0] = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0], 
            [0, 0, -1, 0],
            [0, 0, 0, -1]
        ], dtype=torch.complex64)
        
        # γ¹, γ², γ³ (space-like) 
        gamma[1] = torch.tensor([
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [-1, 0, 0, 0]
        ], dtype=torch.complex64)
        
        gamma[2] = torch.tensor([
            [0, 0, 0, -1j],
            [0, 0, 1j, 0],
            [0, 1j, 0, 0],
            [-1j, 0, 0, 0]
        ], dtype=torch.complex64)
        
        gamma[3] = torch.tensor([
            [0, 0, 1, 0],
            [0, 0, 0, -1],
            [-1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=torch.complex64)
        
        return gamma
    
    def forward(self, x, energy_scale=None):
        # KAN forward pass
        kan_output = x
        for kan_layer in self.kan_layers:
            kan_output = kan_layer(kan_output)
            kan_output = F.gelu(kan_output)
        
        # Convert to complex Dirac spinor
        dirac_real = kan_output.view(-1, 4, 1)
        dirac_field = torch.complex(dirac_real, torch.zeros_like(dirac_real))
        
        # θ parameter with running
        if energy_scale is not None:
            log_energy = torch.log10(energy_scale)
            running_coeff = self.theta_running_net(log_energy)
            theta = torch.exp(self.theta_base_log + running_coeff.squeeze())
        else:
            theta = torch.exp(self.theta_base_log)
        
        return dirac_field, theta

# ===================================================================
# 🏆 Advanced Physics Loss
# ===================================================================

class AdvancedPhysicsLoss(nn.Module):
    """Advanced physics-constrained loss function"""
    def __init__(self, config: HybridNKATConfig):
        super().__init__()
        self.config = config
        
    def spectral_dimension_loss(self, dirac_field, target_dim=4.0):
        """Enhanced spectral dimension estimation"""
        field_magnitudes = torch.abs(dirac_field)
        component_vars = torch.var(field_magnitudes, dim=0)
        total_var = torch.sum(component_vars)
        entropy_term = -torch.sum(component_vars * torch.log(component_vars + 1e-8))
        estimated_dim = 4.0 * torch.sigmoid(entropy_term / total_var)
        
        return F.smooth_l1_loss(estimated_dim, torch.tensor(target_dim, device=dirac_field.device))
    
    def jacobi_constraint_loss(self, dirac_field):
        """Jacobi identity constraint"""
        anticommutator = torch.sum(dirac_field * dirac_field.conj(), dim=1).real
        return torch.mean(anticommutator**2)
    
    def connes_distance_loss(self, dirac_field, coordinates):
        """Connes distance consistency"""
        batch_size = coordinates.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=dirac_field.device)
        
        coord_diff = coordinates.unsqueeze(1) - coordinates.unsqueeze(0)
        euclidean_dist = torch.norm(coord_diff, dim=2)
        
        field_diff = dirac_field.unsqueeze(1) - dirac_field.unsqueeze(0)
        dirac_dist = torch.norm(field_diff, dim=2)
        
        consistency_loss = F.mse_loss(dirac_dist.flatten(), euclidean_dist.flatten())
        return consistency_loss
    
    def theta_running_loss(self, theta_values, energy_scale):
        """θ-running constraint loss with proper tensor dimensions"""
        if energy_scale is None:
            return torch.tensor(0.0, device=theta_values.device)
        
        log_energy = torch.log10(energy_scale)
        running_target = -0.1 * log_energy
        
        # 🔧 テンソル次元を合わせる
        theta_log = torch.log10(theta_values + 1e-80)
        
        # theta_valuesが[batch_size]の場合、running_targetも[batch_size]にする
        if theta_log.dim() == 1 and running_target.dim() == 2:
            running_target = running_target.squeeze(-1)
        # theta_valuesが[batch_size, 1]の場合、running_targetも[batch_size, 1]にする
        elif theta_log.dim() == 2 and running_target.dim() == 1:
            running_target = running_target.unsqueeze(-1)
        
        return F.mse_loss(theta_log, running_target)
    
    def forward(self, dirac_field, theta, coordinates, energy_scale=None):
        """Combined physics loss"""
        spectral_loss = self.spectral_dimension_loss(dirac_field, self.config.target_spectral_dim)
        jacobi_loss = self.jacobi_constraint_loss(dirac_field) 
        connes_loss = self.connes_distance_loss(dirac_field, coordinates)
        theta_running_loss = self.theta_running_loss(theta, energy_scale)
        
        total_loss = (
            self.config.weight_spectral_dim * spectral_loss +
            self.config.weight_jacobi * jacobi_loss +
            self.config.weight_connes * connes_loss +
            self.config.weight_running * theta_running_loss
        )
        
        return {
            'total': total_loss,
            'spectral_dim': spectral_loss,
            'jacobi': jacobi_loss,
            'connes': connes_loss,
            'theta_running': theta_running_loss
        }

# ===================================================================
# 🔧 Quick Optuna Objective
# ===================================================================

def quick_objective(trial, config: HybridNKATConfig):
    """軽量化Optuna objective function"""
    
    # Hyperparameter suggestions
    lr = trial.suggest_float('learning_rate', 1e-4, 5e-4, log=True)
    weight_spectral = trial.suggest_float('weight_spectral_dim', 10.0, 30.0)
    weight_running = trial.suggest_float('weight_running', 1.0, 8.0)
    batch_size = trial.suggest_categorical('batch_size', [16, 20, 24, 28])
    
    # Update config
    config.learning_rate = lr
    config.weight_spectral_dim = weight_spectral
    config.weight_running = weight_running
    config.batch_size = batch_size
    
    # Quick training
    model = HybridNKATModel(config).to(device)
    criterion = AdvancedPhysicsLoss(config)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # 軽量データ
    num_samples = config.optuna_samples  # 400サンプル
    train_coords = torch.randn(num_samples, 4, device=device) * 2 * np.pi
    energy_scales = torch.logspace(10, 19, num_samples, device=device).unsqueeze(1)
    
    # 高速学習ループ
    model.train()
    final_spectral_loss = float('inf')
    
    for epoch in range(config.optuna_quick_epochs):  # 5エポック
        total_loss = 0
        
        for i in range(0, num_samples, batch_size):
            batch_coords = train_coords[i:i+batch_size]
            batch_energy = energy_scales[i:i+batch_size]
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast() if scaler else torch.no_grad():
                dirac_field, theta = model(batch_coords, batch_energy)
                losses = criterion(dirac_field, theta, batch_coords, batch_energy)
            
            if scaler:
                scaler.scale(losses['total']).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                losses['total'].backward()
                optimizer.step()
            
            total_loss += losses['total'].item()
            final_spectral_loss = losses['spectral_dim'].item()
    
    return final_spectral_loss

# ===================================================================
# 💾 Checkpoint Manager (簡略版)
# ===================================================================

class QuickCheckpointManager:
    """簡略版チェックポイント管理"""
    
    def __init__(self, config: HybridNKATConfig):
        self.config = config
        self.checkpoint_dir = config.checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.last_emergency_save = time.time()
        
    def save_checkpoint(self, epoch, model, optimizer, scheduler, history, config, 
                       is_emergency=False, best_loss=None):
        """チェックポイント保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if is_emergency:
            filename = f"quick_emergency_epoch_{epoch}_{timestamp}.pth"
        else:
            filename = f"quick_checkpoint_epoch_{epoch}_{timestamp}.pth"
            
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': config,
            'history': history,
            'timestamp': timestamp,
            'best_loss': best_loss,
            'random_state': torch.get_rng_state(),
            'cuda_random_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }
        
        try:
            torch.save(checkpoint_data, checkpoint_path)
            
            # メタデータ保存
            meta_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.json")
            meta_data = {
                'latest_checkpoint': checkpoint_path,
                'epoch': epoch,
                'timestamp': timestamp,
                'spectral_dim': history['spectral_dim_estimates'][-1] if history['spectral_dim_estimates'] else None,
                'total_loss': history['total_loss'][-1] if history['total_loss'] else None
            }
            
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta_data, f, indent=2, ensure_ascii=False)
                
            print(f"💾 チェックポイント保存: {os.path.basename(checkpoint_path)}")
            return checkpoint_path
            
        except Exception as e:
            print(f"⚠️ チェックポイント保存失敗: {str(e)}")
            return None
    
    def load_latest_checkpoint(self):
        """最新チェックポイント読み込み"""
        meta_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.json")
        
        if not os.path.exists(meta_path):
            return None
            
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta_data = json.load(f)
                
            checkpoint_path = meta_data['latest_checkpoint']
            
            if not os.path.exists(checkpoint_path):
                print(f"⚠️ チェックポイントファイルが見つかりません")
                return None
                
            checkpoint_data = torch.load(checkpoint_path, map_location=device)
            print(f"📂 チェックポイント読み込み: エポック {checkpoint_data['epoch']}")
            
            return checkpoint_data
            
        except Exception as e:
            print(f"⚠️ チェックポイント読み込み失敗: {str(e)}")
            return None
    
    def should_emergency_save(self):
        """緊急保存チェック"""
        if not self.config.auto_backup:
            return False
            
        current_time = time.time()
        elapsed_minutes = (current_time - self.last_emergency_save) / 60
        
        if elapsed_minutes >= self.config.emergency_save_interval:
            self.last_emergency_save = current_time
            return True
            
        return False

# ===================================================================
# 🚀 Quick Training Function
# ===================================================================

def train_quick_nkat(config: HybridNKATConfig, use_optuna: bool = True):
    """Quick版 NKAT training"""
    
    print("🎯 Quick NKAT最適化開始（軽量版）")
    print(f"🔥 設定: バッチ{config.batch_size}, エポック{config.num_epochs}")
    print(f"🔧 Optuna: {config.n_trials}回試行, {config.optuna_timeout}秒制限")
    
    # チェックポイントマネージャー
    checkpoint_manager = QuickCheckpointManager(config)
    
    best_config = config
    start_epoch = 0
    resume_data = None
    
    # レジューム処理
    if config.resume_from_checkpoint:
        print("🔄 チェックポイントから復旧中...")
        resume_data = checkpoint_manager.load_latest_checkpoint()
        
    if resume_data:
        print(f"✅ エポック {resume_data['epoch']} から再開")
        best_config = resume_data['config']
        start_epoch = resume_data['epoch'] + 1
        
        # ランダム状態復元
        torch.set_rng_state(resume_data['random_state'])
        if resume_data['cuda_random_state'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(resume_data['cuda_random_state'])
    else:
        print("🆕 新規訓練開始")
        
        if use_optuna:
            print("🔬 Quick Optuna最適化実行中...")
            
            study = optuna.create_study(
                direction='minimize',
                study_name=config.study_name
            )
            
            # Quick Optuna with progress
            with tqdm(total=config.n_trials, desc="🔍 Quick Optuna", 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} trials [{elapsed}<{remaining}]') as pbar:
                
                def callback(study, trial):
                    pbar.set_postfix({
                        'Best': f'{study.best_value:.6f}' if study.best_value else 'N/A',
                        'Current': f'{trial.value:.6f}' if trial.value else 'Failed'
                    })
                    pbar.update(1)
                
                study.optimize(
                    lambda trial: quick_objective(trial, config),
                    n_trials=config.n_trials,
                    timeout=config.optuna_timeout,  # 5分制限
                    callbacks=[callback]
                )
            
            # 最適パラメータ
            best_params = study.best_params
            print(f"🏆 最適パラメータ: {best_params}")
            
            # 設定更新
            for key, value in best_params.items():
                setattr(best_config, key, value)
    
    # モデル初期化
    print("🚀 最終訓練開始（Quick版）")
    
    model = HybridNKATModel(best_config).to(device)
    criterion = AdvancedPhysicsLoss(best_config)
    optimizer = AdamW(model.parameters(), lr=best_config.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=best_config.num_epochs)
    
    # レジューム時の状態復元
    if resume_data:
        model.load_state_dict(resume_data['model_state_dict'])
        optimizer.load_state_dict(resume_data['optimizer_state_dict'])
        scheduler.load_state_dict(resume_data['scheduler_state_dict'])
        history = resume_data['history']
        print(f"📊 学習履歴復元: {len(history['total_loss'])}エポック分")
    else:
        # Training history
        history = {
            'total_loss': [],
            'spectral_dim_loss': [],
            'jacobi_loss': [],
            'connes_loss': [],
            'theta_running_loss': [],
            'theta_values': [],
            'spectral_dim_estimates': [],
            'gpu_memory_usage': [],
            'training_speed': []
        }
    
    # Training data
    num_samples = 4000 if torch.cuda.is_available() else 800  # 中程度のサイズ
    train_coords = torch.randn(num_samples, 4, device=device) * 2 * np.pi
    energy_scales = torch.logspace(10, 19, num_samples, device=device).unsqueeze(1)
    
    print(f"🔥 Training データ: {num_samples:,}サンプル")
    
    # Training loop
    remaining_epochs = best_config.num_epochs - start_epoch
    
    with tqdm(total=remaining_epochs, desc="🎯 Quick NKAT最適化", 
             bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} epochs [{elapsed}<{remaining}]') as main_pbar:
        
        start_time = time.time()
        
        try:
            for epoch in range(start_epoch, best_config.num_epochs):
                epoch_start = time.time()
                model.train()
                
                # Training
                for i in range(0, len(train_coords), best_config.batch_size):
                    batch_coords = train_coords[i:i+best_config.batch_size]
                    batch_energy = energy_scales[i:i+best_config.batch_size]
                    
                    optimizer.zero_grad()
                    
                    with torch.cuda.amp.autocast() if scaler else torch.no_grad():
                        dirac_field, theta = model(batch_coords, batch_energy)
                        losses = criterion(dirac_field, theta, batch_coords, batch_energy)
                    
                    if scaler:
                        scaler.scale(losses['total']).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        losses['total'].backward()
                        optimizer.step()
                
                scheduler.step()
                
                # Evaluation
                model.eval()
                with torch.no_grad():
                    eval_coords = train_coords[:best_config.batch_size]
                    eval_energy = energy_scales[:best_config.batch_size]
                    eval_dirac, eval_theta = model(eval_coords, eval_energy)
                    eval_losses = criterion(eval_dirac, eval_theta, eval_coords, eval_energy)
                    
                    # Spectral dimension estimation
                    field_magnitudes = torch.abs(eval_dirac.squeeze())
                    component_vars = torch.var(field_magnitudes, dim=0)
                    total_var = torch.sum(component_vars)
                    entropy_term = -torch.sum(component_vars * torch.log(component_vars + 1e-8))
                    estimated_dim = (4.0 * torch.sigmoid(entropy_term / total_var)).item()
                
                # Update history
                history['total_loss'].append(eval_losses['total'].item())
                history['spectral_dim_loss'].append(eval_losses['spectral_dim'].item())
                history['jacobi_loss'].append(eval_losses['jacobi'].item())
                history['connes_loss'].append(eval_losses['connes'].item())
                history['theta_running_loss'].append(eval_losses['theta_running'].item())
                history['theta_values'].append(eval_theta.mean().item())
                history['spectral_dim_estimates'].append(estimated_dim)
                
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.memory_allocated() / 1e9
                    history['gpu_memory_usage'].append(gpu_mem)
                else:
                    history['gpu_memory_usage'].append(0)
                
                epoch_time = time.time() - epoch_start
                history['training_speed'].append(epoch_time)
                
                # Progress update
                convergence_rate = abs(estimated_dim - best_config.target_spectral_dim)
                
                main_pbar.set_postfix({
                    'Loss': f'{history["total_loss"][-1]:.4f}',
                    'Spec_Dim': f'{estimated_dim:.3f}→{best_config.target_spectral_dim:.1f}',
                    'Conv': f'{convergence_rate:.3f}',
                    'θ': f'{eval_theta.mean().item():.1e}',
                    'Speed': f'{epoch_time:.1f}s',
                })
                
                # チェックポイント保存
                should_save_checkpoint = False
                is_emergency = False
                
                if (epoch + 1) % config.checkpoint_freq == 0:
                    should_save_checkpoint = True
                
                if checkpoint_manager.should_emergency_save():
                    should_save_checkpoint = True
                    is_emergency = True
                
                if abs(estimated_dim - best_config.target_spectral_dim) < best_config.spectral_dim_tolerance:
                    should_save_checkpoint = True
                    main_pbar.write(f"\n🎊 目標達成！ Epoch {epoch+1}: スペクトル次元 {estimated_dim:.4f}")
                
                if should_save_checkpoint:
                    checkpoint_manager.save_checkpoint(
                        epoch, model, optimizer, scheduler, history, best_config,
                        is_emergency=is_emergency, best_loss=history['total_loss'][-1]
                    )
                
                # Early stopping
                if abs(estimated_dim - best_config.target_spectral_dim) < best_config.spectral_dim_tolerance:
                    break
                
                main_pbar.update(1)
                
        except KeyboardInterrupt:
            print("\n⚠️ ユーザー中断検出")
            checkpoint_manager.save_checkpoint(
                epoch, model, optimizer, scheduler, history, best_config,
                is_emergency=True, best_loss=history['total_loss'][-1] if history['total_loss'] else float('inf')
            )
            print("💾 緊急チェックポイント保存完了")
            raise
    
    # 最終保存
    final_checkpoint = checkpoint_manager.save_checkpoint(
        epoch, model, optimizer, scheduler, history, best_config,
        is_emergency=False, best_loss=history['total_loss'][-1] if history['total_loss'] else float('inf')
    )
    
    final_time = time.time() - start_time
    print(f"✅ Quick NKAT最適化完了！ 総時間: {final_time/60:.1f}分")
    print(f"🏆 最終スペクトル次元: {history['spectral_dim_estimates'][-1]:.4f}")
    
    return model, history, best_config

# ===================================================================
# 🎨 Quick Results Plotting
# ===================================================================

def plot_quick_results(history, config, save_path=None):
    """Quick結果プロット"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    epochs = list(range(1, len(history['total_loss']) + 1))
    
    # 1. Loss curves
    axes[0, 0].plot(epochs, history['total_loss'], 'b-', label='Total Loss', alpha=0.8)
    axes[0, 0].plot(epochs, history['spectral_dim_loss'], 'r-', label='Spectral Dim', alpha=0.7)
    axes[0, 0].set_title('🔥 Quick Loss Evolution')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Spectral dimension
    axes[0, 1].plot(epochs, history['spectral_dim_estimates'], 'g-', linewidth=2, label='Estimated')
    axes[0, 1].axhline(y=config.target_spectral_dim, color='orange', linestyle='--', label='Target (4.0)')
    axes[0, 1].fill_between(epochs, 
                           config.target_spectral_dim - config.spectral_dim_tolerance,
                           config.target_spectral_dim + config.spectral_dim_tolerance,
                           alpha=0.2, color='orange', label='Tolerance')
    axes[0, 1].set_title('🎯 Spectral Dimension Evolution')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Spectral Dimension')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. θ parameter
    axes[1, 0].plot(epochs, history['theta_values'], 'purple', linewidth=2)
    axes[1, 0].set_title('📐 θ Parameter Evolution')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('θ Value')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Performance metrics
    if history['gpu_memory_usage']:
        axes[1, 1].plot(epochs, history['gpu_memory_usage'], 'cyan', linewidth=2, label='GPU Memory (GB)')
    axes[1, 1].plot(epochs, history['training_speed'], 'red', linewidth=2, label='Speed (s/epoch)')
    axes[1, 1].set_title('⚡ Performance Metrics')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 結果プロット保存: {save_path}")
    
    plt.show()

# ===================================================================
# 🚀 Main Execution
# ===================================================================

def main():
    """Main execution with quick optimization"""
    import argparse
    
    # コマンドライン引数
    parser = argparse.ArgumentParser(description='Quick NKAT Deep Learning Optimization')
    parser.add_argument('--resume', action='store_true', help='チェックポイントから再開')
    parser.add_argument('--checkpoint-freq', type=int, default=5, help='チェックポイント頻度')
    parser.add_argument('--emergency-interval', type=int, default=30, help='緊急保存間隔（分）')
    parser.add_argument('--checkpoint-dir', type=str, default='./nkat_checkpoints_quick', help='チェックポイントディレクトリ')
    parser.add_argument('--no-optuna', action='store_true', help='Optuna最適化をスキップ')
    parser.add_argument('--epochs', type=int, default=50, help='学習エポック数（Quick版デフォルト）')
    parser.add_argument('--batch-size', type=int, default=24, help='バッチサイズ')
    parser.add_argument('--trials', type=int, default=10, help='Optuna試行回数（Quick版）')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🌌 NKAT Quick Optimization (軽量・高速版)")
    print("=" * 80)
    
    # 設定
    config = HybridNKATConfig()
    config.resume_from_checkpoint = args.resume
    config.checkpoint_freq = args.checkpoint_freq
    config.emergency_save_interval = args.emergency_interval
    config.checkpoint_dir = args.checkpoint_dir
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.n_trials = args.trials
    
    print(f"🔧 Quick設定:")
    print(f"   レジューム: {'ON' if config.resume_from_checkpoint else 'OFF'}")
    print(f"   エポック: {config.num_epochs}")
    print(f"   バッチサイズ: {config.batch_size}")
    print(f"   Optuna試行: {config.n_trials}回 {'(スキップ)' if args.no_optuna else ''}")
    print(f"   タイムアウト: {config.optuna_timeout}秒")
    
    # GPU確認
    if torch.cuda.is_available():
        print(f"💾 GPU使用量: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    start_time = time.time()
    
    try:
        # Quick training
        model, history, final_config = train_quick_nkat(
            config, 
            use_optuna=not args.no_optuna
        )
        
        # 結果分析
        elapsed_time = time.time() - start_time
        final_spec_dim = history['spectral_dim_estimates'][-1]
        final_theta = history['theta_values'][-1]
        improvement = (6.05 - final_spec_dim) / 6.05 * 100
        convergence_achieved = abs(final_spec_dim - config.target_spectral_dim) < config.spectral_dim_tolerance
        
        print("\n" + "="*80)
        print("🏆 Quick最適化完了！")
        print("="*80)
        print(f"⏱️ 実行時間: {elapsed_time/60:.1f} 分")
        print(f"🎯 最終スペクトル次元: {final_spec_dim:.3f} (目標: {config.target_spectral_dim})")
        print(f"📐 最終θ値: {final_theta:.2e}")
        print(f"📈 改善度: {improvement:.1f}%")
        print(f"✅ 収束判定: {'成功' if convergence_achieved else '継続推奨'}")
        
        # 結果保存
        results_path = os.path.join(work_dir, 'nkat_quick_results.json')
        final_metrics = {
            'final_spectral_dimension': final_spec_dim,
            'target_spectral_dimension': config.target_spectral_dim,
            'final_theta': final_theta,
            'improvement_percentage': improvement,
            'convergence_achieved': convergence_achieved,
            'training_time_minutes': elapsed_time / 60
        }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(final_metrics, f, indent=2, ensure_ascii=False)
        print(f"📊 結果保存: {results_path}")
        
        # プロット生成
        plot_path = os.path.join(work_dir, 'nkat_quick_results.png')
        plot_quick_results(history, final_config, plot_path)
        
        # 継続推奨メッセージ
        if not convergence_achieved:
            print("\n🔄 学習継続推奨:")
            print(f"   py -3 {os.path.basename(__file__)} --resume --epochs 100")
        else:
            print("\n🎊 最適化成功！次は通常版での詳細実行を推奨")
            
    except Exception as e:
        print(f"❌ エラー発生: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 GPUメモリクリーンアップ完了")
    
    print("\n🎊 Quick最適化完了！")

if __name__ == "__main__":
    main()
