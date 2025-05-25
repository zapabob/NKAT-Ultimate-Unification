#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 NKAT理論統合PI-KANネットワーク（修正版）
Physics-Informed Kolmogorov-Arnold Network with NKAT Theory (Fixed)

Author: NKAT Research Team
Date: 2025-05-24
Version: 1.1 - Fixed Deep Learning Integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict, Any
import warnings
from pathlib import Path
import json
import time
from dataclasses import dataclass
from tqdm import tqdm
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

@dataclass
class PIKANConfig:
    """PI-KANネットワーク設定"""
    input_dim: int = 4
    hidden_dims: List[int] = None
    output_dim: int = 1
    theta_parameter: float = 1e-25
    kappa_parameter: float = 1e-15
    physics_weight: float = 1.0
    learning_rate: float = 1e-3
    batch_size: int = 64
    num_epochs: int = 500
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 128, 64]

class NKATActivation(nn.Module):
    """
    NKAT理論に基づく活性化関数
    """
    
    def __init__(self, theta: float, kappa: float):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(theta, dtype=torch.float32))
        self.kappa = nn.Parameter(torch.tensor(kappa, dtype=torch.float32))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        NKAT修正を含む活性化関数
        """
        # 標準的なReLU
        standard_activation = F.relu(x)
        
        # NKAT修正項
        theta_correction = self.theta * x * torch.sin(x)
        kappa_correction = self.kappa * x**2 * torch.cos(x)
        
        # 修正された活性化
        modified_activation = standard_activation + theta_correction + kappa_correction
        
        return modified_activation

class SimplifiedKANLayer(nn.Module):
    """
    簡略化されたKolmogorov-Arnold層
    """
    
    def __init__(self, input_dim: int, output_dim: int, theta: float, kappa: float):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 線形変換
        self.linear = nn.Linear(input_dim, output_dim)
        
        # NKAT活性化関数
        self.nkat_activation = NKATActivation(theta, kappa)
        
        # 非線形変換のための追加層
        self.nonlinear = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Tanh(),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        簡略化されたKAN層の計算
        """
        # 線形変換
        linear_out = self.linear(x)
        
        # NKAT活性化
        activated = self.nkat_activation(linear_out)
        
        # 非線形変換
        output = self.nonlinear(activated)
        
        return output

class PIKANNetwork(nn.Module):
    """
    簡略化されたPhysics-Informed Kolmogorov-Arnold Network
    """
    
    def __init__(self, config: PIKANConfig):
        super().__init__()
        self.config = config
        
        # ネットワーク層の構築
        self.layers = nn.ModuleList()
        
        # 入力層
        current_dim = config.input_dim
        for hidden_dim in config.hidden_dims:
            layer = SimplifiedKANLayer(
                current_dim, hidden_dim, 
                config.theta_parameter, config.kappa_parameter
            )
            self.layers.append(layer)
            current_dim = hidden_dim
        
        # 出力層
        self.output_layer = nn.Linear(current_dim, config.output_dim)
        
        # 物理制約項の重み
        self.physics_weight = config.physics_weight
        
        logger.info(f"🧠 簡略化PI-KANネットワーク初期化完了: {len(self.layers)+1}層")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ネットワークの順伝播
        """
        for layer in self.layers:
            x = layer(x)
        
        # 出力層
        output = self.output_layer(x)
        
        return output
    
    def compute_physics_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        NKAT理論に基づく物理制約損失の計算（簡略版）
        """
        x_copy = x.clone().detach().requires_grad_(True)
        y = self.forward(x_copy)
        
        # 1階微分の計算
        grad_outputs = torch.ones_like(y)
        try:
            gradients = torch.autograd.grad(
                outputs=y, inputs=x_copy, grad_outputs=grad_outputs,
                create_graph=True, retain_graph=True, allow_unused=True
            )[0]
            
            if gradients is None:
                return torch.tensor(0.0, device=x.device)
            
            # 簡略化された物理制約
            # NKAT理論による修正項
            theta = self.config.theta_parameter
            kappa = self.config.kappa_parameter
            
            # θ-変形による非可換補正
            if x.shape[1] >= 2:
                x_coord = x_copy[:, 0]
                y_coord = x_copy[:, 1]
                theta_term = theta * (x_coord * gradients[:, 1] - y_coord * gradients[:, 0])
            else:
                theta_term = torch.zeros(x.shape[0], device=x.device)
            
            # κ-変形による補正
            kappa_term = kappa * torch.sum(gradients**2, dim=1)
            
            # 物理制約残差
            pde_residual = theta_term + kappa_term
            
            # 物理制約損失
            physics_loss = torch.mean(pde_residual**2)
            
            return physics_loss
            
        except Exception as e:
            logger.warning(f"⚠️ 物理損失計算エラー: {e}")
            return torch.tensor(0.0, device=x.device)
    
    def compute_total_loss(self, x: torch.Tensor, y_true: torch.Tensor, 
                          y_pred: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        総損失の計算
        """
        # データ損失（MSE）
        data_loss = F.mse_loss(y_pred, y_true)
        
        # 物理制約損失
        physics_loss = self.compute_physics_loss(x)
        
        # 総損失
        total_loss = data_loss + self.physics_weight * physics_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'data_loss': data_loss.item(),
            'physics_loss': physics_loss.item()
        }
        
        return total_loss, loss_dict

class NKATDataGenerator:
    """
    NKAT理論に基づく訓練データ生成器
    """
    
    def __init__(self, config: PIKANConfig):
        self.config = config
        self.theta = config.theta_parameter
        self.kappa = config.kappa_parameter
        
    def generate_analytical_solution(self, x: torch.Tensor) -> torch.Tensor:
        """
        NKAT理論の解析解（簡略版）
        """
        if x.shape[1] >= 2:
            x_coord = x[:, 0]
            y_coord = x[:, 1]
            
            # 標準的な解
            standard_solution = torch.exp(-(x_coord**2 + y_coord**2) / 4)
            
            # NKAT修正（小さな補正）
            theta_correction = self.theta * 1e20 * x_coord * y_coord  # スケール調整
            kappa_correction = self.kappa * 1e10 * (x_coord**2 - y_coord**2)  # スケール調整
            
            modified_solution = standard_solution * (1 + theta_correction + kappa_correction)
            
            return modified_solution.unsqueeze(1)
        else:
            # 1次元の場合
            x_coord = x[:, 0]
            solution = torch.exp(-x_coord**2 / 4) * (1 + self.theta * 1e20 * x_coord)
            return solution.unsqueeze(1)
    
    def generate_training_data(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        訓練データの生成
        """
        # ランダムな入力点の生成（範囲を制限）
        x = torch.randn(num_samples, self.config.input_dim, device=device) * 1.5
        
        # 対応する解析解
        y = self.generate_analytical_solution(x)
        
        return x, y

def train_pikan_network(config: PIKANConfig) -> Tuple[PIKANNetwork, Dict]:
    """
    PI-KANネットワークの訓練
    """
    logger.info("🚀 PI-KANネットワーク訓練開始...")
    
    # ネットワークの初期化
    network = PIKANNetwork(config).to(device)
    
    # オプティマイザー
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    
    # データ生成器
    data_generator = NKATDataGenerator(config)
    
    # 訓練履歴
    history = {
        'total_loss': [],
        'data_loss': [],
        'physics_loss': [],
        'learning_rate': []
    }
    
    # 訓練ループ
    network.train()
    for epoch in tqdm(range(config.num_epochs), desc="PI-KAN訓練"):
        # バッチデータの生成
        x_batch, y_batch = data_generator.generate_training_data(config.batch_size)
        
        # 順伝播
        y_pred = network(x_batch)
        
        # 損失計算
        total_loss, loss_dict = network.compute_total_loss(x_batch, y_batch, y_pred)
        
        # 逆伝播
        optimizer.zero_grad()
        total_loss.backward()
        
        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # 履歴の記録
        for key, value in loss_dict.items():
            history[key].append(value)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # ログ出力
        if epoch % 100 == 0:
            logger.info(f"Epoch {epoch}: Total Loss = {total_loss:.6f}, "
                       f"Data Loss = {loss_dict['data_loss']:.6f}, "
                       f"Physics Loss = {loss_dict['physics_loss']:.6f}")
    
    logger.info("✅ PI-KANネットワーク訓練完了")
    return network, history

def evaluate_pikan_network(network: PIKANNetwork, config: PIKANConfig) -> Dict:
    """
    PI-KANネットワークの評価
    """
    logger.info("📊 PI-KANネットワーク評価開始...")
    
    network.eval()
    data_generator = NKATDataGenerator(config)
    
    # テストデータの生成
    x_test, y_true = data_generator.generate_training_data(1000)
    
    with torch.no_grad():
        y_pred = network(x_test)
        
        # 評価指標の計算
        mse = F.mse_loss(y_pred, y_true).item()
        mae = F.l1_loss(y_pred, y_true).item()
        
        # 相関係数
        y_true_np = y_true.cpu().numpy().flatten()
        y_pred_np = y_pred.cpu().numpy().flatten()
        
        # NaNチェック
        valid_mask = ~(np.isnan(y_true_np) | np.isnan(y_pred_np))
        if np.sum(valid_mask) > 1:
            correlation = np.corrcoef(y_true_np[valid_mask], y_pred_np[valid_mask])[0, 1]
        else:
            correlation = 0.0
        
        # 物理制約の満足度
        physics_loss = network.compute_physics_loss(x_test).item()
    
    evaluation_results = {
        'mse': mse,
        'mae': mae,
        'correlation': correlation if not np.isnan(correlation) else 0.0,
        'physics_constraint_violation': physics_loss,
        'test_samples': len(x_test)
    }
    
    logger.info(f"📊 評価結果: MSE={mse:.6f}, MAE={mae:.6f}, "
               f"相関={correlation:.4f}, 物理制約違反={physics_loss:.6f}")
    
    return evaluation_results

def demonstrate_pikan_applications():
    """
    PI-KANネットワークの応用デモンストレーション
    """
    print("=" * 80)
    print("🎯 NKAT理論統合PI-KANネットワーク（修正版）")
    print("=" * 80)
    print("📅 実行日時:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("🧠 深層学習: Physics-Informed Kolmogorov-Arnold Network")
    print("🔬 物理理論: NKAT (Non-commutative Kappa-deformed Algebra Theory)")
    print("=" * 80)
    
    all_results = {}
    
    # 1. 2次元NKAT問題
    print("\n🔍 1. 2次元NKAT問題の解決")
    print("問題：非可換時空での場の方程式")
    
    config_2d = PIKANConfig(
        input_dim=2,
        hidden_dims=[32, 64, 32],
        output_dim=1,
        theta_parameter=1e-25,
        kappa_parameter=1e-15,
        physics_weight=0.01,
        learning_rate=1e-3,
        batch_size=64,
        num_epochs=300
    )
    
    # 訓練
    network_2d, history_2d = train_pikan_network(config_2d)
    
    # 評価
    eval_results_2d = evaluate_pikan_network(network_2d, config_2d)
    
    print(f"✅ 2次元問題結果:")
    print(f"   MSE: {eval_results_2d['mse']:.6f}")
    print(f"   相関: {eval_results_2d['correlation']:.4f}")
    print(f"   物理制約違反: {eval_results_2d['physics_constraint_violation']:.6f}")
    
    all_results['2d_problem'] = {
        'config': config_2d.__dict__,
        'evaluation': eval_results_2d,
        'training_history': history_2d
    }
    
    # 2. 4次元時空問題
    print("\n🔍 2. 4次元時空NKAT問題の解決")
    print("問題：Minkowski時空での修正場の方程式")
    
    config_4d = PIKANConfig(
        input_dim=4,
        hidden_dims=[64, 128, 64],
        output_dim=1,
        theta_parameter=1e-25,
        kappa_parameter=1e-15,
        physics_weight=0.01,
        learning_rate=1e-3,
        batch_size=32,
        num_epochs=200
    )
    
    # 訓練
    network_4d, history_4d = train_pikan_network(config_4d)
    
    # 評価
    eval_results_4d = evaluate_pikan_network(network_4d, config_4d)
    
    print(f"✅ 4次元問題結果:")
    print(f"   MSE: {eval_results_4d['mse']:.6f}")
    print(f"   相関: {eval_results_4d['correlation']:.4f}")
    print(f"   物理制約違反: {eval_results_4d['physics_constraint_violation']:.6f}")
    
    all_results['4d_problem'] = {
        'config': config_4d.__dict__,
        'evaluation': eval_results_4d,
        'training_history': history_4d
    }
    
    # 3. 統合結果の表示
    print("\n📊 3. PI-KAN性能比較")
    print("=" * 50)
    
    problems = ['2次元', '4次元']
    mse_values = [eval_results_2d['mse'], eval_results_4d['mse']]
    correlations = [eval_results_2d['correlation'], eval_results_4d['correlation']]
    
    for i, problem in enumerate(problems):
        print(f"{problem}問題: MSE={mse_values[i]:.6f}, 相関={correlations[i]:.4f}")
    
    # 4. 結果の保存
    with open('pikan_network_results_fixed.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    
    print("\n💾 結果を 'pikan_network_results_fixed.json' に保存しました")
    
    # 5. 可視化
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 訓練履歴（2次元）
        epochs_2d = range(len(history_2d['total_loss']))
        ax1.plot(epochs_2d, history_2d['total_loss'], 'b-', label='総損失', linewidth=2)
        ax1.plot(epochs_2d, history_2d['data_loss'], 'g-', label='データ損失', linewidth=2)
        ax1.plot(epochs_2d, history_2d['physics_loss'], 'r-', label='物理制約損失', linewidth=2)
        ax1.set_xlabel('エポック')
        ax1.set_ylabel('損失')
        ax1.set_title('2次元PI-KAN訓練履歴')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 訓練履歴（4次元）
        epochs_4d = range(len(history_4d['total_loss']))
        ax2.plot(epochs_4d, history_4d['total_loss'], 'b-', label='総損失', linewidth=2)
        ax2.plot(epochs_4d, history_4d['data_loss'], 'g-', label='データ損失', linewidth=2)
        ax2.plot(epochs_4d, history_4d['physics_loss'], 'r-', label='物理制約損失', linewidth=2)
        ax2.set_xlabel('エポック')
        ax2.set_ylabel('損失')
        ax2.set_title('4次元PI-KAN訓練履歴')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # 性能比較
        metrics = ['MSE', '相関係数']
        values_2d = [eval_results_2d['mse'], eval_results_2d['correlation']]
        values_4d = [eval_results_4d['mse'], eval_results_4d['correlation']]
        
        x_pos = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax3.bar(x_pos - width/2, values_2d, width, label='2次元', alpha=0.7, color='blue')
        bars2 = ax3.bar(x_pos + width/2, values_4d, width, label='4次元', alpha=0.7, color='red')
        
        ax3.set_xlabel('評価指標')
        ax3.set_ylabel('値')
        ax3.set_title('PI-KAN性能比較')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(metrics)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 物理制約違反比較
        physics_violations = [eval_results_2d['physics_constraint_violation'], 
                            eval_results_4d['physics_constraint_violation']]
        
        bars = ax4.bar(problems, physics_violations, alpha=0.7, color=['blue', 'red'])
        ax4.set_ylabel('物理制約違反')
        ax4.set_title('物理制約満足度')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('pikan_network_analysis_fixed.png', dpi=300, bbox_inches='tight')
        print("📊 グラフを 'pikan_network_analysis_fixed.png' に保存しました")
        plt.show()
        
    except Exception as e:
        logger.warning(f"⚠️ 可視化エラー: {e}")
    
    # 6. ネットワークの保存
    torch.save(network_2d.state_dict(), 'pikan_2d_model_fixed.pth')
    torch.save(network_4d.state_dict(), 'pikan_4d_model_fixed.pth')
    print("💾 訓練済みモデルを保存しました")
    
    return all_results, network_2d, network_4d

if __name__ == "__main__":
    """
    PI-KANネットワークの実行
    """
    try:
        results, model_2d, model_4d = demonstrate_pikan_applications()
        print("🎉 PI-KANネットワークの実装と訓練が完了しました！")
    except Exception as e:
        logger.error(f"❌ 実行エラー: {e}")
        print(f"❌ エラーが発生しました: {e}") 