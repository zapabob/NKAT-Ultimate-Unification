#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 NKAT理論統合PI-KANネットワーク
Physics-Informed Kolmogorov-Arnold Network with NKAT Theory

Author: NKAT Research Team
Date: 2025-05-24
Version: 1.0 - Deep Learning Integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict, Any, Callable
import warnings
from pathlib import Path
import json
import time
from dataclasses import dataclass
from tqdm import tqdm, trange
import logging
from collections import defaultdict
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

@dataclass
class PIKANConfig:
    """PI-KANネットワーク設定"""
    input_dim: int = 4  # 時空次元
    hidden_dims: List[int] = None
    output_dim: int = 1
    num_basis_functions: int = 10
    theta_parameter: float = 1e-25
    kappa_parameter: float = 1e-15
    physics_weight: float = 1.0
    learning_rate: float = 1e-3
    batch_size: int = 64
    num_epochs: int = 1000
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 128, 64]

class NKATBasisFunction(nn.Module):
    """
    NKAT理論に基づく基底関数
    
    非可換時空構造を反映した基底関数の実装
    """
    
    def __init__(self, input_dim: int, num_basis: int, theta: float, kappa: float):
        super().__init__()
        self.input_dim = input_dim
        self.num_basis = num_basis
        self.theta = theta
        self.kappa = kappa
        
        # 基底関数のパラメータ
        self.centers = nn.Parameter(torch.randn(num_basis, input_dim))
        self.scales = nn.Parameter(torch.ones(num_basis, input_dim))
        self.weights = nn.Parameter(torch.randn(num_basis))
        
        # NKAT理論パラメータ
        self.theta_tensor = nn.Parameter(torch.tensor(theta, dtype=torch.float32))
        self.kappa_tensor = nn.Parameter(torch.tensor(kappa, dtype=torch.float32))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        NKAT修正を含む基底関数の計算
        
        Args:
            x: 入力座標 [batch_size, input_dim]
        
        Returns:
            基底関数の値 [batch_size, num_basis]
        """
        batch_size = x.shape[0]
        
        # 標準的なガウス基底関数
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)  # [batch_size, num_basis, input_dim]
        scaled_diff = diff / (self.scales.unsqueeze(0) + 1e-8)
        gaussian = torch.exp(-0.5 * torch.sum(scaled_diff**2, dim=2))  # [batch_size, num_basis]
        
        # NKAT理論による修正
        # θ-変形：非可換性による補正
        if self.input_dim >= 2:
            # [x, y] = iθ の効果
            x_coord = x[:, 0].unsqueeze(1)  # [batch_size, 1]
            y_coord = x[:, 1].unsqueeze(1) if self.input_dim > 1 else torch.zeros_like(x_coord)
            
            theta_correction = self.theta_tensor * x_coord * y_coord
            theta_modification = torch.exp(theta_correction)  # [batch_size, 1]
            
            gaussian = gaussian * theta_modification
        
        # κ-変形：Minkowski時空の変形
        if self.input_dim >= 4:
            # 時間座標（通常は最初の座標）
            t_coord = x[:, 0].unsqueeze(1)
            
            # 空間座標の二乗和
            spatial_coords = x[:, 1:]
            spatial_norm_sq = torch.sum(spatial_coords**2, dim=1, keepdim=True)
            
            # Minkowski計量の修正
            kappa_correction = self.kappa_tensor * (t_coord**2 - spatial_norm_sq)
            kappa_modification = torch.exp(kappa_correction)  # [batch_size, 1]
            
            gaussian = gaussian * kappa_modification
        
        # 重み付き和
        output = gaussian * self.weights.unsqueeze(0)  # [batch_size, num_basis]
        
        return output

class KolmogorovArnoldLayer(nn.Module):
    """
    Kolmogorov-Arnold表現に基づく層
    
    f(x) = Σ_i φ_i(Σ_j ψ_{i,j}(x_j))
    """
    
    def __init__(self, input_dim: int, output_dim: int, num_basis: int, 
                 theta: float, kappa: float):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_basis = num_basis
        
        # 内側の関数 ψ_{i,j}(x_j)
        self.inner_functions = nn.ModuleList([
            NKATBasisFunction(1, num_basis, theta, kappa) 
            for _ in range(input_dim)
        ])
        
        # 外側の関数 φ_i
        self.outer_function = NKATBasisFunction(input_dim * num_basis, output_dim, theta, kappa)
        
        # 線形変換
        self.linear = nn.Linear(output_dim * self.outer_function.num_basis, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Kolmogorov-Arnold表現の計算
        """
        batch_size = x.shape[0]
        
        # 各入力次元に対して内側の関数を適用
        inner_outputs = []
        for i in range(self.input_dim):
            x_i = x[:, i:i+1]  # [batch_size, 1]
            inner_out = self.inner_functions[i](x_i)  # [batch_size, num_basis]
            inner_outputs.append(inner_out)
        
        # 内側の関数の出力を結合
        combined_inner = torch.cat(inner_outputs, dim=1)  # [batch_size, input_dim * num_basis]
        
        # 外側の関数を適用
        outer_output = self.outer_function(combined_inner)  # [batch_size, output_dim * num_basis]
        
        # 線形変換で最終出力
        output = self.linear(outer_output)  # [batch_size, output_dim]
        
        return output

class PIKANNetwork(nn.Module):
    """
    Physics-Informed Kolmogorov-Arnold Network
    
    NKAT理論の物理法則を組み込んだKANネットワーク
    """
    
    def __init__(self, config: PIKANConfig):
        super().__init__()
        self.config = config
        
        # KANレイヤーの構築
        self.kan_layers = nn.ModuleList()
        
        # 入力層
        current_dim = config.input_dim
        for hidden_dim in config.hidden_dims:
            layer = KolmogorovArnoldLayer(
                current_dim, hidden_dim, config.num_basis_functions,
                config.theta_parameter, config.kappa_parameter
            )
            self.kan_layers.append(layer)
            current_dim = hidden_dim
        
        # 出力層
        output_layer = KolmogorovArnoldLayer(
            current_dim, config.output_dim, config.num_basis_functions,
            config.theta_parameter, config.kappa_parameter
        )
        self.kan_layers.append(output_layer)
        
        # 物理制約項の重み
        self.physics_weight = config.physics_weight
        
        logger.info(f"🧠 PI-KANネットワーク初期化完了: {len(self.kan_layers)}層")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ネットワークの順伝播
        """
        for layer in self.kan_layers:
            x = layer(x)
            x = F.relu(x)  # 活性化関数
        
        return x
    
    def compute_physics_loss(self, x: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        NKAT理論に基づく物理制約損失の計算
        """
        batch_size = x.shape[0]
        
        # 勾配の計算（自動微分）
        x.requires_grad_(True)
        y = self.forward(x)
        
        # 1階微分
        grad_outputs = torch.ones_like(y)
        gradients = torch.autograd.grad(
            outputs=y, inputs=x, grad_outputs=grad_outputs,
            create_graph=True, retain_graph=True
        )[0]
        
        # 2階微分（ラプラシアン）
        laplacian = 0
        for i in range(x.shape[1]):
            grad_i = gradients[:, i]
            grad2_i = torch.autograd.grad(
                outputs=grad_i, inputs=x, grad_outputs=torch.ones_like(grad_i),
                create_graph=True, retain_graph=True
            )[0][:, i]
            laplacian += grad2_i
        
        # NKAT理論による修正項
        theta = self.config.theta_parameter
        kappa = self.config.kappa_parameter
        
        # θ-変形による非可換補正
        if x.shape[1] >= 2:
            x_coord = x[:, 0]
            y_coord = x[:, 1]
            theta_term = theta * (x_coord * gradients[:, 1] - y_coord * gradients[:, 0])
        else:
            theta_term = torch.zeros(batch_size, device=x.device)
        
        # κ-変形によるMinkowski補正
        if x.shape[1] >= 4:
            t_coord = x[:, 0]
            spatial_coords = x[:, 1:]
            spatial_laplacian = torch.sum(gradients[:, 1:]**2, dim=1)
            kappa_term = kappa * (gradients[:, 0]**2 - spatial_laplacian)
        else:
            kappa_term = torch.zeros(batch_size, device=x.device)
        
        # 修正されたPDE: (∇² + θ-項 + κ-項)u = 0
        pde_residual = laplacian + theta_term + kappa_term
        
        # 物理制約損失
        physics_loss = torch.mean(pde_residual**2)
        
        return physics_loss
    
    def compute_total_loss(self, x: torch.Tensor, y_true: torch.Tensor, 
                          y_pred: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        総損失の計算（データ損失 + 物理制約損失）
        """
        # データ損失（MSE）
        data_loss = F.mse_loss(y_pred, y_true)
        
        # 物理制約損失
        physics_loss = self.compute_physics_loss(x, y_pred)
        
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
        NKAT理論の解析解（近似）
        """
        # 簡単な解析解の例：修正された調和振動子
        if x.shape[1] >= 2:
            x_coord = x[:, 0]
            y_coord = x[:, 1]
            
            # 標準的な解
            standard_solution = torch.exp(-(x_coord**2 + y_coord**2) / 2)
            
            # NKAT修正
            theta_correction = self.theta * x_coord * y_coord
            kappa_correction = self.kappa * (x_coord**2 - y_coord**2)
            
            modified_solution = standard_solution * torch.exp(theta_correction + kappa_correction)
            
            return modified_solution.unsqueeze(1)
        else:
            # 1次元の場合
            x_coord = x[:, 0]
            solution = torch.exp(-x_coord**2 / 2) * (1 + self.theta * x_coord + self.kappa * x_coord**2)
            return solution.unsqueeze(1)
    
    def generate_training_data(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        訓練データの生成
        """
        # ランダムな入力点の生成
        x = torch.randn(num_samples, self.config.input_dim, device=device) * 2
        
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.8)
    
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
        scheduler.step(total_loss)
        
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
        correlation = np.corrcoef(y_true_np, y_pred_np)[0, 1]
        
        # 物理制約の満足度
        physics_loss = network.compute_physics_loss(x_test, y_pred).item()
    
    evaluation_results = {
        'mse': mse,
        'mae': mae,
        'correlation': correlation,
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
    print("🎯 NKAT理論統合PI-KANネットワーク")
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
        num_basis_functions=8,
        theta_parameter=1e-3,  # 可視化のため大きめの値
        kappa_parameter=1e-2,
        physics_weight=0.1,
        learning_rate=1e-3,
        batch_size=64,
        num_epochs=500
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
        num_basis_functions=10,
        theta_parameter=1e-4,
        kappa_parameter=1e-3,
        physics_weight=0.2,
        learning_rate=1e-3,
        batch_size=32,
        num_epochs=300
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
    with open('pikan_network_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    
    print("\n💾 結果を 'pikan_network_results.json' に保存しました")
    
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
        
        # 値をバーの上に表示
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{height:.4f}', ha='center', va='bottom')
        
        # 物理制約違反比較
        physics_violations = [eval_results_2d['physics_constraint_violation'], 
                            eval_results_4d['physics_constraint_violation']]
        
        bars = ax4.bar(problems, physics_violations, alpha=0.7, color=['blue', 'red'])
        ax4.set_ylabel('物理制約違反')
        ax4.set_title('物理制約満足度')
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        
        # 値をバーの上に表示
        for bar, value in zip(bars, physics_violations):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                    f'{value:.2e}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('pikan_network_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 グラフを 'pikan_network_analysis.png' に保存しました")
        plt.show()
        
    except Exception as e:
        logger.warning(f"⚠️ 可視化エラー: {e}")
    
    # 6. ネットワークの保存
    torch.save(network_2d.state_dict(), 'pikan_2d_model.pth')
    torch.save(network_4d.state_dict(), 'pikan_4d_model.pth')
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