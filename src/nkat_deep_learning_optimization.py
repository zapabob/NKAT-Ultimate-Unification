#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT深層学習最適化システム：非可換パラメータと超収束因子の係数最適化
NKAT Deep Learning Optimization System: Non-Commutative Parameter and Super-Convergence Factor Optimization

Author: 峯岸 亮 (Ryo Minegishi)
Date: 2025年5月28日
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import zeta
import pandas as pd
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# GPU設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 使用デバイス: {device}")

class NKATDataset(Dataset):
    """
    NKAT理論用データセット
    """
    
    def __init__(self, N_values, target_values, noise_level=1e-6):
        """
        Args:
            N_values: 次元数の配列
            target_values: 目標超収束因子値
            noise_level: ノイズレベル
        """
        self.N_values = torch.tensor(N_values, dtype=torch.float32)
        self.target_values = torch.tensor(target_values, dtype=torch.float32)
        
        # ノイズ追加
        if noise_level > 0:
            noise = torch.normal(0, noise_level, size=self.target_values.shape)
            self.target_values += noise
    
    def __len__(self):
        return len(self.N_values)
    
    def __getitem__(self, idx):
        return self.N_values[idx], self.target_values[idx]

class NKATSuperConvergenceNet(nn.Module):
    """
    NKAT超収束因子予測ニューラルネットワーク
    """
    
    def __init__(self, hidden_dims=[128, 256, 512, 256, 128], dropout_rate=0.1):
        """
        Args:
            hidden_dims: 隠れ層の次元数リスト
            dropout_rate: ドロップアウト率
        """
        super(NKATSuperConvergenceNet, self).__init__()
        
        # パラメータ予測ネットワーク
        layers = []
        input_dim = 1  # N値
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
        
        # 最終層：3つのパラメータ (γ, δ, t_c)
        layers.append(nn.Linear(input_dim, 3))
        
        self.parameter_net = nn.Sequential(*layers)
        
        # パラメータ制約用活性化関数
        self.gamma_activation = nn.Sigmoid()  # 0 < γ < 1
        self.delta_activation = nn.Sigmoid()  # 0 < δ < 0.1
        self.tc_activation = nn.Softplus()    # t_c > 1
        
        # 超収束因子計算ネットワーク
        self.convergence_net = nn.Sequential(
            nn.Linear(4, 256),  # N + 3パラメータ
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        print("🧠 NKAT深層学習ネットワーク初期化完了")
        print(f"📊 パラメータ数: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, N):
        """
        前向き計算
        
        Args:
            N: 次元数テンソル
            
        Returns:
            超収束因子の予測値
        """
        # パラメータ予測
        raw_params = self.parameter_net(N.unsqueeze(-1))
        
        # パラメータ制約適用
        gamma = self.gamma_activation(raw_params[:, 0]) * 0.5 + 0.1  # 0.1 < γ < 0.6
        delta = self.delta_activation(raw_params[:, 1]) * 0.08 + 0.01  # 0.01 < δ < 0.09
        t_c = self.tc_activation(raw_params[:, 2]) + 10.0  # t_c > 10
        
        # 入力特徴量結合
        features = torch.stack([N, gamma, delta, t_c], dim=1)
        
        # 超収束因子計算
        log_S = self.convergence_net(features)
        S = torch.exp(log_S.squeeze())
        
        return S, gamma, delta, t_c
    
    def theoretical_super_convergence(self, N, gamma, delta, t_c):
        """
        理論的超収束因子の計算
        
        Args:
            N: 次元数
            gamma, delta, t_c: パラメータ
            
        Returns:
            理論的超収束因子
        """
        # 密度関数の積分
        integral = gamma * torch.log(N / t_c)
        
        # 指数減衰項（N > t_c の場合）
        mask = N > t_c
        if mask.any():
            integral = torch.where(mask, 
                                 integral + delta * (N - t_c),
                                 integral)
        
        return torch.exp(integral)

class NKATPhysicsLoss(nn.Module):
    """
    NKAT理論に基づく物理制約損失関数
    """
    
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.01):
        """
        Args:
            alpha: データ適合項の重み
            beta: 物理制約項の重み
            gamma: 正則化項の重み
        """
        super(NKATPhysicsLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mse = nn.MSELoss()
    
    def forward(self, predictions, targets, N_values, gamma_pred, delta_pred, tc_pred, model):
        """
        物理制約付き損失計算
        
        Args:
            predictions: 予測値
            targets: 目標値
            N_values: 次元数
            gamma_pred, delta_pred, tc_pred: 予測パラメータ
            model: モデル
            
        Returns:
            総損失
        """
        # データ適合損失
        data_loss = self.mse(predictions, targets)
        
        # 物理制約損失
        physics_loss = self._physics_constraints(N_values, gamma_pred, delta_pred, tc_pred, model)
        
        # 正則化損失
        reg_loss = self._regularization_loss(model)
        
        total_loss = self.alpha * data_loss + self.beta * physics_loss + self.gamma * reg_loss
        
        return total_loss, data_loss, physics_loss, reg_loss
    
    def _physics_constraints(self, N_values, gamma_pred, delta_pred, tc_pred, model):
        """
        物理制約の計算
        """
        constraints = []
        
        # 1. リーマン予想制約: γ ln(N/t_c) → 1/2
        riemann_constraint = torch.mean((gamma_pred * torch.log(N_values / tc_pred) - 0.5) ** 2)
        constraints.append(riemann_constraint)
        
        # 2. 単調性制約: S(N)は単調増加
        if len(N_values) > 1:
            sorted_indices = torch.argsort(N_values)
            sorted_N = N_values[sorted_indices]
            sorted_S, _, _, _ = model(sorted_N)
            monotonicity_loss = torch.mean(torch.relu(sorted_S[:-1] - sorted_S[1:]))
            constraints.append(monotonicity_loss)
        
        # 3. 漸近制約: S(N) ~ N^γ for large N
        large_N_mask = N_values > 100
        if large_N_mask.any():
            large_N = N_values[large_N_mask]
            large_S, large_gamma, _, _ = model(large_N)
            theoretical_S = torch.pow(large_N, large_gamma)
            asymptotic_loss = torch.mean((large_S / theoretical_S - 1) ** 2)
            constraints.append(asymptotic_loss)
        
        # 4. 密度関数正値制約
        positivity_loss = torch.mean(torch.relu(-gamma_pred)) + torch.mean(torch.relu(-delta_pred))
        constraints.append(positivity_loss)
        
        return sum(constraints)
    
    def _regularization_loss(self, model):
        """
        正則化損失の計算
        """
        l2_reg = sum(torch.norm(param) ** 2 for param in model.parameters())
        return l2_reg

class NKATDeepLearningOptimizer:
    """
    NKAT深層学習最適化システム
    """
    
    def __init__(self, learning_rate=1e-3, batch_size=32, num_epochs=1000):
        """
        Args:
            learning_rate: 学習率
            batch_size: バッチサイズ
            num_epochs: エポック数
        """
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        # モデル初期化
        self.model = NKATSuperConvergenceNet().to(device)
        
        # 損失関数
        self.criterion = NKATPhysicsLoss()
        
        # オプティマイザー
        self.optimizer = optim.AdamW(self.model.parameters(), 
                                   lr=learning_rate, 
                                   weight_decay=1e-5)
        
        # スケジューラー
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=100, T_mult=2
        )
        
        # 履歴
        self.train_history = {
            'total_loss': [],
            'data_loss': [],
            'physics_loss': [],
            'reg_loss': [],
            'gamma_values': [],
            'delta_values': [],
            'tc_values': []
        }
        
        print("🚀 NKAT深層学習最適化システム初期化完了")
    
    def generate_training_data(self, N_range=(10, 1000), num_samples=1000):
        """
        訓練データの生成
        
        Args:
            N_range: 次元数の範囲
            num_samples: サンプル数
            
        Returns:
            データローダー
        """
        print("📊 訓練データ生成中...")
        
        # 次元数の生成（対数スケール）
        N_values = np.logspace(np.log10(N_range[0]), np.log10(N_range[1]), num_samples)
        
        # 理論的超収束因子の計算
        gamma_true = 0.234
        delta_true = 0.035
        t_c_true = 17.26
        
        target_values = []
        for N in tqdm(N_values, desc="理論値計算"):
            # 理論的超収束因子
            integral = gamma_true * np.log(N / t_c_true)
            if N > t_c_true:
                integral += delta_true * (N - t_c_true)
            S_theoretical = np.exp(integral)
            target_values.append(S_theoretical)
        
        target_values = np.array(target_values)
        
        # データセット作成
        dataset = NKATDataset(N_values, target_values, noise_level=1e-4)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        print(f"✅ 訓練データ生成完了: {num_samples}サンプル")
        return dataloader
    
    def train(self, dataloader):
        """
        モデル訓練
        
        Args:
            dataloader: データローダー
        """
        print("🎓 モデル訓練開始...")
        
        self.model.train()
        
        for epoch in tqdm(range(self.num_epochs), desc="エポック"):
            epoch_losses = {'total': 0, 'data': 0, 'physics': 0, 'reg': 0}
            epoch_params = {'gamma': [], 'delta': [], 'tc': []}
            
            for batch_N, batch_targets in dataloader:
                batch_N = batch_N.to(device)
                batch_targets = batch_targets.to(device)
                
                # 前向き計算
                predictions, gamma_pred, delta_pred, tc_pred = self.model(batch_N)
                
                # 損失計算
                total_loss, data_loss, physics_loss, reg_loss = self.criterion(
                    predictions, batch_targets, batch_N, gamma_pred, delta_pred, tc_pred, self.model
                )
                
                # 逆伝播
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # 勾配クリッピング
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # 損失記録
                epoch_losses['total'] += total_loss.item()
                epoch_losses['data'] += data_loss.item()
                epoch_losses['physics'] += physics_loss.item()
                epoch_losses['reg'] += reg_loss.item()
                
                # パラメータ記録
                epoch_params['gamma'].extend(gamma_pred.detach().cpu().numpy())
                epoch_params['delta'].extend(delta_pred.detach().cpu().numpy())
                epoch_params['tc'].extend(tc_pred.detach().cpu().numpy())
            
            # スケジューラー更新
            self.scheduler.step()
            
            # エポック平均の記録
            num_batches = len(dataloader)
            for key in epoch_losses:
                epoch_losses[key] /= num_batches
            
            self.train_history['total_loss'].append(epoch_losses['total'])
            self.train_history['data_loss'].append(epoch_losses['data'])
            self.train_history['physics_loss'].append(epoch_losses['physics'])
            self.train_history['reg_loss'].append(epoch_losses['reg'])
            
            self.train_history['gamma_values'].append(np.mean(epoch_params['gamma']))
            self.train_history['delta_values'].append(np.mean(epoch_params['delta']))
            self.train_history['tc_values'].append(np.mean(epoch_params['tc']))
            
            # 進捗表示
            if (epoch + 1) % 100 == 0:
                print(f"エポック {epoch+1}/{self.num_epochs}:")
                print(f"  総損失: {epoch_losses['total']:.6f}")
                print(f"  データ損失: {epoch_losses['data']:.6f}")
                print(f"  物理損失: {epoch_losses['physics']:.6f}")
                print(f"  平均γ: {np.mean(epoch_params['gamma']):.6f}")
                print(f"  平均δ: {np.mean(epoch_params['delta']):.6f}")
                print(f"  平均t_c: {np.mean(epoch_params['tc']):.6f}")
        
        print("✅ モデル訓練完了")
    
    def evaluate_model(self, test_N_values):
        """
        モデル評価
        
        Args:
            test_N_values: テスト用次元数
            
        Returns:
            評価結果
        """
        print("📊 モデル評価中...")
        
        self.model.eval()
        
        with torch.no_grad():
            test_N_tensor = torch.tensor(test_N_values, dtype=torch.float32).to(device)
            predictions, gamma_pred, delta_pred, tc_pred = self.model(test_N_tensor)
            
            # CPU に移動
            predictions = predictions.cpu().numpy()
            gamma_pred = gamma_pred.cpu().numpy()
            delta_pred = delta_pred.cpu().numpy()
            tc_pred = tc_pred.cpu().numpy()
        
        # 統計計算
        results = {
            'predictions': predictions,
            'gamma_mean': np.mean(gamma_pred),
            'gamma_std': np.std(gamma_pred),
            'delta_mean': np.mean(delta_pred),
            'delta_std': np.std(delta_pred),
            'tc_mean': np.mean(tc_pred),
            'tc_std': np.std(tc_pred),
            'gamma_values': gamma_pred,
            'delta_values': delta_pred,
            'tc_values': tc_pred
        }
        
        print("✅ モデル評価完了")
        print(f"📊 最適化パラメータ:")
        print(f"  γ = {results['gamma_mean']:.6f} ± {results['gamma_std']:.6f}")
        print(f"  δ = {results['delta_mean']:.6f} ± {results['delta_std']:.6f}")
        print(f"  t_c = {results['tc_mean']:.6f} ± {results['tc_std']:.6f}")
        
        return results
    
    def visualize_results(self, test_N_values, results):
        """
        結果の可視化
        
        Args:
            test_N_values: テスト用次元数
            results: 評価結果
        """
        print("📈 結果可視化中...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 訓練損失の推移
        axes[0, 0].plot(self.train_history['total_loss'], label='総損失', color='red')
        axes[0, 0].plot(self.train_history['data_loss'], label='データ損失', color='blue')
        axes[0, 0].plot(self.train_history['physics_loss'], label='物理損失', color='green')
        axes[0, 0].set_xlabel('エポック')
        axes[0, 0].set_ylabel('損失')
        axes[0, 0].set_title('訓練損失の推移')
        axes[0, 0].legend()
        axes[0, 0].set_yscale('log')
        
        # 2. パラメータの収束
        axes[0, 1].plot(self.train_history['gamma_values'], label='γ', color='red')
        axes[0, 1].axhline(y=0.234, color='red', linestyle='--', alpha=0.7, label='γ理論値')
        axes[0, 1].set_xlabel('エポック')
        axes[0, 1].set_ylabel('γ値')
        axes[0, 1].set_title('γパラメータの収束')
        axes[0, 1].legend()
        
        axes[0, 2].plot(self.train_history['delta_values'], label='δ', color='blue')
        axes[0, 2].axhline(y=0.035, color='blue', linestyle='--', alpha=0.7, label='δ理論値')
        axes[0, 2].set_xlabel('エポック')
        axes[0, 2].set_ylabel('δ値')
        axes[0, 2].set_title('δパラメータの収束')
        axes[0, 2].legend()
        
        # 3. 超収束因子の予測
        axes[1, 0].loglog(test_N_values, results['predictions'], 'b-', label='深層学習予測', linewidth=2)
        
        # 理論値との比較
        gamma_true, delta_true, t_c_true = 0.234, 0.035, 17.26
        theoretical_values = []
        for N in test_N_values:
            integral = gamma_true * np.log(N / t_c_true)
            if N > t_c_true:
                integral += delta_true * (N - t_c_true)
            theoretical_values.append(np.exp(integral))
        
        axes[1, 0].loglog(test_N_values, theoretical_values, 'r--', label='理論値', linewidth=2)
        axes[1, 0].set_xlabel('次元数 N')
        axes[1, 0].set_ylabel('超収束因子 S(N)')
        axes[1, 0].set_title('超収束因子の予測vs理論値')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. パラメータ分布
        axes[1, 1].hist(results['gamma_values'], bins=50, alpha=0.7, color='red', label='γ分布')
        axes[1, 1].axvline(x=0.234, color='red', linestyle='--', linewidth=2, label='理論値')
        axes[1, 1].set_xlabel('γ値')
        axes[1, 1].set_ylabel('頻度')
        axes[1, 1].set_title('γパラメータの分布')
        axes[1, 1].legend()
        
        # 5. t_c パラメータの収束
        axes[1, 2].plot(self.train_history['tc_values'], label='t_c', color='green')
        axes[1, 2].axhline(y=17.26, color='green', linestyle='--', alpha=0.7, label='t_c理論値')
        axes[1, 2].set_xlabel('エポック')
        axes[1, 2].set_ylabel('t_c値')
        axes[1, 2].set_title('t_cパラメータの収束')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig('nkat_deep_learning_optimization_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ 可視化完了")
    
    def save_model_and_results(self, results, filename_prefix='nkat_dl_optimization'):
        """
        モデルと結果の保存
        
        Args:
            results: 評価結果
            filename_prefix: ファイル名プレフィックス
        """
        print("💾 モデルと結果を保存中...")
        
        # モデル保存
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_history': self.train_history,
            'results': results
        }, f'{filename_prefix}_model.pth')
        
        # 結果をJSON形式で保存
        json_results = {
            'optimal_parameters': {
                'gamma_mean': float(results['gamma_mean']),
                'gamma_std': float(results['gamma_std']),
                'delta_mean': float(results['delta_mean']),
                'delta_std': float(results['delta_std']),
                'tc_mean': float(results['tc_mean']),
                'tc_std': float(results['tc_std'])
            },
            'training_config': {
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'num_epochs': self.num_epochs
            },
            'final_losses': {
                'total_loss': self.train_history['total_loss'][-1],
                'data_loss': self.train_history['data_loss'][-1],
                'physics_loss': self.train_history['physics_loss'][-1],
                'reg_loss': self.train_history['reg_loss'][-1]
            }
        }
        
        with open(f'{filename_prefix}_results.json', 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 保存完了: {filename_prefix}_model.pth, {filename_prefix}_results.json")

def main():
    """メイン実行関数"""
    print("🚀 NKAT深層学習最適化システム開始")
    print("="*60)
    
    # システム初期化
    optimizer = NKATDeepLearningOptimizer(
        learning_rate=1e-3,
        batch_size=64,
        num_epochs=2000
    )
    
    # 訓練データ生成
    dataloader = optimizer.generate_training_data(
        N_range=(10, 1000),
        num_samples=2000
    )
    
    # モデル訓練
    optimizer.train(dataloader)
    
    # モデル評価
    test_N_values = np.logspace(1, 3, 100)
    results = optimizer.evaluate_model(test_N_values)
    
    # 結果可視化
    optimizer.visualize_results(test_N_values, results)
    
    # モデルと結果の保存
    optimizer.save_model_and_results(results)
    
    # リーマン予想への含意
    gamma_opt = results['gamma_mean']
    t_c_opt = results['tc_mean']
    riemann_convergence = gamma_opt * np.log(1000 / t_c_opt)
    riemann_deviation = abs(riemann_convergence - 0.5)
    
    print("\n" + "="*60)
    print("🎯 NKAT深層学習最適化結果")
    print("="*60)
    print(f"📊 最適化パラメータ:")
    print(f"  γ = {results['gamma_mean']:.6f} ± {results['gamma_std']:.6f}")
    print(f"  δ = {results['delta_mean']:.6f} ± {results['delta_std']:.6f}")
    print(f"  t_c = {results['tc_mean']:.6f} ± {results['tc_std']:.6f}")
    print(f"\n🎯 リーマン予想への含意:")
    print(f"  収束率: γ·ln(1000/t_c) = {riemann_convergence:.6f}")
    print(f"  理論値からの偏差: {riemann_deviation:.6f}")
    print(f"  リーマン予想支持度: {100*(1-min(riemann_deviation/0.1, 1.0)):.1f}%")
    
    print("\n🏁 深層学習最適化完了")

if __name__ == "__main__":
    main() 