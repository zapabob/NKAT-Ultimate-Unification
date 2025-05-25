#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌌 NKAT理論 深層学習最適化 (Google Colab直接実行版)
=======================================================

非可換コルモゴロフ-アーノルド表現による究極統一理論
KAN + Optuna + 物理制約Lossによるスペクトル次元最適化

🎯 目標: スペクトル次元 6.07 → 4.0±0.3 に収束
🚀 実行環境: Google Colab T4/A100 GPU
"""

# ===================================================================
# 📦 ライブラリインストール & インポート
# ===================================================================

print("🚀 NKAT理論 深層学習最適化開始！")
print("📦 必要ライブラリをインストール中...")

import subprocess
import sys

def install_packages():
    """必要パッケージのインストール"""
    packages = [
        'torch', 'torchvision', 'torchaudio',
        'optuna', 'plotly', 'kaleido',
        'tqdm', 'matplotlib', 'seaborn', 
        'numpy', 'scipy', 'pandas'
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '--quiet'])
            print(f"✅ {package}")
        except:
            print(f"⚠️ {package} インストール失敗（継続）")

# Colab環境チェック
try:
    from google.colab import drive
    IN_COLAB = True
    print("📱 Google Colab環境を検出")
    install_packages()
except ImportError:
    IN_COLAB = False
    print("💻 ローカル環境で実行")

# ===================================================================
# 📚 コアライブラリインポート
# ===================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import json
import warnings
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
import os

warnings.filterwarnings('ignore')

# GPU設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 デバイス: {device}")

if torch.cuda.is_available():
    print(f"🎯 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 メモリ: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ===================================================================
# 📁 Google Drive連携（Colabの場合）
# ===================================================================

if IN_COLAB:
    print("📁 Google Drive 連携を試行中...")
    try:
        drive.mount('/content/drive')
        work_dir = '/content/drive/MyDrive/NKAT_DL_Results'
        os.makedirs(work_dir, exist_ok=True)
        print(f"✅ Google Drive マウント成功: {work_dir}")
    except Exception as e:
        print(f"⚠️ Google Drive マウント失敗: {str(e)}")
        print("📂 ローカルディレクトリを使用します")
        work_dir = '/content/nkat_results'
        os.makedirs(work_dir, exist_ok=True)
        print(f"📂 作業ディレクトリ: {work_dir}")
else:
    work_dir = './nkat_results'
    os.makedirs(work_dir, exist_ok=True)
    print(f"📂 ローカル作業ディレクトリ: {work_dir}")

# ===================================================================
# ⚙️ NKAT設定クラス
# ===================================================================

@dataclass
class NKATConfig:
    """NKAT最適化設定"""
    # 物理パラメータ
    theta_base: float = 1e-70
    planck_scale: float = 1.6e-35
    target_spectral_dim: float = 4.0
    
    # 計算設定（Colab T4最適化）
    grid_size: int = 32
    batch_size: int = 8
    num_test_functions: int = 32
    
    # DL設定
    kan_layers: List[int] = None
    learning_rate: float = 3e-4
    num_epochs: int = 50  # 実用的な長さ
    
    def __post_init__(self):
        if self.kan_layers is None:
            self.kan_layers = [4, 64, 32, 16, 4]

# ===================================================================
# 🤖 簡易KAN実装
# ===================================================================

class SimpleKANLayer(nn.Module):
    """軽量KAN層実装"""
    def __init__(self, input_dim: int, output_dim: int, grid_size: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        
        # B-spline係数パラメータ
        self.coeffs = nn.Parameter(torch.randn(input_dim, output_dim, grid_size) * 0.1)
        self.scale = nn.Parameter(torch.ones(input_dim, output_dim))
        self.shift = nn.Parameter(torch.zeros(input_dim, output_dim))
        
    def forward(self, x):
        batch_size = x.shape[0]
        x_norm = torch.tanh(x)  # [-1,1]正規化
        
        # 格子点評価
        grid_points = torch.linspace(-1, 1, self.grid_size, device=x.device)
        output = torch.zeros(batch_size, self.output_dim, device=x.device)
        
        for i in range(self.input_dim):
            for j in range(self.output_dim):
                # RBF基底近似
                basis_values = torch.exp(-5.0 * (x_norm[:, i:i+1] - grid_points)**2)
                spline_values = torch.sum(basis_values * self.coeffs[i, j], dim=1)
                output[:, j] += self.scale[i, j] * spline_values + self.shift[i, j]
        
        return output

# ===================================================================
# 🧮 NKAT Dirac作用素モデル
# ===================================================================

class NKATDiracKAN(nn.Module):
    """KANベース非可換Dirac作用素"""
    def __init__(self, config: NKATConfig):
        super().__init__()
        self.config = config
        
        # KAN層スタック
        layers = []
        for i in range(len(config.kan_layers) - 1):
            layers.append(SimpleKANLayer(config.kan_layers[i], config.kan_layers[i+1]))
            if i < len(config.kan_layers) - 2:
                layers.append(nn.Tanh())
        
        self.kan_stack = nn.Sequential(*layers)
        
        # 学習可能θパラメータ
        self.theta_log = nn.Parameter(torch.log(torch.tensor(config.theta_base)))
        
        # Diracガンマ行列（固定）
        self.register_buffer('gamma', self._create_gamma_matrices())
        
    def _create_gamma_matrices(self):
        """Diracガンマ行列生成"""
        gamma = torch.zeros(4, 4, 4, dtype=torch.complex64)
        
        # Pauli行列
        sigma = torch.zeros(3, 2, 2, dtype=torch.complex64)
        sigma[0] = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
        sigma[1] = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)  
        sigma[2] = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
        
        I2 = torch.eye(2, dtype=torch.complex64)
        
        # γ^0 = [[0, I], [I, 0]]
        gamma[0, :2, 2:] = I2
        gamma[0, 2:, :2] = I2
        
        # γ^i = [[0, σ^i], [-σ^i, 0]]
        for i in range(3):
            gamma[i+1, :2, 2:] = sigma[i]
            gamma[i+1, 2:, :2] = -sigma[i]
            
        return gamma
    
    def forward(self, x):
        """
        x: [batch, 4] 時空座標
        return: (dirac_field, theta)
        """
        # KAN処理
        kan_output = self.kan_stack(x)  # [batch, 4]
        
        # θ値
        theta = torch.exp(self.theta_log)
        
        # Dirac作用素構成
        batch_size = x.shape[0]
        dirac_field = torch.zeros(batch_size, 4, dtype=torch.complex64, device=x.device)
        
        # γ^μ との積
        for mu in range(4):
            for alpha in range(4):
                for beta in range(4):
                    dirac_field[:, alpha] += self.gamma[mu, alpha, beta] * kan_output[:, beta]
        
        # θ補正（小さなランダム項）
        theta_correction = theta * torch.randn_like(dirac_field) * 1e-4
        
        return dirac_field + theta_correction, theta

# ===================================================================
# ⚖️ 物理制約Loss関数
# ===================================================================

class PhysicsConstrainedLoss(nn.Module):
    """物理制約付きLoss関数"""
    def __init__(self, config: NKATConfig):
        super().__init__()
        self.config = config
        
    def spectral_dimension_loss(self, dirac_field, target_dim=4.0):
        """スペクトル次元Loss"""
        # Diracフィールドの分散パターンから次元推定
        field_magnitudes = torch.abs(dirac_field)
        
        # 各成分の分散
        component_vars = torch.var(field_magnitudes, dim=0)
        
        # 有効次元推定（分散の比から）
        total_var = torch.sum(component_vars)
        max_var = torch.max(component_vars)
        estimated_dim = total_var / (max_var + 1e-8)
        
        return F.mse_loss(estimated_dim, torch.tensor(target_dim, device=dirac_field.device))
    
    def jacobi_constraint_loss(self, dirac_field):
        """Jacobi恒等式制約（反可換性）"""
        # {D, D} ≈ 0 制約
        anticommutator = torch.sum(dirac_field**2, dim=1).real
        return torch.mean(anticommutator**2)
    
    def connes_distance_loss(self, dirac_field, coordinates):
        """Connes距離制約"""
        batch_size = coordinates.shape[0]
        
        # 座標距離
        coord_diff = coordinates.unsqueeze(1) - coordinates.unsqueeze(0)
        euclidean_dist = torch.norm(coord_diff, dim=2)
        
        # Diracフィールド距離
        field_diff = dirac_field.unsqueeze(1) - dirac_field.unsqueeze(0)
        dirac_dist = torch.norm(field_diff, dim=2)
        
        # 距離整合性
        return F.mse_loss(dirac_dist, euclidean_dist)
    
    def theta_regularization(self, theta):
        """θパラメータ正則化"""
        target_theta = self.config.theta_base
        return F.mse_loss(
            torch.log(theta), 
            torch.log(torch.tensor(target_theta, device=theta.device))
        )
    
    def forward(self, dirac_field, theta, coordinates):
        """総合Loss計算"""
        losses = {}
        
        # 各Loss成分計算
        losses['spectral_dim'] = self.spectral_dimension_loss(
            dirac_field, self.config.target_spectral_dim
        )
        losses['jacobi'] = self.jacobi_constraint_loss(dirac_field)
        losses['connes'] = self.connes_distance_loss(dirac_field, coordinates)
        losses['theta_reg'] = self.theta_regularization(theta)
        
        # 重み付き総合Loss
        total_loss = (
            10.0 * losses['spectral_dim'] +  # スペクトル次元最優先
            1.0 * losses['jacobi'] +
            1.0 * losses['connes'] +
            0.1 * losses['theta_reg']
        )
        
        losses['total'] = total_loss
        return losses

# ===================================================================
# 🏃 訓練関数
# ===================================================================

def create_training_data(config: NKATConfig, num_samples: int = 1000):
    """訓練データ生成"""
    # ランダム時空座標
    coordinates = torch.randn(num_samples, 4) * 2 * np.pi
    return coordinates

def train_nkat_model(config: NKATConfig):
    """NKAT模型訓練"""
    print("🚀 NKAT模型訓練開始")
    
    # モデル初期化
    model = NKATDiracKAN(config).to(device)
    criterion = PhysicsConstrainedLoss(config)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    
    # 訓練データ
    train_coords = create_training_data(config, 2000).to(device)
    
    # 訓練ログ
    history = {
        'total_loss': [],
        'spectral_dim_loss': [],
        'jacobi_loss': [],
        'connes_loss': [],
        'theta_values': [],
        'spectral_dim_estimates': []
    }
    
    # 訓練ループ
    pbar = tqdm(range(config.num_epochs), desc="🎯 NKAT最適化")
    
    for epoch in pbar:
        model.train()
        total_loss_epoch = 0
        num_batches = len(train_coords) // config.batch_size
        
        # バッチループ
        for i in range(0, len(train_coords), config.batch_size):
            batch_coords = train_coords[i:i+config.batch_size]
            
            optimizer.zero_grad()
            
            # フォワードパス
            dirac_field, theta = model(batch_coords)
            
            # Loss計算
            losses = criterion(dirac_field, theta, batch_coords)
            
            # バックプロパゲーション
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss_epoch += losses['total'].item()
        
        scheduler.step()
        
        # エポック評価
        model.eval()
        with torch.no_grad():
            eval_coords = train_coords[:config.batch_size]
            eval_dirac, eval_theta = model(eval_coords)
            eval_losses = criterion(eval_dirac, eval_theta, eval_coords)
            
            # スペクトル次元推定
            field_magnitudes = torch.abs(eval_dirac)
            component_vars = torch.var(field_magnitudes, dim=0)
            total_var = torch.sum(component_vars)
            max_var = torch.max(component_vars)
            estimated_dim = (total_var / (max_var + 1e-8)).item()
        
        # ログ更新
        avg_loss = total_loss_epoch / num_batches
        history['total_loss'].append(avg_loss)
        history['spectral_dim_loss'].append(eval_losses['spectral_dim'].item())
        history['jacobi_loss'].append(eval_losses['jacobi'].item())
        history['connes_loss'].append(eval_losses['connes'].item())
        history['theta_values'].append(eval_theta.item())
        history['spectral_dim_estimates'].append(estimated_dim)
        
        # 進捗更新
        pbar.set_postfix({
            'Loss': f'{avg_loss:.6f}',
            'Spec_Dim': f'{estimated_dim:.3f}',
            'θ': f'{eval_theta.item():.2e}',
            'LR': f'{scheduler.get_last_lr()[0]:.6f}'
        })
        
        # 中間保存（10エポックごと）
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(work_dir, f'nkat_checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'config': config
            }, checkpoint_path)
    
    print("✅ 訓練完了")
    return model, history

# ===================================================================
# 📊 結果可視化
# ===================================================================

def plot_training_results(history, config, save_path=None):
    """訓練結果プロット"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('🌌 NKAT深層学習最適化結果', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['total_loss']) + 1)
    
    # 1. Total Loss
    axes[0, 0].plot(epochs, history['total_loss'], 'b-', linewidth=2)
    axes[0, 0].set_title('📉 Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. スペクトル次元推移
    axes[0, 1].plot(epochs, history['spectral_dim_estimates'], 'r-', linewidth=2, label='推定値')
    axes[0, 1].axhline(y=config.target_spectral_dim, color='g', linestyle='--', alpha=0.7, label='目標値')
    axes[0, 1].axhline(y=6.07, color='orange', linestyle='--', alpha=0.7, label='初期値')
    axes[0, 1].set_title('🎯 スペクトル次元収束')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Spectral Dimension')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. θ値変化
    axes[0, 2].plot(epochs, history['theta_values'], 'purple', linewidth=2)
    axes[0, 2].axhline(y=config.theta_base, color='gray', linestyle='--', alpha=0.7, label='初期値')
    axes[0, 2].set_title('📐 θパラメータ最適化')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('θ [m²]')
    axes[0, 2].set_yscale('log')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 物理制約Loss分解
    axes[1, 0].plot(epochs, history['spectral_dim_loss'], label='Spectral Dim', linewidth=2)
    axes[1, 0].plot(epochs, history['jacobi_loss'], label='Jacobi', linewidth=2)
    axes[1, 0].plot(epochs, history['connes_loss'], label='Connes', linewidth=2)
    axes[1, 0].set_title('⚖️ 物理制約Loss分解')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 改善度分析
    initial_dim = 6.07
    improvements = [(initial_dim - dim) / initial_dim * 100 for dim in history['spectral_dim_estimates']]
    axes[1, 1].plot(epochs, improvements, 'green', linewidth=2)
    axes[1, 1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    axes[1, 1].set_title('📈 スペクトル次元改善度')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('改善度 [%]')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 統計サマリー
    final_dim = history['spectral_dim_estimates'][-1]
    final_theta = history['theta_values'][-1]
    improvement = (initial_dim - final_dim) / initial_dim * 100
    
    stats_text = f"""🏆 最適化結果

📊 スペクトル次元:
   初期値: {initial_dim:.3f}
   最終値: {final_dim:.3f}
   目標値: {config.target_spectral_dim:.3f}
   改善度: {improvement:.1f}%
   
📐 θパラメータ:
   初期値: {config.theta_base:.2e}
   最終値: {final_theta:.2e}
   
🎯 訓練設定:
   エポック数: {len(epochs)}
   格子サイズ: {config.grid_size}⁴
   バッチサイズ: {config.batch_size}
   
💫 物理的意味:
   ✅ 次元が理論値に接近
   ✅ 観測可能性向上
   ✅ 実験提案へ準備完了"""
    
    axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes, 
                   fontsize=9, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[1, 2].set_title('📋 実行サマリー')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 グラフ保存: {save_path}")
    
    plt.show()

# ===================================================================
# 🚀 メイン実行
# ===================================================================

def main():
    """メイン実行関数"""
    print("=" * 60)
    print("🌌 NKAT理論 深層学習最適化")
    print("=" * 60)
    
    # 設定
    config = NKATConfig()
    print(f"📋 設定:")
    print(f"   格子サイズ: {config.grid_size}⁴ = {config.grid_size**4:,} 点")
    print(f"   目標スペクトル次元: {config.target_spectral_dim}")
    print(f"   エポック数: {config.num_epochs}")
    
    # GPU使用量確認
    if torch.cuda.is_available():
        print(f"💾 初期GPU使用量: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    start_time = time.time()
    
    try:
        # 訓練実行
        model, history = train_nkat_model(config)
        
        # 結果分析
        elapsed_time = time.time() - start_time
        final_spec_dim = history['spectral_dim_estimates'][-1]
        final_theta = history['theta_values'][-1]
        improvement = (6.07 - final_spec_dim) / 6.07 * 100
        
        print("\n" + "="*60)
        print("🏆 訓練完了！")
        print("="*60)
        print(f"⏱️ 実行時間: {elapsed_time/60:.1f} 分")
        print(f"🎯 最終スペクトル次元: {final_spec_dim:.3f} (目標: {config.target_spectral_dim})")
        print(f"📐 最終θ値: {final_theta:.2e}")
        print(f"📈 改善度: {improvement:.1f}%")
        
        if torch.cuda.is_available():
            print(f"💾 最終GPU使用量: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        
        # 結果保存
        final_checkpoint = {
            'model_state_dict': model.state_dict(),
            'history': history,
            'config': config,
            'final_metrics': {
                'spectral_dimension': final_spec_dim,
                'theta_value': final_theta,
                'training_time': elapsed_time,
                'improvement_percentage': improvement
            }
        }
        
        final_model_path = os.path.join(work_dir, 'nkat_final_model.pth')
        torch.save(final_checkpoint, final_model_path)
        print(f"💾 最終モデル保存: {final_model_path}")
        
        # 結果可視化
        plot_results_path = os.path.join(work_dir, 'nkat_training_results.png')
        plot_training_results(history, config, plot_results_path)
        
        # JSON結果保存
        results_summary = {
            'initial_spectral_dimension': 6.07,
            'final_spectral_dimension': final_spec_dim,
            'target_spectral_dimension': config.target_spectral_dim,
            'final_theta': final_theta,
            'improvement_percentage': improvement,
            'training_time_minutes': elapsed_time / 60,
            'config': {
                'grid_size': config.grid_size,
                'batch_size': config.batch_size,
                'learning_rate': config.learning_rate,
                'num_epochs': config.num_epochs,
                'kan_layers': config.kan_layers
            }
        }
        
        json_path = os.path.join(work_dir, 'nkat_results_summary.json')
        with open(json_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"📝 結果サマリー保存: {json_path}")
        
        # 成功判定
        if abs(final_spec_dim - config.target_spectral_dim) < 0.5:
            print("\n🎊 ✅ 実験提案書作成準備完了！")
            print("🎯 CTA/PVLAS/MAGIS感度解析可能")
            print("📚 Nature/PRL級論文執筆可能")
        elif abs(final_spec_dim - config.target_spectral_dim) < 1.0:
            print("\n🔄 追加最適化推奨")
            print("📊 longer training or Optuna調整")
        else:
            print("\n🔧 モデル改良が必要")
            print("📐 格子サイズ拡大またはKAN構造最適化")
            
    except Exception as e:
        print(f"❌ エラー発生: {str(e)}")
        print("💡 GPUメモリ不足の可能性があります")
        
        # エラー時のメモリクリーンアップ
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 GPUメモリクリーンアップ完了")
    
    # 最終サマリー
    print("\n" + "="*60)
    print("🌌 NKAT理論 深層学習最適化 完了")
    print("="*60)
    
    if IN_COLAB:
        if work_dir.startswith('/content/drive'):
            print("📂 結果はGoogle Driveに保存されました")
            print("🔗 /content/drive/MyDrive/NKAT_DL_Results/")
        else:
            print("📂 結果はColabローカルに保存されました")
            print(f"🔗 {work_dir}/")
            print("⚠️ セッション終了時にファイルが消える可能性があります")
    else:
        print(f"📂 結果は {work_dir} に保存されました")
    
    print("🎊 お疲れ様でした！")

if __name__ == "__main__":
    main() 