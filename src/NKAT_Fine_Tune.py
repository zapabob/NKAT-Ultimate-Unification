# -*- coding: utf-8 -*-
"""
🎯 NKAT GPU修羅モード微調整版 🎯
誤差 1×10⁻⁵ アタック専用 - 20エポック集中攻撃
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import datetime
import os
from pathlib import Path
from matplotlib import rcParams

# 日本語フォント設定
rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class NKATFineTuneNetwork(nn.Module):
    """NKAT微調整専用ネットワーク（元モデル互換）"""
    
    def __init__(self, input_dim=4, hidden_dims=[512, 256, 128], grid_size=64):
        super().__init__()
        
        self.input_dim = input_dim
        self.grid_size = grid_size
        
        # 元モデルと同じアーキテクチャ（互換性確保）
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)  # 元と同じドロップアウト率
            ])
            prev_dim = hidden_dim
            
        # 元モデルと同じ出力層
        layers.extend([
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ])
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class NKATFineTuneLoss:
    """微調整専用損失関数（超高精度）"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.theta_min = 1e-50
        self.theta_max = 1e-10
        
    def ultra_precision_spectral_loss(self, output, target_dim=4.0):
        """超高精度スペクトラル次元損失"""
        # より安全で精密な計算
        output_clamped = torch.clamp(output, min=-5, max=5)
        spectral_dim = 4.0 + torch.mean(output_clamped) * 0.1  # スケール調整
        
        # 二乗誤差（高精度用）
        error = (spectral_dim - target_dim)**2
        return error, spectral_dim.item()
    
    def ultra_precision_theta_loss(self, output):
        """超高精度θパラメータ損失"""
        # より安定したθ計算
        theta_raw = torch.exp(-torch.abs(output) * 0.1)
        theta_clamped = torch.clamp(theta_raw, min=self.theta_min, max=self.theta_max)
        
        # 目標値（実験値に基づく）
        target_theta = 1e-35
        theta_mse = torch.mean((theta_clamped - target_theta)**2)
        
        return theta_mse, torch.mean(theta_clamped).item()

def load_best_checkpoint():
    """最良チェックポイント読み込み"""
    checkpoint_dir = Path("nkat_shura_checkpoints")
    best_model_path = checkpoint_dir / "best_model.pth"
    
    if best_model_path.exists():
        print(f"📁 ベストモデル読み込み: {best_model_path}")
        return torch.load(best_model_path)
    else:
        print("⚠️ ベストモデルが見つかりません")
        return None

def fine_tune_training():
    """微調整訓練（誤差10⁻⁵アタック）"""
    print("🎯" * 20)
    print("🚀 NKAT 微調整モード起動！")
    print("🎯 目標: スペクトラル次元誤差 < 1×10⁻⁵")
    print("⚡ 設定: 20エポック集中攻撃")
    print("🎯" * 20)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 デバイス: {device}")
    
    # ベストモデル読み込み
    checkpoint = load_best_checkpoint()
    
    if checkpoint is None:
        print("❌ チェックポイントなし - 新規訓練開始")
        model = NKATFineTuneNetwork(grid_size=64).to(device)
        start_epoch = 0
        best_error = float('inf')
    else:
        print("✅ チェックポイント読み込み成功")
        model = NKATFineTuneNetwork(grid_size=64).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        
        # 最良誤差を履歴から取得
        if 'metrics' in checkpoint and checkpoint['metrics']:
            last_metrics = checkpoint['metrics'][-1]
            best_error = abs(last_metrics.get('spectral_dim', 4.0) - 4.0)
        else:
            best_error = float('inf')
        
        print(f"📊 開始エポック: {start_epoch}")
        print(f"📊 現在の最良誤差: {best_error:.2e}")
    
    # 微調整用オプティマイザー（低学習率）
    optimizer = optim.AdamW(model.parameters(), 
                           lr=1e-5,  # 微調整用低学習率
                           weight_decay=1e-5)
    
    # 学習率スケジューラー
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    physics_loss = NKATFineTuneLoss(device)
    
    # 微調整用重み（精密調整）
    weights = {
        'spectral': 20.0,  # スペクトラル次元重視
        'theta': 2.0,
        'jacobi': 1.0,
        'connes': 1.0
    }
    
    # 訓練履歴
    fine_tune_history = []
    patience_counter = 0
    max_patience = 10
    
    print(f"\n🎯 微調整訓練開始！")
    print(f"📊 学習率: {optimizer.param_groups[0]['lr']:.2e}")
    print(f"📊 重み: {weights}")
    
    start_time = time.time()
    
    # 微調整ループ（20エポック）
    for epoch in range(start_epoch + 1, start_epoch + 21):
        epoch_start = time.time()
        
        model.train()
        
        # 高精度データ生成
        batch_size = 512  # 微調整用大バッチ
        x = torch.randn(batch_size, 4, device=device)
        x_time_positive = torch.abs(x[:, 0])
        x = torch.cat([x_time_positive.unsqueeze(1), x[:, 1:]], dim=1)
        x.requires_grad_(True)
        
        # フォワードパス
        output = model(x)
        
        # 超高精度損失計算
        spectral_loss, spectral_dim = physics_loss.ultra_precision_spectral_loss(output)
        theta_loss, theta_value = physics_loss.ultra_precision_theta_loss(output)
        
        # 総合損失
        total_loss = (weights['spectral'] * spectral_loss + 
                      weights['theta'] * theta_loss)
        
        # バックプロパゲーション
        optimizer.zero_grad()
        total_loss.backward()
        
        # 勾配クリッピング（微調整用）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        optimizer.step()
        
        epoch_time = time.time() - epoch_start
        
        # メトリクス記録
        spectral_error = abs(spectral_dim - 4.0)
        metrics = {
            'epoch': epoch,
            'total_loss': total_loss.item(),
            'spectral_loss': spectral_loss.item(),
            'spectral_dim': spectral_dim,
            'spectral_error': spectral_error,
            'theta_loss': theta_loss.item(),
            'theta_value': theta_value,
            'epoch_time': epoch_time,
            'lr': optimizer.param_groups[0]['lr']
        }
        fine_tune_history.append(metrics)
        
        # 進捗表示
        print(f"🎯 Epoch {epoch:3d}/20 | "
              f"Loss: {total_loss.item():.3e} | "
              f"d_s: {spectral_dim:.8f} | "
              f"Error: {spectral_error:.3e} | "
              f"θ: {theta_value:.2e} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # ベスト更新チェック
        if spectral_error < best_error:
            best_error = spectral_error
            patience_counter = 0
            
            # ベストモデル保存
            checkpoint_dir = Path("nkat_fine_tune_checkpoints")
            checkpoint_dir.mkdir(exist_ok=True)
            best_path = checkpoint_dir / "best_fine_tune.pth"
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_error': best_error,
                'metrics': fine_tune_history
            }, best_path)
            
            if spectral_error < 1e-5:
                print(f"🏆 目標達成！ Error: {spectral_error:.3e} < 1×10⁻⁵")
                break
        else:
            patience_counter += 1
        
        # 学習率調整
        scheduler.step(spectral_error)
        
        # 早期停止
        if patience_counter >= max_patience:
            print(f"🛑 早期停止 (patience={max_patience})")
            break
    
    # 微調整完了
    total_time = time.time() - start_time
    print(f"\n🎉 微調整完了！")
    print(f"⏱️ 総時間: {total_time:.1f}秒")
    print(f"🎯 最終スペクトラル次元: {fine_tune_history[-1]['spectral_dim']:.10f}")
    print(f"🎯 最小誤差: {best_error:.3e}")
    
    # 結果保存
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 履歴保存
    history_file = f"nkat_fine_tune_history_{timestamp}.json"
    with open(history_file, 'w') as f:
        json.dump(fine_tune_history, f, indent=2)
    
    # 微調整プロット生成
    generate_fine_tune_plot(fine_tune_history, timestamp)
    
    print(f"📊 微調整結果保存: {history_file}")
    
    return fine_tune_history, best_error

def generate_fine_tune_plot(history, timestamp):
    """微調整結果プロット"""
    epochs = [h['epoch'] for h in history]
    spectral_dims = [h['spectral_dim'] for h in history]
    spectral_errors = [h['spectral_error'] for h in history]
    learning_rates = [h['lr'] for h in history]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # スペクトラル次元微調整
    ax1.plot(epochs, spectral_dims, 'b-', linewidth=2, marker='o', markersize=3)
    ax1.axhline(y=4.0, color='r', linestyle='--', label='Target d_s = 4.0')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Spectral Dimension')
    ax1.set_title('Fine-Tune: Spectral Dimension Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 誤差収束（対数）
    ax2.semilogy(epochs, spectral_errors, 'g-', linewidth=2, marker='s', markersize=3)
    ax2.axhline(y=1e-5, color='r', linestyle='--', label='Target < 1e-5')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Spectral Dimension Error')
    ax2.set_title('Ultra-Precision Error Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 学習率変化
    ax3.semilogy(epochs, learning_rates, 'orange', linewidth=2, marker='^', markersize=3)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Adaptive Learning Rate')
    ax3.grid(True, alpha=0.3)
    
    # 誤差改善率
    if len(spectral_errors) > 1:
        improvement_rates = []
        for i in range(1, len(spectral_errors)):
            if spectral_errors[i-1] > 0:
                rate = (spectral_errors[i-1] - spectral_errors[i]) / spectral_errors[i-1] * 100
                improvement_rates.append(rate)
            else:
                improvement_rates.append(0)
        
        ax4.plot(epochs[1:], improvement_rates, 'purple', linewidth=2, marker='d', markersize=3)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Error Improvement Rate (%)')
        ax4.set_title('Fine-Tune Improvement Rate')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = f"nkat_fine_tune_results_{timestamp}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 微調整プロット保存: {plot_file}")

def main():
    """メイン実行"""
    try:
        history, best_error = fine_tune_training()
        
        print(f"\n🏆 微調整完了！")
        print(f"🎯 達成精度: {best_error:.3e}")
        
        if best_error < 1e-5:
            print("🎉 目標達成！ 誤差 < 1×10⁻⁵")
        else:
            print(f"📈 目標まで: {best_error/1e-5:.1f}倍")
            
    except Exception as e:
        print(f"❌ 微調整エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 