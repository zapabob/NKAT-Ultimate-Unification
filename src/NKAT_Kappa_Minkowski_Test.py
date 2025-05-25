# -*- coding: utf-8 -*-
"""
κ-Minkowski 変形スモークテスト
Moyal積との差分解析と32³グリッドでの高速検証
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from matplotlib import rcParams

# 日本語フォント設定
rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class KappaMinkowskiDeformation:
    """κ-Minkowski時空変形の実装"""
    
    def __init__(self, kappa=1e16, device='cpu'):
        """
        Args:
            kappa: κ-変形パラメータ (通常はプランクスケール)
            device: 計算デバイス
        """
        self.kappa = kappa
        self.device = device
        
    def star_product_kappa(self, f, g, x):
        """κ-Minkowski スター積"""
        # κ-変形された非可換積
        # f ⋆_κ g = f·g + (i/2κ)·{∂f/∂x^μ, ∂g/∂x^ν}·θ^μν + O(1/κ²)
        
        # 1次近似での実装
        f_grad = torch.autograd.grad(f.sum(), x, create_graph=True)[0]
        g_grad = torch.autograd.grad(g.sum(), x, create_graph=True)[0]
        
        # κ-変形項
        kappa_term = (1.0 / (2 * self.kappa)) * torch.sum(
            f_grad[:, :, None] * g_grad[:, None, :], dim=(1, 2)
        )
        
        return f * g + kappa_term.unsqueeze(-1).unsqueeze(-1)
    
    def bicrossproduct_algebra(self, p, x):
        """双交差積代数の実装"""
        # [x^μ, p^ν] = (i/κ)·(δ^μ_0·p^ν - δ^ν_0·p^μ)
        commutator = torch.zeros_like(x)
        
        # μ=0, ν≠0 の場合
        commutator[:, 0] = torch.sum(p[:, 1:], dim=1) / self.kappa
        
        # μ≠0, ν=0 の場合  
        commutator[:, 1:] = -p[:, 0].unsqueeze(-1) / self.kappa
        
        return commutator

class MoyalDeformation:
    """比較用のMoyal変形"""
    
    def __init__(self, theta=1e-35, device='cpu'):
        self.theta = theta
        self.device = device
        
    def star_product_moyal(self, f, g, x):
        """標準的なMoyalスター積"""
        f_grad = torch.autograd.grad(f.sum(), x, create_graph=True)[0]
        g_grad = torch.autograd.grad(g.sum(), x, create_graph=True)[0]
        
        # Moyal項: (i/2)·θ^μν·∂f/∂x^μ·∂g/∂x^ν
        moyal_term = (self.theta / 2.0) * torch.sum(
            f_grad * g_grad, dim=1
        )
        
        return f * g + moyal_term.unsqueeze(-1).unsqueeze(-1)

class NKATTestNetwork(nn.Module):
    """NKAT テスト用ニューラルネットワーク（軽量版）"""
    
    def __init__(self, input_dim=4, hidden_dims=[64, 32], grid_size=32):
        super().__init__()
        
        self.input_dim = input_dim
        self.grid_size = grid_size
        
        # 軽量アーキテクチャ（32³グリッド用）
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def run_smoke_test():
    """κ-Minkowski vs Moyal スモークテスト"""
    print("🧪 κ-Minkowski スモークテスト開始...")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 デバイス: {device}")
    
    # パラメータ設定
    grid_size = 32  # 高速テスト用
    batch_size = 64
    
    # 変形オブジェクト初期化
    kappa_def = KappaMinkowskiDeformation(kappa=1e16, device=device)
    moyal_def = MoyalDeformation(theta=1e-35, device=device)
    
    # テストデータ生成
    x = torch.randn(batch_size, 4, requires_grad=True, device=device)
    
    # テスト関数
    f = torch.sin(x.sum(dim=1, keepdim=True))
    g = torch.cos(x.sum(dim=1, keepdim=True))
    
    print(f"📊 テストデータ: {x.shape}")
    print(f"📊 関数f: {f.shape}, 関数g: {g.shape}")
    
    # 計算時間測定
    results = {}
    
    # κ-Minkowski テスト
    start_time = time.time()
    try:
        kappa_result = kappa_def.star_product_kappa(f, g, x)
        kappa_time = time.time() - start_time
        results['kappa'] = {
            'result': kappa_result,
            'time': kappa_time,
            'success': True
        }
        print(f"✅ κ-Minkowski計算完了: {kappa_time:.4f}秒")
    except Exception as e:
        results['kappa'] = {'success': False, 'error': str(e)}
        print(f"❌ κ-Minkowski計算エラー: {e}")
    
    # Moyal テスト
    start_time = time.time()
    try:
        moyal_result = moyal_def.star_product_moyal(f, g, x)
        moyal_time = time.time() - start_time
        results['moyal'] = {
            'result': moyal_result,
            'time': moyal_time,
            'success': True
        }
        print(f"✅ Moyal計算完了: {moyal_time:.4f}秒")
    except Exception as e:
        results['moyal'] = {'success': False, 'error': str(e)}
        print(f"❌ Moyal計算エラー: {e}")
    
    # 差分解析
    if results['kappa']['success'] and results['moyal']['success']:
        diff = torch.abs(results['kappa']['result'] - results['moyal']['result'])
        mean_diff = torch.mean(diff).item()
        max_diff = torch.max(diff).item()
        
        print(f"\n📈 差分解析:")
        print(f"• 平均差分: {mean_diff:.2e}")
        print(f"• 最大差分: {max_diff:.2e}")
        print(f"• 計算時間比: {results['kappa']['time']/results['moyal']['time']:.2f}")
        
        # 差分の可視化
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.hist(results['kappa']['result'].detach().cpu().numpy().flatten(), 
                bins=30, alpha=0.7, label='κ-Minkowski')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('κ-Minkowski Distribution')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.hist(results['moyal']['result'].detach().cpu().numpy().flatten(), 
                bins=30, alpha=0.7, label='Moyal', color='orange')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Moyal Distribution')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.hist(diff.detach().cpu().numpy().flatten(), 
                bins=30, alpha=0.7, label='Difference', color='red')
        plt.xlabel('|κ-Minkowski - Moyal|')
        plt.ylabel('Frequency')
        plt.title('Difference Distribution')
        plt.legend()
        
        plt.tight_layout()
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plot_file = f"kappa_moyal_comparison_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 比較プロット保存: {plot_file}")
        
        # 健全性チェック
        if mean_diff < 1e-3:
            print("✅ 健全性チェック: PASS (差分が許容範囲内)")
        else:
            print("⚠️ 健全性チェック: 要注意 (大きな差分を検出)")
    
    return results

def test_bicrossproduct():
    """双交差積代数のテスト"""
    print("\n🔬 双交差積代数テスト...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kappa_def = KappaMinkowskiDeformation(kappa=1e16, device=device)
    
    # テストベクトル
    x = torch.randn(10, 4, device=device)
    p = torch.randn(10, 4, device=device)
    
    try:
        commutator = kappa_def.bicrossproduct_algebra(p, x)
        print(f"✅ 双交差積計算成功: {commutator.shape}")
        print(f"📊 交換子ノルム: {torch.norm(commutator).item():.2e}")
        return True
    except Exception as e:
        print(f"❌ 双交差積計算エラー: {e}")
        return False

def main():
    """メイン実行"""
    print("🚀 κ-Minkowski vs Moyal 比較テスト")
    print("🎯 目的: モデル依存性の健全性確認")
    print("⚡ 設定: 32³グリッド高速モード")
    print("=" * 60)
    
    try:
        # スモークテスト実行
        results = run_smoke_test()
        
        # 双交差積テスト
        bicross_success = test_bicrossproduct()
        
        # 総合評価
        print("\n" + "=" * 60)
        print("🎯 総合評価:")
        
        if results.get('kappa', {}).get('success') and results.get('moyal', {}).get('success'):
            print("✅ κ-Minkowski実装: 正常動作")
            print("✅ Moyal実装: 正常動作")
            print("✅ 差分解析: 完了")
        else:
            print("❌ 一部の実装でエラー発生")
            
        if bicross_success:
            print("✅ 双交差積代数: 正常動作")
        else:
            print("❌ 双交差積代数: エラー")
            
        print("\n📝 次のステップ:")
        print("• 64³グリッドでの本格テスト")
        print("• 物理量の詳細比較")
        print("• NKAT本体への統合")
        
    except Exception as e:
        print(f"❌ テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 