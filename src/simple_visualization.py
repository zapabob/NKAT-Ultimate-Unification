#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 NKAT理論結果のシンプル視覚化システム
Simple NKAT Theory Results Visualization

最小限のライブラリで基本的な可視化を実現
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# 日本語フォント設定
plt.rcParams['font.family'] = ['MS Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_results():
    """結果ファイルの読み込み"""
    try:
        with open('high_precision_riemann_results.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("⚠️ 結果ファイルが見つかりません。サンプルデータを使用します。")
        return create_sample_data()

def create_sample_data():
    """サンプルデータの生成"""
    gamma_values = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062]
    return {
        'gamma_values': gamma_values,
        'statistics': {
            'spectral_dimension_mean': [0.005646] * len(gamma_values),
            'real_part_mean': [0.002823] * len(gamma_values),
            'convergence_mean': [0.497177] * len(gamma_values)
        },
        'overall_statistics': {
            'mean_convergence': 0.49717717,
            'success_rate': 0.0
        }
    }

def create_simple_visualization():
    """シンプルな可視化の生成"""
    print("🎯 NKAT理論：シンプル視覚化レポート")
    print("=" * 50)
    
    # データの読み込み
    results = load_results()
    
    # データの抽出
    gamma_values = results['gamma_values']
    spectral_dims = results['statistics']['spectral_dimension_mean']
    real_parts = results['statistics']['real_part_mean']
    convergences = results['statistics']['convergence_mean']
    
    # 理論値
    theoretical_spectral = [1.0] * len(gamma_values)
    theoretical_real = [0.5] * len(gamma_values)
    
    # 図1: スペクトル次元比較
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # スペクトル次元比較
    x_pos = np.arange(len(gamma_values))
    width = 0.35
    
    ax1.bar(x_pos - width/2, spectral_dims, width, label='計算結果', color='skyblue', alpha=0.7)
    ax1.bar(x_pos + width/2, theoretical_spectral, width, label='理論期待値', color='orange', alpha=0.7)
    ax1.set_xlabel('γ値インデックス')
    ax1.set_ylabel('スペクトル次元 d_s')
    ax1.set_title('🎯 スペクトル次元比較')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 実部比較
    ax2.plot(gamma_values, real_parts, 'bo-', label='計算された実部', markersize=8)
    ax2.axhline(y=0.5, color='red', linestyle='--', label='理論値 (1/2)', linewidth=2)
    ax2.set_xlabel('γ値')
    ax2.set_ylabel('Re(s)')
    ax2.set_title('📊 実部の比較')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 収束性分析
    ax3.plot(gamma_values, convergences, 'ro-', markersize=8, linewidth=2)
    ax3.axhline(y=0.1, color='green', linestyle='--', alpha=0.7, label='成功基準')
    ax3.axhline(y=0.01, color='blue', linestyle='--', alpha=0.7, label='高精度基準')
    ax3.set_xlabel('γ値')
    ax3.set_ylabel('|Re(s) - 1/2|')
    ax3.set_title('⚠️ 収束性分析')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 改良ロードマップ
    phases = ['現在', '短期', '中期', '長期', '理想']
    targets = [0.497, 0.1, 0.05, 0.01, 0.001]
    colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
    
    bars = ax4.bar(phases, targets, color=colors, alpha=0.7)
    ax4.set_ylabel('|Re(s) - 1/2|')
    ax4.set_title('🚀 改良ロードマップ')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    # 値の表示
    for bar, val in zip(bars, targets):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('nkat_simple_analysis.png', dpi=300, bbox_inches='tight')
    print("💾 グラフを 'nkat_simple_analysis.png' に保存しました")
    plt.show()
    
    # 統計情報の表示
    print("\n📊 数値的結果サマリー:")
    print(f"検証済みγ値数: {len(gamma_values)}")
    print(f"平均スペクトル次元: {np.mean(spectral_dims):.6f}")
    print(f"理論期待値: 1.000000")
    print(f"差異倍率: {1.0 / np.mean(spectral_dims):.1f}倍")
    
    print(f"\n🔍 収束性解析:")
    stats = results.get('overall_statistics', {})
    print(f"平均収束率: {stats.get('mean_convergence', 0):.8f}")
    print(f"成功率: {stats.get('success_rate', 0)*100:.2f}%")
    
    improvement_needed = stats.get('mean_convergence', 0.5) / 0.01
    print(f"高精度達成に必要な改良: {improvement_needed:.1f}倍")
    
    print(f"\n🎯 主要な改良点:")
    print("1. ハミルトニアン主対角項の正規化")
    print("2. スペクトル次元計算アルゴリズムの修正")
    print("3. 数値安定性の向上")
    print("4. 理論的制約の導入")
    
    print(f"\n🏆 NKAT理論の現状評価:")
    if stats.get('mean_convergence', 1) < 0.1:
        print("✅ 成功：理論的期待値に近い結果")
    elif stats.get('mean_convergence', 1) < 0.3:
        print("⚠️ 改良が必要：部分的に有望な結果")
    else:
        print("❌ 大幅な改良が必要：系統的な問題が存在")
    
    print("\n🎉 シンプル視覚化完了！")

if __name__ == "__main__":
    try:
        create_simple_visualization()
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc() 