#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 NKAT理論結果の視覚化システム
NKAT Theory Results Visualization System

機能:
1. スペクトル次元の比較可視化
2. 収束性の分析グラフ
3. 理論値との差異の表示
4. 研究進捗の可視化

Author: NKAT Research Team
Date: 2025-05-26
Version: 1.0
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple
import seaborn as sns

# 日本語フォント設定
plt.rcParams['font.family'] = ['MS Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)

class NKATResultsVisualizer:
    """
    NKAT理論結果の総合的可視化システム
    """
    
    def __init__(self, results_file: str = 'high_precision_riemann_results.json'):
        self.results_file = results_file
        self.results = self.load_results()
        
    def load_results(self) -> Dict:
        """結果ファイルの読み込み"""
        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️ 結果ファイル {self.results_file} が見つかりません")
            return self.create_sample_data()
    
    def create_sample_data(self) -> Dict:
        """サンプルデータの生成（ファイルが存在しない場合）"""
        gamma_values = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062]
        return {
            'gamma_values': gamma_values,
            'statistics': {
                'spectral_dimension_mean': [0.005646] * len(gamma_values),
                'spectral_dimension_std': [0.0] * len(gamma_values),
                'real_part_mean': [0.002823] * len(gamma_values),
                'convergence_mean': [0.497177] * len(gamma_values)
            },
            'overall_statistics': {
                'mean_convergence': 0.49717717,
                'success_rate': 0.0
            }
        }
    
    def plot_spectral_dimension_comparison(self):
        """スペクトル次元の理論値との比較"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        gamma_values = self.results['gamma_values']
        spectral_dims = self.results['statistics']['spectral_dimension_mean']
        theoretical_values = [1.0] * len(gamma_values)  # 理論期待値
        
        # 左側: 現在の結果 vs 理論値
        x_pos = np.arange(len(gamma_values))
        width = 0.35
        
        bars1 = ax1.bar(x_pos - width/2, spectral_dims, width, 
                       label='現在の結果', color='skyblue', alpha=0.7)
        bars2 = ax1.bar(x_pos + width/2, theoretical_values, width, 
                       label='理論期待値', color='orange', alpha=0.7)
        
        ax1.set_xlabel('リーマンゼータ零点 γ値')
        ax1.set_ylabel('スペクトル次元 d_s')
        ax1.set_title('🎯 スペクトル次元: 現在結果 vs 理論期待値')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f'{γ:.3f}' for γ in gamma_values], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 値を棒グラフ上に表示
        for bar, val in zip(bars1, spectral_dims):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=8)
        
        # 右側: 対数スケールでの比較
        ax2.semilogy(gamma_values, spectral_dims, 'o-', label='現在の結果', 
                    color='skyblue', markersize=8, linewidth=2)
        ax2.semilogy(gamma_values, theoretical_values, 's--', label='理論期待値', 
                    color='orange', markersize=8, linewidth=2)
        
        ax2.set_xlabel('リーマンゼータ零点 γ値')
        ax2.set_ylabel('スペクトル次元 d_s (対数スケール)')
        ax2.set_title('📊 対数スケールでの比較')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('nkat_spectral_dimension_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_convergence_analysis(self):
        """収束性の分析グラフ"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        gamma_values = self.results['gamma_values']
        convergences = self.results['statistics']['convergence_mean']
        real_parts = self.results['statistics']['real_part_mean']
        
        # 左上: 収束性の可視化
        ax1.plot(gamma_values, convergences, 'ro-', markersize=8, linewidth=2, 
                label='|Re(s) - 1/2|')
        ax1.axhline(y=0.1, color='green', linestyle='--', alpha=0.7, 
                   label='成功基準 (0.1)')
        ax1.axhline(y=0.01, color='blue', linestyle='--', alpha=0.7, 
                   label='高精度基準 (0.01)')
        
        ax1.set_xlabel('γ値')
        ax1.set_ylabel('|Re(s) - 1/2|')
        ax1.set_title('🎯 収束性分析')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 右上: 実部の分布
        ax2.plot(gamma_values, real_parts, 'bo-', markersize=8, linewidth=2,
                label='計算された実部')
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7,
                   label='理論値 (Re = 1/2)')
        
        ax2.set_xlabel('γ値')
        ax2.set_ylabel('Re(s)')
        ax2.set_title('📊 実部の分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 左下: 差異の可視化
        differences = [abs(re - 0.5) for re in real_parts]
        colors = ['red' if diff > 0.1 else 'orange' if diff > 0.01 else 'green' 
                 for diff in differences]
        
        bars = ax3.bar(range(len(gamma_values)), differences, color=colors, alpha=0.7)
        ax3.set_xlabel('γ値インデックス')
        ax3.set_ylabel('|Re(s) - 1/2|')
        ax3.set_title('⚠️ 理論値からの差異')
        ax3.set_xticks(range(len(gamma_values)))
        ax3.set_xticklabels([f'{γ:.3f}' for γ in gamma_values], rotation=45)
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # 右下: 統計的サマリー
        stats = self.results.get('overall_statistics', {})
        labels = ['平均収束率', '成功率 (%)', '高精度成功率 (%)']
        values = [
            stats.get('mean_convergence', 0),
            stats.get('success_rate', 0) * 100,
            stats.get('high_precision_success_rate', 0) * 100
        ]
        
        bars = ax4.bar(labels, values, color=['skyblue', 'lightgreen', 'lightcoral'])
        ax4.set_ylabel('値')
        ax4.set_title('📈 統計的サマリー')
        ax4.grid(True, alpha=0.3)
        
        # 値を棒グラフ上に表示
        for bar, val in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                    f'{val:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('nkat_convergence_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_theoretical_landscape(self):
        """理論的期待値の風景図"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # γ値の範囲を拡張
        gamma_extended = np.linspace(10, 50, 100)
        
        # 理論的期待値（臨界線上ではRe(s) = 1/2）
        theoretical_real_parts = np.full_like(gamma_extended, 0.5)
        
        # 現在の結果の外挿
        gamma_values = self.results['gamma_values']
        real_parts = self.results['statistics']['real_part_mean']
        
        # プロット
        ax.plot(gamma_extended, theoretical_real_parts, 'r-', linewidth=3,
               label='理論的期待値 (Re = 1/2)', alpha=0.8)
        ax.plot(gamma_values, real_parts, 'bo', markersize=10,
               label='NKAT計算結果', alpha=0.8)
        
        # 信頼区間の表示
        upper_bound = np.full_like(gamma_extended, 0.6)
        lower_bound = np.full_like(gamma_extended, 0.4)
        ax.fill_between(gamma_extended, lower_bound, upper_bound, 
                       alpha=0.2, color='green', label='許容範囲 (±0.1)')
        
        # より厳しい基準
        upper_tight = np.full_like(gamma_extended, 0.51)
        lower_tight = np.full_like(gamma_extended, 0.49)
        ax.fill_between(gamma_extended, lower_tight, upper_tight, 
                       alpha=0.3, color='blue', label='高精度範囲 (±0.01)')
        
        ax.set_xlabel('リーマンゼータ零点 γ値')
        ax.set_ylabel('Re(s)')
        ax.set_title('🌟 NKAT理論: 理論的期待値との比較風景')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 注釈の追加
        ax.annotate('リーマン予想:\n全ての零点は\nRe(s) = 1/2 上にある', 
                   xy=(35, 0.5), xytext=(40, 0.3),
                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.savefig('nkat_theoretical_landscape.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_improvement_roadmap(self):
        """改良ロードマップの可視化"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 現在の状況と目標
        current_convergence = self.results.get('overall_statistics', {}).get('mean_convergence', 0.497)
        
        milestones = [
            ('現在', current_convergence, 'red'),
            ('短期目標\n(1-2週間)', 0.1, 'orange'),
            ('中期目標\n(1-2ヶ月)', 0.05, 'yellow'),
            ('長期目標\n(3-6ヶ月)', 0.01, 'lightgreen'),
            ('理想的目標', 0.001, 'green')
        ]
        
        phases = [m[0] for m in milestones]
        convergences = [m[1] for m in milestones]
        colors = [m[2] for m in milestones]
        
        # 改良ロードマップの描画
        bars = ax.bar(phases, convergences, color=colors, alpha=0.7)
        ax.set_ylabel('|Re(s) - 1/2| (対数スケール)')
        ax.set_title('🚀 NKAT理論改良ロードマップ')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # 値を棒グラフ上に表示
        for bar, val in zip(bars, convergences):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                   f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 成功基準線の追加
        ax.axhline(y=0.1, color='blue', linestyle='--', alpha=0.7, 
                  label='成功基準 (0.1)')
        ax.axhline(y=0.01, color='purple', linestyle='--', alpha=0.7, 
                  label='高精度基準 (0.01)')
        
        ax.legend()
        
        # 改良点の注釈
        improvements = [
            "現在:\n- 基本NKAT実装\n- 系統的バイアス有り",
            "短期:\n- ハミルトニアン改良\n- 正規化修正",
            "中期:\n- 数値安定性向上\n- 多精度演算",
            "長期:\n- 理論的精密化\n- 量子補正項",
            "理想:\n- 完全な理論一致\n- 論文発表レベル"
        ]
        
        for i, (bar, improvement) in enumerate(zip(bars, improvements)):
            if i % 2 == 0:
                y_pos = max(convergences) * 0.5
            else:
                y_pos = max(convergences) * 0.1
            
            ax.annotate(improvement, 
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(bar.get_x() + bar.get_width()/2, y_pos),
                       arrowprops=dict(arrowstyle='->', alpha=0.5),
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
                       fontsize=8, ha='center')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('nkat_improvement_roadmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_comprehensive_report(self):
        """総合レポートの生成"""
        print("=" * 80)
        print("🎯 NKAT理論によるリーマン予想検証：総合視覚化レポート")
        print("=" * 80)
        
        # 基本統計の表示
        stats = self.results.get('overall_statistics', {})
        print(f"📊 現在の実行結果:")
        print(f"  平均収束率: {stats.get('mean_convergence', 0):.8f}")
        print(f"  成功率: {stats.get('success_rate', 0)*100:.2f}%")
        print(f"  検証済みγ値数: {len(self.results['gamma_values'])}")
        
        print(f"\n🔍 理論的分析:")
        theoretical_expected = 0.0  # 理論的には完全に1/2に収束すべき
        current_avg = stats.get('mean_convergence', 0.5)
        improvement_needed = current_avg / 0.01 if current_avg > 0 else float('inf')
        
        print(f"  理論期待値からの差異: {current_avg:.6f}")
        print(f"  高精度達成に必要な改良率: {improvement_needed:.1f}倍")
        
        print(f"\n🚀 可視化グラフの生成:")
        
        # 各種グラフの生成
        try:
            print("  1. スペクトル次元比較グラフ...")
            self.plot_spectral_dimension_comparison()
            
            print("  2. 収束性分析グラフ...")
            self.plot_convergence_analysis()
            
            print("  3. 理論的期待値風景図...")
            self.plot_theoretical_landscape()
            
            print("  4. 改良ロードマップ...")
            self.plot_improvement_roadmap()
            
            print("✅ 全ての可視化が完了しました！")
            
        except Exception as e:
            print(f"❌ 可視化エラー: {e}")
        
        print(f"\n💾 生成されたファイル:")
        print("  - nkat_spectral_dimension_comparison.png")
        print("  - nkat_convergence_analysis.png") 
        print("  - nkat_theoretical_landscape.png")
        print("  - nkat_improvement_roadmap.png")
        
        print(f"\n🎉 NKAT理論の視覚化レポートが完了しました！")

def main():
    """メイン実行関数"""
    try:
        visualizer = NKATResultsVisualizer()
        visualizer.generate_comprehensive_report()
    except Exception as e:
        print(f"❌ 実行エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 