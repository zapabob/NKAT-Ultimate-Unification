#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 NKAT v11.0 結果可視化システム
Visualization System for NKAT v11.0 Rigorous Mathematical Verification Results

Author: NKAT Research Consortium
Date: 2025-05-26
Version: 11.0 - Results Visualization
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import Dict, List, Any

# 日本語フォント設定
plt.rcParams['font.family'] = ['MS Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12

# スタイル設定
sns.set_style("whitegrid")
sns.set_palette("husl")

class NKATResultsVisualizer:
    """NKAT v11.0結果可視化クラス"""
    
    def __init__(self, results_file: str = None):
        self.results_file = results_file or self._find_latest_results()
        self.results = self._load_results()
        
    def _find_latest_results(self) -> str:
        """最新の結果ファイルを検索"""
        results_dir = Path("rigorous_verification_results")
        if not results_dir.exists():
            raise FileNotFoundError("結果ディレクトリが見つかりません")
        
        json_files = list(results_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError("結果ファイルが見つかりません")
        
        # 最新ファイルを選択
        latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
        return str(latest_file)
    
    def _load_results(self) -> Dict:
        """結果ファイルの読み込み"""
        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise RuntimeError(f"結果ファイル読み込みエラー: {e}")
    
    def create_comprehensive_visualization(self):
        """包括的可視化の作成"""
        fig = plt.figure(figsize=(20, 16))
        
        # 1. 臨界線収束解析
        ax1 = plt.subplot(2, 3, 1)
        self._plot_critical_line_convergence(ax1)
        
        # 2. スペクトル次元分布
        ax2 = plt.subplot(2, 3, 2)
        self._plot_spectral_dimension_distribution(ax2)
        
        # 3. GUE統計距離
        ax3 = plt.subplot(2, 3, 3)
        self._plot_gue_statistical_distance(ax3)
        
        # 4. レベル間隔統計
        ax4 = plt.subplot(2, 3, 4)
        self._plot_level_spacing_statistics(ax4)
        
        # 5. 収束度vs γ値
        ax5 = plt.subplot(2, 3, 5)
        self._plot_convergence_vs_gamma(ax5)
        
        # 6. 総合評価レーダーチャート
        ax6 = plt.subplot(2, 3, 6, projection='polar')
        self._plot_comprehensive_radar(ax6)
        
        plt.tight_layout()
        
        # 保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"nkat_v11_comprehensive_analysis_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 包括的解析図を保存: {filename}")
        
        plt.show()
    
    def _plot_critical_line_convergence(self, ax):
        """臨界線収束解析プロット"""
        spectral_analysis = self.results['critical_line_verification']['spectral_analysis']
        
        gamma_values = [item['gamma'] for item in spectral_analysis]
        convergences = [item['convergence_to_half'] for item in spectral_analysis]
        
        ax.scatter(gamma_values, convergences, alpha=0.7, s=100, c='blue', edgecolors='black')
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='理論値 (Re(s)=1/2)')
        
        # 平均線
        mean_convergence = np.mean(convergences)
        ax.axhline(y=mean_convergence, color='green', linestyle='-', alpha=0.7, 
                  label=f'平均収束度: {mean_convergence:.4f}')
        
        ax.set_xlabel('γ値')
        ax.set_ylabel('収束度 (|Re(s) - 0.5|)')
        ax.set_title('🎯 臨界線収束解析\n非可換KA理論による検証')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_spectral_dimension_distribution(self, ax):
        """スペクトル次元分布プロット"""
        spectral_analysis = self.results['critical_line_verification']['spectral_analysis']
        
        spectral_dims = [item['spectral_dimension'] for item in spectral_analysis]
        
        ax.hist(spectral_dims, bins=8, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(x=np.mean(spectral_dims), color='red', linestyle='--', 
                  label=f'平均: {np.mean(spectral_dims):.6f}')
        ax.axvline(x=np.median(spectral_dims), color='green', linestyle='--', 
                  label=f'中央値: {np.median(spectral_dims):.6f}')
        
        ax.set_xlabel('スペクトル次元')
        ax.set_ylabel('頻度')
        ax.set_title('📊 スペクトル次元分布\n量子GUE統計解析')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_gue_statistical_distance(self, ax):
        """GUE統計距離プロット"""
        spectral_analysis = self.results['critical_line_verification']['spectral_analysis']
        
        gamma_values = [item['gamma'] for item in spectral_analysis]
        gue_distances = [item['gue_statistical_distance'] for item in spectral_analysis]
        
        ax.plot(gamma_values, gue_distances, 'o-', linewidth=2, markersize=8, 
               color='purple', alpha=0.8)
        
        # 平均線
        mean_distance = np.mean(gue_distances)
        ax.axhline(y=mean_distance, color='orange', linestyle='--', 
                  label=f'平均距離: {mean_distance:.2f}')
        
        ax.set_xlabel('γ値')
        ax.set_ylabel('GUE統計距離')
        ax.set_title('🔬 量子GUE統計距離\nWigner-Dyson理論との比較')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_level_spacing_statistics(self, ax):
        """レベル間隔統計プロット"""
        spectral_analysis = self.results['critical_line_verification']['spectral_analysis']
        
        # 正規化分散の抽出
        normalized_variances = [item['level_spacing_stats']['normalized_variance'] 
                              for item in spectral_analysis]
        theoretical_var = spectral_analysis[0]['level_spacing_stats']['theoretical_variance']
        
        gamma_values = [item['gamma'] for item in spectral_analysis]
        
        ax.semilogy(gamma_values, normalized_variances, 'o-', linewidth=2, markersize=8, 
                   color='red', alpha=0.8, label='観測値')
        ax.axhline(y=theoretical_var, color='blue', linestyle='--', linewidth=2,
                  label=f'GUE理論値: {theoretical_var:.3f}')
        
        ax.set_xlabel('γ値')
        ax.set_ylabel('正規化分散 (対数スケール)')
        ax.set_title('📈 レベル間隔統計\nGUE理論との比較')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_convergence_vs_gamma(self, ax):
        """収束度 vs γ値の関係プロット"""
        spectral_analysis = self.results['critical_line_verification']['spectral_analysis']
        
        gamma_values = [item['gamma'] for item in spectral_analysis]
        convergences = [item['convergence_to_half'] for item in spectral_analysis]
        spectral_dims = [item['spectral_dimension'] for item in spectral_analysis]
        
        # カラーマップでスペクトル次元を表現
        scatter = ax.scatter(gamma_values, convergences, c=spectral_dims, 
                           s=150, alpha=0.8, cmap='viridis', edgecolors='black')
        
        # カラーバー
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('スペクトル次元')
        
        # 理論値線
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, 
                  label='理論値 (Re(s)=1/2)')
        
        ax.set_xlabel('γ値')
        ax.set_ylabel('収束度')
        ax.set_title('🎯 収束度 vs γ値\nスペクトル次元による色分け')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_comprehensive_radar(self, ax):
        """総合評価レーダーチャート"""
        # 評価項目
        categories = [
            '数学的厳密性',
            '証明完全性', 
            '統計的有意性',
            '臨界線収束',
            'GUE適合性',
            'スペクトル一貫性'
        ]
        
        # スコア計算
        rigor_score = self.results['mathematical_rigor_score']
        completeness_score = self.results['proof_completeness']
        significance_score = self.results['statistical_significance']
        
        # 臨界線収束スコア (1 - 平均収束度)
        spectral_analysis = self.results['critical_line_verification']['spectral_analysis']
        convergences = [item['convergence_to_half'] for item in spectral_analysis]
        critical_score = 1.0 - np.mean(convergences)
        
        # GUE適合性スコア (KS p値ベース)
        gue_correlation = self.results['gue_correlation_analysis']
        gue_score = max(0, 1.0 - abs(gue_correlation['ks_statistic']))
        
        # スペクトル一貫性スコア (分散の逆数ベース)
        spectral_dims = [item['spectral_dimension'] for item in spectral_analysis]
        spectral_consistency = 1.0 / (1.0 + np.std(spectral_dims))
        
        scores = [
            rigor_score,
            completeness_score,
            significance_score,
            critical_score,
            gue_score,
            spectral_consistency
        ]
        
        # レーダーチャート作成
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        scores += scores[:1]  # 閉じるために最初の値を追加
        angles += angles[:1]
        
        ax.plot(angles, scores, 'o-', linewidth=3, color='blue', alpha=0.8)
        ax.fill(angles, scores, alpha=0.25, color='blue')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title('🏆 NKAT v11.0 総合評価\n数理的精緻化検証結果', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # グリッド
        ax.grid(True, alpha=0.3)
        
        # スコア値をテキストで表示
        for angle, score, category in zip(angles[:-1], scores[:-1], categories):
            ax.text(angle, score + 0.05, f'{score:.3f}', 
                   ha='center', va='center', fontsize=9, fontweight='bold')
    
    def create_detailed_analysis_plots(self):
        """詳細解析プロットの作成"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. γ値別詳細統計
        self._plot_gamma_detailed_stats(axes[0, 0])
        
        # 2. 理論値との偏差解析
        self._plot_theoretical_deviation_analysis(axes[0, 1])
        
        # 3. 時系列収束解析
        self._plot_convergence_timeline(axes[1, 0])
        
        # 4. 統計的有意性解析
        self._plot_statistical_significance_analysis(axes[1, 1])
        
        plt.tight_layout()
        
        # 保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"nkat_v11_detailed_analysis_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📈 詳細解析図を保存: {filename}")
        
        plt.show()
    
    def _plot_gamma_detailed_stats(self, ax):
        """γ値別詳細統計プロット"""
        spectral_analysis = self.results['critical_line_verification']['spectral_analysis']
        
        gamma_values = [item['gamma'] for item in spectral_analysis]
        real_parts = [item['real_part'] for item in spectral_analysis]
        
        # エラーバー付きプロット
        errors = [abs(rp - 0.5) for rp in real_parts]
        
        ax.errorbar(gamma_values, real_parts, yerr=errors, 
                   fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, 
                  label='理論値 (Re(s)=1/2)')
        
        ax.set_xlabel('γ値')
        ax.set_ylabel('実部 Re(s)')
        ax.set_title('🎯 γ値別実部詳細統計\nエラーバー付き')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_theoretical_deviation_analysis(self, ax):
        """理論値との偏差解析プロット"""
        spectral_analysis = self.results['critical_line_verification']['spectral_analysis']
        
        # Wigner-Dyson理論値との偏差
        wigner_deviations = [item['level_spacing_stats']['wigner_dyson_deviation'] 
                           for item in spectral_analysis]
        variance_deviations = [item['level_spacing_stats']['variance_deviation'] 
                             for item in spectral_analysis]
        
        gamma_values = [item['gamma'] for item in spectral_analysis]
        
        ax2 = ax.twinx()
        
        line1 = ax.plot(gamma_values, wigner_deviations, 'o-', color='blue', 
                       label='Wigner-Dyson偏差', linewidth=2, markersize=6)
        line2 = ax2.plot(gamma_values, variance_deviations, 's-', color='red', 
                        label='分散偏差', linewidth=2, markersize=6)
        
        ax.set_xlabel('γ値')
        ax.set_ylabel('Wigner-Dyson偏差', color='blue')
        ax2.set_ylabel('分散偏差', color='red')
        ax.set_title('📊 理論値との偏差解析\n二軸プロット')
        
        # 凡例統合
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        
        ax.grid(True, alpha=0.3)
    
    def _plot_convergence_timeline(self, ax):
        """時系列収束解析プロット"""
        spectral_analysis = self.results['critical_line_verification']['spectral_analysis']
        
        # γ値順にソート
        sorted_analysis = sorted(spectral_analysis, key=lambda x: x['gamma'])
        
        convergences = [item['convergence_to_half'] for item in sorted_analysis]
        cumulative_mean = np.cumsum(convergences) / np.arange(1, len(convergences) + 1)
        
        ax.plot(range(1, len(convergences) + 1), convergences, 'o-', 
               alpha=0.7, label='個別収束度', linewidth=2, markersize=6)
        ax.plot(range(1, len(cumulative_mean) + 1), cumulative_mean, 's-', 
               color='red', label='累積平均', linewidth=3, markersize=8)
        
        ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, 
                  label='理論値 (0.5)')
        
        ax.set_xlabel('γ値インデックス (昇順)')
        ax.set_ylabel('収束度')
        ax.set_title('📈 時系列収束解析\n累積平均による安定性評価')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_statistical_significance_analysis(self, ax):
        """統計的有意性解析プロット"""
        # 各種統計指標の可視化
        rigor_score = self.results['mathematical_rigor_score']
        completeness_score = self.results['proof_completeness']
        significance_score = self.results['statistical_significance']
        
        spectral_analysis = self.results['critical_line_verification']['spectral_analysis']
        convergences = [item['convergence_to_half'] for item in spectral_analysis]
        critical_line_property = np.mean(convergences)
        
        metrics = ['数学的厳密性', '証明完全性', '統計的有意性', '臨界線性質']
        scores = [rigor_score, completeness_score, significance_score, 1.0 - critical_line_property]
        colors = ['blue', 'green', 'orange', 'purple']
        
        bars = ax.bar(metrics, scores, color=colors, alpha=0.7, edgecolor='black')
        
        # 値をバーの上に表示
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('スコア')
        ax.set_title('📊 統計的有意性解析\n各種評価指標')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 回転したラベル
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def generate_summary_report(self):
        """サマリーレポートの生成"""
        print("=" * 80)
        print("🎯 NKAT v11.0 数理的精緻化検証 - サマリーレポート")
        print("=" * 80)
        
        # 基本情報
        print(f"📅 検証日時: {self.results['verification_timestamp']}")
        print(f"🔬 手法: 非可換コルモゴロフ・アーノルド × 量子GUE")
        print(f"📊 演算子次元: {self.results['noncommutative_ka_structure']['dimension']}")
        print(f"🎯 非可換パラメータ: {self.results['noncommutative_ka_structure']['noncomm_parameter']}")
        
        # 検証結果
        print("\n📈 検証結果:")
        print(f"  数学的厳密性: {self.results['mathematical_rigor_score']:.3f}")
        print(f"  証明完全性: {self.results['proof_completeness']:.3f}")
        print(f"  統計的有意性: {self.results['statistical_significance']:.3f}")
        
        # 臨界線検証
        critical_results = self.results['critical_line_verification']
        print(f"\n🎯 臨界線検証:")
        print(f"  検証γ値数: {len(critical_results['spectral_analysis'])}")
        print(f"  平均収束度: {critical_results['critical_line_property']:.6f}")
        print(f"  検証成功: {critical_results['verification_success']}")
        
        # GUE相関
        gue_results = self.results['gue_correlation_analysis']
        print(f"\n🔬 量子GUE相関:")
        print(f"  KS統計量: {gue_results['ks_statistic']:.3f}")
        print(f"  p値: {gue_results['ks_pvalue']:.6f}")
        print(f"  分布類似性: {gue_results['distributions_similar']}")
        
        # 統計サマリー
        spectral_analysis = critical_results['spectral_analysis']
        convergences = [item['convergence_to_half'] for item in spectral_analysis]
        spectral_dims = [item['spectral_dimension'] for item in spectral_analysis]
        
        print(f"\n📊 統計サマリー:")
        print(f"  収束度 - 平均: {np.mean(convergences):.6f}, 標準偏差: {np.std(convergences):.6f}")
        print(f"  スペクトル次元 - 平均: {np.mean(spectral_dims):.6f}, 標準偏差: {np.std(spectral_dims):.6f}")
        
        print("=" * 80)

def main():
    """メイン実行関数"""
    try:
        print("🎯 NKAT v11.0 結果可視化システム開始")
        
        # 可視化システム初期化
        visualizer = NKATResultsVisualizer()
        
        # サマリーレポート生成
        visualizer.generate_summary_report()
        
        # 包括的可視化
        print("\n📊 包括的可視化を作成中...")
        visualizer.create_comprehensive_visualization()
        
        # 詳細解析プロット
        print("\n📈 詳細解析プロットを作成中...")
        visualizer.create_detailed_analysis_plots()
        
        print("\n🎉 NKAT v11.0 結果可視化完了！")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")

if __name__ == "__main__":
    main() 