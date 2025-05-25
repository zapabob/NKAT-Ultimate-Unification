#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT v11 詳細収束分析システム
0.497762収束結果の深掘り分析

作成者: NKAT Research Team
作成日: 2025年5月26日
バージョン: v11.0
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

class ConvergenceAnalyzer:
    """収束分析クラス"""
    
    def __init__(self, results_file='high_precision_riemann_results.json'):
        """初期化"""
        self.results_file = results_file
        self.data = self.load_data()
        self.analysis_results = {}
        
    def load_data(self):
        """データを読み込み"""
        if not os.path.exists(self.results_file):
            raise FileNotFoundError(f"結果ファイルが見つかりません: {self.results_file}")
        
        with open(self.results_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def analyze_convergence_patterns(self):
        """収束パターンを分析"""
        print("🔍 収束パターン分析を開始...")
        
        if 'convergence_to_half_all' not in self.data:
            print("❌ 収束データが見つかりません")
            return
        
        convergence_data = np.array(self.data['convergence_to_half_all'])
        
        # 基本統計
        mean_convergence = np.mean(convergence_data)
        std_convergence = np.std(convergence_data)
        min_convergence = np.min(convergence_data)
        max_convergence = np.max(convergence_data)
        
        # 理論値からの偏差
        theoretical_value = 0.5
        deviation = abs(mean_convergence - theoretical_value)
        relative_error = (deviation / theoretical_value) * 100
        
        # 安定性指標
        coefficient_of_variation = (std_convergence / mean_convergence) * 100
        
        self.analysis_results['convergence_patterns'] = {
            'mean_convergence': mean_convergence,
            'std_convergence': std_convergence,
            'min_convergence': min_convergence,
            'max_convergence': max_convergence,
            'theoretical_deviation': deviation,
            'relative_error_percent': relative_error,
            'coefficient_of_variation': coefficient_of_variation,
            'data_shape': convergence_data.shape,
            'total_samples': convergence_data.size
        }
        
        print(f"✅ 平均収束度: {mean_convergence:.8f}")
        print(f"✅ 標準偏差: {std_convergence:.8f}")
        print(f"✅ 理論値偏差: {deviation:.8f}")
        print(f"✅ 相対誤差: {relative_error:.4f}%")
        print(f"✅ 変動係数: {coefficient_of_variation:.6f}%")
        
        return self.analysis_results['convergence_patterns']
    
    def analyze_gamma_dependency(self):
        """γ値依存性を分析"""
        print("\n🔍 γ値依存性分析を開始...")
        
        if 'gamma_values' not in self.data or 'convergence_to_half_all' not in self.data:
            print("❌ 必要なデータが見つかりません")
            return
        
        gamma_values = np.array(self.data['gamma_values'])
        convergence_data = np.array(self.data['convergence_to_half_all'])
        
        # 各γ値に対する収束度の平均
        gamma_convergence_means = []
        for i in range(len(gamma_values)):
            if i < convergence_data.shape[1]:
                mean_conv = np.mean(convergence_data[:, i])
                gamma_convergence_means.append(mean_conv)
        
        gamma_convergence_means = np.array(gamma_convergence_means)
        
        # 相関分析
        if len(gamma_values) == len(gamma_convergence_means):
            correlation, p_value = stats.pearsonr(gamma_values, gamma_convergence_means)
            
            # 線形回帰
            slope, intercept, r_value, p_value_reg, std_err = stats.linregress(
                gamma_values, gamma_convergence_means
            )
            
            # 多項式フィッティング
            poly_coeffs = np.polyfit(gamma_values, gamma_convergence_means, 2)
            poly_func = np.poly1d(poly_coeffs)
            
            self.analysis_results['gamma_dependency'] = {
                'gamma_values': gamma_values.tolist(),
                'convergence_means': gamma_convergence_means.tolist(),
                'correlation': correlation,
                'correlation_p_value': p_value,
                'linear_slope': slope,
                'linear_intercept': intercept,
                'r_squared': r_value**2,
                'polynomial_coefficients': poly_coeffs.tolist()
            }
            
            print(f"✅ γ値との相関: {correlation:.6f} (p={p_value:.6f})")
            print(f"✅ 線形回帰 R²: {r_value**2:.6f}")
            print(f"✅ 線形傾き: {slope:.8f}")
        
        return self.analysis_results.get('gamma_dependency', {})
    
    def analyze_theoretical_comparison(self):
        """理論値との比較分析"""
        print("\n🔍 理論値比較分析を開始...")
        
        if 'convergence_to_half_all' not in self.data:
            print("❌ 収束データが見つかりません")
            return
        
        convergence_data = np.array(self.data['convergence_to_half_all']).flatten()
        theoretical_value = 0.5
        
        # 統計的検定
        # 一標本t検定（理論値との比較）
        t_stat, t_p_value = stats.ttest_1samp(convergence_data, theoretical_value)
        
        # 正規性検定
        shapiro_stat, shapiro_p = stats.shapiro(convergence_data[:5000] if len(convergence_data) > 5000 else convergence_data)
        
        # 信頼区間
        confidence_level = 0.95
        degrees_freedom = len(convergence_data) - 1
        confidence_interval = stats.t.interval(
            confidence_level, degrees_freedom,
            loc=np.mean(convergence_data),
            scale=stats.sem(convergence_data)
        )
        
        # 効果量（Cohen's d）
        cohens_d = (np.mean(convergence_data) - theoretical_value) / np.std(convergence_data)
        
        self.analysis_results['theoretical_comparison'] = {
            'theoretical_value': theoretical_value,
            'sample_mean': np.mean(convergence_data),
            'sample_std': np.std(convergence_data),
            't_statistic': t_stat,
            't_p_value': t_p_value,
            'shapiro_statistic': shapiro_stat,
            'shapiro_p_value': shapiro_p,
            'confidence_interval': confidence_interval,
            'cohens_d': cohens_d,
            'sample_size': len(convergence_data)
        }
        
        print(f"✅ t検定統計量: {t_stat:.6f} (p={t_p_value:.6f})")
        print(f"✅ 正規性検定: {shapiro_stat:.6f} (p={shapiro_p:.6f})")
        print(f"✅ 95%信頼区間: [{confidence_interval[0]:.6f}, {confidence_interval[1]:.6f}]")
        print(f"✅ Cohen's d: {cohens_d:.6f}")
        
        return self.analysis_results['theoretical_comparison']
    
    def generate_improvement_suggestions(self):
        """改善提案を生成"""
        print("\n💡 改善提案を生成中...")
        
        suggestions = []
        
        # 収束パターン分析に基づく提案
        if 'convergence_patterns' in self.analysis_results:
            conv_analysis = self.analysis_results['convergence_patterns']
            
            if conv_analysis['relative_error_percent'] > 1.0:
                suggestions.append({
                    'category': '精度改善',
                    'priority': 'high',
                    'suggestion': 'より高精度な数値計算手法の導入を検討',
                    'details': f"現在の相対誤差: {conv_analysis['relative_error_percent']:.4f}%"
                })
            
            if conv_analysis['coefficient_of_variation'] > 0.01:
                suggestions.append({
                    'category': '安定性向上',
                    'priority': 'medium',
                    'suggestion': '計算の安定性向上のためのアルゴリズム最適化',
                    'details': f"変動係数: {conv_analysis['coefficient_of_variation']:.6f}%"
                })
        
        # γ値依存性分析に基づく提案
        if 'gamma_dependency' in self.analysis_results:
            gamma_analysis = self.analysis_results['gamma_dependency']
            
            if abs(gamma_analysis.get('correlation', 0)) > 0.5:
                suggestions.append({
                    'category': 'γ値最適化',
                    'priority': 'medium',
                    'suggestion': 'γ値選択の最適化による収束性向上',
                    'details': f"γ値相関: {gamma_analysis.get('correlation', 0):.6f}"
                })
        
        # 理論値比較に基づく提案
        if 'theoretical_comparison' in self.analysis_results:
            theory_analysis = self.analysis_results['theoretical_comparison']
            
            if theory_analysis['t_p_value'] < 0.05:
                suggestions.append({
                    'category': '理論検証',
                    'priority': 'high',
                    'suggestion': '理論値との有意差の原因調査と修正',
                    'details': f"t検定 p値: {theory_analysis['t_p_value']:.6f}"
                })
        
        # 一般的な提案
        suggestions.extend([
            {
                'category': '計算資源',
                'priority': 'low',
                'suggestion': 'より多くのサンプル数での検証実行',
                'details': '統計的信頼性の向上'
            },
            {
                'category': '手法拡張',
                'priority': 'medium',
                'suggestion': '異なる数値積分手法との比較検証',
                'details': '手法の妥当性確認'
            }
        ])
        
        self.analysis_results['improvement_suggestions'] = suggestions
        
        print(f"✅ {len(suggestions)}件の改善提案を生成しました")
        
        return suggestions
    
    def create_comprehensive_visualization(self):
        """包括的可視化を作成"""
        print("\n📊 包括的可視化を作成中...")
        
        # 出力ディレクトリ作成
        output_dir = "convergence_analysis_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # 大きなフィギュアサイズ設定
        fig = plt.figure(figsize=(20, 16))
        
        # サブプロット配置
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. 収束度ヒストグラム
        ax1 = fig.add_subplot(gs[0, 0])
        if 'convergence_to_half_all' in self.data:
            convergence_data = np.array(self.data['convergence_to_half_all']).flatten()
            ax1.hist(convergence_data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.axvline(0.5, color='red', linestyle='--', label='理論値 (0.5)')
            ax1.axvline(np.mean(convergence_data), color='orange', linestyle='-', label=f'平均値 ({np.mean(convergence_data):.6f})')
            ax1.set_xlabel('収束度')
            ax1.set_ylabel('頻度')
            ax1.set_title('収束度分布')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. γ値依存性
        ax2 = fig.add_subplot(gs[0, 1])
        if 'gamma_dependency' in self.analysis_results:
            gamma_dep = self.analysis_results['gamma_dependency']
            gamma_vals = gamma_dep['gamma_values']
            conv_means = gamma_dep['convergence_means']
            
            ax2.scatter(gamma_vals, conv_means, alpha=0.7, s=100, color='green')
            
            # 線形回帰線
            if 'linear_slope' in gamma_dep:
                x_line = np.linspace(min(gamma_vals), max(gamma_vals), 100)
                y_line = gamma_dep['linear_slope'] * x_line + gamma_dep['linear_intercept']
                ax2.plot(x_line, y_line, 'r--', label=f"線形回帰 (R²={gamma_dep.get('r_squared', 0):.4f})")
            
            ax2.set_xlabel('γ値')
            ax2.set_ylabel('平均収束度')
            ax2.set_title('γ値依存性')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. 理論値との偏差
        ax3 = fig.add_subplot(gs[0, 2])
        if 'convergence_to_half_all' in self.data:
            convergence_data = np.array(self.data['convergence_to_half_all']).flatten()
            deviations = convergence_data - 0.5
            
            ax3.hist(deviations, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
            ax3.axvline(0, color='black', linestyle='-', label='理論値からの偏差=0')
            ax3.axvline(np.mean(deviations), color='blue', linestyle='--', label=f'平均偏差 ({np.mean(deviations):.6f})')
            ax3.set_xlabel('理論値からの偏差')
            ax3.set_ylabel('頻度')
            ax3.set_title('理論値偏差分布')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. 時系列プロット（収束度）
        ax4 = fig.add_subplot(gs[1, :])
        if 'convergence_to_half_all' in self.data:
            convergence_data = np.array(self.data['convergence_to_half_all'])
            
            for i in range(min(convergence_data.shape[0], 5)):  # 最初の5系列
                ax4.plot(convergence_data[i], alpha=0.7, label=f'系列 {i+1}')
            
            ax4.axhline(0.5, color='red', linestyle='--', label='理論値 (0.5)')
            ax4.set_xlabel('γ値インデックス')
            ax4.set_ylabel('収束度')
            ax4.set_title('収束度時系列')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. 統計サマリー
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.axis('off')
        
        if 'convergence_patterns' in self.analysis_results:
            conv_stats = self.analysis_results['convergence_patterns']
            stats_text = f"""
統計サマリー
━━━━━━━━━━━━━━━━━━━━
平均収束度: {conv_stats['mean_convergence']:.8f}
標準偏差: {conv_stats['std_convergence']:.8f}
最小値: {conv_stats['min_convergence']:.8f}
最大値: {conv_stats['max_convergence']:.8f}
理論値偏差: {conv_stats['theoretical_deviation']:.8f}
相対誤差: {conv_stats['relative_error_percent']:.4f}%
変動係数: {conv_stats['coefficient_of_variation']:.6f}%
サンプル数: {conv_stats['total_samples']:,}
            """
            ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace')
        
        # 6. 品質評価
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis('off')
        
        # 品質スコア計算
        if 'convergence_patterns' in self.analysis_results:
            conv_stats = self.analysis_results['convergence_patterns']
            
            # 収束スコア（理論値に近いほど高い）
            convergence_score = 1 - abs(conv_stats['mean_convergence'] - 0.5) * 2
            
            # 一貫性スコア（標準偏差が小さいほど高い）
            consistency_score = 1 - min(conv_stats['std_convergence'] * 1000, 1)
            
            # 総合品質スコア
            overall_quality = (convergence_score + consistency_score) / 2
            
            quality_text = f"""
品質評価
━━━━━━━━━━━━━━━━━━━━
収束スコア: {convergence_score:.6f}
一貫性スコア: {consistency_score:.6f}
総合品質: {overall_quality:.6f}

評価:
"""
            if overall_quality > 0.95:
                quality_text += "🎉 優秀"
            elif overall_quality > 0.9:
                quality_text += "✅ 良好"
            elif overall_quality > 0.8:
                quality_text += "⚠️ 普通"
            else:
                quality_text += "❌ 要改善"
            
            ax6.text(0.1, 0.9, quality_text, transform=ax6.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace')
        
        # 7. 改善提案
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        
        if 'improvement_suggestions' in self.analysis_results:
            suggestions = self.analysis_results['improvement_suggestions']
            
            suggestion_text = "改善提案\n━━━━━━━━━━━━━━━━━━━━\n"
            for i, suggestion in enumerate(suggestions[:5]):  # 最初の5件
                priority_icon = "🔴" if suggestion['priority'] == 'high' else "🟡" if suggestion['priority'] == 'medium' else "🟢"
                suggestion_text += f"{priority_icon} {suggestion['category']}\n"
                suggestion_text += f"   {suggestion['suggestion']}\n\n"
            
            ax7.text(0.1, 0.9, suggestion_text, transform=ax7.transAxes, fontsize=9,
                    verticalalignment='top')
        
        # 8. スペクトル分析
        ax8 = fig.add_subplot(gs[3, :])
        if 'spectral_dimensions_all' in self.data:
            spectral_data = np.array(self.data['spectral_dimensions_all'])
            
            for i in range(min(spectral_data.shape[0], 3)):
                ax8.plot(spectral_data[i], alpha=0.7, marker='o', label=f'スペクトル次元 {i+1}')
            
            ax8.set_xlabel('γ値インデックス')
            ax8.set_ylabel('スペクトル次元')
            ax8.set_title('スペクトル次元分析')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
        
        # タイトル設定
        fig.suptitle('NKAT v11 詳細収束分析レポート\n0.497762収束結果の包括的分析', 
                     fontsize=16, fontweight='bold')
        
        # 保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"detailed_convergence_analysis_{timestamp}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        print(f"✅ 可視化を保存しました: {output_file}")
        
        return output_file
    
    def save_analysis_results(self):
        """分析結果を保存"""
        output_dir = "convergence_analysis_results"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"convergence_analysis_{timestamp}.json")
        
        # メタデータ追加
        self.analysis_results['metadata'] = {
            'analysis_timestamp': timestamp,
            'source_file': self.results_file,
            'analyzer_version': 'v11.0'
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ 分析結果を保存しました: {output_file}")
        
        return output_file
    
    def run_complete_analysis(self):
        """完全分析を実行"""
        print("🚀 NKAT v11 詳細収束分析を開始します...")
        print("=" * 60)
        
        try:
            # 各分析を実行
            self.analyze_convergence_patterns()
            self.analyze_gamma_dependency()
            self.analyze_theoretical_comparison()
            self.generate_improvement_suggestions()
            
            # 可視化作成
            visualization_file = self.create_comprehensive_visualization()
            
            # 結果保存
            results_file = self.save_analysis_results()
            
            print("\n" + "=" * 60)
            print("🎉 分析完了!")
            print(f"📊 可視化: {visualization_file}")
            print(f"📄 結果: {results_file}")
            
            return {
                'visualization': visualization_file,
                'results': results_file,
                'analysis_data': self.analysis_results
            }
            
        except Exception as e:
            print(f"❌ 分析中にエラーが発生しました: {e}")
            return None

def main():
    """メイン実行関数"""
    print("NKAT v11 詳細収束分析システム")
    print("0.497762収束結果の深掘り分析")
    print("=" * 50)
    
    try:
        # アナライザー初期化
        analyzer = ConvergenceAnalyzer()
        
        # 完全分析実行
        results = analyzer.run_complete_analysis()
        
        if results:
            print("\n📈 分析サマリー:")
            if 'convergence_patterns' in analyzer.analysis_results:
                conv_stats = analyzer.analysis_results['convergence_patterns']
                print(f"   平均収束度: {conv_stats['mean_convergence']:.8f}")
                print(f"   相対誤差: {conv_stats['relative_error_percent']:.4f}%")
                print(f"   品質評価: ", end="")
                
                convergence_score = 1 - abs(conv_stats['mean_convergence'] - 0.5) * 2
                consistency_score = 1 - min(conv_stats['std_convergence'] * 1000, 1)
                overall_quality = (convergence_score + consistency_score) / 2
                
                if overall_quality > 0.95:
                    print("🎉 優秀")
                elif overall_quality > 0.9:
                    print("✅ 良好")
                else:
                    print("⚠️ 要改善")
        
    except Exception as e:
        print(f"❌ エラー: {e}")

if __name__ == "__main__":
    main() 