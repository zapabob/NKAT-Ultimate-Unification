#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 NKAT v11 詳細収束分析システム - 0.497762収束結果の深掘り
NKAT v11 Detailed Convergence Analyzer - Deep Analysis of 0.497762 Convergence

Author: NKAT Research Consortium
Date: 2025-05-26
Version: 11.0 - Detailed Convergence Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
import warnings
from typing import Dict, List, Tuple, Optional
import logging

# 日本語フォント設定
plt.rcParams['font.family'] = ['MS Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NKATConvergenceAnalyzer:
    """NKAT v11 詳細収束分析クラス"""
    
    def __init__(self):
        self.results_paths = [
            "rigorous_verification_results",
            "enhanced_verification_results",
            "../rigorous_verification_results"
        ]
        self.output_dir = Path("convergence_analysis_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # 分析結果保存用
        self.analysis_results = {
            "timestamp": datetime.now().isoformat(),
            "convergence_analysis": {},
            "statistical_analysis": {},
            "theoretical_comparison": {},
            "improvement_suggestions": []
        }
    
    def load_latest_results(self) -> Optional[Dict]:
        """最新の厳密検証結果を読み込み"""
        try:
            for results_path in self.results_paths:
                path = Path(results_path)
                if path.exists():
                    files = list(path.glob("*rigorous_verification*.json"))
                    if files:
                        latest_file = max(files, key=lambda x: x.stat().st_mtime)
                        logger.info(f"📁 最新結果ファイル読み込み: {latest_file}")
                        with open(latest_file, 'r', encoding='utf-8') as f:
                            return json.load(f)
            return None
        except Exception as e:
            logger.error(f"結果読み込みエラー: {e}")
            return None
    
    def extract_convergence_data(self, results: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """収束データの抽出"""
        if 'critical_line_verification' not in results:
            raise ValueError("臨界線検証データが見つかりません")
        
        spectral_analysis = results['critical_line_verification'].get('spectral_analysis', [])
        if not spectral_analysis:
            raise ValueError("スペクトル解析データが見つかりません")
        
        gamma_values = np.array([item['gamma'] for item in spectral_analysis])
        convergences = np.array([item['convergence_to_half'] for item in spectral_analysis])
        real_parts = np.array([item['real_part'] for item in spectral_analysis])
        
        logger.info(f"📊 データ抽出完了: {len(gamma_values)}個のγ値")
        logger.info(f"🎯 平均収束度: {np.mean(convergences):.8f}")
        
        return gamma_values, convergences, real_parts
    
    def analyze_convergence_pattern(self, gamma_values: np.ndarray, 
                                  convergences: np.ndarray) -> Dict:
        """収束パターンの詳細分析"""
        logger.info("🔍 収束パターン分析開始...")
        
        analysis = {
            "basic_statistics": {
                "mean": float(np.mean(convergences)),
                "std": float(np.std(convergences)),
                "min": float(np.min(convergences)),
                "max": float(np.max(convergences)),
                "median": float(np.median(convergences)),
                "q25": float(np.percentile(convergences, 25)),
                "q75": float(np.percentile(convergences, 75))
            },
            "theoretical_deviation": {
                "mean_deviation_from_half": float(np.mean(np.abs(convergences - 0.5))),
                "max_deviation_from_half": float(np.max(np.abs(convergences - 0.5))),
                "relative_error": float(np.mean(np.abs(convergences - 0.5)) / 0.5 * 100)
            },
            "stability_metrics": {
                "coefficient_of_variation": float(np.std(convergences) / np.mean(convergences)),
                "range": float(np.max(convergences) - np.min(convergences)),
                "iqr": float(np.percentile(convergences, 75) - np.percentile(convergences, 25))
            }
        }
        
        # 収束性の品質評価
        mean_conv = analysis["basic_statistics"]["mean"]
        if mean_conv > 0.497:
            quality = "優秀"
        elif mean_conv > 0.495:
            quality = "良好"
        elif mean_conv > 0.49:
            quality = "普通"
        else:
            quality = "要改善"
        
        analysis["quality_assessment"] = {
            "overall_quality": quality,
            "convergence_score": float(1.0 - np.mean(np.abs(convergences - 0.5))),
            "consistency_score": float(1.0 / (1.0 + np.std(convergences)))
        }
        
        logger.info(f"✅ 収束パターン分析完了: 品質={quality}")
        return analysis
    
    def analyze_gamma_dependency(self, gamma_values: np.ndarray, 
                                convergences: np.ndarray) -> Dict:
        """γ値依存性の分析"""
        logger.info("📈 γ値依存性分析開始...")
        
        # 相関分析
        correlation = np.corrcoef(gamma_values, convergences)[0, 1]
        
        # 線形回帰
        slope, intercept, r_value, p_value, std_err = stats.linregress(gamma_values, convergences)
        
        # 多項式フィッティング
        poly_coeffs = np.polyfit(gamma_values, convergences, 2)
        poly_func = np.poly1d(poly_coeffs)
        
        # 残差分析
        linear_pred = slope * gamma_values + intercept
        poly_pred = poly_func(gamma_values)
        
        linear_residuals = convergences - linear_pred
        poly_residuals = convergences - poly_pred
        
        analysis = {
            "correlation": {
                "pearson_correlation": float(correlation),
                "correlation_strength": "強い" if abs(correlation) > 0.7 else "中程度" if abs(correlation) > 0.3 else "弱い"
            },
            "linear_regression": {
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_value**2),
                "p_value": float(p_value),
                "standard_error": float(std_err)
            },
            "polynomial_fit": {
                "coefficients": [float(c) for c in poly_coeffs],
                "linear_rmse": float(np.sqrt(np.mean(linear_residuals**2))),
                "polynomial_rmse": float(np.sqrt(np.mean(poly_residuals**2)))
            },
            "trend_analysis": {
                "overall_trend": "増加" if slope > 0 else "減少" if slope < 0 else "平坦",
                "trend_significance": "有意" if p_value < 0.05 else "非有意"
            }
        }
        
        logger.info(f"✅ γ値依存性分析完了: 相関={correlation:.4f}")
        return analysis
    
    def theoretical_comparison(self, convergences: np.ndarray) -> Dict:
        """理論値との比較分析"""
        logger.info("🎯 理論値比較分析開始...")
        
        theoretical_value = 0.5
        deviations = np.abs(convergences - theoretical_value)
        
        # 統計的検定
        t_stat, t_p_value = stats.ttest_1samp(convergences, theoretical_value)
        
        # 正規性検定
        shapiro_stat, shapiro_p = stats.shapiro(convergences)
        
        # 信頼区間
        confidence_interval = stats.t.interval(0.95, len(convergences)-1, 
                                             loc=np.mean(convergences), 
                                             scale=stats.sem(convergences))
        
        analysis = {
            "deviation_statistics": {
                "mean_absolute_deviation": float(np.mean(deviations)),
                "max_absolute_deviation": float(np.max(deviations)),
                "min_absolute_deviation": float(np.min(deviations)),
                "std_deviation": float(np.std(deviations))
            },
            "statistical_tests": {
                "t_test": {
                    "statistic": float(t_stat),
                    "p_value": float(t_p_value),
                    "significant_difference": bool(t_p_value < 0.05)
                },
                "normality_test": {
                    "shapiro_statistic": float(shapiro_stat),
                    "shapiro_p_value": float(shapiro_p),
                    "is_normal": bool(shapiro_p > 0.05)
                }
            },
            "confidence_interval": {
                "lower_bound": float(confidence_interval[0]),
                "upper_bound": float(confidence_interval[1]),
                "contains_theoretical": bool(confidence_interval[0] <= theoretical_value <= confidence_interval[1])
            },
            "precision_metrics": {
                "relative_precision": float(np.std(convergences) / np.mean(convergences) * 100),
                "accuracy": float(1.0 - np.mean(deviations)),
                "precision_score": float(1.0 / (1.0 + np.std(convergences)))
            }
        }
        
        logger.info(f"✅ 理論値比較完了: 精度={analysis['precision_metrics']['accuracy']:.6f}")
        return analysis
    
    def generate_improvement_suggestions(self, convergence_analysis: Dict, 
                                       gamma_analysis: Dict, 
                                       theoretical_analysis: Dict) -> List[str]:
        """改善提案の生成"""
        suggestions = []
        
        # 収束性に基づく提案
        mean_conv = convergence_analysis["basic_statistics"]["mean"]
        if mean_conv < 0.498:
            suggestions.append("🔧 ハミルトニアン次元を増加させて精度向上を図る")
            suggestions.append("⚙️ 非可換パラメータθの最適化を実施")
        
        # 安定性に基づく提案
        cv = convergence_analysis["stability_metrics"]["coefficient_of_variation"]
        if cv > 0.01:
            suggestions.append("📊 数値安定性向上のため正則化項を調整")
            suggestions.append("🎯 適応的次元調整アルゴリズムの改良")
        
        # γ値依存性に基づく提案
        if abs(gamma_analysis["correlation"]["pearson_correlation"]) > 0.5:
            suggestions.append("📈 γ値依存性を考慮した適応的パラメータ調整")
            suggestions.append("🔄 γ値範囲別の最適化戦略の実装")
        
        # 理論値からの偏差に基づく提案
        if theoretical_analysis["deviation_statistics"]["mean_absolute_deviation"] > 0.003:
            suggestions.append("🎯 高精度演算ライブラリの導入検討")
            suggestions.append("💻 GPU計算精度の向上")
        
        # 統計的有意性に基づく提案
        if theoretical_analysis["statistical_tests"]["t_test"]["significant_difference"]:
            suggestions.append("📊 系統的バイアスの調査と補正")
            suggestions.append("🔬 理論モデルの再検討")
        
        if not suggestions:
            suggestions.append("🎉 現在の収束性は優秀です！さらなる精度向上のため大規模検証を推奨")
        
        return suggestions
    
    def create_comprehensive_visualization(self, gamma_values: np.ndarray, 
                                         convergences: np.ndarray, 
                                         real_parts: np.ndarray):
        """包括的可視化の作成"""
        logger.info("📊 包括的可視化作成開始...")
        
        # 図のセットアップ
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. 収束性の時系列プロット
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(gamma_values, convergences, 'o-', color='red', linewidth=2, markersize=8, label='収束度')
        ax1.axhline(y=0.5, color='green', linestyle='--', linewidth=2, label='理論値 (1/2)')
        ax1.fill_between(gamma_values, convergences - np.std(convergences), 
                        convergences + np.std(convergences), alpha=0.3, color='red')
        ax1.set_xlabel('γ値')
        ax1.set_ylabel('収束度')
        ax1.set_title('🎯 臨界線収束性の詳細分析', fontsize=16, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 収束度のヒストグラム
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(convergences, bins=15, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(x=np.mean(convergences), color='red', linestyle='--', 
                   label=f'平均: {np.mean(convergences):.6f}')
        ax2.axvline(x=0.5, color='green', linestyle='--', label='理論値: 0.5')
        ax2.set_xlabel('収束度')
        ax2.set_ylabel('頻度')
        ax2.set_title('📊 収束度分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 実部の分析
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(gamma_values, real_parts, 's-', color='purple', linewidth=2, markersize=6)
        ax3.axhline(y=0.5, color='green', linestyle='--', linewidth=2)
        ax3.set_xlabel('γ値')
        ax3.set_ylabel('実部')
        ax3.set_title('🔬 実部の変化')
        ax3.grid(True, alpha=0.3)
        
        # 4. 理論値からの偏差
        ax4 = fig.add_subplot(gs[1, 2])
        deviations = np.abs(convergences - 0.5)
        ax4.plot(gamma_values, deviations, '^-', color='orange', linewidth=2, markersize=6)
        ax4.set_xlabel('γ値')
        ax4.set_ylabel('|収束度 - 1/2|')
        ax4.set_title('📏 理論値からの偏差')
        ax4.grid(True, alpha=0.3)
        
        # 5. 相関分析
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.scatter(gamma_values, convergences, c=range(len(gamma_values)), 
                   cmap='viridis', s=100, alpha=0.7)
        z = np.polyfit(gamma_values, convergences, 1)
        p = np.poly1d(z)
        ax5.plot(gamma_values, p(gamma_values), "r--", alpha=0.8, linewidth=2)
        ax5.set_xlabel('γ値')
        ax5.set_ylabel('収束度')
        ax5.set_title('📈 γ値 vs 収束度')
        ax5.grid(True, alpha=0.3)
        
        # 6. 統計的品質評価
        ax6 = fig.add_subplot(gs[2, 1])
        metrics = ['平均', '標準偏差', '最小値', '最大値']
        values = [np.mean(convergences), np.std(convergences), 
                 np.min(convergences), np.max(convergences)]
        colors = ['blue', 'orange', 'green', 'red']
        bars = ax6.bar(metrics, values, color=colors, alpha=0.7)
        ax6.set_ylabel('値')
        ax6.set_title('📊 統計的品質指標')
        ax6.grid(True, alpha=0.3)
        
        # 値をバーの上に表示
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.6f}', ha='center', va='bottom', fontsize=10)
        
        # 7. 収束性の時間発展（移動平均）
        ax7 = fig.add_subplot(gs[2, 2])
        window_size = min(5, len(convergences)//2)
        if window_size >= 2:
            moving_avg = pd.Series(convergences).rolling(window=window_size, center=True).mean()
            ax7.plot(gamma_values, convergences, 'o', alpha=0.5, color='gray', label='生データ')
            ax7.plot(gamma_values, moving_avg, '-', color='red', linewidth=3, label=f'移動平均({window_size})')
            ax7.axhline(y=0.5, color='green', linestyle='--', linewidth=2, label='理論値')
            ax7.set_xlabel('γ値')
            ax7.set_ylabel('収束度')
            ax7.set_title('📈 収束性の平滑化トレンド')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        
        # 8. 精度評価サマリー
        ax8 = fig.add_subplot(gs[3, :])
        ax8.axis('off')
        
        # サマリーテキスト
        summary_text = f"""
        🎯 NKAT v11 詳細収束分析サマリー
        
        📊 基本統計:
        • 平均収束度: {np.mean(convergences):.8f}
        • 標準偏差: {np.std(convergences):.8f}
        • 理論値からの平均偏差: {np.mean(np.abs(convergences - 0.5)):.8f}
        • 相対誤差: {np.mean(np.abs(convergences - 0.5)) / 0.5 * 100:.4f}%
        
        🎯 品質評価:
        • 精度スコア: {1.0 - np.mean(np.abs(convergences - 0.5)):.6f}
        • 一貫性スコア: {1.0 / (1.0 + np.std(convergences)):.6f}
        • 全体品質: {"優秀" if np.mean(convergences) > 0.497 else "良好" if np.mean(convergences) > 0.495 else "普通"}
        
        📈 統計的有意性:
        • γ値との相関: {np.corrcoef(gamma_values, convergences)[0, 1]:.4f}
        • 変動係数: {np.std(convergences) / np.mean(convergences):.6f}
        """
        
        ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.suptitle('🚀 NKAT v11 包括的収束分析ダッシュボード', fontsize=20, fontweight='bold')
        
        # 保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"nkat_v11_comprehensive_convergence_analysis_{timestamp}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"📊 可視化保存完了: {output_file}")
        
        plt.show()
        return output_file
    
    def run_comprehensive_analysis(self) -> Dict:
        """包括的分析の実行"""
        logger.info("🚀 NKAT v11 詳細収束分析開始")
        print("=" * 80)
        print("🎯 NKAT v11 詳細収束分析システム")
        print("=" * 80)
        print(f"📅 分析開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🔬 目標: 0.497762収束結果の深掘り分析")
        print("=" * 80)
        
        # 1. データ読み込み
        results = self.load_latest_results()
        if not results:
            raise ValueError("検証結果が見つかりません")
        
        # 2. データ抽出
        gamma_values, convergences, real_parts = self.extract_convergence_data(results)
        
        # 3. 収束パターン分析
        convergence_analysis = self.analyze_convergence_pattern(gamma_values, convergences)
        self.analysis_results["convergence_analysis"] = convergence_analysis
        
        # 4. γ値依存性分析
        gamma_analysis = self.analyze_gamma_dependency(gamma_values, convergences)
        self.analysis_results["gamma_dependency"] = gamma_analysis
        
        # 5. 理論値比較
        theoretical_analysis = self.theoretical_comparison(convergences)
        self.analysis_results["theoretical_comparison"] = theoretical_analysis
        
        # 6. 改善提案生成
        suggestions = self.generate_improvement_suggestions(
            convergence_analysis, gamma_analysis, theoretical_analysis
        )
        self.analysis_results["improvement_suggestions"] = suggestions
        
        # 7. 可視化作成
        visualization_file = self.create_comprehensive_visualization(
            gamma_values, convergences, real_parts
        )
        self.analysis_results["visualization_file"] = str(visualization_file)
        
        # 8. 結果保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"nkat_v11_detailed_analysis_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)
        
        # 9. 結果表示
        self.display_analysis_summary()
        
        logger.info(f"💾 分析結果保存: {results_file}")
        print(f"\n💾 詳細分析結果を保存しました: {results_file}")
        print("🎉 NKAT v11 詳細収束分析完了！")
        
        return self.analysis_results
    
    def display_analysis_summary(self):
        """分析サマリーの表示"""
        print("\n" + "=" * 80)
        print("📊 NKAT v11 詳細収束分析結果")
        print("=" * 80)
        
        # 基本統計
        conv_stats = self.analysis_results["convergence_analysis"]["basic_statistics"]
        print(f"\n🎯 基本統計:")
        print(f"  平均収束度: {conv_stats['mean']:.8f}")
        print(f"  標準偏差: {conv_stats['std']:.8f}")
        print(f"  最良収束: {conv_stats['min']:.8f}")
        print(f"  最悪収束: {conv_stats['max']:.8f}")
        
        # 品質評価
        quality = self.analysis_results["convergence_analysis"]["quality_assessment"]
        print(f"\n📈 品質評価:")
        print(f"  全体品質: {quality['overall_quality']}")
        print(f"  収束スコア: {quality['convergence_score']:.6f}")
        print(f"  一貫性スコア: {quality['consistency_score']:.6f}")
        
        # 理論値比較
        theoretical = self.analysis_results["theoretical_comparison"]
        print(f"\n🎯 理論値比較:")
        print(f"  平均絶対偏差: {theoretical['deviation_statistics']['mean_absolute_deviation']:.8f}")
        print(f"  相対誤差: {theoretical['precision_metrics']['relative_precision']:.4f}%")
        print(f"  精度: {theoretical['precision_metrics']['accuracy']:.6f}")
        
        # 改善提案
        print(f"\n🔧 改善提案:")
        for i, suggestion in enumerate(self.analysis_results["improvement_suggestions"], 1):
            print(f"  {i}. {suggestion}")
        
        print("=" * 80)

def main():
    """メイン実行関数"""
    try:
        analyzer = NKATConvergenceAnalyzer()
        results = analyzer.run_comprehensive_analysis()
        return results
    except Exception as e:
        logger.error(f"分析エラー: {e}")
        print(f"❌ 分析エラー: {e}")
        return None

if __name__ == "__main__":
    main() 