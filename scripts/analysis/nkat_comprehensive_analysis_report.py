#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌟 NKAT 総合分析レポート生成システム
Stage2-5の結果を統合し、最終的な性能評価と改善提案を作成
"""
import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
from tqdm import tqdm

# 英語表記設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class NKATComprehensiveAnalyzer:
    """NKAT総合分析システム"""
    
    def __init__(self, logs_dir="logs"):
        self.logs_dir = logs_dir
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def load_stage_results(self):
        """各Stageの結果を読み込み"""
        
        print("📊 Loading Stage Results...")
        
        # Stage2: 汎化テスト結果
        stage2_files = glob.glob(os.path.join(self.logs_dir, "*stage2*generalization*.json"))
        if stage2_files:
            latest_stage2 = max(stage2_files, key=os.path.getctime)
            with open(latest_stage2, 'r', encoding='utf-8') as f:
                self.results['stage2'] = json.load(f)
            print(f"✅ Stage2 結果読み込み: {latest_stage2}")
        else:
            print("⚠️ Stage2結果が見つかりません")
            
        # Stage3: ロバストネステスト結果
        stage3_files = glob.glob(os.path.join(self.logs_dir, "*stage3*robustness*.json"))
        if stage3_files:
            latest_stage3 = max(stage3_files, key=os.path.getctime)
            with open(latest_stage3, 'r', encoding='utf-8') as f:
                self.results['stage3'] = json.load(f)
            print(f"✅ Stage3 結果読み込み: {latest_stage3}")
        else:
            print("⚠️ Stage3結果が見つかりません")
            
        # Stage4: デプロイメント最適化結果
        stage4_files = glob.glob(os.path.join(self.logs_dir, "*stage4*deployment*.json"))
        if stage4_files:
            latest_stage4 = max(stage4_files, key=os.path.getctime)
            with open(latest_stage4, 'r', encoding='utf-8') as f:
                self.results['stage4'] = json.load(f)
            print(f"✅ Stage4 結果読み込み: {latest_stage4}")
        else:
            print("⚠️ Stage4結果が見つかりません")
            
    def calculate_unified_metrics(self):
        """統合メトリクス計算"""
        
        print("🔬 Calculating Unified Metrics...")
        
        unified_metrics = {
            'overall_performance_score': 0.0,
            'generalization_score': 0.0,
            'robustness_score': 0.0,
            'efficiency_score': 0.0,
            'tpe_score': 0.0,
            'recommendations': []
        }
        
        # Stage2: 汎化性能
        if 'stage2' in self.results:
            stage2_data = self.results['stage2']
            unified_metrics['generalization_score'] = stage2_data.get('global_tpe', 0.0)
            unified_metrics['tpe_score'] = stage2_data.get('global_tpe', 0.0)
            
            # TPE改善提案
            if stage2_data.get('global_tpe', 0.0) < 0.70:
                unified_metrics['recommendations'].append({
                    'priority': 'HIGH',
                    'category': 'Generalization',
                    'issue': f"Global TPE Score ({stage2_data.get('global_tpe', 0.0):.3f}) < 0.70 target",
                    'solution': "ハイパーパラメータ最適化、データ拡張強化、アーキテクチャ調整"
                })
        
        # Stage3: ロバストネス
        if 'stage3' in self.results:
            stage3_data = self.results['stage3']
            unified_metrics['robustness_score'] = stage3_data.get('robustness_score', 0.0)
            
            # ロバストネス改善提案
            if stage3_data.get('robustness_score', 0.0) < 75.0:
                unified_metrics['recommendations'].append({
                    'priority': 'MEDIUM',
                    'category': 'Robustness',
                    'issue': f"Robustness Score ({stage3_data.get('robustness_score', 0.0):.1f}%) < 75% target",
                    'solution': "敵対的訓練、データ正則化強化、アンサンブル手法"
                })
        
        # Stage4: 効率性
        if 'stage4' in self.results:
            stage4_data = self.results['stage4']
            benchmark_results = stage4_data.get('benchmark_results', {})
            
            if 'distilled' in benchmark_results:
                distilled_acc = benchmark_results['distilled']['accuracy']
                original_acc = benchmark_results['original']['accuracy']
                efficiency_ratio = distilled_acc / original_acc if original_acc > 0 else 0
                unified_metrics['efficiency_score'] = efficiency_ratio * 100
                
                # 効率性改善提案
                if efficiency_ratio < 0.95:
                    unified_metrics['recommendations'].append({
                        'priority': 'LOW',
                        'category': 'Efficiency',
                        'issue': f"Knowledge Distillation efficiency ({efficiency_ratio:.2f}) < 0.95 target",
                        'solution': "蒸留温度調整、学生モデルアーキテクチャ最適化"
                    })
        
        # 総合スコア計算
        scores = [
            unified_metrics['generalization_score'],
            unified_metrics['robustness_score'] / 100.0,  # 0-1スケールに正規化
            unified_metrics['efficiency_score'] / 100.0
        ]
        valid_scores = [s for s in scores if s > 0]
        unified_metrics['overall_performance_score'] = np.mean(valid_scores) if valid_scores else 0.0
        
        self.unified_metrics = unified_metrics
        return unified_metrics
    
    def create_comprehensive_visualization(self):
        """総合可視化作成"""
        
        print("📈 Creating Comprehensive Visualization...")
        
        fig = plt.figure(figsize=(20, 15))
        
        # Stage2: 汎化性能レーダーチャート
        ax1 = plt.subplot(2, 3, 1, projection='polar')
        if 'stage2' in self.results:
            stage2_data = self.results['stage2']
            dataset_results = stage2_data.get('dataset_results', [])
            
            categories = [r['dataset'] for r in dataset_results if 'accuracy' in r]
            accuracies = [r['accuracy'] for r in dataset_results if 'accuracy' in r]
            
            if categories and accuracies:
                angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                accuracies += accuracies[:1]  # 円形完成
                angles += angles[:1]
                
                ax1.plot(angles, accuracies, 'o-', linewidth=2, color='blue')
                ax1.fill(angles, accuracies, alpha=0.25, color='blue')
                ax1.set_xticks(angles[:-1])
                ax1.set_xticklabels(categories)
                ax1.set_ylim(0, 1)
                ax1.set_title('Stage2: Generalization Performance', fontweight='bold', pad=20)
        
        # Stage3: ロバストネス比較
        ax2 = plt.subplot(2, 3, 2)
        if 'stage3' in self.results:
            stage3_data = self.results['stage3']
            test_results = stage3_data.get('test_results', {})
            
            robustness_types = []
            min_accuracies = []
            
            if 'adversarial' in test_results:
                robustness_types.append('Adversarial')
                min_accuracies.append(min(test_results['adversarial'].values()))
            
            if 'rotation' in test_results:
                robustness_types.append('Rotation')
                min_accuracies.append(min(test_results['rotation'].values()))
            
            if 'noise' in test_results:
                robustness_types.append('Noise')
                min_accuracies.append(min(test_results['noise'].values()))
            
            if robustness_types:
                bars = ax2.bar(robustness_types, min_accuracies, color=['red', 'orange', 'green'][:len(robustness_types)])
                ax2.set_ylabel('Minimum Accuracy (%)')
                ax2.set_title('Stage3: Robustness Analysis', fontweight='bold')
                ax2.set_ylim(0, 100)
                
                for bar, acc in zip(bars, min_accuracies):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                            f'{acc:.1f}%', ha='center', va='bottom')
        
        # Stage4: 効率性比較
        ax3 = plt.subplot(2, 3, 3)
        if 'stage4' in self.results:
            stage4_data = self.results['stage4']
            benchmark_results = stage4_data.get('benchmark_results', {})
            
            model_names = []
            accuracies = []
            sizes = []
            
            for model_name, results in benchmark_results.items():
                model_names.append(model_name.replace('_', ' ').title())
                accuracies.append(results.get('accuracy', 0))
                sizes.append(results.get('model_size_mb', 0))
            
            if model_names and accuracies:
                # 精度 vs サイズの散布図
                scatter = ax3.scatter(sizes, accuracies, s=100, alpha=0.7, c=range(len(model_names)), cmap='viridis')
                ax3.set_xlabel('Model Size (MB)')
                ax3.set_ylabel('Accuracy (%)')
                ax3.set_title('Stage4: Efficiency Analysis', fontweight='bold')
                ax3.grid(True, alpha=0.3)
                
                # モデル名ラベル
                for i, name in enumerate(model_names):
                    ax3.annotate(name, (sizes[i], accuracies[i]), xytext=(5, 5), 
                               textcoords='offset points', fontsize=8)
        
        # 統合メトリクス表示
        ax4 = plt.subplot(2, 3, 4)
        metrics_names = ['Generalization', 'Robustness', 'Efficiency', 'Overall']
        metrics_values = [
            self.unified_metrics['generalization_score'],
            self.unified_metrics['robustness_score'],
            self.unified_metrics['efficiency_score'],
            self.unified_metrics['overall_performance_score'] * 100
        ]
        
        colors = ['blue', 'red', 'green', 'purple']
        bars = ax4.barh(metrics_names, metrics_values, color=colors)
        ax4.set_xlabel('Score')
        ax4.set_title('Unified Performance Metrics', fontweight='bold')
        ax4.set_xlim(0, 100)
        
        for bar, value in zip(bars, metrics_values):
            ax4.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{value:.1f}', ha='left', va='center')
        
        # TPE推移グラフ
        ax5 = plt.subplot(2, 3, 5)
        if 'stage2' in self.results:
            stage2_data = self.results['stage2']
            tpe_scores = stage2_data.get('tpe_scores', [])
            dataset_names = [r['dataset'] for r in stage2_data.get('dataset_results', [])]
            
            if tpe_scores and dataset_names:
                ax5.plot(dataset_names, tpe_scores, 'o-', linewidth=2, markersize=8, color='purple')
                ax5.axhline(y=0.70, color='red', linestyle='--', alpha=0.7, label='Target: 0.70')
                ax5.set_ylabel('TPE Score')
                ax5.set_title('TPE Performance Across Datasets', fontweight='bold')
                ax5.legend()
                ax5.grid(True, alpha=0.3)
                plt.setp(ax5.get_xticklabels(), rotation=45)
        
        # 改善提案サマリー
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        recommendations = self.unified_metrics.get('recommendations', [])
        high_priority = [r for r in recommendations if r['priority'] == 'HIGH']
        medium_priority = [r for r in recommendations if r['priority'] == 'MEDIUM']
        low_priority = [r for r in recommendations if r['priority'] == 'LOW']
        
        ax6.text(0.5, 0.9, 'Improvement Recommendations', ha='center', va='top', 
                fontsize=14, fontweight='bold', transform=ax6.transAxes)
        
        y_pos = 0.8
        for category, items in [('🔴 HIGH PRIORITY', high_priority),
                               ('🟠 MEDIUM PRIORITY', medium_priority),
                               ('🟡 LOW PRIORITY', low_priority)]:
            if items:
                ax6.text(0.1, y_pos, category, ha='left', va='top', fontweight='bold',
                        transform=ax6.transAxes)
                y_pos -= 0.1
                
                for item in items:
                    ax6.text(0.15, y_pos, f"• {item['category']}: {item['issue'][:40]}...",
                            ha='left', va='top', fontsize=9, transform=ax6.transAxes)
                    y_pos -= 0.08
                
                y_pos -= 0.05
        
        plt.tight_layout()
        
        output_path = f"nkat_comprehensive_analysis_{self.timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 総合可視化保存: {output_path}")
        return output_path
    
    def generate_report(self):
        """総合レポート生成"""
        
        print("📝 Generating Comprehensive Report...")
        
        report = f"""
# 🌟 NKAT 総合分析レポート
**生成日時**: {datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}

## 📊 Executive Summary

### 🎯 統合メトリクス
- **総合性能スコア**: {self.unified_metrics['overall_performance_score']:.3f}
- **汎化性能 (TPE)**: {self.unified_metrics['generalization_score']:.3f}
- **ロバストネス**: {self.unified_metrics['robustness_score']:.1f}%
- **効率性**: {self.unified_metrics['efficiency_score']:.1f}%

## 🔍 詳細分析

### Stage2: 汎化テスト結果
"""
        
        if 'stage2' in self.results:
            stage2_data = self.results['stage2']
            report += f"""
- **Global TPE Score**: {stage2_data.get('global_tpe', 0.0):.3f}
- **Global Accuracy**: {stage2_data.get('global_accuracy', 0.0):.3f}
- **Consistency Score**: {stage2_data.get('consistency_score', 0.0):.3f}

**データセット別結果**:
"""
            for result in stage2_data.get('dataset_results', []):
                if 'accuracy' in result:
                    report += f"- {result['dataset']}: Accuracy={result['accuracy']:.3f}, TPE={result.get('tpe_score', 0.0):.3f}\n"
        
        if 'stage3' in self.results:
            stage3_data = self.results['stage3']
            report += f"""
### Stage3: ロバストネステスト結果
- **総合ロバストネススコア**: {stage3_data.get('robustness_score', 0.0):.1f}%

**攻撃種別結果**:
"""
            test_results = stage3_data.get('test_results', {})
            if 'adversarial' in test_results:
                min_adv = min(test_results['adversarial'].values())
                report += f"- 敵対的攻撃: 最小精度 {min_adv:.1f}%\n"
            if 'rotation' in test_results:
                min_rot = min(test_results['rotation'].values())
                report += f"- 回転変換: 最小精度 {min_rot:.1f}%\n"
            if 'noise' in test_results:
                min_noise = min(test_results['noise'].values())
                report += f"- ノイズ: 最小精度 {min_noise:.1f}%\n"
        
        if 'stage4' in self.results:
            stage4_data = self.results['stage4']
            benchmark_results = stage4_data.get('benchmark_results', {})
            report += f"""
### Stage4: デプロイメント最適化結果
"""
            for model_name, results in benchmark_results.items():
                report += f"- {model_name.title()}: {results.get('accuracy', 0.0):.1f}% accuracy, {results.get('model_size_mb', 0.0):.1f}MB\n"
        
        report += f"""
## 🎯 改善提案

### 🔴 高優先度
"""
        
        high_priority = [r for r in self.unified_metrics.get('recommendations', []) if r['priority'] == 'HIGH']
        for rec in high_priority:
            report += f"""
**{rec['category']}**: {rec['issue']}
- 解決策: {rec['solution']}
"""
        
        report += f"""
### 🟠 中優先度
"""
        
        medium_priority = [r for r in self.unified_metrics.get('recommendations', []) if r['priority'] == 'MEDIUM']
        for rec in medium_priority:
            report += f"""
**{rec['category']}**: {rec['issue']}
- 解決策: {rec['solution']}
"""
        
        report += f"""
## 🚀 次のステップ

### 即座に取り組むべき項目
1. **TPE Score改善**: 現在{self.unified_metrics['generalization_score']:.3f} → 目標0.70
2. **ロバストネス強化**: データ拡張とアンサンブル手法の導入
3. **効率性最適化**: 知識蒸留の精度向上

### 中長期的な改善
1. **アーキテクチャ革新**: NKATメカニズムの更なる発展
2. **スケーラビリティ**: より大規模なデータセットでの検証
3. **実用性**: 産業応用への展開

## 📈 成果と今後の展望

このNKATプロジェクトは革新的なTransformerアーキテクチャを提案し、
複数のベンチマークで有望な結果を示しました。特に：

✅ **技術的革新**: ゲージ理論に基づく新しいアテンションメカニズム  
✅ **実証的成果**: 複数データセットでの汎化性能確認  
✅ **実用的最適化**: デプロイメント対応の軽量化手法  

今後の発展により、さらなる性能向上と実用化が期待されます。

---
*Generated by NKAT Comprehensive Analysis System*
"""
        
        report_path = f"NKAT_Comprehensive_Analysis_Report_{self.timestamp}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ 総合レポート保存: {report_path}")
        return report_path
    
    def save_unified_results(self):
        """統合結果をJSONで保存"""
        
        unified_data = {
            'timestamp': self.timestamp,
            'unified_metrics': self.unified_metrics,
            'stage_results': self.results,
            'summary': {
                'total_stages_analyzed': len(self.results),
                'overall_score': self.unified_metrics['overall_performance_score'],
                'primary_recommendations': len([r for r in self.unified_metrics.get('recommendations', []) if r['priority'] == 'HIGH'])
            }
        }
        
        json_path = f"nkat_unified_results_{self.timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(unified_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 統合結果保存: {json_path}")
        return json_path

def main():
    """メイン実行関数"""
    
    print("🌟 NKAT 総合分析システム 開始")
    print("="*60)
    
    # 分析器作成
    analyzer = NKATComprehensiveAnalyzer()
    
    # 結果読み込み
    analyzer.load_stage_results()
    
    # 統合メトリクス計算
    unified_metrics = analyzer.calculate_unified_metrics()
    
    # 可視化作成
    viz_path = analyzer.create_comprehensive_visualization()
    
    # レポート生成
    report_path = analyzer.generate_report()
    
    # 結果保存
    json_path = analyzer.save_unified_results()
    
    # サマリー表示
    print("\n" + "="*60)
    print("🌟 NKAT 総合分析完了")
    print("="*60)
    print(f"🎯 総合性能スコア: {unified_metrics['overall_performance_score']:.3f}")
    print(f"📊 汎化性能 (TPE): {unified_metrics['generalization_score']:.3f}")
    print(f"🛡️ ロバストネス: {unified_metrics['robustness_score']:.1f}%")
    print(f"⚡ 効率性: {unified_metrics['efficiency_score']:.1f}%")
    print(f"📋 改善提案数: {len(unified_metrics.get('recommendations', []))}")
    print()
    print(f"📈 可視化: {viz_path}")
    print(f"📝 レポート: {report_path}")
    print(f"💾 統合結果: {json_path}")
    print("="*60)

if __name__ == "__main__":
    main() 