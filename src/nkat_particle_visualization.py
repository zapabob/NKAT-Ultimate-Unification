#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT理論粒子予測結果の包括的可視化システム
6種類の予測粒子の質量階層、検出可能性、理論的意義を可視化

Author: NKAT研究チーム
Date: 2025-06-01
Version: 1.0 - 粒子可視化特化版
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 日本語対応フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (16, 12)

class NKATParticleVisualizer:
    """NKAT粒子予測結果可視化システム"""
    
    def __init__(self, results_file):
        """初期化"""
        print("🎨 NKAT粒子可視化システム初期化開始")
        
        # 結果データ読み込み
        with open(results_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # カラーパレット設定
        self.colors = {
            'NQG': '#FF6B6B',  # 赤系（重力）
            'NCM': '#4ECDC4',  # 青緑系（ヒッグス）
            'QIM': '#45B7D1',  # 青系（情報）
            'TPO': '#96CEB4',  # 緑系（位相）
            'HDC': '#FFEAA7',  # 黄系（高次元）
            'QEP': '#DDA0DD'   # 紫系（エントロピー）
        }
        
        print("✅ 可視化システム初期化完了")
    
    def create_mass_spectrum_plot(self):
        """質量スペクトラム可視化"""
        print("📊 質量スペクトラム可視化作成中...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # データ準備
        particles = list(self.data['mass_spectrum'].keys())
        masses = [self.data['mass_spectrum'][p] for p in particles]
        log_masses = [np.log10(max(1e-50, m)) for m in masses]
        
        # 上段：線形スケール（対数変換済み）
        bars1 = ax1.bar(particles, log_masses, 
                       color=[self.colors[p] for p in particles],
                       alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax1.set_ylabel('Log₁₀(質量 [GeV])', fontsize=12, fontweight='bold')
        ax1.set_title('NKAT理論予測粒子の質量階層', fontsize=16, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 質量値をバーの上に表示
        for i, (bar, mass) in enumerate(zip(bars1, masses)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{mass:.2e} GeV', ha='center', va='bottom', 
                    fontsize=9, rotation=45)
        
        # 下段：標準模型粒子との比較
        sm_particles = ['electron', 'muon', 'proton', 'W_boson', 'higgs', 'top_quark']
        sm_masses = [0.511e-3, 0.106, 0.938, 80.4, 125.1, 173.0]
        sm_log_masses = [np.log10(m) for m in sm_masses]
        
        # NKAT粒子（検出可能範囲のみ）
        detectable_particles = []
        detectable_masses = []
        for p, m in zip(particles, masses):
            if 1e-30 < m < 1e20:  # 検出可能範囲
                detectable_particles.append(p)
                detectable_masses.append(np.log10(m))
        
        # 標準模型プロット
        ax2.scatter(range(len(sm_particles)), sm_log_masses, 
                   s=100, c='gray', marker='o', label='標準模型粒子', alpha=0.7)
        
        # NKAT粒子プロット
        if detectable_particles:
            nkat_x = [len(sm_particles) + i for i in range(len(detectable_particles))]
            ax2.scatter(nkat_x, detectable_masses,
                       s=150, c=[self.colors[p] for p in detectable_particles],
                       marker='*', label='NKAT予測粒子', alpha=0.9)
        
        # ラベル設定
        all_labels = sm_particles + detectable_particles
        ax2.set_xticks(range(len(all_labels)))
        ax2.set_xticklabels(all_labels, rotation=45, ha='right')
        ax2.set_ylabel('Log₁₀(質量 [GeV])', fontsize=12, fontweight='bold')
        ax2.set_title('標準模型粒子とNKAT予測粒子の比較', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_detection_feasibility_plot(self):
        """検出可能性評価可視化"""
        print("🔍 検出可能性評価可視化作成中...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 検出手法別感度マップ
        detection_data = {}
        for particle, data in self.data['detectability_summary'].items():
            detection_data[particle] = {}
            if isinstance(data, dict):
                for method, sensitivity in data.items():
                    if isinstance(sensitivity, (int, float)):
                        detection_data[particle][method] = -np.log10(abs(sensitivity))
                    elif isinstance(sensitivity, dict) and 'sensitivity' in sensitivity:
                        detection_data[particle][method] = -np.log10(abs(sensitivity['sensitivity']))
        
        # ヒートマップ用データ準備
        if detection_data:
            particles = list(detection_data.keys())
            methods = set()
            for p_data in detection_data.values():
                methods.update(p_data.keys())
            methods = list(methods)
            
            heatmap_data = np.zeros((len(particles), len(methods)))
            for i, particle in enumerate(particles):
                for j, method in enumerate(methods):
                    if method in detection_data[particle]:
                        heatmap_data[i, j] = detection_data[particle][method]
            
            # ヒートマップ作成
            im = ax1.imshow(heatmap_data, cmap='viridis', aspect='auto')
            ax1.set_xticks(range(len(methods)))
            ax1.set_xticklabels(methods, rotation=45, ha='right')
            ax1.set_yticks(range(len(particles)))
            ax1.set_yticklabels(particles)
            ax1.set_title('検出感度マップ (-log₁₀(感度))', fontweight='bold')
            plt.colorbar(im, ax=ax1)
        
        # 質量 vs 検出可能性
        masses = []
        detectabilities = []
        particle_names = []
        
        for particle in self.data['mass_spectrum'].keys():
            mass = self.data['mass_spectrum'][particle]
            if 1e-35 < mass < 1e25:  # 表示範囲
                masses.append(mass)
                
                # 検出可能性スコア計算
                if particle in self.data['detectability_summary']:
                    det_data = self.data['detectability_summary'][particle]
                    if isinstance(det_data, dict):
                        # 感度値の平均を取る
                        sensitivities = []
                        for val in det_data.values():
                            if isinstance(val, (int, float)):
                                sensitivities.append(-np.log10(abs(val)))
                            elif isinstance(val, dict) and 'sensitivity' in val:
                                sensitivities.append(-np.log10(abs(val['sensitivity'])))
                        detectability = np.mean(sensitivities) if sensitivities else 5
                    else:
                        detectability = 5
                else:
                    detectability = 5
                
                detectabilities.append(detectability)
                particle_names.append(particle)
        
        # 散布図
        if masses and detectabilities:
            scatter = ax2.scatter(masses, detectabilities, 
                                c=[self.colors[p] for p in particle_names],
                                s=200, alpha=0.7, edgecolors='black')
            
            for i, name in enumerate(particle_names):
                ax2.annotate(name, (masses[i], detectabilities[i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=10, fontweight='bold')
            
            ax2.set_xscale('log')
            ax2.set_xlabel('質量 [GeV]', fontweight='bold')
            ax2.set_ylabel('検出可能性スコア', fontweight='bold')
            ax2.set_title('質量 vs 検出可能性', fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        # 実験手法別検出確率
        experiments = ['LHC', 'LIGO', 'Cosmic Ray', 'Quantum Lab', 'Precision Test']
        detection_probs = []
        
        for exp in experiments:
            prob = 0
            count = 0
            for particle, data in self.data['detectability_summary'].items():
                if isinstance(data, dict):
                    for method, val in data.items():
                        if exp.lower() in method.lower():
                            if isinstance(val, (int, float)):
                                prob += min(1.0, abs(val) * 1e10)
                                count += 1
                            elif isinstance(val, dict) and 'sensitivity' in val:
                                prob += min(1.0, abs(val['sensitivity']) * 1e10)
                                count += 1
            
            if count > 0:
                detection_probs.append(prob / count)
            else:
                detection_probs.append(0.1)
        
        bars = ax3.bar(experiments, detection_probs, 
                      color=plt.cm.Set3(np.linspace(0, 1, len(experiments))),
                      alpha=0.8, edgecolor='black')
        
        ax3.set_ylabel('検出確率', fontweight='bold')
        ax3.set_title('実験手法別検出確率', fontweight='bold')
        ax3.set_ylim(0, 1)
        
        for bar, prob in zip(bars, detection_probs):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 理論的意義レーダーチャート
        categories = ['Quantum Gravity', 'Higgs Mechanism', 'Information Theory', 
                     'Topology', 'Extra Dimensions', 'Entropy Processing']
        
        # 各粒子の理論的重要度スコア
        importance_scores = {
            'NQG': [10, 2, 3, 4, 6, 5],
            'NCM': [3, 10, 4, 5, 3, 4],
            'QIM': [4, 3, 10, 6, 5, 8],
            'TPO': [5, 6, 5, 10, 4, 6],
            'HDC': [6, 4, 5, 7, 10, 5],
            'QEP': [7, 5, 9, 6, 5, 10]
        }
        
        # レーダーチャート作成
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 閉じるため
        
        ax4 = plt.subplot(2, 2, 4, projection='polar')
        
        for particle, scores in importance_scores.items():
            scores += scores[:1]  # 閉じるため
            ax4.plot(angles, scores, 'o-', linewidth=2, 
                    label=particle, color=self.colors[particle], alpha=0.7)
            ax4.fill(angles, scores, alpha=0.1, color=self.colors[particle])
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(categories)
        ax4.set_ylim(0, 10)
        ax4.set_title('理論的重要度レーダーチャート', fontweight='bold', pad=20)
        ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        return fig
    
    def create_unification_analysis_plot(self):
        """統一理論解析可視化"""
        print("🌌 統一理論解析可視化作成中...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 質量階層とエネルギースケール
        energy_scales = {
            'Electroweak': 100,
            'GUT': 1e16,
            'Planck': 1e19,
            'NKAT Unification': self.data['unification_analysis']['coupling_unification']['nkat_unification_scale']
        }
        
        scales = list(energy_scales.keys())
        energies = [energy_scales[s] for s in scales]
        log_energies = [np.log10(max(1e-50, e)) for e in energies]
        
        bars = ax1.bar(scales, log_energies, 
                      color=['blue', 'green', 'red', 'purple'],
                      alpha=0.7, edgecolor='black')
        
        ax1.set_ylabel('Log₁₀(エネルギー [GeV])', fontweight='bold')
        ax1.set_title('統一理論エネルギースケール', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, energy in zip(bars, energies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{energy:.2e}', ha='center', va='bottom', fontsize=9)
        
        # 対称性構造
        symmetries = ['SU(3)', 'SU(2)', 'U(1)', 'E₈', 'Non-commutative', 'SUSY', 'Extra Dim']
        importance = [8, 7, 6, 10, 9, 7, 8]
        
        ax2.barh(symmetries, importance, 
                color=plt.cm.viridis(np.linspace(0, 1, len(symmetries))),
                alpha=0.8, edgecolor='black')
        
        ax2.set_xlabel('理論的重要度', fontweight='bold')
        ax2.set_title('対称性構造の重要度', fontweight='bold')
        
        # 宇宙論的影響
        cosmo_effects = list(self.data['cosmological_impact'].keys())
        if 'phase_transitions' in cosmo_effects:
            cosmo_effects.remove('phase_transitions')
        
        effect_strengths = []
        for effect in cosmo_effects:
            if effect == 'dark_matter_candidates':
                effect_strengths.append(len(self.data['cosmological_impact'][effect]))
            else:
                effect_strengths.append(5)  # デフォルト値
        
        wedges, texts, autotexts = ax3.pie(effect_strengths, labels=cosmo_effects, 
                                          autopct='%1.1f%%', startangle=90,
                                          colors=plt.cm.Set3(np.linspace(0, 1, len(cosmo_effects))))
        
        ax3.set_title('宇宙論的影響の分布', fontweight='bold')
        
        # 技術応用ポテンシャル
        tech_apps = self.data['technological_applications']
        app_categories = list(tech_apps.keys())
        app_scores = []
        
        for category in app_categories:
            # 各カテゴリの応用数をスコアとする
            score = len(tech_apps[category]) if isinstance(tech_apps[category], dict) else 3
            app_scores.append(score)
        
        bars = ax4.bar(app_categories, app_scores,
                      color=['gold', 'lightcoral', 'lightblue'],
                      alpha=0.8, edgecolor='black')
        
        ax4.set_ylabel('応用ポテンシャル', fontweight='bold')
        ax4.set_title('技術応用ポテンシャル', fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        
        for bar, score in zip(bars, app_scores):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{score}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def create_comprehensive_summary(self):
        """包括的サマリー可視化"""
        print("📋 包括的サマリー可視化作成中...")
        
        fig = plt.figure(figsize=(20, 14))
        
        # グリッドレイアウト設定
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. 粒子一覧表
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.axis('off')
        
        # テーブルデータ準備
        table_data = []
        for particle, data in self.data['predicted_particles'].items():
            mass = data['mass_gev']
            spin = data['spin']
            role = data.get('theoretical_role', data.get('theoretical_significance', 'Unknown'))
            table_data.append([particle, f'{mass:.2e} GeV', str(spin), role[:30] + '...'])
        
        table = ax1.table(cellText=table_data,
                         colLabels=['粒子', '質量', 'スピン', '理論的役割'],
                         cellLoc='center',
                         loc='center',
                         colColours=['lightblue']*4)
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax1.set_title('NKAT予測粒子一覧', fontsize=16, fontweight='bold', pad=20)
        
        # 2. 質量分布ヒストグラム
        ax2 = fig.add_subplot(gs[0, 2:])
        masses = [self.data['mass_spectrum'][p] for p in self.data['mass_spectrum'].keys()]
        log_masses = [np.log10(max(1e-50, m)) for m in masses]
        
        ax2.hist(log_masses, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Log₁₀(質量 [GeV])', fontweight='bold')
        ax2.set_ylabel('粒子数', fontweight='bold')
        ax2.set_title('質量分布', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. 統一スケール比較
        ax3 = fig.add_subplot(gs[1, :2])
        
        unif_data = self.data['unification_analysis']['coupling_unification']
        scales = ['Electroweak', 'GUT', 'Planck', 'NKAT']
        scale_values = [unif_data['electroweak_scale'], unif_data['gut_scale'], 
                       unif_data['planck_scale'], unif_data['nkat_unification_scale']]
        
        log_scales = [np.log10(max(1e-50, s)) for s in scale_values]
        
        bars = ax3.bar(scales, log_scales, 
                      color=['blue', 'green', 'red', 'purple'],
                      alpha=0.7, edgecolor='black')
        
        ax3.set_ylabel('Log₁₀(エネルギー [GeV])', fontweight='bold')
        ax3.set_title('統一エネルギースケール比較', fontweight='bold')
        
        # 4. 検出可能性サマリー
        ax4 = fig.add_subplot(gs[1, 2:])
        
        particles = list(self.data['mass_spectrum'].keys())
        detectability_scores = []
        
        for particle in particles:
            if particle in self.data['detectability_summary']:
                # 簡単な検出可能性スコア
                mass = self.data['mass_spectrum'][particle]
                if 1e-15 < mass < 1e15:  # 検出可能範囲
                    score = 0.8
                elif 1e-20 < mass < 1e20:
                    score = 0.5
                else:
                    score = 0.2
            else:
                score = 0.3
            
            detectability_scores.append(score)
        
        bars = ax4.bar(particles, detectability_scores,
                      color=[self.colors[p] for p in particles],
                      alpha=0.8, edgecolor='black')
        
        ax4.set_ylabel('検出可能性', fontweight='bold')
        ax4.set_title('粒子別検出可能性', fontweight='bold')
        ax4.set_ylim(0, 1)
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. 理論的フレームワーク
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        framework_text = f"""
NKAT理論的フレームワーク:
• 基礎理論: {self.data['theoretical_framework']['base_theory']}
• 対称性群: {self.data['theoretical_framework']['symmetry_group']}
• 次元構造: {self.data['theoretical_framework']['dimension']}
• 基本スケール: {self.data['theoretical_framework']['fundamental_scale']} m²

質量階層統計:
• 最小質量: {self.data['unification_analysis']['mass_range_gev']['minimum']:.2e} GeV
• 最大質量: {self.data['unification_analysis']['mass_range_gev']['maximum']:.2e} GeV
• 質量範囲: {self.data['unification_analysis']['mass_range_gev']['span_orders']:.1f} 桁

宇宙論的意義:
• 暗黒物質候補: {', '.join(self.data['cosmological_impact']['dark_matter_candidates'])}
• 暗黒エネルギー機構: {self.data['cosmological_impact']['dark_energy_mechanism']}
• インフレーション駆動: {self.data['cosmological_impact']['inflation_driver']}
        """
        
        ax5.text(0.05, 0.95, framework_text, transform=ax5.transAxes,
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('NKAT理論粒子予測 - 包括的解析サマリー', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        return fig
    
    def generate_all_visualizations(self):
        """全ての可視化を生成"""
        print("🚀 全可視化生成開始")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 各可視化を生成・保存
        visualizations = [
            ('mass_spectrum', self.create_mass_spectrum_plot),
            ('detection_feasibility', self.create_detection_feasibility_plot),
            ('unification_analysis', self.create_unification_analysis_plot),
            ('comprehensive_summary', self.create_comprehensive_summary)
        ]
        
        saved_files = []
        
        for name, create_func in visualizations:
            try:
                print(f"📊 {name} 可視化作成中...")
                fig = create_func()
                filename = f"nkat_particle_{name}_{timestamp}.png"
                fig.savefig(filename, dpi=300, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                plt.close(fig)
                saved_files.append(filename)
                print(f"✅ {filename} 保存完了")
            except Exception as e:
                print(f"❌ {name} 可視化エラー: {e}")
        
        print(f"\n🎯 可視化完了: {len(saved_files)} ファイル生成")
        for file in saved_files:
            print(f"  📁 {file}")
        
        return saved_files

def main():
    """メイン実行関数"""
    print("🎨 NKAT理論粒子予測可視化システム")
    print("=" * 60)
    
    try:
        # 最新の結果ファイルを探す
        import glob
        result_files = glob.glob("nkat_particle_predictions_*.json")
        if not result_files:
            print("❌ 結果ファイルが見つかりません")
            return
        
        latest_file = max(result_files)
        print(f"📂 使用ファイル: {latest_file}")
        
        # 可視化システム初期化
        visualizer = NKATParticleVisualizer(latest_file)
        
        # 全可視化生成
        saved_files = visualizer.generate_all_visualizations()
        
        print("\n" + "=" * 60)
        print("✅ NKAT粒子予測可視化完了")
        print("=" * 60)
        
        return saved_files
        
    except Exception as e:
        print(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main() 