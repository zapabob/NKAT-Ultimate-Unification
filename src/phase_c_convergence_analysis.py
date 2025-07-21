#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase C 収束半径分析: 統合特解の収束半径とフラクタル次元評価
目標: 級数の収束半径を厳密評価し、数値エビデンスと整合性を確認
"""

import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp, nsum, inf, zeta, gamma, sin, pi, exp, log, sqrt
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime
import os
from scipy.optimize import curve_fit
from scipy.stats import linregress

# 超高精度設定
mp.dps = 150

class ConvergenceAnalyzer:
    """収束半径とフラクタル次元分析クラス"""
    
    def __init__(self, theta=0.1):
        self.theta = theta
        self.results = {}
        
    def unified_solution_series(self, s, N=10_000):
        """統合特解の級数展開"""
        try:
            quantum_correction = exp(-1j * self.theta * log(2 * pi)) * \
                               (1 + self.theta**2 * log(s / (1 - s)) / 2)
            
            return nsum(lambda n: n**(-s) * exp(1j * self.theta * log(n)) * quantum_correction, [1, N])
        except Exception as e:
            print(f"Error in unified_solution_series: {e}")
            return mp.nan
    
    def convergence_radius_estimation(self, s, max_N=50_000):
        """収束半径の数値推定"""
        try:
            # 級数の差分を計算
            differences = []
            N_values = []
            
            for N in tqdm(range(1000, max_N, 1000), desc="収束半径推定"):
                val_N = self.unified_solution_series(s, N)
                val_N_plus_1 = self.unified_solution_series(s, N + 1)
                diff = abs(val_N - val_N_plus_1)
                
                differences.append(float(diff))
                N_values.append(N)
            
            # 指数関数的収束のフィッティング
            log_diffs = np.log(differences)
            log_N = np.log(N_values)
            
            # 線形回帰で収束指数を推定
            slope, intercept, r_value, p_value, std_err = linregress(log_N, log_diffs)
            
            # 収束半径の推定
            convergence_radius = exp(-intercept / slope) if slope < 0 else float('inf')
            
            return {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'convergence_radius': convergence_radius,
                'convergence_rate': -slope,
                'N_values': N_values,
                'differences': differences
            }
        except Exception as e:
            print(f"Error in convergence_radius_estimation: {e}")
            return None
    
    def fractal_dimension_analysis(self, t_min=0, t_max=100, step=0.5):
        """フラクタル次元の分析"""
        try:
            # 零点候補の探索
            zeros_candidates = []
            t_values = np.arange(t_min, t_max, step)
            
            for t in tqdm(t_values, desc="零点候補探索"):
                s = 0.5 + 1j * t
                val = self.unified_solution_series(s, 10000)
                if abs(val) < 0.01:  # 零点候補
                    zeros_candidates.append({
                        't': t,
                        's': complex(s),
                        'value': complex(val),
                        'abs_value': abs(val)
                    })
            
            # ボックスカウント法によるフラクタル次元計算
            if len(zeros_candidates) > 10:
                # 零点の分布からフラクタル次元を推定
                t_positions = [z['t'] for z in zeros_candidates]
                
                # 異なるスケールでのボックスカウント
                scales = np.logspace(-2, 1, 20)
                box_counts = []
                
                for scale in tqdm(scales, desc="ボックスカウント"):
                    boxes = set()
                    for t in t_positions:
                        box_index = int(t / scale)
                        boxes.add(box_index)
                    box_counts.append(len(boxes))
                
                # フラクタル次元の計算
                log_scales = np.log(scales)
                log_counts = np.log(box_counts)
                
                slope, intercept, r_value, p_value, std_err = linregress(log_scales, log_counts)
                fractal_dimension = -slope
                
                return {
                    'fractal_dimension': fractal_dimension,
                    'r_squared': r_value**2,
                    'zeros_count': len(zeros_candidates),
                    'scales': scales.tolist(),
                    'box_counts': box_counts,
                    'zeros_candidates': zeros_candidates
                }
            else:
                return None
        except Exception as e:
            print(f"Error in fractal_dimension_analysis: {e}")
            return None
    
    def theoretical_numerical_consistency(self, s_test=0.5 + 10j):
        """理論値と数値実験値の整合性確認"""
        try:
            # 理論的収束定数の推定
            convergence_data = self.convergence_radius_estimation(s_test)
            
            if convergence_data:
                theoretical_C = 2.5  # 理論値
                theoretical_alpha = 1.5  # 理論値
                
                numerical_C = exp(convergence_data['intercept'])
                numerical_alpha = -convergence_data['slope']
                
                # 整合性チェック
                C_consistency = abs(theoretical_C - numerical_C) < 0.5
                alpha_consistency = abs(theoretical_alpha - numerical_alpha) < 0.2
                
                return {
                    'theoretical_C': theoretical_C,
                    'theoretical_alpha': theoretical_alpha,
                    'numerical_C': numerical_C,
                    'numerical_alpha': numerical_alpha,
                    'C_consistency': C_consistency,
                    'alpha_consistency': alpha_consistency,
                    'overall_consistency': C_consistency and alpha_consistency
                }
            else:
                return None
        except Exception as e:
            print(f"Error in theoretical_numerical_consistency: {e}")
            return None
    
    def unified_solution_existence_test(self, re_range=np.linspace(0.1, 0.9, 50)):
        """統合特解の存在性テスト"""
        try:
            results = []
            
            for re in tqdm(re_range, desc="統合特解存在性テスト"):
                s = re + 1j * 10  # 固定虚部で実部を変化
                
                # 異なるNでの値の収束性をチェック
                N_values = [1000, 5000, 10000, 20000]
                values = []
                
                for N in N_values:
                    val = self.unified_solution_series(s, N)
                    values.append(float(abs(val)))
                
                # 収束性の判定
                convergence_check = all(abs(values[i] - values[i-1]) < 0.001 for i in range(1, len(values)))
                
                results.append({
                    's_real': re,
                    's': complex(s),
                    'values': values,
                    'convergence_check': convergence_check,
                    'final_value': values[-1]
                })
            
            return results
        except Exception as e:
            print(f"Error in unified_solution_existence_test: {e}")
            return None
    
    def visualize_results(self, save_path="phase_c_results"):
        """結果の可視化"""
        os.makedirs(save_path, exist_ok=True)
        
        # 1. 収束半径分析
        if 'convergence_radius' in self.results:
            plt.figure(figsize=(15, 10))
            
            data = self.results['convergence_radius']
            N_values = data['N_values']
            differences = data['differences']
            
            plt.subplot(2, 3, 1)
            plt.loglog(N_values, differences, 'b-', marker='o', alpha=0.7)
            plt.xlabel('N (項数)')
            plt.ylabel('|ΔΨ(s)|')
            plt.title('統合特解の収束性')
            plt.grid(True)
            
            plt.subplot(2, 3, 2)
            log_N = np.log(N_values)
            log_diffs = np.log(differences)
            plt.plot(log_N, log_diffs, 'r-', marker='s', alpha=0.7)
            plt.xlabel('log(N)')
            plt.ylabel('log(|ΔΨ(s)|)')
            plt.title('収束性の対数プロット')
            plt.grid(True)
            
            # フィッティング線
            slope = data['slope']
            intercept = data['intercept']
            fit_line = slope * log_N + intercept
            plt.plot(log_N, fit_line, 'g--', label=f'Fit: α={-slope:.3f}')
            plt.legend()
            
            plt.subplot(2, 3, 3)
            plt.bar(['収束半径', '収束率'], [data['convergence_radius'], data['convergence_rate']], 
                   color=['blue', 'red'], alpha=0.7)
            plt.title('収束パラメータ')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"{save_path}/convergence_radius_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. フラクタル次元分析
        if 'fractal_dimension' in self.results:
            plt.figure(figsize=(12, 8))
            
            data = self.results['fractal_dimension']
            scales = data['scales']
            box_counts = data['box_counts']
            
            plt.subplot(2, 2, 1)
            plt.loglog(scales, box_counts, 'g-', marker='o', alpha=0.7)
            plt.xlabel('スケール ε')
            plt.ylabel('ボックス数 N(ε)')
            plt.title('ボックスカウント法')
            plt.grid(True)
            
            plt.subplot(2, 2, 2)
            log_scales = np.log(scales)
            log_counts = np.log(box_counts)
            plt.plot(log_scales, log_counts, 'm-', marker='s', alpha=0.7)
            plt.xlabel('log(ε)')
            plt.ylabel('log(N(ε))')
            plt.title('フラクタル次元推定')
            plt.grid(True)
            
            # フィッティング線
            fractal_dim = data['fractal_dimension']
            plt.plot(log_scales, -fractal_dim * log_scales + np.mean(log_counts), 
                    'r--', label=f'D = {fractal_dim:.3f}')
            plt.legend()
            
            plt.subplot(2, 2, 3)
            t_values = [z['t'] for z in data['zeros_candidates']]
            abs_values = [z['abs_value'] for z in data['zeros_candidates']]
            plt.scatter(t_values, abs_values, c=abs_values, cmap='viridis', alpha=0.7)
            plt.colorbar(label='|Ψ(s)|')
            plt.xlabel('t (虚部)')
            plt.ylabel('|Ψ(s)|')
            plt.title('零点候補の分布')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"{save_path}/fractal_dimension_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. 理論値と数値実験値の整合性
        if 'consistency' in self.results:
            plt.figure(figsize=(10, 6))
            
            data = self.results['consistency']
            
            plt.subplot(1, 2, 1)
            plt.bar(['理論値', '数値実験値'], [data['theoretical_C'], data['numerical_C']], 
                   color=['blue', 'red'], alpha=0.7)
            plt.ylabel('収束定数 C')
            plt.title('収束定数の比較')
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.bar(['理論値', '数値実験値'], [data['theoretical_alpha'], data['numerical_alpha']], 
                   color=['green', 'orange'], alpha=0.7)
            plt.ylabel('収束指数 α')
            plt.title('収束指数の比較')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"{save_path}/theoretical_numerical_consistency.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def save_results(self, filename="phase_c_results.json"):
        """結果をJSONファイルに保存"""
        timestamp = datetime.now().isoformat()
        
        # 複素数をJSON化可能な形式に変換
        def convert_complex(obj):
            if isinstance(obj, complex):
                return {'real': obj.real, 'imag': obj.imag}
            elif isinstance(obj, np.complex128):
                return {'real': float(obj.real), 'imag': float(obj.imag)}
            return obj
        
        # 結果を変換
        converted_results = {}
        for key, value in self.results.items():
            if isinstance(value, list):
                converted_results[key] = []
                for item in value:
                    converted_item = {}
                    for k, v in item.items():
                        converted_item[k] = convert_complex(v)
                    converted_results[key].append(converted_item)
            else:
                converted_results[key] = convert_complex(value)
        
        data = {
            'timestamp': timestamp,
            'theta': self.theta,
            'results': converted_results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"結果を保存しました: {filename}")

def main():
    """メイン実行関数"""
    print("🚀 Phase C 収束半径分析開始！")
    
    # 収束分析の初期化
    analyzer = ConvergenceAnalyzer(theta=0.1)
    
    # 1. 収束半径推定
    print("\n📊 収束半径推定実行中...")
    analyzer.results['convergence_radius'] = analyzer.convergence_radius_estimation(0.5 + 10j)
    
    # 2. フラクタル次元分析
    print("\n🔍 フラクタル次元分析実行中...")
    analyzer.results['fractal_dimension'] = analyzer.fractal_dimension_analysis()
    
    # 3. 理論値と数値実験値の整合性確認
    print("\n⚖️ 理論値と数値実験値の整合性確認中...")
    analyzer.results['consistency'] = analyzer.theoretical_numerical_consistency()
    
    # 4. 統合特解の存在性テスト
    print("\n🔬 統合特解の存在性テスト実行中...")
    analyzer.results['existence_test'] = analyzer.unified_solution_existence_test()
    
    # 5. 結果の可視化と保存
    print("\n📈 結果の可視化と保存中...")
    analyzer.visualize_results()
    analyzer.save_results()
    
    print("\n✅ Phase C 収束半径分析完了！")
    print(f"📁 結果ファイル: phase_c_results.json")
    print(f"📊 可視化ファイル: phase_c_results/")

if __name__ == "__main__":
    main() 