#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase A 数値実験: 非可換ゼータ関数の数値検証
目標: ヒートカーネル展開テストと零点探索
"""

import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp, nsum, inf, zeta, findroot
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime
import os

# 高精度設定
mp.dps = 50

class NoncommutativeZeta:
    """非可換ゼータ関数の数値実装"""
    
    def __init__(self, theta=0.1):
        self.theta = theta
        self.results = {}
    
    def zeta_nc(self, s, N=10_000):
        """非可換ゼータ関数の数値計算"""
        try:
            return nsum(lambda n: n**(-s) * mp.ej(self.theta * mp.log(n)), [1, N])
        except Exception as e:
            print(f"Error in zeta_nc: {e}")
            return mp.nan
    
    def zeta_classical(self, s):
        """古典ゼータ関数（比較用）"""
        try:
            return zeta(s)
        except Exception as e:
            print(f"Error in zeta_classical: {e}")
            return mp.nan
    
    def convergence_test(self, s, max_N=10000):
        """収束性テスト"""
        results = []
        for N in tqdm(range(100, max_N, 100), desc="収束テスト"):
            try:
                val = self.zeta_nc(s, N)
                results.append({
                    'N': N,
                    'value': float(val),
                    'abs_value': float(abs(val))
                })
            except Exception as e:
                print(f"Error at N={N}: {e}")
                continue
        
        return results
    
    def critical_line_search(self, t_min=0, t_max=100, step=0.1):
        """臨界線上の零点探索"""
        zeros = []
        t_values = np.arange(t_min, t_max, step)
        
        for t in tqdm(t_values, desc="臨界線零点探索"):
            s = 0.5 + 1j * t
            try:
                val = self.zeta_nc(s, 10000)
                if abs(val) < 0.01:  # 零点候補
                    zeros.append({
                        't': t,
                        's': complex(s),
                        'value': complex(val),
                        'abs_value': abs(val)
                    })
            except Exception as e:
                print(f"Error at t={t}: {e}")
                continue
        
        return zeros
    
    def theta_limit_test(self, s, theta_range=np.linspace(0, 0.5, 50)):
        """θ→0 限界テスト"""
        results = []
        
        for theta in tqdm(theta_range, desc="θ限界テスト"):
            self.theta = theta
            try:
                nc_val = self.zeta_nc(s, 10000)
                classical_val = self.zeta_classical(s)
                diff = abs(nc_val - classical_val)
                
                results.append({
                    'theta': theta,
                    'nc_value': complex(nc_val),
                    'classical_value': complex(classical_val),
                    'difference': float(diff)
                })
            except Exception as e:
                print(f"Error at theta={theta}: {e}")
                continue
        
        return results
    
    def visualize_results(self, save_path="phase_a_results"):
        """結果の可視化"""
        os.makedirs(save_path, exist_ok=True)
        
        # 1. 収束性プロット
        if 'convergence' in self.results:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            N_values = [r['N'] for r in self.results['convergence']]
            abs_values = [r['abs_value'] for r in self.results['convergence']]
            plt.semilogy(N_values, abs_values, 'b-', label='|ζ_θ(s)|')
            plt.xlabel('N (項数)')
            plt.ylabel('|ζ_θ(s)|')
            plt.title('非可換ゼータ関数の収束性')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 2, 2)
            differences = [abs(abs_values[i] - abs_values[i-1]) for i in range(1, len(abs_values))]
            plt.semilogy(N_values[1:], differences, 'r-', label='差分')
            plt.xlabel('N (項数)')
            plt.ylabel('|Δζ_θ(s)|')
            plt.title('収束差分')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"{save_path}/convergence_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. θ限界テスト
        if 'theta_limit' in self.results:
            plt.figure(figsize=(10, 6))
            theta_values = [r['theta'] for r in self.results['theta_limit']]
            differences = [r['difference'] for r in self.results['theta_limit']]
            
            plt.semilogy(theta_values, differences, 'g-', marker='o')
            plt.xlabel('θ (非可換パラメータ)')
            plt.ylabel('|ζ_θ(s) - ζ(s)|')
            plt.title('θ→0 限界での古典ゼータとの差分')
            plt.grid(True)
            plt.savefig(f"{save_path}/theta_limit_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. 臨界線零点
        if 'critical_zeros' in self.results:
            plt.figure(figsize=(12, 6))
            t_values = [r['t'] for r in self.results['critical_zeros']]
            abs_values = [r['abs_value'] for r in self.results['critical_zeros']]
            
            plt.subplot(1, 2, 1)
            plt.scatter(t_values, abs_values, c=abs_values, cmap='viridis', alpha=0.7)
            plt.colorbar(label='|ζ_θ(s)|')
            plt.xlabel('t (虚部)')
            plt.ylabel('|ζ_θ(s)|')
            plt.title('臨界線上の零点候補')
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.hist(abs_values, bins=20, alpha=0.7, color='orange')
            plt.xlabel('|ζ_θ(s)|')
            plt.ylabel('頻度')
            plt.title('零点候補の分布')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"{save_path}/critical_zeros_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def save_results(self, filename="phase_a_results.json"):
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
    print("🚀 Phase A 数値実験開始！")
    
    # 非可換ゼータ関数の初期化
    nzeta = NoncommutativeZeta(theta=0.1)
    
    # 1. 収束性テスト
    print("\n📊 収束性テスト実行中...")
    s_test = 2 + 3j  # テスト用の複素数
    nzeta.results['convergence'] = nzeta.convergence_test(s_test, max_N=5000)
    
    # 2. θ限界テスト
    print("\n🔄 θ限界テスト実行中...")
    nzeta.results['theta_limit'] = nzeta.theta_limit_test(s_test)
    
    # 3. 臨界線零点探索
    print("\n🔍 臨界線零点探索実行中...")
    nzeta.results['critical_zeros'] = nzeta.critical_line_search(t_min=0, t_max=50, step=0.5)
    
    # 4. 結果の可視化と保存
    print("\n📈 結果の可視化と保存中...")
    nzeta.visualize_results()
    nzeta.save_results()
    
    print("\n✅ Phase A 数値実験完了！")
    print(f"📁 結果ファイル: phase_a_results.json")
    print(f"📊 可視化ファイル: phase_a_results/")

if __name__ == "__main__":
    main() 