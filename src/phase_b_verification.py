#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase B 高精度検証: 関数方程式の数値検証
目標: zetaθ(s) = χ(s) · zeta_{-θ}(1-s) の厳密検証
"""

import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp, nsum, inf, zeta, gamma, sin, pi, exp, log, sqrt
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime
import os

# 超高精度設定
mp.dps = 100

class FunctionalEquationVerifier:
    """関数方程式の高精度検証クラス"""
    
    def __init__(self, theta=0.1):
        self.theta = theta
        self.results = {}
        
    def zeta_nc(self, s, N=10_000):
        """非可換ゼータ関数の高精度計算"""
        try:
            return nsum(lambda n: n**(-s) * exp(1j * self.theta * log(n)), [1, N])
        except Exception as e:
            print(f"Error in zeta_nc: {e}")
            return mp.nan
    
    def chi_factor(self, s):
        """関数方程式のχ因子"""
        try:
            gamma_factor = gamma((1 - s) / 2) / gamma(s / 2)
            chi = (2 * pi) ** (s - 1) * sin(pi * s / 2) * gamma_factor
            return chi
        except Exception as e:
            print(f"Error in chi_factor: {e}")
            return mp.nan
    
    def quantum_gravity_correction(self, s):
        """量子重力補正"""
        try:
            correction = exp(-1j * self.theta * log(2 * pi)) * \
                       (1 + self.theta**2 * log(s / (1 - s)) / 2)
            return correction
        except Exception as e:
            print(f"Error in quantum_gravity_correction: {e}")
            return mp.nan
    
    def unified_zeta(self, s, N=10_000):
        """統合ゼータ関数（量子重力補正込み）"""
        try:
            zeta_val = self.zeta_nc(s, N)
            correction = self.quantum_gravity_correction(s)
            return zeta_val * correction
        except Exception as e:
            print(f"Error in unified_zeta: {e}")
            return mp.nan
    
    def verify_functional_equation(self, s, N=10_000):
        """関数方程式の検証"""
        try:
            # 左辺: zetaθ(s)
            left_side = self.unified_zeta(s, N)
            
            # 右辺: χ(s) * zeta_{-θ}(1-s)
            chi = self.chi_factor(s)
            right_side = chi * self.unified_zeta(1 - s, N)
            
            # 差分計算
            difference = abs(left_side - right_side)
            relative_error = difference / abs(left_side) if left_side != 0 else float('inf')
            
            return {
                's': complex(s),
                'left_side': complex(left_side),
                'right_side': complex(right_side),
                'difference': float(difference),
                'relative_error': float(relative_error),
                'chi_factor': complex(chi),
                'quantum_correction': complex(self.quantum_gravity_correction(s))
            }
        except Exception as e:
            print(f"Error in verify_functional_equation: {e}")
            return None
    
    def critical_line_verification(self, t_min=0, t_max=100, step=1.0):
        """臨界線上の関数方程式検証"""
        results = []
        t_values = np.arange(t_min, t_max, step)
        
        for t in tqdm(t_values, desc="臨界線関数方程式検証"):
            s = 0.5 + 1j * t
            result = self.verify_functional_equation(s)
            if result:
                results.append(result)
        
        return results
    
    def analytic_continuation_test(self, re_range=np.linspace(-2, 3, 50)):
        """解析接続の検証"""
        results = []
        
        for re in tqdm(re_range, desc="解析接続検証"):
            s = re + 1j * 10  # 固定虚部で実部を変化
            result = self.verify_functional_equation(s)
            if result:
                results.append(result)
        
        return results
    
    def quantum_correction_analysis(self, theta_range=np.linspace(0, 0.5, 20)):
        """量子重力補正の影響分析"""
        results = []
        
        for theta in tqdm(theta_range, desc="量子重力補正分析"):
            self.theta = theta
            s = 0.5 + 1j * 10  # 臨界線上の点
            result = self.verify_functional_equation(s)
            if result:
                result['theta'] = theta
                results.append(result)
        
        return results
    
    def visualize_results(self, save_path="phase_b_results"):
        """結果の可視化"""
        os.makedirs(save_path, exist_ok=True)
        
        # 1. 関数方程式検証プロット
        if 'functional_equation' in self.results:
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 3, 1)
            differences = [r['difference'] for r in self.results['functional_equation']]
            t_values = [r['s'].imag for r in self.results['functional_equation']]
            plt.semilogy(t_values, differences, 'b-', marker='o', alpha=0.7)
            plt.xlabel('t (虚部)')
            plt.ylabel('|左辺 - 右辺|')
            plt.title('関数方程式検証')
            plt.grid(True)
            
            plt.subplot(2, 3, 2)
            relative_errors = [r['relative_error'] for r in self.results['functional_equation']]
            plt.semilogy(t_values, relative_errors, 'r-', marker='s', alpha=0.7)
            plt.xlabel('t (虚部)')
            plt.ylabel('相対誤差')
            plt.title('相対誤差')
            plt.grid(True)
            
            plt.subplot(2, 3, 3)
            chi_factors = [abs(r['chi_factor']) for r in self.results['functional_equation']]
            plt.plot(t_values, chi_factors, 'g-', marker='^', alpha=0.7)
            plt.xlabel('t (虚部)')
            plt.ylabel('|χ(s)|')
            plt.title('χ因子の絶対値')
            plt.grid(True)
            
            plt.subplot(2, 3, 4)
            quantum_corrections = [abs(r['quantum_correction']) for r in self.results['functional_equation']]
            plt.plot(t_values, quantum_corrections, 'm-', marker='d', alpha=0.7)
            plt.xlabel('t (虚部)')
            plt.ylabel('|量子重力補正|')
            plt.title('量子重力補正')
            plt.grid(True)
            
            plt.subplot(2, 3, 5)
            left_sides = [abs(r['left_side']) for r in self.results['functional_equation']]
            right_sides = [abs(r['right_side']) for r in self.results['functional_equation']]
            plt.semilogy(t_values, left_sides, 'b-', label='左辺', alpha=0.7)
            plt.semilogy(t_values, right_sides, 'r--', label='右辺', alpha=0.7)
            plt.xlabel('t (虚部)')
            plt.ylabel('絶対値')
            plt.title('左辺 vs 右辺')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 3, 6)
            plt.scatter([r['s'].real for r in self.results['functional_equation']], 
                       [r['s'].imag for r in self.results['functional_equation']], 
                       c=differences, cmap='viridis', alpha=0.7)
            plt.colorbar(label='|左辺 - 右辺|')
            plt.xlabel('Re(s)')
            plt.ylabel('Im(s)')
            plt.title('複素平面での誤差分布')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"{save_path}/functional_equation_verification.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. 解析接続テスト
        if 'analytic_continuation' in self.results:
            plt.figure(figsize=(12, 8))
            
            re_values = [r['s'].real for r in self.results['analytic_continuation']]
            differences = [r['difference'] for r in self.results['analytic_continuation']]
            
            plt.subplot(2, 2, 1)
            plt.semilogy(re_values, differences, 'b-', marker='o', alpha=0.7)
            plt.xlabel('Re(s)')
            plt.ylabel('|左辺 - 右辺|')
            plt.title('解析接続検証')
            plt.grid(True)
            
            plt.subplot(2, 2, 2)
            relative_errors = [r['relative_error'] for r in self.results['analytic_continuation']]
            plt.semilogy(re_values, relative_errors, 'r-', marker='s', alpha=0.7)
            plt.xlabel('Re(s)')
            plt.ylabel('相対誤差')
            plt.title('解析接続相対誤差')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"{save_path}/analytic_continuation_verification.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. 量子重力補正分析
        if 'quantum_correction' in self.results:
            plt.figure(figsize=(10, 6))
            
            theta_values = [r['theta'] for r in self.results['quantum_correction']]
            differences = [r['difference'] for r in self.results['quantum_correction']]
            
            plt.semilogy(theta_values, differences, 'g-', marker='o', alpha=0.7)
            plt.xlabel('θ (非可換パラメータ)')
            plt.ylabel('|左辺 - 右辺|')
            plt.title('量子重力補正の影響')
            plt.grid(True)
            plt.savefig(f"{save_path}/quantum_correction_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def save_results(self, filename="phase_b_results.json"):
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
    print("🚀 Phase B 高精度検証開始！")
    
    # 関数方程式検証の初期化
    verifier = FunctionalEquationVerifier(theta=0.1)
    
    # 1. 臨界線上の関数方程式検証
    print("\n📊 臨界線上の関数方程式検証実行中...")
    verifier.results['functional_equation'] = verifier.critical_line_verification(t_min=0, t_max=50, step=1.0)
    
    # 2. 解析接続テスト
    print("\n🔄 解析接続テスト実行中...")
    verifier.results['analytic_continuation'] = verifier.analytic_continuation_test()
    
    # 3. 量子重力補正分析
    print("\n⚛️ 量子重力補正分析実行中...")
    verifier.results['quantum_correction'] = verifier.quantum_correction_analysis()
    
    # 4. 結果の可視化と保存
    print("\n📈 結果の可視化と保存中...")
    verifier.visualize_results()
    verifier.save_results()
    
    print("\n✅ Phase B 高精度検証完了！")
    print(f"📁 結果ファイル: phase_b_results.json")
    print(f"📊 可視化ファイル: phase_b_results/")

if __name__ == "__main__":
    main() 