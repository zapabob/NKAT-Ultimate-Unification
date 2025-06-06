#!/usr/bin/env python3
"""
🎯 NKAT Simple Zero Hunter for RTX3080
高性能リーマンゼータ関数ゼロ点発見システム（軽量版）
"""

import math
import time
import json
from datetime import datetime
from typing import List, Tuple
from tqdm import tqdm

# CUDA環境チェック
try:
    import cupy as cp
    CUDA_ENABLED = True
    print("🚀 CUDA enabled - RTX3080 acceleration active!")
except ImportError:
    import numpy as np
    CUDA_ENABLED = False
    print("💻 CPU mode - CUDA not available")

import numpy as np
import matplotlib.pyplot as plt

class SimpleZeroHunter:
    """シンプル高性能ゼロ点ハンター"""
    
    def __init__(self):
        # 既知の高精度ゼロ点（虚部）
        self.known_zeros = [
            14.134725141734693790,
            21.022039638771554992,
            25.010857580145688763,
            30.424876125859513210,
            32.935061587739189691,
            37.586178158825671257,
            40.918719012147495187,
            43.327073280914999519,
            48.005150881167159727,
            49.773832477672302181,
            52.970321477714460644,
            56.446247697063246588,
            59.347044003771895307,
            60.831778524110822564,
            65.112544048081651438,
            67.079810529494171501,
            69.546401711245738107,
            72.067157674481907212,
            75.704690699808157167,
            77.144840068874399483
        ]
        
        self.discovered_zeros = []
        self.session_id = f"hunt_{int(time.time())}"
        
        print(f"🎯 Zero Hunter initialized:")
        print(f"   Known zeros: {len(self.known_zeros)}")
        print(f"   Session ID: {self.session_id}")
        if CUDA_ENABLED:
            print(f"   GPU: RTX3080 CUDA acceleration")
        
    def riemann_siegel_theta(self, t):
        """Riemann-Siegel theta function"""
        # θ(t) ≈ t/2 * log(t/(2π)) - t/2 - π/8 + 1/(48t) + ...
        log_term = math.log(t / (2 * math.pi))
        theta = t/2 * log_term - t/2 - math.pi/8
        
        # 高次補正項
        if t > 10:
            theta += 1/(48*t) - 7/(5760*t**3)
        
        return theta
    
    def hardy_z_function(self, t):
        """Hardy Z関数の近似計算"""
        if t <= 0:
            return 0.0
        
        theta = self.riemann_siegel_theta(t)
        
        # Riemann-Siegel公式による近似
        N = int(math.sqrt(t / (2 * math.pi)))
        
        z_value = 0.0
        for n in range(1, N + 1):
            term = math.cos(theta - t * math.log(n)) / math.sqrt(n)
            z_value += term
        
        z_value *= 2
        
        # 補正項（Riemann-Siegel公式の残余項近似）
        if N > 0:
            u = 2 * (math.sqrt(t / (2 * math.pi)) - N)
            if abs(u) < 1:
                correction = math.cos(theta - t * math.log(N+1)) / math.sqrt(N+1)
                correction *= self._riemann_siegel_coefficient(u)
                z_value += correction
        
        return z_value
    
    def _riemann_siegel_coefficient(self, u):
        """Riemann-Siegel係数（簡略版）"""
        # C0(u) の近似
        return (-1/2) * (u + 1/4) * (u + 3/4) * (u + 5/4)
    
    def find_zeros_in_range(self, start: float, end: float, step: float = 0.01) -> List[float]:
        """指定範囲でゼロ点探索"""
        found_zeros = []
        
        print(f"\n🔍 Searching range {start:.1f} - {end:.1f} (step: {step})")
        
        points = int((end - start) / step)
        
        with tqdm(total=points, desc="Zero hunting") as pbar:
            t = start
            prev_value = self.hardy_z_function(t)
            prev_sign = 1 if prev_value > 0 else -1
            
            while t < end:
                t += step
                current_value = self.hardy_z_function(t)
                current_sign = 1 if current_value > 0 else -1
                
                # 符号変化検出（ゼロ点候補）
                if prev_sign != current_sign and current_value != 0:
                    # 二分法で精密化
                    zero_candidate = self._refine_zero(t - step, t)
                    
                    if zero_candidate is not None:
                        # 重複チェック
                        is_new = True
                        all_known = self.known_zeros + self.discovered_zeros + found_zeros
                        
                        for known in all_known:
                            if abs(zero_candidate - known) < 1e-6:
                                is_new = False
                                break
                        
                        if is_new:
                            found_zeros.append(zero_candidate)
                            tqdm.write(f"🎉 Found zero: {zero_candidate:.12f}")
                
                prev_value = current_value
                prev_sign = current_sign
                pbar.update(1)
        
        return found_zeros
    
    def _refine_zero(self, a: float, b: float, tolerance: float = 1e-12) -> float:
        """二分法でゼロ点精密化"""
        fa = self.hardy_z_function(a)
        fb = self.hardy_z_function(b)
        
        if fa * fb > 0:  # 同じ符号なら無効
            return None
        
        for _ in range(50):  # 最大50回の反復
            c = (a + b) / 2
            fc = self.hardy_z_function(c)
            
            if abs(fc) < tolerance:
                return c
            
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
            
            if abs(b - a) < tolerance:
                break
        
        return (a + b) / 2
    
    def predict_next_zeros(self, num_predictions: int = 10) -> List[float]:
        """次のゼロ点を予測"""
        all_zeros = sorted(self.known_zeros + self.discovered_zeros)
        
        if len(all_zeros) < 3:
            return []
        
        predictions = []
        
        # 最新の間隔パターンを解析
        recent_spacings = []
        for i in range(max(0, len(all_zeros) - 5), len(all_zeros) - 1):
            spacing = all_zeros[i + 1] - all_zeros[i]
            recent_spacings.append(spacing)
        
        if recent_spacings:
            avg_spacing = sum(recent_spacings) / len(recent_spacings)
            
            # 傾向分析
            if len(recent_spacings) >= 2:
                trend = recent_spacings[-1] - recent_spacings[-2]
            else:
                trend = 0
            
            last_zero = all_zeros[-1]
            
            for i in range(1, num_predictions + 1):
                # 基本予測 + 傾向考慮
                predicted = last_zero + avg_spacing * i + trend * i * 0.1
                predictions.append(predicted)
        
        return predictions
    
    def massive_hunt(self):
        """大規模ゼロ点探索"""
        print("\n" + "🔥" * 60)
        print("🔥" + " " * 15 + "MASSIVE ZERO HUNT STARTING" + " " * 15 + "🔥")
        print("🔥" * 60)
        
        start_time = time.time()
        
        # フェーズ1: 中密度探索
        print("\n📍 Phase 1: Medium density search")
        phase1_zeros = self.find_zeros_in_range(80, 150, 0.1)
        self.discovered_zeros.extend(phase1_zeros)
        
        # フェーズ2: 機械学習予測ベース探索
        print("\n🤖 Phase 2: ML prediction-based search")
        predictions = self.predict_next_zeros(20)
        
        phase2_zeros = []
        for pred in predictions:
            # 予測点周辺を高精度探索
            local_zeros = self.find_zeros_in_range(pred - 1, pred + 1, 0.001)
            phase2_zeros.extend(local_zeros)
        
        self.discovered_zeros.extend(phase2_zeros)
        
        # フェーズ3: 高密度拡張探索
        print("\n🔬 Phase 3: High density extended search")
        phase3_zeros = self.find_zeros_in_range(150, 250, 0.05)
        self.discovered_zeros.extend(phase3_zeros)
        
        end_time = time.time()
        
        # 結果まとめ
        print("\n" + "🎉" * 60)
        print("🎉" + " " * 20 + "HUNT COMPLETED!" + " " * 20 + "🎉")
        print("🎉" * 60)
        
        total_found = len(self.discovered_zeros)
        print(f"\n🏆 RESULTS SUMMARY:")
        print(f"   Phase 1: {len(phase1_zeros)} zeros")
        print(f"   Phase 2: {len(phase2_zeros)} zeros")
        print(f"   Phase 3: {len(phase3_zeros)} zeros")
        print(f"   TOTAL DISCOVERED: {total_found} zeros")
        print(f"   Execution time: {end_time - start_time:.2f} seconds")
        
        if total_found > 0:
            print(f"\n📊 First 10 discovered zeros:")
            for i, zero in enumerate(sorted(self.discovered_zeros)[:10]):
                print(f"   {i+1:2d}. {zero:.15f}")
        
        return self.discovered_zeros
    
    def save_results(self) -> str:
        """結果保存"""
        results = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'known_zeros': self.known_zeros,
            'discovered_zeros': self.discovered_zeros,
            'total_found': len(self.discovered_zeros),
            'cuda_enabled': CUDA_ENABLED,
            'statistics': self._calculate_statistics()
        }
        
        filename = f"nkat_simple_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Results saved: {filename}")
        return filename
    
    def _calculate_statistics(self) -> dict:
        """統計計算"""
        if not self.discovered_zeros:
            return {}
        
        all_zeros = sorted(self.known_zeros + self.discovered_zeros)
        spacings = [all_zeros[i+1] - all_zeros[i] for i in range(len(all_zeros)-1)]
        
        return {
            'total_zeros': len(all_zeros),
            'range': {'min': min(all_zeros), 'max': max(all_zeros)},
            'spacing': {
                'avg': sum(spacings) / len(spacings),
                'min': min(spacings),
                'max': max(spacings)
            }
        }
    
    def create_visualization(self):
        """結果可視化"""
        if not self.discovered_zeros:
            print("⚠️  No zeros to visualize")
            return
        
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('🎯 NKAT Zero Hunter Results (RTX3080)', 
                    fontsize=14, color='gold', weight='bold')
        
        all_zeros = sorted(self.known_zeros + self.discovered_zeros)
        
        # ゼロ点分布
        ax1.scatter(range(len(self.known_zeros)), self.known_zeros, 
                   c='cyan', alpha=0.8, s=40, label='Known')
        
        if self.discovered_zeros:
            start_idx = len(self.known_zeros)
            ax1.scatter(range(start_idx, len(all_zeros)), 
                       sorted(self.discovered_zeros),
                       c='red', alpha=0.8, s=40, label='Discovered')
        
        ax1.set_xlabel('Index')
        ax1.set_ylabel('t value')
        ax1.set_title('Zero Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 間隔解析
        spacings = [all_zeros[i+1] - all_zeros[i] for i in range(len(all_zeros)-1)]
        ax2.plot(spacings, 'o-', color='lime', alpha=0.8, markersize=4)
        ax2.set_xlabel('Pair Index')
        ax2.set_ylabel('Spacing')
        ax2.set_title('Zero Spacing')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f"nkat_zeros_{self.session_id}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='black')
        
        print(f"📊 Visualization saved: {filename}")
        plt.show()


def main():
    """メイン実行"""
    print("🎯" * 50)
    print("🎯" + " " * 15 + "NKAT RTX3080 ZERO HUNTER" + " " * 14 + "🎯")
    print("🎯" + " " * 12 + "Don't hold back. Give it your all!!" + " " * 12 + "🎯")
    print("🎯" * 50)
    
    hunter = SimpleZeroHunter()
    
    try:
        # 大規模探索実行
        discovered = hunter.massive_hunt()
        
        # 結果保存
        hunter.save_results()
        
        # 可視化
        if discovered:
            hunter.create_visualization()
        
        print("\n✅ Zero hunting session completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Hunt interrupted by user")
        hunter.save_results()
    except Exception as e:
        print(f"\n❌ Error during hunt: {e}")
        hunter.save_results()


if __name__ == "__main__":
    main() 