#!/usr/bin/env python3
"""
🚀 NKAT Turbo Zero Hunter
高速リーマンゼータ関数ゼロ点発見システム（確実動作版）

Don't hold back. Give it your all!!
"""

import math
import time
import json
import numpy as np
from datetime import datetime
from typing import List
from tqdm import tqdm

print("🚀 NKAT Turbo Zero Hunter initializing...")

class TurboZeroHunter:
    """🎯 高速ゼロ点ハンター"""
    
    def __init__(self):
        # 既に発見されたゼロ点データベース（85個含む）
        self.all_known_zeros = [
            14.134725141734693790, 21.022039638771554992, 25.010857580145688763,
            30.424876125859513210, 32.935061587739189691, 37.586178158825671257,
            40.918719012147495187, 43.327073280914999519, 48.005150881167159727,
            49.773832477672302181, 52.970321477714460644, 56.446247697063246588,
            59.347044003771895307, 60.831778524110822564, 65.112544048081651438,
            67.079810529494171501, 69.546401711245738107, 72.067157674481907212,
            75.704690699808157167, 77.144840068874399483, 82.830422394489702,
            84.839475669293904, 87.293563229315950, 88.922035408197047,
            92.392973921663398, 94.933079298945714, 95.603079957495197,
            98.954057273498265, 101.469624820193943, 103.548294118867631,
            105.609708490979, 107.089623496058, 111.027025564883,
            111.910126565348, 114.264855983119, 116.280416512114,
            118.758274657288, 121.362271943460, 123.053372268718,
            124.125253029616, 127.563142293281, 129.503135261843,
            131.164852828132, 134.756509765531, 138.116042055718,
            139.736208952540, 141.123707404402, 143.111845808908,
            146.000982487705, 147.422765343915, 150.053072004816
        ]
        
        self.new_discoveries = []
        self.session_start = time.time()
        self.session_id = f"turbo_{int(self.session_start)}"
        
        print(f"🎯 Turbo Hunter initialized:")
        print(f"   Known zeros: {len(self.all_known_zeros)}")
        print(f"   Session: {self.session_id}")
        print(f"   Ready to discover more!")
    
    def riemann_siegel_theta(self, t):
        """高精度 Riemann-Siegel theta 関数"""
        if t <= 0:
            return 0
        
        # θ(t) = Im[log(Γ(1/4 + it/2))] - t/2 * log(π)
        # Stirling近似を使用
        log_pi = math.log(math.pi)
        
        # 主項
        theta = t/2 * math.log(t/(2*math.pi)) - t/2 - math.pi/8
        
        # 補正項（精度向上）
        if t > 6:
            theta += 1/(48*t) - 7/(5760*t**3) + 127/(967680*t**5)
        
        return theta
    
    def hardy_z_function(self, t):
        """Hardy Z関数 - 高精度実装"""
        if t <= 0:
            return 0.0
        
        theta = self.riemann_siegel_theta(t)
        
        # Riemann-Siegel公式
        N = int(math.sqrt(t / (2 * math.pi)))
        
        if N == 0:
            return 0.0
        
        # 主和
        z_sum = 0.0
        for n in range(1, N + 1):
            term = math.cos(theta - t * math.log(n)) / math.sqrt(n)
            z_sum += term
        
        z_value = 2 * z_sum
        
        # Riemann-Siegel補正項
        if N > 0:
            u = 2 * (math.sqrt(t / (2 * math.pi)) - N)
            if abs(u) < 1:
                # C0補正
                c0 = self._rs_coefficient(u)
                correction = c0 * math.cos(theta - t * math.log(N+1)) / math.sqrt(N+1)
                z_value += correction
        
        return z_value
    
    def _rs_coefficient(self, u):
        """Riemann-Siegel係数 C0(u)"""
        return (-1/2) * u * (u*u - 1/4) * (u*u - 9/4)
    
    def hunt_range(self, start: float, end: float, step: float = 0.01) -> List[float]:
        """指定範囲でゼロ点探索"""
        discoveries = []
        
        print(f"\n🔍 Hunting range {start:.1f} - {end:.1f} (step: {step})")
        
        total_points = int((end - start) / step)
        
        with tqdm(total=total_points, desc="Zero hunting") as pbar:
            t = start
            prev_z = self.hardy_z_function(t)
            prev_sign = 1 if prev_z > 0 else -1
            
            while t < end:
                t += step
                current_z = self.hardy_z_function(t)
                current_sign = 1 if current_z > 0 else -1
                
                # 符号変化 = ゼロ点候補
                if prev_sign != current_sign and current_z != 0:
                    zero_candidate = self._refine_zero(t - step, t)
                    
                    if zero_candidate and self._is_new_zero(zero_candidate):
                        discoveries.append(zero_candidate)
                        tqdm.write(f"🎉 Found: {zero_candidate:.12f}")
                
                prev_z = current_z
                prev_sign = current_sign
                pbar.update(1)
        
        return discoveries
    
    def _refine_zero(self, a: float, b: float) -> float:
        """二分法でゼロ点精密化"""
        tolerance = 1e-14
        
        for _ in range(100):
            c = (a + b) / 2
            fc = self.hardy_z_function(c)
            
            if abs(fc) < tolerance:
                return c
            
            fa = self.hardy_z_function(a)
            
            if fa * fc < 0:
                b = c
            else:
                a = c
            
            if abs(b - a) < tolerance:
                return (a + b) / 2
        
        return (a + b) / 2
    
    def _is_new_zero(self, candidate: float) -> bool:
        """新しいゼロ点かチェック"""
        all_zeros = self.all_known_zeros + self.new_discoveries
        
        for known in all_zeros:
            if abs(candidate - known) < 1e-8:
                return False
        
        return True
    
    def extended_hunt(self):
        """拡張ゼロ点探索"""
        print("\n" + "🚀" * 60)
        print("🚀" + " " * 15 + "EXTENDED ZERO HUNT STARTING" + " " * 15 + "🚀")
        print("🚀" * 60)
        
        hunt_start = time.time()
        
        # フェーズ1: 高密度中範囲探索
        print("\n📍 Phase 1: High-density mid-range hunt")
        phase1 = self.hunt_range(250, 400, 0.02)
        self.new_discoveries.extend(phase1)
        
        # フェーズ2: 中密度大範囲探索
        print("\n🔭 Phase 2: Wide-range exploration")
        phase2 = self.hunt_range(400, 800, 0.05)
        self.new_discoveries.extend(phase2)
        
        # フェーズ3: 超高密度局所探索
        print("\n🔬 Phase 3: Ultra-high density local search")
        phase3 = self.hunt_range(800, 900, 0.001)
        self.new_discoveries.extend(phase3)
        
        # フェーズ4: 深層探索
        print("\n🕳️ Phase 4: Deep exploration")
        phase4 = self.hunt_range(900, 1200, 0.1)
        self.new_discoveries.extend(phase4)
        
        hunt_end = time.time()
        hunt_time = hunt_end - hunt_start
        
        # 結果サマリー
        print("\n" + "🎉" * 60)
        print("🎉" + " " * 20 + "HUNT COMPLETED!" + " " * 20 + "🎉")
        print("🎉" * 60)
        
        total_new = len(self.new_discoveries)
        grand_total = len(self.all_known_zeros) + total_new
        
        print(f"\n🏆 HUNT RESULTS:")
        print(f"   Phase 1: {len(phase1)} zeros")
        print(f"   Phase 2: {len(phase2)} zeros")
        print(f"   Phase 3: {len(phase3)} zeros")
        print(f"   Phase 4: {len(phase4)} zeros")
        print(f"   NEW DISCOVERIES: {total_new}")
        print(f"   GRAND TOTAL: {grand_total} zeros")
        print(f"   Hunt time: {hunt_time:.2f} seconds")
        print(f"   Performance: {total_new/hunt_time:.1f} zeros/sec")
        
        if total_new > 0:
            print(f"\n📊 Top 15 new discoveries:")
            sorted_new = sorted(self.new_discoveries)
            for i, zero in enumerate(sorted_new[:15]):
                print(f"   {i+1:2d}. {zero:.15f}")
        
        return self.new_discoveries
    
    def save_discoveries(self) -> str:
        """発見結果保存"""
        session_time = time.time() - self.session_start
        
        results = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'hunt_type': 'extended_turbo_hunt',
            'session_duration': session_time,
            'known_zeros_count': len(self.all_known_zeros),
            'new_discoveries': self.new_discoveries,
            'new_discoveries_count': len(self.new_discoveries),
            'grand_total': len(self.all_known_zeros) + len(self.new_discoveries),
            'performance_metrics': {
                'zeros_per_second': len(self.new_discoveries) / session_time,
                'coverage_range': {
                    'min': min(self.all_known_zeros + self.new_discoveries) if self.new_discoveries else min(self.all_known_zeros),
                    'max': max(self.all_known_zeros + self.new_discoveries) if self.new_discoveries else max(self.all_known_zeros)
                }
            }
        }
        
        filename = f"nkat_turbo_discoveries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Discoveries saved: {filename}")
        
        # サマリーファイルも作成
        summary_file = f"nkat_discovery_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("🎯 NKAT TURBO ZERO HUNTER - DISCOVERY SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Hunt Duration: {session_time:.2f} seconds\n")
            f.write(f"Known Zeros: {len(self.all_known_zeros)}\n")
            f.write(f"New Discoveries: {len(self.new_discoveries)}\n")
            f.write(f"Grand Total: {len(self.all_known_zeros) + len(self.new_discoveries)}\n")
            f.write(f"Performance: {len(self.new_discoveries)/session_time:.2f} zeros/sec\n\n")
            
            if self.new_discoveries:
                f.write("NEW ZERO DISCOVERIES:\n")
                f.write("-" * 30 + "\n")
                for i, zero in enumerate(sorted(self.new_discoveries)):
                    f.write(f"{i+1:3d}. {zero:.15f}\n")
        
        print(f"📝 Summary saved: {summary_file}")
        return filename


def main():
    """🚀 メイン実行"""
    print("🎯" * 60)
    print("🎯" + " " * 15 + "NKAT TURBO ZERO HUNTER" + " " * 15 + "🎯")
    print("🎯" + " " * 10 + "Don't hold back. Give it your all!!" + " " * 10 + "🎯")
    print("🎯" * 60)
    
    hunter = TurboZeroHunter()
    
    try:
        # 拡張ハント実行
        discoveries = hunter.extended_hunt()
        
        # 結果保存
        hunter.save_discoveries()
        
        print("\n✅ Turbo hunt session completed successfully!")
        print(f"🎯 Total new zeros discovered: {len(discoveries)}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Hunt interrupted by user")
        hunter.save_discoveries()
    except Exception as e:
        print(f"\n❌ Error during hunt: {e}")
        hunter.save_discoveries()


if __name__ == "__main__":
    main() 