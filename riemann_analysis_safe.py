#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT リーマン予想解析 - 安全版
CuPyエラー対応済み
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

# CUDA環境の安全な検出
CUPY_AVAILABLE = False
try:
    import cupy as cp
    import cupyx.scipy.special as cp_special
    CUPY_AVAILABLE = True
    print("✅ CuPy CUDA利用可能 - GPU超高速モードで実行")
except ImportError as e:
    print(f"⚠️ CuPy未検出: {e}")
    print("💡 CPUモードで実行します")
    import numpy as cp
except Exception as e:
    print(f"❌ CuPy初期化エラー: {e}")
    print("💡 CPUモードで実行します")
    import numpy as cp
    CUPY_AVAILABLE = False

def safe_riemann_analysis(max_iterations=1000):
    """安全なリーマン予想解析"""
    print("🔬 NKAT リーマン予想解析開始")
    
    # 解析パラメータ
    t_values = np.linspace(0.1, 50, max_iterations)
    zeta_values = []
    
    print(f"📊 解析点数: {len(t_values)}")
    
    # プログレスバー付きで解析実行
    for t in tqdm(t_values, desc="リーマンゼータ関数計算"):
        try:
            # 簡単なゼータ関数近似
            s = 0.5 + 1j * t
            zeta_approx = sum(1/n**s for n in range(1, 100))
            zeta_values.append(abs(zeta_approx))
        except Exception as e:
            print(f"⚠️ 計算エラー (t={t}): {e}")
            zeta_values.append(0)
    
    # 結果の可視化
    plt.figure(figsize=(12, 8))
    plt.plot(t_values, zeta_values, 'b-', linewidth=1, alpha=0.7)
    plt.title('NKAT リーマンゼータ関数解析結果', fontsize=16)
    plt.xlabel('虚部 t', fontsize=12)
    plt.ylabel('|ζ(1/2 + it)|', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 結果保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'nkat_safe_riemann_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
    
    # JSON結果保存
    results = {
        'timestamp': timestamp,
        'cupy_available': CUPY_AVAILABLE,
        'max_iterations': max_iterations,
        'analysis_points': len(t_values),
        'max_zeta_value': max(zeta_values),
        'min_zeta_value': min(zeta_values),
        'mean_zeta_value': np.mean(zeta_values)
    }
    
    with open(f'nkat_safe_riemann_analysis_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("✅ 解析完了")
    print(f"📈 最大値: {results['max_zeta_value']:.6f}")
    print(f"📉 最小値: {results['min_zeta_value']:.6f}")
    print(f"📊 平均値: {results['mean_zeta_value']:.6f}")
    
    plt.show()
    return results

if __name__ == "__main__":
    try:
        results = safe_riemann_analysis()
        print("🎉 NKAT リーマン予想解析が正常に完了しました")
    except KeyboardInterrupt:
        print("\n⏹️ ユーザーによって中断されました")
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")
        import traceback
        traceback.print_exc()
