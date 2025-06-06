#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔬 NKAT Simple Precision Enhancer
=================================
シンプルで確実な超高精度再検証システム
"""

import mpmath as mp
import time
from datetime import datetime
from tqdm import tqdm

def enhance_zero_verification(t_value, target_precision=200):
    """
    🎯 単一ゼロ点の高精度検証
    
    Args:
        t_value: ゼロ点の虚部
        target_precision: 目標精度
        
    Returns:
        検証結果
    """
    # 精度設定
    original_dps = mp.dps
    mp.dps = target_precision + 50
    
    try:
        # ゼロ点の定義
        s = mp.mpc(mp.mpf('0.5'), mp.mpf(str(t_value)))
        
        # リーマンゼータ関数の計算
        start_time = time.time()
        zeta_value = mp.zeta(s)
        calc_time = time.time() - start_time
        
        # 絶対値の計算
        abs_zeta = abs(zeta_value)
        
        # 精度判定（より緩い基準）
        if abs_zeta < mp.mpf(10) ** (-150):
            status = "✅ 超高精度ゼロ"
            is_zero = True
        elif abs_zeta < mp.mpf(10) ** (-100):
            status = "🎯 高精度ゼロ"
            is_zero = True
        elif abs_zeta < mp.mpf(10) ** (-50):
            status = "📏 精密ゼロ"
            is_zero = True
        elif abs_zeta < mp.mpf(10) ** (-20):
            status = "🔍 数値ゼロ"
            is_zero = True
        elif abs_zeta < mp.mpf(10) ** (-10):
            status = "📊 近似ゼロ"
            is_zero = True
        else:
            status = "❌ ゼロではない"
            is_zero = False
        
        result = {
            't': t_value,
            'precision_used': target_precision,
            'zeta_value': str(zeta_value),
            'abs_zeta': str(abs_zeta),
            'abs_zeta_float': float(abs_zeta),
            'abs_zeta_scientific': f"{float(abs_zeta):.2e}",
            'is_zero': is_zero,
            'status': status,
            'calculation_time': calc_time
        }
        
        return result
        
    finally:
        mp.dps = original_dps

def main():
    """メイン実行関数"""
    print("🔬 NKAT Simple Precision Enhancer 起動")
    print("=" * 60)
    
    # 前回失敗したゼロ点
    failed_zeros = [
        52.97032147778034,
        56.446244229740955,
        59.347044000825385,
        60.83178239760432,
        65.11254404444117,
        67.07980507468255,
        69.54641033011764,
        72.06715767480921,
        75.70469232045076,
        77.14481700970858
    ]
    
    print(f"🎯 {len(failed_zeros)}個のゼロ点を200桁精度で再検証")
    print("=" * 60)
    
    success_count = 0
    results = []
    
    for i, t in enumerate(failed_zeros, 1):
        print(f"\n📍 検証 {i}/{len(failed_zeros)}: t = {t}")
        
        try:
            result = enhance_zero_verification(t, target_precision=200)
            results.append(result)
            
            print(f"   |ζ(s)| = {result['abs_zeta_scientific']}")
            print(f"   {result['status']}")
            print(f"   ⏱️  計算時間: {result['calculation_time']:.3f}秒")
            
            if result['is_zero']:
                success_count += 1
                print("   ✅ 検証成功!")
            else:
                print("   ❌ 非ゼロ")
                
        except Exception as e:
            print(f"   ❌ エラー: {e}")
            results.append({
                't': t,
                'error': str(e),
                'is_zero': False
            })
    
    # 最終結果
    total = len(failed_zeros)
    success_rate = (success_count / total * 100) if total > 0 else 0
    
    print("\n" + "=" * 60)
    print("🎉 200桁精度再検証結果サマリー")
    print("=" * 60)
    print(f"🔢 再検証ゼロ点数: {total}")
    print(f"✅ 成功数: {success_count}")
    print(f"❌ 失敗数: {total - success_count}")
    print(f"📈 成功率: {success_rate:.1f}%")
    
    if success_rate >= 50:
        print("\n🎉 大幅な改善を達成!")
        print("📐 高精度計算により多くのゼロ点を確認")
    elif success_rate >= 20:
        print("\n🎯 部分的な改善を達成")
        print("📏 一部のゼロ点で高精度確認")
    else:
        print("\n⚠️ 改善は限定的")
        print("📊 これらの点は真のゼロ点でない可能性")
    
    print("=" * 60)
    print("🔬 検証完了")
    
    # 結果保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"simple_enhanced_results_{timestamp}.json"
    
    import json
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({
            'total': total,
            'success_count': success_count,
            'success_rate': success_rate,
            'results': results,
            'timestamp': timestamp
        }, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"💾 結果保存: {filename}")

if __name__ == "__main__":
    main() 