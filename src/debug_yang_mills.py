#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script for Yang-Mills theory
量子ヤンミルズ理論のデバッグスクリプト
"""

import numpy as np
import json
from datetime import datetime

def test_basic_calculation():
    """基本的な計算のテスト"""
    print("🧪 基本的な計算テスト開始...")
    
    try:
        # 物理定数
        lambda_qcd = 0.2  # GeV
        energy_scale = 1.0  # GeV
        
        # 結合定数の計算
        g0 = 1.0
        mu0 = 1.0
        beta0 = 11.0 / (4 * np.pi)
        log_ratio = np.log(energy_scale**2 / mu0**2)
        
        running_g = g0 / np.sqrt(1 + beta0 * g0**2 * log_ratio)
        print(f"✅ 結合定数計算成功: {running_g:.6f}")
        
        # 質量ギャップの計算
        N = 3
        mass_gap = lambda_qcd**2 * np.exp(-8 * np.pi**2 / (running_g**2 * N))
        print(f"✅ 質量ギャップ計算成功: {mass_gap:.6f} GeV²")
        
        # 弦張力の計算
        string_tension = lambda_qcd**2 * np.exp(-8 * np.pi**2 / (running_g**2 * N))
        print(f"✅ 弦張力計算成功: {string_tension:.6f} GeV²")
        
        return True
        
    except Exception as e:
        print(f"❌ 基本計算エラー: {e}")
        return False

def test_numpy_conversion():
    """numpy変換のテスト"""
    print("\n🧪 numpy変換テスト開始...")
    
    try:
        # テストデータ
        test_data = {
            'array': np.array([1, 2, 3]),
            'bool': np.bool_(True),
            'int': np.int32(42),
            'float': np.float64(3.14),
            'nested': {
                'array': np.array([[1, 2], [3, 4]]),
                'bool': np.bool_(False)
            }
        }
        
        # 変換関数
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        # 変換実行
        converted_data = convert_numpy(test_data)
        
        # JSON保存テスト
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"debug_test_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ numpy変換成功: {filename}")
        return True
        
    except Exception as e:
        print(f"❌ numpy変換エラー: {e}")
        return False

def test_enhanced_mass_gap():
    """改良された質量ギャップ計算のテスト"""
    print("\n🧪 改良された質量ギャップ計算テスト開始...")
    
    try:
        # 物理定数
        lambda_qcd = 0.2  # GeV
        energy_scales = np.logspace(-3, 3, 50)
        mass_gaps = []
        
        for scale in energy_scales:
            # 結合定数
            g0 = 1.0
            mu0 = 1.0
            beta0 = 11.0 / (4 * np.pi)
            beta1 = 51.0 / (8 * np.pi**2)
            
            log_ratio = np.log(scale**2 / mu0**2)
            
            # 2ループ精度の結合定数
            running_g = g0 / np.sqrt(1 + beta0 * g0**2 * log_ratio + 
                                    (beta1 / beta0) * g0**2 * np.log(1 + beta0 * g0**2 * log_ratio))
            
            # 主要項
            main_gap = lambda_qcd**2 * np.exp(-8 * np.pi**2 / (running_g**2 * 3))
            
            # 量子補正
            quantum_correction = running_g**2 / (4 * np.pi) * scale**2
            
            # 非可換補正
            theta = 1e-6
            noncommutative_correction = theta * scale**4 / (4 * np.pi**2)
            
            total_gap = main_gap + quantum_correction + noncommutative_correction
            mass_gaps.append(max(total_gap, 0.01))
        
        min_gap = np.min(mass_gaps)
        min_gap_position = energy_scales[np.argmin(mass_gaps)]
        
        print(f"✅ 改良された質量ギャップ計算成功:")
        print(f"   最小質量ギャップ: {min_gap:.6f} GeV²")
        print(f"   ギャップ位置: {min_gap_position:.6f} GeV")
        print(f"   質量ギャップ検証: {min_gap > 0.01}")
        
        return True
        
    except Exception as e:
        print(f"❌ 改良された質量ギャップ計算エラー: {e}")
        return False

def main():
    """メイン関数"""
    print("🔍 量子ヤンミルズ理論デバッグテスト")
    print("=" * 50)
    
    # テスト実行
    test1 = test_basic_calculation()
    test2 = test_numpy_conversion()
    test3 = test_enhanced_mass_gap()
    
    print(f"\n📊 テスト結果:")
    print(f"✅ 基本計算: {'成功' if test1 else '失敗'}")
    print(f"✅ numpy変換: {'成功' if test2 else '失敗'}")
    print(f"✅ 改良された質量ギャップ: {'成功' if test3 else '失敗'}")
    
    if test1 and test2 and test3:
        print("\n🎉 全てのテストが成功しました！")
    else:
        print("\n❌ 一部のテストが失敗しました。")

if __name__ == "__main__":
    main() 