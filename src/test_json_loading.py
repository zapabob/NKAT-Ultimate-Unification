#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最新JSONファイル読み込み機能のテスト
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_enhanced_large_scale_verification():
    """nkat_v11_enhanced_large_scale_verification.pyのテスト"""
    try:
        from nkat_v11_enhanced_large_scale_verification import LargeScaleGammaChallengeIntegrator
        
        print("🔍 LargeScaleGammaChallengeIntegratorテスト開始...")
        integrator = LargeScaleGammaChallengeIntegrator()
        
        # データ読み込みテスト
        data = integrator._load_gamma_challenge_data()
        if data:
            print(f"✅ データ読み込み成功")
            if 'results' in data:
                print(f"📊 結果数: {len(data['results']):,}")
            else:
                print(f"📊 データキー: {list(data.keys())}")
        else:
            print("⚠️ データが見つかりませんでした")
        
        # 高品質γ値抽出テスト
        high_quality_gammas = integrator.extract_high_quality_gammas(min_quality=0.95, max_count=50)
        print(f"📈 高品質γ値抽出: {len(high_quality_gammas)}個")
        if high_quality_gammas:
            print(f"📊 範囲: {min(high_quality_gammas):.6f} - {max(high_quality_gammas):.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False

def test_gamma_challenge_robust():
    """nkat_10000_gamma_challenge_robust.pyのテスト"""
    try:
        from nkat_10000_gamma_challenge_robust import NKAT10KGammaChallenge, RobustRecoveryManager
        
        print("\n🔍 NKAT10KGammaChallengeテスト開始...")
        recovery_manager = RobustRecoveryManager()
        challenge_system = NKAT10KGammaChallenge(recovery_manager)
        
        # 最新データ読み込みテスト
        existing_data = challenge_system.load_latest_gamma_data()
        if existing_data:
            print(f"✅ 既存データ読み込み成功")
            if 'results' in existing_data:
                print(f"📊 結果数: {len(existing_data['results']):,}")
                
                # 完了済みγ値抽出テスト
                completed_gammas = challenge_system.extract_completed_gammas(existing_data)
                print(f"📈 完了済みγ値: {len(completed_gammas):,}個")
            else:
                print(f"📊 データキー: {list(existing_data.keys())}")
        else:
            print("⚠️ 既存データが見つかりませんでした")
        
        # γ値生成テスト（少数で）
        gamma_values = challenge_system.generate_gamma_values(count=100, exclude_completed=True)
        print(f"📊 γ値生成テスト: {len(gamma_values)}個")
        
        return True
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False

if __name__ == "__main__":
    print("🚀 最新JSONファイル読み込み機能テスト開始")
    print("=" * 60)
    
    # テスト実行
    test1_result = test_enhanced_large_scale_verification()
    test2_result = test_gamma_challenge_robust()
    
    print("\n" + "=" * 60)
    print("📊 テスト結果:")
    print(f"  Enhanced Large Scale Verification: {'✅ 成功' if test1_result else '❌ 失敗'}")
    print(f"  Gamma Challenge Robust: {'✅ 成功' if test2_result else '❌ 失敗'}")
    
    if test1_result and test2_result:
        print("🎉 全テスト成功！最新JSONファイル活用機能が正常に動作しています")
    else:
        print("⚠️ 一部テストが失敗しました") 