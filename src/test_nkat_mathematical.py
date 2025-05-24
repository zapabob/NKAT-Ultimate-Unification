#!/usr/bin/env python3
"""
🌌 NKAT数学的厳密版テストスクリプト
非可換コルモゴロフ・アーノルド理論の動作確認
"""

import sys
import torch
import numpy as np
from datetime import datetime

# NKAT数学的厳密版をインポート
try:
    import codecs
    with codecs.open('NKAT_DL_Hybrid_Colab.py', 'r', 'utf-8') as f:
        nkat_code = f.read()
    exec(nkat_code)
    print("✅ NKAT数学的厳密版インポート成功")
except Exception as e:
    print(f"❌ インポートエラー: {e}")
    sys.exit(1)

def test_nkat_configuration():
    """NKAT設定のテスト"""
    print("\n🔧 NKAT設定テスト開始...")
    
    try:
        config = ColabNKATConfig()
        print(f"   κ-パラメータ: {config.kappa_parameter:.2e}")
        print(f"   目標スペクトル次元: {config.target_spectral_dim}")
        print(f"   KAN層構成: {config.kan_layers}")
        print(f"   κ-変形B-スプライン: {config.kappa_deformed_splines}")
        print("✅ NKAT設定テスト成功")
        return config
    except Exception as e:
        print(f"❌ 設定テストエラー: {e}")
        return None

def test_kappa_deformed_spline():
    """κ-変形B-スプラインのテスト"""
    print("\n🌌 κ-変形B-スプラインテスト開始...")
    
    try:
        spline = KappaDeformedBSpline(grid_size=16, spline_order=3, kappa_param=1.6e-35)
        
        # テスト入力
        test_input = torch.randn(4, 4) * 0.1
        basis_output = spline.kappa_deformed_basis(test_input)
        
        print(f"   入力形状: {test_input.shape}")
        print(f"   基底関数出力形状: {basis_output.shape}")
        print(f"   基底関数値範囲: [{basis_output.min():.4f}, {basis_output.max():.4f}]")
        print("✅ κ-変形B-スプラインテスト成功")
        return True
    except Exception as e:
        print(f"❌ κ-変形B-スプラインテストエラー: {e}")
        return False

def test_mathematical_kan_layer():
    """数学的KAN層のテスト"""
    print("\n🧠 数学的KAN層テスト開始...")
    
    try:
        config = ColabNKATConfig()
        kan_layer = MathematicalKANLayer(4, 8, config)
        
        # テスト入力（非可換座標）
        test_coords = torch.randn(8, 4) * 0.1
        output = kan_layer(test_coords)
        
        print(f"   入力形状: {test_coords.shape}")
        print(f"   出力形状: {output.shape}")
        print(f"   出力値範囲: [{output.min():.4f}, {output.max():.4f}]")
        print("✅ 数学的KAN層テスト成功")
        return True
    except Exception as e:
        print(f"❌ 数学的KAN層テストエラー: {e}")
        return False

def test_physics_loss():
    """物理情報損失関数のテスト"""
    print("\n🔬 物理情報損失関数テスト開始...")
    
    try:
        config = ColabNKATConfig()
        physics_loss = MathematicalPhysicsLoss(config)
        
        # テストデータ
        batch_size = 4
        model_output = torch.randn(batch_size, 4) * 0.1
        coordinates = torch.randn(batch_size, 4) * 0.1
        coordinates.requires_grad_(True)
        
        # 損失計算
        total_loss, loss_details = physics_loss(model_output, coordinates)
        
        print(f"   総合損失: {total_loss.item():.6f}")
        print(f"   スペクトル次元損失: {loss_details['spectral'].item():.6f}")
        print(f"   ヤコビ損失: {loss_details['jacobi'].item():.6f}")
        print(f"   コンヌ距離損失: {loss_details['connes'].item():.6f}")
        print(f"   θランニング損失: {loss_details['theta_running'].item():.6f}")
        
        if 'spectral_dims' in loss_details:
            print(f"   計算されたスペクトル次元: {loss_details['spectral_dims'].item():.6f}")
        
        print("✅ 物理情報損失関数テスト成功")
        return True
    except Exception as e:
        print(f"❌ 物理情報損失関数テストエラー: {e}")
        return False

def test_experimental_predictions():
    """実験的予測計算のテスト"""
    print("\n🌌 実験的予測計算テスト開始...")
    
    try:
        config = ColabNKATConfig()
        predictor = ExperimentalPredictionCalculator(config)
        
        # テストデータ
        model_output = torch.randn(4, 4) * 0.1
        coordinates = torch.randn(4, 4) * 0.1
        
        # γ線時間遅延テスト
        photon_energy = 1e12  # eV
        distance = 1e25  # m
        time_delay = predictor.compute_gamma_ray_time_delay(
            model_output, coordinates, photon_energy, distance
        )
        print(f"   γ線時間遅延: {time_delay.mean().item():.2e} 秒")
        
        # 真空複屈折テスト
        magnetic_field = 1.0  # T
        prop_length = 1e6  # m
        phase_diff = predictor.compute_vacuum_birefringence(
            model_output, coordinates, magnetic_field, prop_length
        )
        print(f"   真空複屈折位相差: {phase_diff.mean().item():.2e} ラジアン")
        
        print("✅ 実験的予測計算テスト成功")
        return True
    except Exception as e:
        print(f"❌ 実験的予測計算テストエラー: {e}")
        return False

def test_mathematical_nkat_model():
    """統合NKATモデルのテスト"""
    print("\n🌌 統合NKATモデルテスト開始...")
    
    try:
        config = ColabNKATConfig()
        # 軽量設定でテスト
        config.kan_layers = [4, 16, 8, 4]
        config.num_epochs = 2
        
        model = MathematicalNKATModel(config)
        
        # テスト入力
        test_coords = torch.randn(4, 4) * 0.1
        energy_scales = torch.ones(4)
        
        # 順伝播テスト
        output = model(test_coords, energy_scales)
        
        print(f"   場の出力形状: {output['field_output'].shape}")
        print(f"   物理損失: {output['physics_loss'].item():.6f}")
        print(f"   損失詳細キー: {list(output['loss_details'].keys())}")
        
        print("✅ 統合NKATモデルテスト成功")
        return True
    except Exception as e:
        print(f"❌ 統合NKATモデルテストエラー: {e}")
        return False

def main():
    """メインテスト関数"""
    print("🌌" + "="*60)
    print("🌌 NKAT数学的厳密版テスト実行")
    print("🌌" + "="*60)
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PyTorch版: {torch.__version__}")
    print(f"CUDA利用可能: {torch.cuda.is_available()}")
    
    # テスト実行
    tests = [
        ("NKAT設定", test_nkat_configuration),
        ("κ-変形B-スプライン", test_kappa_deformed_spline),
        ("数学的KAN層", test_mathematical_kan_layer),
        ("物理情報損失関数", test_physics_loss),
        ("実験的予測計算", test_experimental_predictions),
        ("統合NKATモデル", test_mathematical_nkat_model),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}テスト中に予期しないエラー: {e}")
            results.append((test_name, False))
    
    # 結果サマリー
    print("\n🌌" + "="*60)
    print("🌌 テスト結果サマリー")
    print("🌌" + "="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 成功" if result else "❌ 失敗"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🌌 総合結果: {passed}/{total} テスト成功")
    
    if passed == total:
        print("🎉 すべてのテストが成功しました！")
        print("🌌 NKAT数学的厳密版は正常に動作しています。")
    else:
        print("⚠️  一部のテストが失敗しました。")
    
    print("\n🌌 NKAT数学的厳密版テスト完了")

if __name__ == "__main__":
    main() 