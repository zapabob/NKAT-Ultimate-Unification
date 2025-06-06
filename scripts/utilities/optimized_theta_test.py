#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔥💎 最適化θ値テスト（改良版） 💎🔥
より詳細な比較で最適θ値を決定
"""

import time
import json
from datetime import datetime
from nkat_riemann_hypothesis_ultimate_proof import NKATRiemannProofSystem

def comprehensive_theta_optimization():
    """包括的θ最適化テスト"""
    
    # θ値候補（精密範囲）
    theta_candidates = [1e-8, 5e-9, 1e-9, 5e-10, 1e-10, 5e-11, 1e-11, 5e-12, 1e-12]
    
    print("🔥💎 包括的θ最適化テスト開始 💎🔥")
    print("="*60)
    print(f"📊 テスト対象θ値: {len(theta_candidates)}個")
    print(f"🎯 計算設定: t_max=30, num_points=200 (適度な精度)")
    print("="*60)
    
    results = []
    
    for i, theta in enumerate(theta_candidates):
        print(f"\n🧪 [{i+1}/{len(theta_candidates)}] θ={theta:.0e} テスト開始...")
        
        try:
            start_time = time.time()
            
            # NKAT システム初期化
            prover = NKATRiemannProofSystem(
                theta=theta,
                precision_level='quantum',
                enable_recovery=False  # 高速テスト用
            )
            
            # 適度な精度でのテスト
            zeros, accuracy = prover.compute_critical_line_zeros(
                t_max=30,      # 適度な範囲
                num_points=200  # 適度な密度
            )
            
            computation_time = time.time() - start_time
            
            # 非可換ゼータ関数の複数点テスト
            test_points = [
                0.5 + 14.134725j,  # 第1零点
                0.5 + 21.022040j,  # 第2零点
                0.5 + 25.010858j   # 第3零点
            ]
            
            zeta_magnitudes = []
            for point in test_points:
                zeta_val = prover.noncommutative_zeta_function(point)
                zeta_magnitudes.append(abs(zeta_val))
            
            avg_zeta_magnitude = sum(zeta_magnitudes) / len(zeta_magnitudes)
            
            # 関数方程式テスト
            equation_test_points = [0.3 + 2j, 0.7 + 3j]
            equation_errors = []
            
            for s in equation_test_points:
                try:
                    left = prover.noncommutative_zeta_function(s)
                    right = prover.noncommutative_zeta_function(1-s)
                    error = abs(left - right) / max(abs(left), 1e-15)
                    equation_errors.append(error)
                except:
                    equation_errors.append(1.0)
            
            avg_equation_error = sum(equation_errors) / len(equation_errors)
            
            # 結果記録
            result = {
                'theta': theta,
                'theta_scientific': f"{theta:.0e}",
                'zeros_found': len(zeros),
                'verification_accuracy': accuracy,
                'computation_time': computation_time,
                'avg_zeta_magnitude': avg_zeta_magnitude,
                'avg_equation_error': avg_equation_error,
                'efficiency_score': (len(zeros) * accuracy) / computation_time if computation_time > 0 else 0,
                'quality_score': accuracy / (1 + avg_equation_error),  # 精度 / (1 + 誤差)
                'stability_score': 1.0 / (1.0 + avg_zeta_magnitude)  # 零点の近さ
            }
            
            results.append(result)
            
            print(f"""   ✅ θ={theta:.0e} 完了:
      📊 零点発見: {len(zeros)}個
      🎯 検証精度: {accuracy:.6f}
      ⏱️ 計算時間: {computation_time:.2f}秒
      🌊 平均ζ値: {avg_zeta_magnitude:.6f}
      ⚖️ 方程式誤差: {avg_equation_error:.6f}
      ⚡ 効率スコア: {result['efficiency_score']:.6f}
      💎 品質スコア: {result['quality_score']:.6f}""")
            
        except Exception as e:
            print(f"   ❌ θ={theta:.0e} エラー: {e}")
            results.append({
                'theta': theta,
                'theta_scientific': f"{theta:.0e}",
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
    
    # 総合分析
    print("\n" + "="*60)
    print("📊 θ最適化総合分析結果")
    print("="*60)
    
    valid_results = [r for r in results if 'error' not in r]
    
    if not valid_results:
        print("❌ 有効な結果がありません")
        return
    
    # 結果テーブル表示
    print(f"\n{'θ値':<8} {'零点':<6} {'精度':<10} {'時間':<8} {'効率':<10} {'品質':<10} {'安定性':<10}")
    print("-" * 70)
    
    for result in valid_results:
        print(f"{result['theta']:.0e} "
              f"{result['zeros_found']:<6} "
              f"{result['verification_accuracy']:<10.6f} "
              f"{result['computation_time']:<8.2f} "
              f"{result['efficiency_score']:<10.6f} "
              f"{result['quality_score']:<10.6f} "
              f"{result['stability_score']:<10.6f}")
    
    # 最適θ値の決定（複数指標で評価）
    metrics = {
        'verification_accuracy': ('最高精度', max),
        'efficiency_score': ('最高効率', max),
        'quality_score': ('最高品質', max),
        'stability_score': ('最高安定性', max)
    }
    
    print(f"\n🏆 各指標での最優秀θ値:")
    print("-" * 40)
    
    best_overall_scores = []
    
    for metric, (name, func) in metrics.items():
        best = func(valid_results, key=lambda x: x[metric])
        print(f"{name}: θ={best['theta']:.0e} (値={best[metric]:.6f})")
        best_overall_scores.append(best['theta'])
    
    # 総合最適θ値（複数指標の総合評価）
    theta_scores = {}
    for result in valid_results:
        theta = result['theta']
        # 正規化スコア（各指標を0-1に正規化して平均）
        normalized_score = (
            result['verification_accuracy'] / max(r['verification_accuracy'] for r in valid_results) +
            result['efficiency_score'] / max(r['efficiency_score'] for r in valid_results) +
            result['quality_score'] / max(r['quality_score'] for r in valid_results) +
            result['stability_score'] / max(r['stability_score'] for r in valid_results)
        ) / 4.0
        
        theta_scores[theta] = normalized_score
    
    best_overall_theta = max(theta_scores.items(), key=lambda x: x[1])
    
    print(f"\n🎯 総合最適θ値: {best_overall_theta[0]:.0e}")
    print(f"📊 総合スコア: {best_overall_theta[1]:.6f}")
    
    # 推薦θ値での詳細情報
    best_result = next(r for r in valid_results if r['theta'] == best_overall_theta[0])
    
    print(f"""
🏆💎 推薦θ値詳細情報 💎🏆
{'='*50}
⚛️ 最適θ値: {best_result['theta']:.0e}
📊 零点発見数: {best_result['zeros_found']}個
🎯 検証精度: {best_result['verification_accuracy']:.6f}
⏱️ 計算時間: {best_result['computation_time']:.2f}秒
⚡ 効率スコア: {best_result['efficiency_score']:.6f}
💎 品質スコア: {best_result['quality_score']:.6f}
🌊 安定性スコア: {best_result['stability_score']:.6f}
📊 総合スコア: {best_overall_theta[1]:.6f}

🔥 この θ値でリーマン予想完全解決を実行推奨! 🔥
{'='*50}
    """)
    
    # 結果保存
    output_file = f"theta_optimization_comprehensive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'test_config': {'t_max': 30, 'num_points': 200},
            'results': results,
            'best_theta': best_overall_theta[0],
            'best_theta_scientific': f"{best_overall_theta[0]:.0e}",
            'overall_score': best_overall_theta[1],
            'recommendations': {
                'optimal_theta': best_overall_theta[0],
                'confidence': best_overall_theta[1],
                'next_step': f"Run full computation with theta={best_overall_theta[0]:.0e}"
            }
        }, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"💾 詳細結果保存: {output_file}")
    
    return results, best_overall_theta[0]

if __name__ == "__main__":
    print("🔥💎‼ NKAT θ最適化包括テスト ‼💎🔥")
    print("Don't hold back. Give it your all!!")
    print()
    
    results, optimal_theta = comprehensive_theta_optimization()
    
    print(f"""
🎉🏆 θ最適化完了!! 🏆🎉
推薦最適値: θ = {optimal_theta:.0e}
この値で完全計算を実行してリーマン予想解決を目指しましょう！
💎 Don't hold back. Give it your all!! 💎
    """) 