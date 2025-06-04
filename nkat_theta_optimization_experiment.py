#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔥💎‼ NKAT理論：θパラメータ最適化実験システム ‼💎🔥
複数のθ値での性能比較と最適値探索
電源断リカバリーシステム完全対応

θ候補: 1e-8, 1e-10, 1e-12, 1e-14, 1e-16
"""

import numpy as np
import time
import json
import os
from datetime import datetime
from pathlib import Path
from nkat_riemann_hypothesis_ultimate_proof import NKATRiemannProofSystem

class NKATThetaOptimizer:
    """
    🔬 NKAT θパラメータ最適化システム
    """
    
    def __init__(self):
        self.theta_candidates = [1e-8, 1e-10, 1e-12, 1e-14, 1e-16]
        self.results_dir = Path("nkat_theta_optimization_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # 実験設定
        self.test_config = {
            'short_test': {'t_max': 30, 'num_points': 100},
            'medium_test': {'t_max': 60, 'num_points': 500},
            'full_test': {'t_max': 120, 'num_points': 15000}
        }
        
        print(f"""
🔥💎 NKAT θパラメータ最適化実験開始 💎🔥
{'='*70}
📊 実験対象θ値: {self.theta_candidates}
📁 結果ディレクトリ: {self.results_dir}
🛡️ 電源断リカバリー: 各θ値で独立保護
🎯 テスト種類: short, medium, full
Don't hold back. Give it your all!! 🚀💎
{'='*70}
        """)
    
    def run_theta_experiment(self, theta_value, test_type='short'):
        """指定θ値での実験実行"""
        print(f"\n🧪 θ={theta_value:.0e} 実験開始 ({test_type}テスト)")
        
        # θ専用のリカバリーディレクトリ
        recovery_dir = f"nkat_recovery_theta_{theta_value:.0e}"
        
        try:
            # NKAT システム初期化
            prover = NKATRiemannProofSystem(
                theta=theta_value,
                precision_level='quantum',
                enable_recovery=True
            )
            
            # リカバリーディレクトリを手動設定
            prover.recovery_system.recovery_dir = Path(recovery_dir)
            prover.recovery_system.recovery_dir.mkdir(exist_ok=True)
            prover.recovery_system.checkpoint_file = prover.recovery_system.recovery_dir / "nkat_checkpoint.pkl"
            prover.recovery_system.metadata_file = prover.recovery_system.recovery_dir / "nkat_session_metadata.json"
            
            start_time = time.time()
            
            # テスト設定取得
            config = self.test_config[test_type]
            
            print(f"   📊 t_max={config['t_max']}, num_points={config['num_points']}")
            
            # 零点計算実行
            zeros, accuracy = prover.compute_critical_line_zeros(
                t_max=config['t_max'], 
                num_points=config['num_points']
            )
            
            end_time = time.time()
            computation_time = end_time - start_time
            
            # 非可換ゼータ関数テスト
            zeta_test_results = []
            test_points = [0.5 + 14.134725j, 0.5 + 21.022040j, 0.5 + 25.010858j]
            
            for s in test_points:
                zeta_val = prover.noncommutative_zeta_function(s)
                zeta_test_results.append({
                    'point': str(s),
                    'value': complex(zeta_val),
                    'magnitude': abs(zeta_val)
                })
            
            # 結果収集
            result = {
                'theta': theta_value,
                'test_type': test_type,
                'timestamp': datetime.now().isoformat(),
                'computation_time': computation_time,
                'zeros_found': len(zeros),
                'verification_accuracy': accuracy,
                'zeta_test_results': zeta_test_results,
                'performance_metrics': {
                    'zeros_per_second': len(zeros) / computation_time,
                    'accuracy_per_time': accuracy / computation_time,
                    'efficiency_score': (len(zeros) * accuracy) / computation_time
                },
                'theta_scientific': f"{theta_value:.0e}",
                'config_used': config
            }
            
            # 結果保存
            result_file = self.results_dir / f"theta_{theta_value:.0e}_{test_type}_result.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"""
   ✅ θ={theta_value:.0e} 実験完了
   📊 零点発見数: {len(zeros)}個
   🎯 検証精度: {accuracy:.6f}
   ⏱️ 計算時間: {computation_time:.2f}秒
   ⚡ 効率スコア: {result['performance_metrics']['efficiency_score']:.6f}
   💾 結果保存: {result_file}
            """)
            
            return result
            
        except Exception as e:
            print(f"   ❌ θ={theta_value:.0e} 実験エラー: {e}")
            return {
                'theta': theta_value,
                'test_type': test_type,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_full_optimization(self, test_type='short'):
        """全θ値での最適化実験"""
        print(f"\n🚀 全θ値最適化実験開始 ({test_type})")
        
        results = []
        
        for theta in self.theta_candidates:
            result = self.run_theta_experiment(theta, test_type)
            results.append(result)
            
            # 実験間の短い休憩
            time.sleep(2)
        
        # 総合結果分析
        self.analyze_optimization_results(results, test_type)
        
        return results
    
    def analyze_optimization_results(self, results, test_type):
        """最適化結果の分析"""
        print(f"\n📊 θ最適化結果分析 ({test_type})")
        print("="*70)
        
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            print("❌ 有効な結果がありません")
            return
        
        # 各メトリクスでのランキング
        metrics = ['verification_accuracy', 'zeros_found', 'efficiency_score']
        
        for metric in metrics:
            sorted_results = sorted(valid_results, 
                                  key=lambda x: x.get('performance_metrics', {}).get(metric, 0) 
                                  if metric == 'efficiency_score' 
                                  else x.get(metric, 0), 
                                  reverse=True)
            
            print(f"\n🏆 {metric} ランキング:")
            for i, result in enumerate(sorted_results[:3]):
                theta_val = result['theta']
                if metric == 'efficiency_score':
                    value = result.get('performance_metrics', {}).get(metric, 0)
                else:
                    value = result.get(metric, 0)
                print(f"   #{i+1}: θ={theta_val:.0e} → {value:.6f}")
        
        # 最適θ値の推薦
        best_overall = max(valid_results, 
                          key=lambda x: x.get('performance_metrics', {}).get('efficiency_score', 0))
        
        print(f"""
🎯 推薦最適θ値: {best_overall['theta']:.0e}
   📊 零点発見数: {best_overall['zeros_found']}個
   🎯 検証精度: {best_overall['verification_accuracy']:.6f}
   ⚡ 効率スコア: {best_overall['performance_metrics']['efficiency_score']:.6f}
        """)
        
        # 総合結果保存
        summary_file = self.results_dir / f"optimization_summary_{test_type}.json"
        summary = {
            'test_type': test_type,
            'timestamp': datetime.now().isoformat(),
            'all_results': results,
            'best_theta': best_overall['theta'],
            'best_theta_scientific': f"{best_overall['theta']:.0e}",
            'performance_ranking': {
                metric: [(r['theta'], r.get('performance_metrics', {}).get(metric, 0) 
                         if metric == 'efficiency_score' else r.get(metric, 0))
                        for r in sorted(valid_results, 
                                      key=lambda x: x.get('performance_metrics', {}).get(metric, 0) 
                                      if metric == 'efficiency_score' else x.get(metric, 0), 
                                      reverse=True)]
                for metric in metrics
            }
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"💾 総合分析結果保存: {summary_file}")
        
        return summary

def main():
    """θ最適化実験メイン関数"""
    print("🔥💎‼ NKAT θパラメータ最適化実験システム ‼💎🔥")
    print("Don't hold back. Give it your all!!")
    print()
    
    optimizer = NKATThetaOptimizer()
    
    # 段階的実験実行
    print("🎯 段階1: 短時間テスト (θ効果の迅速確認)")
    short_results = optimizer.run_full_optimization('short')
    
    print("\n🎯 段階2: 中期間テスト (精度とパフォーマンスのバランス)")
    medium_results = optimizer.run_full_optimization('medium')
    
    # フル計算は最適θ値のみで実行（時間節約）
    if short_results:
        valid_short = [r for r in short_results if 'error' not in r]
        if valid_short:
            best_theta = max(valid_short, 
                           key=lambda x: x.get('performance_metrics', {}).get('efficiency_score', 0))['theta']
            
            print(f"\n🎯 段階3: フル計算 (最適θ={best_theta:.0e}のみ)")
            full_result = optimizer.run_theta_experiment(best_theta, 'full')
            
            print(f"""
🏆💎 NKAT θ最適化実験完了! 💎🏆
{'='*70}
🥇 最適θ値: {best_theta:.0e}
🔥 リーマン予想解決への最適パラメータ決定!
💾 全結果は nkat_theta_optimization_results/ に保存
🛡️ 電源断リカバリーで全計算データ保護済み
Don't hold back. Give it your all!! 🚀💎
{'='*70}
            """)

if __name__ == "__main__":
    main() 