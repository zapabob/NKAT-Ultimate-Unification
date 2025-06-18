#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Yang-Mills Mass Gap Computation Runner
=====================================

簡単な実行スクリプト - コマンドライン引数対応
RTX3080 CUDA最適化版

Usage:
    py -3 run_yang_mills.py --gauge 2 --lattice 32 --modes 100
    py -3 run_yang_mills.py --gauge 3 --lattice 64 --modes 200 --alpha 0.3
"""

import argparse
import sys
import os
from pathlib import Path

# メインモジュールをインポート
try:
    from yang_mills_mass_gap_cuda import YangMillsMassGapCUDA, main as full_main
except ImportError:
    print("❌ Error: yang_mills_mass_gap_cuda.py not found!")
    print("Please ensure the main module is in the same directory.")
    sys.exit(1)

def parse_arguments():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(
        description="Yang-Mills Mass Gap Computation via URT + NC-KART",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 基本パラメータ
    parser.add_argument('--gauge', '-N', type=int, default=2,
                       help='Gauge group SU(N)')
    parser.add_argument('--lattice', '-L', type=int, default=32,
                       help='Lattice size (L^4)')
    parser.add_argument('--modes', '-K', type=int, default=100,
                       help='Maximum URT modes (K_max)')
    parser.add_argument('--alpha', '-a', type=float, default=0.5,
                       help='Exponential decay parameter')
    parser.add_argument('--iterations', '-i', type=int, default=30,
                       help='Maximum Dyson-Schwinger iterations')
    
    # 計算オプション
    parser.add_argument('--device', '-d', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Computation device')
    parser.add_argument('--no-checkpoints', action='store_true',
                       help='Disable checkpoint system')
    parser.add_argument('--session-id', type=str, default=None,
                       help='Custom session ID')
    
    # 出力オプション
    parser.add_argument('--output-dir', '-o', type=str, default='.',
                       help='Output directory for results')
    parser.add_argument('--plot', action='store_true',
                       help='Generate convergence plots')
    parser.add_argument('--continuity-test', action='store_true',
                       help='Run theta continuity test')
    
    # 実行モード
    parser.add_argument('--quick', action='store_true',
                       help='Quick test run (small parameters)')
    parser.add_argument('--full', action='store_true',
                       help='Full computation (both SU(2) and SU(3))')
    
    return parser.parse_args()

def quick_test():
    """クイックテスト実行"""
    print("🚀 Quick Test Mode - Yang-Mills Mass Gap")
    print("=" * 50)
    
    # 小さなパラメータでテスト
    ym = YangMillsMassGapCUDA(
        N_gauge=2, 
        lattice_size=16, 
        device='cuda',
        enable_checkpoints=False
    )
    
    results = ym.compute_mass_gap(K_max=20, alpha=0.5, max_iter=10)
    
    print("\n✅ Quick test completed!")
    print(f"Mass Gap: {results['results']['mass_gap']:.4f} GeV")
    print(f"Computation time: {results['computation_time']:.2f} seconds")
    
    return results

def single_computation(args):
    """単一計算の実行"""
    print(f"🔬 SU({args.gauge}) Yang-Mills Mass Gap Computation")
    print("=" * 60)
    
    # 出力ディレクトリ作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 計算実行
    ym = YangMillsMassGapCUDA(
        N_gauge=args.gauge,
        lattice_size=args.lattice,
        device=args.device,
        session_id=args.session_id,
        enable_checkpoints=not args.no_checkpoints
    )
    
    results = ym.compute_mass_gap(
        K_max=args.modes,
        alpha=args.alpha,
        max_iter=args.iterations
    )
    
    # 結果保存
    output_file = output_dir / f"yang_mills_su{args.gauge}_{ym.session_id}.json"
    ym.save_results(results, str(output_file))
    
    # プロット生成
    if args.plot:
        plot_file = output_dir / f"convergence_su{args.gauge}_{ym.session_id}.png"
        ym.plot_convergence(results['convergence_history'], str(plot_file))
    
    # 連続性テスト
    if args.continuity_test:
        print("\n🧪 Running θ → 0 continuity test...")
        thetas, masses = ym.theta_continuity_test()
        
        # 連続性結果保存
        continuity_file = output_dir / f"continuity_test_{ym.session_id}.json"
        continuity_data = {
            'theta_values': thetas,
            'mass_gaps': masses,
            'session_id': ym.session_id
        }
        
        import json
        with open(continuity_file, 'w') as f:
            json.dump(continuity_data, f, indent=2)
        
        print(f"Continuity test results saved: {continuity_file}")
    
    return results

def full_computation(args):
    """完全計算（SU(2) + SU(3)）"""
    print("🚀 Full Yang-Mills Mass Gap Computation")
    print("SU(2) and SU(3) comparison")
    print("=" * 60)
    
    results = {}
    
    # SU(2) 計算
    print("\n🔬 SU(2) Computation")
    args_su2 = argparse.Namespace(**vars(args))
    args_su2.gauge = 2
    results['su2'] = single_computation(args_su2)
    
    # SU(3) 計算
    print("\n🔬 SU(3) Computation")
    args_su3 = argparse.Namespace(**vars(args))
    args_su3.gauge = 3
    results['su3'] = single_computation(args_su3)
    
    # 比較結果表示
    print("\n📊 COMPARISON RESULTS")
    print("=" * 40)
    mg_su2 = results['su2']['results']['mass_gap']
    mg_su3 = results['su3']['results']['mass_gap']
    scaling = mg_su3 / mg_su2
    expected = (3/2)**0.5
    
    print(f"SU(2) Mass Gap: {mg_su2:.4f} GeV")
    print(f"SU(3) Mass Gap: {mg_su3:.4f} GeV")
    print(f"Scaling factor: {scaling:.4f}")
    print(f"Expected √(3/2): {expected:.4f}")
    print(f"Agreement: {abs(scaling - expected) < 0.1}")
    
    # 比較結果保存
    output_dir = Path(args.output_dir)
    comparison_file = output_dir / f"comparison_results_{results['su2']['session_id']}.json"
    
    comparison_data = {
        'su2_mass_gap': mg_su2,
        'su3_mass_gap': mg_su3,
        'scaling_factor': scaling,
        'expected_scaling': expected,
        'agreement': abs(scaling - expected) < 0.1,
        'parameters': vars(args)
    }
    
    import json
    with open(comparison_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\nComparison results saved: {comparison_file}")
    
    return results

def main():
    """メイン実行関数"""
    args = parse_arguments()
    
    print("🎯 Yang-Mills Mass Gap Computation")
    print(f"Parameters: SU({args.gauge}), L={args.lattice}, K={args.modes}")
    print(f"Device: {args.device.upper()}")
    print()
    
    try:
        if args.quick:
            # クイックテスト
            results = quick_test()
        elif args.full:
            # 完全計算
            results = full_computation(args)
        else:
            # 単一計算
            results = single_computation(args)
        
        print("\n✅ All computations completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Computation interrupted by user")
        print("💾 Emergency save should have been triggered")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 