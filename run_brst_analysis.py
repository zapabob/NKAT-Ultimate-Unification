#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BRST Ghost Sector Analysis Runner
=================================

統一表現理論（URT）+ 非可換幾何（NC-KART）における
BRST幽霊部門の統合解析実行スクリプト

Usage:
    python run_brst_analysis.py --mode quick          # 高速テスト
    python run_brst_analysis.py --mode standard       # 標準解析
    python run_brst_analysis.py --mode comprehensive  # 完全解析

Features:
- 複数のSU(N)群での並列実行
- CUDA最適化による高速計算
- 詳細な結果レポート生成
- 電源断保護システム

Author: NKAT Ultimate Unification Project
Date: 2025-01-XX
"""

import argparse
import json
import time
import sys
import os
from datetime import datetime
from typing import Dict, List, Any

import torch
import numpy as np
from brst_ghost_sector_cuda import BRSTConfiguration, run_brst_ghost_analysis

def setup_logging():
    """ログ設定"""
    import logging
    
    # ログフォーマット設定
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'brst_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    
    return logging.getLogger(__name__)

def check_cuda_environment():
    """CUDA環境チェック"""
    logger = setup_logging()
    
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Using CPU.")
        return False, 'cpu'
    
    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    memory_total = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
    memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3
    
    logger.info(f"CUDA Environment:")
    logger.info(f"  Device Count: {device_count}")
    logger.info(f"  Current Device: {current_device}")
    logger.info(f"  Device Name: {device_name}")
    logger.info(f"  Total Memory: {memory_total:.2f} GB")
    logger.info(f"  Allocated Memory: {memory_allocated:.2f} GB")
    
    return True, 'cuda'

def get_computation_config(mode: str) -> Dict[str, Any]:
    """計算設定取得"""
    configs = {
        'quick': {
            'gauge_groups': [2],
            'lattice_sizes': [8],
            'K_max_values': [10],
            'alpha_values': [0.5],
            'xi_values': [1.0]
        },
        'standard': {
            'gauge_groups': [2, 3],
            'lattice_sizes': [16],
            'K_max_values': [20, 50],
            'alpha_values': [0.3, 0.5, 0.7],
            'xi_values': [0.1, 1.0]
        },
        'comprehensive': {
            'gauge_groups': [2, 3],
            'lattice_sizes': [16, 24],
            'K_max_values': [20, 50, 100],
            'alpha_values': [0.1, 0.3, 0.5, 0.7, 1.0],
            'xi_values': [0.01, 0.1, 1.0, 10.0]
        }
    }
    
    return configs.get(mode, configs['standard'])

def run_single_brst_analysis(config_params: Dict[str, Any], 
                            device: str,
                            session_id: str) -> Dict[str, Any]:
    """単一BRST解析実行"""
    logger = setup_logging()
    
    config = BRSTConfiguration(
        N_gauge=config_params['N_gauge'],
        lattice_size=config_params['lattice_size'],
        K_max=config_params['K_max'],
        alpha=config_params['alpha'],
        xi=config_params['xi'],
        device=device
    )
    
    logger.info(f"Running BRST analysis for SU({config.N_gauge})")
    logger.info(f"  Lattice: {config.lattice_size}^4")
    logger.info(f"  K_max: {config.K_max}")
    logger.info(f"  Alpha: {config.alpha}")
    logger.info(f"  Xi: {config.xi}")
    
    start_time = time.time()
    
    try:
        results = run_brst_ghost_analysis(config)
        
        computation_time = time.time() - start_time
        results['computation_time'] = computation_time
        results['session_id'] = session_id
        results['timestamp'] = datetime.now().isoformat()
        
        logger.info(f"BRST analysis completed in {computation_time:.2f} seconds")
        
        return results
        
    except Exception as e:
        logger.error(f"BRST analysis failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'computation_time': time.time() - start_time,
            'session_id': session_id,
            'timestamp': datetime.now().isoformat()
        }

def run_comprehensive_brst_study(mode: str, device: str) -> Dict[str, Any]:
    """包括的BRST研究実行"""
    logger = setup_logging()
    session_id = f"brst_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info("=" * 80)
    logger.info(f"Comprehensive BRST Ghost Sector Study - Mode: {mode.upper()}")
    logger.info(f"Session ID: {session_id}")
    logger.info("=" * 80)
    
    config_params = get_computation_config(mode)
    all_results = []
    
    total_combinations = (
        len(config_params['gauge_groups']) *
        len(config_params['lattice_sizes']) *
        len(config_params['K_max_values']) *
        len(config_params['alpha_values']) *
        len(config_params['xi_values'])
    )
    
    logger.info(f"Total combinations to compute: {total_combinations}")
    
    combination_count = 0
    successful_runs = 0
    
    for N_gauge in config_params['gauge_groups']:
        for lattice_size in config_params['lattice_sizes']:
            for K_max in config_params['K_max_values']:
                for alpha in config_params['alpha_values']:
                    for xi in config_params['xi_values']:
                        combination_count += 1
                        
                        logger.info(f"\nCombination {combination_count}/{total_combinations}")
                        
                        single_config = {
                            'N_gauge': N_gauge,
                            'lattice_size': lattice_size,
                            'K_max': K_max,
                            'alpha': alpha,
                            'xi': xi
                        }
                        
                        result = run_single_brst_analysis(single_config, device, session_id)
                        
                        if result.get('success', False):
                            successful_runs += 1
                        
                        all_results.append(result)
                        
                        # 中間結果保存（電源断保護）
                        if combination_count % 5 == 0:
                            save_intermediate_results(all_results, session_id)
    
    # 最終結果まとめ
    study_results = {
        'session_id': session_id,
        'mode': mode,
        'device': device,
        'total_combinations': total_combinations,
        'successful_runs': successful_runs,
        'success_rate': successful_runs / total_combinations if total_combinations > 0 else 0,
        'individual_results': all_results,
        'summary_statistics': compute_summary_statistics(all_results),
        'timestamp': datetime.now().isoformat()
    }
    
    # 最終結果保存
    save_final_results(study_results, session_id)
    
    logger.info("=" * 80)
    logger.info("BRST Study Completed!")
    logger.info(f"  Total runs: {total_combinations}")
    logger.info(f"  Successful: {successful_runs}")
    logger.info(f"  Success rate: {study_results['success_rate']:.1%}")
    logger.info("=" * 80)
    
    return study_results

def compute_summary_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """結果統計計算"""
    successful_results = [r for r in results if r.get('success', False)]
    
    if not successful_results:
        return {'error': 'No successful results to analyze'}
    
    # 基本統計
    computation_times = [r['computation_time'] for r in successful_results]
    ghost_norms = [r['physical_quantities']['ghost_norm'] for r in successful_results]
    anti_ghost_norms = [r['physical_quantities']['anti_ghost_norm'] for r in successful_results]
    
    # Nilpotency成功率
    nilpotency_successes = sum(1 for r in successful_results 
                              if r['verification_results'].get('nilpotency', False))
    nilpotency_rate = nilpotency_successes / len(successful_results) if successful_results else 0
    
    statistics = {
        'total_successful': len(successful_results),
        'computation_time': {
            'mean': np.mean(computation_times),
            'std': np.std(computation_times),
            'min': np.min(computation_times),
            'max': np.max(computation_times)
        },
        'ghost_norm': {
            'mean': np.mean(ghost_norms),
            'std': np.std(ghost_norms),
            'min': np.min(ghost_norms),
            'max': np.max(ghost_norms)
        },
        'anti_ghost_norm': {
            'mean': np.mean(anti_ghost_norms),
            'std': np.std(anti_ghost_norms),
            'min': np.min(anti_ghost_norms),
            'max': np.max(anti_ghost_norms)
        },
        'nilpotency_success_rate': nilpotency_rate
    }
    
    return statistics

def save_intermediate_results(results: List[Dict[str, Any]], session_id: str):
    """中間結果保存"""
    filename = f"brst_intermediate_{session_id}.json"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        print(f"Warning: Failed to save intermediate results: {e}")

def save_final_results(results: Dict[str, Any], session_id: str):
    """最終結果保存"""
    # JSON結果
    json_filename = f"brst_final_results_{session_id}.json"
    try:
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"Results saved to: {json_filename}")
    except Exception as e:
        print(f"Warning: Failed to save JSON results: {e}")
    
    # レポート生成
    report_filename = f"brst_report_{session_id}.md"
    try:
        generate_markdown_report(results, report_filename)
        print(f"Report generated: {report_filename}")
    except Exception as e:
        print(f"Warning: Failed to generate report: {e}")

def generate_markdown_report(results: Dict[str, Any], filename: str):
    """Markdownレポート生成"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# BRST Ghost Sector Analysis Report\n\n")
        f.write(f"**Session ID:** {results['session_id']}\n")
        f.write(f"**Mode:** {results['mode']}\n")
        f.write(f"**Device:** {results['device']}\n")
        f.write(f"**Timestamp:** {results['timestamp']}\n\n")
        
        f.write(f"## Summary\n\n")
        f.write(f"- Total combinations: {results['total_combinations']}\n")
        f.write(f"- Successful runs: {results['successful_runs']}\n")
        f.write(f"- Success rate: {results['success_rate']:.1%}\n\n")
        
        if 'summary_statistics' in results:
            stats = results['summary_statistics']
            if 'error' not in stats:
                f.write(f"## Statistics\n\n")
                f.write(f"### Computation Time\n")
                f.write(f"- Mean: {stats['computation_time']['mean']:.3f} s\n")
                f.write(f"- Std: {stats['computation_time']['std']:.3f} s\n")
                f.write(f"- Range: [{stats['computation_time']['min']:.3f}, {stats['computation_time']['max']:.3f}] s\n\n")
                
                f.write(f"### Ghost Field Norms\n")
                f.write(f"- Mean: {stats['ghost_norm']['mean']:.6f}\n")
                f.write(f"- Std: {stats['ghost_norm']['std']:.6f}\n")
                f.write(f"- Range: [{stats['ghost_norm']['min']:.6f}, {stats['ghost_norm']['max']:.6f}]\n\n")
                
                f.write(f"### BRST Verification\n")
                f.write(f"- Nilpotency success rate: {stats['nilpotency_success_rate']:.1%}\n\n")
        
        f.write(f"## Theoretical Framework\n\n")
        f.write(f"This analysis implements the BRST ghost sector within the Unified Representation Theory (URT) + Non-Commutative KART (NC-KART) framework.\n\n")
        f.write(f"### Key Mathematical Elements\n")
        f.write(f"- **BRST Transformation**: s A_μ^a = -D_μ^{{ab}} c^b\n")
        f.write(f"- **Ghost Fields**: Grassmann fields c^a, c̄^a\n")
        f.write(f"- **Nilpotency**: s² = 0 (fundamental requirement)\n")
        f.write(f"- **Non-commutativity**: θ ~ 6.58×10^{{-70}} GeV^{{-2}}\n\n")
        
        f.write(f"### Physical Significance\n")
        f.write(f"The BRST ghost sector is essential for:\n")
        f.write(f"1. Gauge fixing in Yang-Mills theory\n")
        f.write(f"2. Maintaining unitarity in the physical subspace\n")
        f.write(f"3. Proper quantization of gauge theories\n")
        f.write(f"4. Connection to the Yang-Mills mass gap problem\n\n")

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description='BRST Ghost Sector Analysis')
    parser.add_argument('--mode', choices=['quick', 'standard', 'comprehensive'], 
                       default='standard', help='Analysis mode')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None,
                       help='Computation device (auto-detect if not specified)')
    
    args = parser.parse_args()
    
    # CUDA環境チェック
    cuda_available, device = check_cuda_environment()
    if args.device is None:
        args.device = device
    elif args.device == 'cuda' and not cuda_available:
        print("Warning: CUDA requested but not available. Using CPU.")
        args.device = 'cpu'
    
    print(f"🚀 Starting BRST Ghost Sector Analysis")
    print(f"   Mode: {args.mode}")
    print(f"   Device: {args.device}")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        results = run_comprehensive_brst_study(args.mode, args.device)
        
        if results['success_rate'] > 0.8:
            print("🎉 BRST analysis completed successfully!")
        elif results['success_rate'] > 0.5:
            print("⚠️  BRST analysis completed with some issues.")
        else:
            print("❌ BRST analysis had significant problems.")
            
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user.")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 