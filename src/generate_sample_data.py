#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 NKAT Dashboard サンプルデータ生成器
Sample Data Generator for NKAT Streamlit Dashboard Testing

Author: NKAT Research Consortium
Date: 2025-05-26
Version: 1.0 - Sample Data Generator
"""

import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import random

def generate_entanglement_sample_data():
    """量子もつれ解析のサンプルデータ生成"""
    gamma_values = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062]
    
    entanglement_metrics = []
    for gamma in gamma_values:
        metrics = {
            "gamma": gamma,
            "concurrence": random.uniform(0.0, 0.1),
            "entanglement_entropy": random.uniform(0.0, 0.01),
            "negativity": random.uniform(0.0, 0.05),
            "quantum_discord": random.uniform(0.0, 0.02),
            "bell_violation": random.uniform(0.0, 0.01),
            "timestamp": datetime.now().isoformat()
        }
        entanglement_metrics.append(metrics)
    
    sample_data = {
        "timestamp": datetime.now().isoformat(),
        "gamma_values": gamma_values,
        "entanglement_metrics": entanglement_metrics,
        "statistics": {
            "mean_concurrence": np.mean([m["concurrence"] for m in entanglement_metrics]),
            "max_concurrence": np.max([m["concurrence"] for m in entanglement_metrics]),
            "mean_entropy": np.mean([m["entanglement_entropy"] for m in entanglement_metrics]),
            "entanglement_detection_rate": 0.2
        }
    }
    
    # 結果ディレクトリ作成
    results_dir = Path("10k_gamma_results")
    results_dir.mkdir(exist_ok=True)
    
    # サンプルデータ保存
    with open(results_dir / "nkat_v91_entanglement_results.json", 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False, default=str)
    
    print("✅ 量子もつれサンプルデータを生成しました")

def generate_10k_gamma_sample_data():
    """10,000γチャレンジのサンプルデータ生成"""
    # 進行中のチェックポイントデータ
    checkpoint_data = {
        "batch_id": 25,
        "gamma_start_idx": 2500,
        "gamma_end_idx": 2600,
        "completed_gammas": [14.134725 + i * 0.1 for i in range(100)],
        "results": [],
        "timestamp": datetime.now().isoformat(),
        "system_state": {
            "cpu_percent": 75.5,
            "memory_percent": 68.2,
            "gpu_memory_used": 8.5,
            "gpu_memory_total": 10.7,
            "timestamp": datetime.now().isoformat()
        },
        "memory_usage": 68.2,
        "gpu_memory": 8.5,
        "total_progress": 26.0
    }
    
    # 結果データ生成
    results = []
    for i in range(2600):
        gamma = 14.134725 + i * 0.1
        spectral_dim = 1.0 + random.uniform(-0.2, 0.2)
        real_part = spectral_dim / 2
        convergence = abs(real_part - 0.5)
        
        result = {
            "gamma": gamma,
            "spectral_dimension": spectral_dim,
            "real_part": real_part,
            "convergence_to_half": convergence,
            "timestamp": datetime.now().isoformat(),
            "batch_id": i // 100,
            "batch_index": i % 100
        }
        results.append(result)
    
    checkpoint_data["results"] = results
    
    # 最終結果データ（完了版）
    final_results = {
        "timestamp": datetime.now().isoformat(),
        "total_gammas_processed": 2600,
        "valid_results": 2580,
        "execution_time_seconds": 450.5,
        "execution_time_formatted": "0h 7m 30.5s",
        "processing_speed_per_gamma": 0.173,
        "success_rate": 0.992,
        "statistics": {
            "mean_spectral_dimension": 1.025,
            "std_spectral_dimension": 0.156,
            "mean_convergence": 0.089,
            "best_convergence": 0.001,
            "worst_convergence": 0.245
        },
        "results": results
    }
    
    # ディレクトリ作成
    results_dir = Path("10k_gamma_results")
    results_dir.mkdir(exist_ok=True)
    
    checkpoint_dir = Path("10k_gamma_checkpoints_production")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # チェックポイント保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_file = checkpoint_dir / f"checkpoint_batch_25_{timestamp}.json"
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False, default=str)
    
    # 最終結果保存
    final_file = results_dir / f"10k_gamma_final_results_{timestamp}.json"
    with open(final_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False, default=str)
    
    print("✅ 10,000γチャレンジサンプルデータを生成しました")

def generate_historical_data():
    """履歴データの生成"""
    results_dir = Path("10k_gamma_results")
    results_dir.mkdir(exist_ok=True)
    
    # 過去数日分のデータ
    for days_ago in range(1, 8):
        date = datetime.now() - timedelta(days=days_ago)
        timestamp = date.strftime("%Y%m%d_%H%M%S")
        
        # 中間結果データ
        intermediate_data = {
            "timestamp": date.isoformat(),
            "batches_completed": random.randint(10, 50),
            "total_results": random.randint(1000, 5000),
            "results": [
                {
                    "gamma": 14.134725 + i * 0.1,
                    "spectral_dimension": 1.0 + random.uniform(-0.3, 0.3),
                    "batch_id": i // 100
                }
                for i in range(random.randint(100, 500))
            ]
        }
        
        intermediate_file = results_dir / f"intermediate_results_batch_{intermediate_data['batches_completed']}_{timestamp}.json"
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            json.dump(intermediate_data, f, indent=2, ensure_ascii=False, default=str)
    
    print("✅ 履歴データを生成しました")

def main():
    """メイン実行関数"""
    print("🧪 NKAT Dashboard サンプルデータ生成開始")
    print("=" * 60)
    
    try:
        # 量子もつれデータ生成
        generate_entanglement_sample_data()
        
        # 10,000γチャレンジデータ生成
        generate_10k_gamma_sample_data()
        
        # 履歴データ生成
        generate_historical_data()
        
        print("=" * 60)
        print("🎉 全サンプルデータの生成が完了しました！")
        print()
        print("📁 生成されたファイル:")
        print("  📊 10k_gamma_results/nkat_v91_entanglement_results.json")
        print("  📊 10k_gamma_results/10k_gamma_final_results_*.json")
        print("  📊 10k_gamma_results/intermediate_results_*.json")
        print("  💾 10k_gamma_checkpoints_production/checkpoint_batch_*.json")
        print()
        print("🚀 ダッシュボードを起動してデータを確認してください:")
        print("   start_dashboard.bat")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")

if __name__ == "__main__":
    main() 