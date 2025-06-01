#!/usr/bin/env python3
# run_optuna_tpe.py
"""
🚀 NKAT-Transformer TPE最適化実行スクリプト
LLMスタイルハイパーパラメータ × Theory-Practical Equilibrium

使用方法:
    py -3 run_optuna_tpe.py --n_trials 50 --timeout 3600
"""

import argparse
import json
import os
import time
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.visualization import plot_optimization_history, plot_param_importances

from optim_tpe import objective, enhanced_objective, print_trial_summary

# 英語グラフ設定（文字化け防止）
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def create_study_configuration(study_name: str = None) -> optuna.Study:
    """
    📚 Optuna Study設定作成
    
    Args:
        study_name: スタディ名
        
    Returns:
        設定済みStudy
    """
    if study_name is None:
        study_name = f"nkat_tpe_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # TPEサンプラー：多変量最適化対応
    sampler = TPESampler(
        multivariate=True,           # 多変量最適化
        group=True,                  # グループ最適化
        warn_independent_sampling=True,
        n_startup_trials=10,         # ランダム初期化トライアル数
        n_ei_candidates=24,          # 期待改善候補数
        gamma=lambda x: min(int(0.25 * x), 25),  # 上位%選択関数
        prior_weight=1.0,            # 事前分布重み
        consider_magic_clip=True,    # Magic clip適用
        consider_endpoints=True      # エンドポイント考慮
    )
    
    # MedianPruner：不要なトライアルの早期停止
    pruner = MedianPruner(
        n_startup_trials=8,          # プルーニング開始トライアル数
        n_warmup_steps=3,            # ウォームアップステップ数
        interval_steps=1,            # 評価間隔
        n_min_trials=5               # 最小比較トライアル数
    )
    
    # SQLiteストレージ（永続化）
    storage_path = f"logs/{study_name}.db"
    os.makedirs("logs", exist_ok=True)
    storage_url = f"sqlite:///{storage_path}"
    
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",        # TPE最大化
        sampler=sampler,
        pruner=pruner,
        storage=storage_url,
        load_if_exists=True         # 既存スタディがあれば継続
    )
    
    return study


def create_optimization_visualizations(study: optuna.Study, timestamp: str):
    """
    📊 最適化結果の可視化
    
    Args:
        study: 完了したOptuna Study
        timestamp: タイムスタンプ
    """
    if len(study.trials) == 0:
        print("⚠️ No trials to visualize.")
        return
    
    # 出力ディレクトリ作成
    viz_dir = f"logs/visualizations_{timestamp}"
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. 最適化履歴
    try:
        fig = plot_optimization_history(study)
        fig.update_layout(
            title="🎯 TPE Optimization History",
            xaxis_title="Trial Number",
            yaxis_title="TPE Score",
            font=dict(size=12)
        )
        fig.write_html(f"{viz_dir}/optimization_history.html")
        fig.write_image(f"{viz_dir}/optimization_history.png", width=800, height=600)
        print(f"✅ Optimization history saved to {viz_dir}/")
    except Exception as e:
        print(f"⚠️ Could not create optimization history: {e}")
    
    # 2. パラメータ重要度
    try:
        fig = plot_param_importances(study)
        fig.update_layout(
            title="🎛️ Hyperparameter Importances",
            xaxis_title="Importance Score",
            font=dict(size=12)
        )
        fig.write_html(f"{viz_dir}/param_importances.html")
        fig.write_image(f"{viz_dir}/param_importances.png", width=800, height=600)
        print(f"✅ Parameter importances saved to {viz_dir}/")
    except Exception as e:
        print(f"⚠️ Could not create parameter importances: {e}")
    
    # 3. カスタム分析プロット
    create_custom_analysis_plots(study, viz_dir)


def create_custom_analysis_plots(study: optuna.Study, output_dir: str):
    """
    📈 カスタム分析プロット作成
    
    Args:
        study: Optuna Study
        output_dir: 出力ディレクトリ
    """
    # データフレーム作成
    trials_df = study.trials_dataframe()
    if trials_df.empty:
        return
    
    # TPE vs Accuracy散布図
    plt.figure(figsize=(12, 8))
    
    # サブプロット1: TPE vs Validation Accuracy
    plt.subplot(2, 2, 1)
    val_accs = [trial.user_attrs.get('val_accuracy', 0) for trial in study.trials]
    tpe_scores = [trial.value for trial in study.trials if trial.value is not None]
    
    if len(val_accs) == len(tpe_scores):
        plt.scatter(val_accs, tpe_scores, alpha=0.6, c='blue')
        plt.xlabel('Validation Accuracy')
        plt.ylabel('TPE Score')
        plt.title('TPE vs Validation Accuracy')
        plt.grid(True, alpha=0.3)
    
    # サブプロット2: Temperature vs TPE
    plt.subplot(2, 2, 2)
    temperatures = trials_df.get('params_temperature', [])
    valid_tpe = [trial.value for trial in study.trials if trial.value is not None]
    
    if len(temperatures) >= len(valid_tpe):
        temperatures = temperatures[:len(valid_tpe)]
        plt.scatter(temperatures, valid_tpe, alpha=0.6, c='red')
        plt.xlabel('Temperature')
        plt.ylabel('TPE Score')
        plt.title('Temperature vs TPE Score')
        plt.grid(True, alpha=0.3)
    
    # サブプロット3: NKAT Strength vs TPE
    plt.subplot(2, 2, 3)
    nkat_strengths = trials_df.get('params_nkat_strength', [])
    
    if len(nkat_strengths) >= len(valid_tpe):
        nkat_strengths = nkat_strengths[:len(valid_tpe)]
        plt.scatter(nkat_strengths, valid_tpe, alpha=0.6, c='green')
        plt.xlabel('NKAT Strength')
        plt.ylabel('TPE Score') 
        plt.title('NKAT Strength vs TPE Score')
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
    
    # サブプロット4: Lambda Theory分布
    plt.subplot(2, 2, 4)
    lambda_theories = [trial.user_attrs.get('lambda_theory', 0) for trial in study.trials]
    lambda_theories = [lt for lt in lambda_theories if lt > 0]
    
    if lambda_theories:
        plt.hist(lambda_theories, bins=20, alpha=0.7, color='purple')
        plt.xlabel('Lambda Theory (NKAT Parameters)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Theory Parameters')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/custom_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Custom analysis plots saved to {output_dir}/custom_analysis.png")


def save_results_summary(study: optuna.Study, timestamp: str, args: argparse.Namespace):
    """
    💾 結果サマリー保存
    
    Args:
        study: 完了したOptuna Study
        timestamp: タイムスタンプ
        args: 実行時引数
    """
    if len(study.trials) == 0:
        print("⚠️ No results to save.")
        return
    
    # 結果ディレクトリ作成
    results_dir = f"logs/results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. CSV形式でトライアル結果保存
    trials_df = study.trials_dataframe()
    trials_df.to_csv(f"{results_dir}/optuna_trials.csv", index=False)
    
    # 2. ベストパラメータをJSON保存
    best_params = {
        'best_tpe_score': study.best_value,
        'best_params': study.best_params,
        'best_trial_attrs': study.best_trial.user_attrs,
        'study_summary': {
            'total_trials': len(study.trials),
            'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'failed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
        },
        'execution_info': {
            'n_trials': args.n_trials,
            'timeout': args.timeout,
            'timestamp': timestamp
        }
    }
    
    with open(f"{results_dir}/best_results.json", 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)
    
    # 3. 実行可能コード保存
    generate_best_model_code(study.best_params, results_dir)
    
    print(f"✅ Results summary saved to {results_dir}/")


def generate_best_model_code(best_params: dict, output_dir: str):
    """
    🔧 ベストパラメータでのモデル実行コード生成
    
    Args:
        best_params: ベストハイパーパラメータ
        output_dir: 出力ディレクトリ
    """
    code_template = f'''#!/usr/bin/env python3
"""
🏆 Best NKAT-Transformer Configuration
Auto-generated from Optuna optimization results
TPE-optimized hyperparameters
"""

import torch
from nkat_transformer.model import NKATVisionTransformer

# 🎯 Best hyperparameters from Optuna
BEST_PARAMS = {best_params}

def create_best_model():
    """最適化されたNKATモデルを作成"""
    model = NKATVisionTransformer(
        img_size=28,
        patch_size=4,
        num_classes=10,
        embed_dim=BEST_PARAMS['embed_dim'],
        depth=BEST_PARAMS['depth'],
        num_heads=8,
        # LLM-style parameters
        temperature=BEST_PARAMS['temperature'],
        top_k=BEST_PARAMS['top_k'] if BEST_PARAMS['top_k'] > 0 else None,
        top_p=BEST_PARAMS['top_p'],
        # NKAT theory parameters
        nkat_strength=BEST_PARAMS['nkat_strength'],
        nkat_decay=BEST_PARAMS['nkat_decay'],
        # Regularization
        dropout_embed=BEST_PARAMS['dropout_embed'],
        dropout_attn=BEST_PARAMS['dropout_attn'],
        label_smoothing=BEST_PARAMS['label_smoothing']
    )
    return model

if __name__ == "__main__":
    model = create_best_model()
    print("🚀 Best NKAT model created!")
    print(f"Total parameters: {{sum(p.numel() for p in model.parameters()):,}}")
    
    # モデル情報表示
    info = model.get_model_info()
    print(f"NKAT parameters: {{info['nkat_parameters']:,}}")
    print(f"NKAT ratio: {{info['nkat_ratio']:.6f}}")
'''
    
    with open(f"{output_dir}/best_model.py", 'w', encoding='utf-8') as f:
        f.write(code_template)
    
    print(f"✅ Best model code saved to {output_dir}/best_model.py")


def main():
    """🚀 メイン実行関数"""
    parser = argparse.ArgumentParser(description="NKAT-Transformer TPE Optimization")
    parser.add_argument("--n_trials", type=int, default=30,
                      help="Number of optimization trials")
    parser.add_argument("--timeout", type=int, default=1800,
                      help="Optimization timeout in seconds")
    parser.add_argument("--study_name", type=str, default=None,
                      help="Custom study name")
    parser.add_argument("--enhanced", action="store_true",
                      help="Use enhanced objective function")
    
    args = parser.parse_args()
    
    # タイムスタンプ生成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("🚀 Starting NKAT-Transformer TPE Optimization")
    print("="*60)
    print(f"📊 Trials: {args.n_trials}")
    print(f"⏱️ Timeout: {args.timeout}s")
    print(f"🎯 Objective: {'Enhanced' if args.enhanced else 'Standard'} TPE")
    print("="*60)
    
    # Study作成
    study = create_study_configuration(args.study_name)
    
    # 目的関数選択
    objective_func = enhanced_objective if args.enhanced else objective
    
    # 最適化実行
    start_time = time.time()
    try:
        study.optimize(
            objective_func, 
            n_trials=args.n_trials, 
            timeout=args.timeout,
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⚠️ Optimization interrupted by user")
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return
    
    optimization_time = time.time() - start_time
    
    # 結果表示
    print_trial_summary(study)
    print(f"\n⏱️ Total optimization time: {optimization_time:.2f}s")
    
    # 結果保存・可視化
    save_results_summary(study, timestamp, args)
    create_optimization_visualizations(study, timestamp)
    
    print(f"\n🎉 Optimization completed! Results saved with timestamp: {timestamp}")


if __name__ == "__main__":
    main() 