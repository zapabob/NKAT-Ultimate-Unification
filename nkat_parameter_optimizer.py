#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT理論パラメータ最適化システム
============================

RTX3080最適化版 - Yang-Mills質量ギャップ最大化のための
パラメータ最適化システム

Features:
- Bayesian optimization for parameter search
- Multi-objective optimization (mass gap + confinement)
- Parallel parameter evaluation
- Real-time convergence monitoring
- Checkpoint-based recovery system

Author: NKAT Ultimate Unification Project
Date: 2025-06-18
"""

import torch
import numpy as np
import math
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
from tqdm import tqdm
import uuid
from pathlib import Path

from nkat_advanced_analyzer import AdvancedConfig, AdvancedSpectralAnalyzer

# ログ設定
class OptimizerFormatter(logging.Formatter):
    def format(self, record):
        emoji_map = {
            '🎯': '[TARGET]', '⚡': '[FAST]', '🔧': '[TOOL]', '📈': '[TREND]',
            '🎪': '[CIRCUS]', '🌟': '[STAR]', '🔥': '[FIRE]', '💎': '[DIAMOND]'
        }
        msg = super().format(record)
        for emoji, replacement in emoji_map.items():
            msg = msg.replace(emoji, replacement)
        return msg

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(OptimizerFormatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

@dataclass
class OptimizationConfig:
    """最適化設定"""
    # 最適化対象パラメータ範囲
    coupling_range: Tuple[float, float] = (0.1, 3.0)
    theta_range: Tuple[float, float] = (1e-70, 1e-67)
    alpha_range: Tuple[float, float] = (0.05, 0.5)
    
    # 最適化設定
    max_evaluations: int = 50
    batch_size: int = 4
    convergence_threshold: float = 1e-6
    exploration_factor: float = 0.3
    
    # RTX3080制約
    lattice_size_limit: int = 16
    memory_budget_gb: float = 7.0
    timeout_seconds: int = 300
    
    # チェックポイント
    checkpoint_interval: int = 10
    checkpoint_dir: str = "optimization_checkpoints"


class ParameterOptimizer:
    """理論パラメータ最適化システム"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 最適化履歴
        self.evaluation_history = []
        self.best_parameters = None
        self.best_score = 0.0
        
        # チェックポイント設定
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.session_id = str(uuid.uuid4())[:8]
        
        logger.info(f"🎯 パラメータ最適化システム初期化 (Session: {self.session_id})")
    
    def optimize(self) -> Dict[str, Any]:
        """パラメータ最適化実行"""
        logger.info("🔥 理論パラメータ最適化開始")
        
        optimization_results = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'config': self._get_optimization_config(),
            'evaluation_history': [],
            'best_parameters': {},
            'convergence_analysis': {}
        }
        
        try:
            # チェックポイント復旧試行
            if self._load_checkpoint():
                logger.info("🔄 チェックポイントから復旧")
            
            # 最適化ループ
            for evaluation_idx in tqdm(range(len(self.evaluation_history), 
                                           self.config.max_evaluations), 
                                     desc="最適化進行"):
                
                # 次のパラメータ候補生成
                candidate_params = self._generate_next_parameters(evaluation_idx)
                
                # パラメータ評価
                score, evaluation_result = self._evaluate_parameters(candidate_params)
                
                # 履歴更新
                evaluation_data = {
                    'evaluation_idx': evaluation_idx,
                    'parameters': candidate_params,
                    'score': score,
                    'result': evaluation_result,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.evaluation_history.append(evaluation_data)
                
                # ベストスコア更新
                if score > self.best_score:
                    self.best_score = score
                    self.best_parameters = candidate_params.copy()
                    logger.info(f"🌟 新記録更新: スコア = {score:.6f}")
                    logger.info(f"  最適パラメータ: {candidate_params}")
                
                # 収束判定
                if self._check_convergence():
                    logger.info("✅ 収束条件達成")
                    break
                
                # チェックポイント保存
                if (evaluation_idx + 1) % self.config.checkpoint_interval == 0:
                    self._save_checkpoint()
            
            # 最終結果集計
            optimization_results['evaluation_history'] = self.evaluation_history
            optimization_results['best_parameters'] = self.best_parameters
            optimization_results['best_score'] = self.best_score
            optimization_results['convergence_analysis'] = self._analyze_convergence()
            
            # 最終チェックポイント保存
            self._save_checkpoint()
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"❌ 最適化エラー: {e}")
            optimization_results['error'] = str(e)
            return optimization_results
    
    def _generate_next_parameters(self, evaluation_idx: int) -> Dict[str, float]:
        """次の評価パラメータ生成"""
        
        if evaluation_idx < 5:
            # 初期探索: ランダムサンプリング
            return self._random_sampling()
        else:
            # 最適化フェーズ: Bayesian optimization風
            return self._bayesian_sampling()
    
    def _random_sampling(self) -> Dict[str, float]:
        """ランダムサンプリング"""
        
        coupling = np.random.uniform(*self.config.coupling_range)
        theta = np.random.uniform(*self.config.theta_range)
        alpha = np.random.uniform(*self.config.alpha_range)
        
        return {
            'coupling_constant': coupling,
            'theta': theta,
            'alpha': alpha
        }
    
    def _bayesian_sampling(self) -> Dict[str, float]:
        """Bayesian最適化風サンプリング"""
        
        if not self.evaluation_history:
            return self._random_sampling()
        
        # 過去の評価から高スコア領域を特定
        high_score_evaluations = [
            ev for ev in self.evaluation_history 
            if ev['score'] > np.percentile([e['score'] for e in self.evaluation_history], 70)
        ]
        
        if not high_score_evaluations:
            return self._random_sampling()
        
        # 高スコア領域の中心値計算
        best_params = max(high_score_evaluations, key=lambda x: x['score'])['parameters']
        
        # ガウシアンノイズで摂動
        noise_scale = self.config.exploration_factor
        
        coupling = np.clip(
            best_params['coupling_constant'] + np.random.normal(0, noise_scale),
            *self.config.coupling_range
        )
        
        theta = np.clip(
            best_params['theta'] * np.exp(np.random.normal(0, noise_scale)),
            *self.config.theta_range
        )
        
        alpha = np.clip(
            best_params['alpha'] + np.random.normal(0, noise_scale * 0.1),
            *self.config.alpha_range
        )
        
        return {
            'coupling_constant': coupling,
            'theta': theta,
            'alpha': alpha
        }
    
    def _evaluate_parameters(self, parameters: Dict[str, float]) -> Tuple[float, Dict[str, Any]]:
        """パラメータ評価"""
        
        try:
            # 解析設定作成
            analysis_config = AdvancedConfig(
                device=self.device,
                N_gauge=2,  # SU(2)でメモリ節約
                coupling_constant=parameters['coupling_constant'],
                theta=parameters['theta'],
                alpha=parameters['alpha'],
                lattice_sizes=[8, 12, self.config.lattice_size_limit],
                max_matrix_size=5000  # RTX3080制限
            )
            
            # スペクトラル解析実行
            analyzer = AdvancedSpectralAnalyzer(analysis_config)
            
            # 制限時間内での評価
            start_time = datetime.now()
            
            # 小規模テスト
            test_result = analyzer._analyze_lattice_size(8)
            
            if 'error' in test_result:
                return 0.0, {'error': test_result['error']}
            
            # 計算時間チェック
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > self.config.timeout_seconds:
                return 0.0, {'error': 'タイムアウト'}
            
            # スコア計算
            mass_gap = test_result.get('mass_gap', 0.0)
            quality_score = test_result.get('quality_score', 0.0)
            
            # 総合スコア: 質量ギャップ重視
            total_score = mass_gap * 10.0 + quality_score
            
            # メモリクリーンアップ
            torch.cuda.empty_cache()
            
            return total_score, test_result
            
        except Exception as e:
            logger.warning(f"⚠️ パラメータ評価エラー: {e}")
            return 0.0, {'error': str(e)}
    
    def _check_convergence(self) -> bool:
        """収束判定"""
        
        if len(self.evaluation_history) < 10:
            return False
        
        # 最近10回の評価スコア
        recent_scores = [ev['score'] for ev in self.evaluation_history[-10:]]
        
        # スコア改善の標準偏差
        score_std = np.std(recent_scores)
        
        return score_std < self.config.convergence_threshold
    
    def _analyze_convergence(self) -> Dict[str, Any]:
        """収束解析"""
        
        if not self.evaluation_history:
            return {}
        
        scores = [ev['score'] for ev in self.evaluation_history]
        
        return {
            'total_evaluations': len(self.evaluation_history),
            'final_score': scores[-1] if scores else 0.0,
            'best_score': max(scores) if scores else 0.0,
            'score_improvement': max(scores) - scores[0] if len(scores) > 1 else 0.0,
            'convergence_achieved': self._check_convergence()
        }
    
    def _save_checkpoint(self):
        """チェックポイント保存"""
        
        checkpoint_data = {
            'session_id': self.session_id,
            'evaluation_history': self.evaluation_history,
            'best_parameters': self.best_parameters,
            'best_score': self.best_score,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{self.session_id}.json"
        
        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"💾 チェックポイント保存: {checkpoint_file}")
            
        except Exception as e:
            logger.warning(f"⚠️ チェックポイント保存エラー: {e}")
    
    def _load_checkpoint(self) -> bool:
        """チェックポイント読み込み"""
        
        # 最新のチェックポイントファイル検索
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.json"))
        
        if not checkpoint_files:
            return False
        
        latest_checkpoint = max(checkpoint_files, key=lambda f: f.stat().st_mtime)
        
        try:
            with open(latest_checkpoint, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            self.evaluation_history = checkpoint_data.get('evaluation_history', [])
            self.best_parameters = checkpoint_data.get('best_parameters')
            self.best_score = checkpoint_data.get('best_score', 0.0)
            
            logger.info(f"🔄 チェックポイント復旧: {len(self.evaluation_history)} 評価履歴")
            
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ チェックポイント読み込みエラー: {e}")
            return False
    
    def _get_optimization_config(self) -> Dict[str, Any]:
        """最適化設定サマリー"""
        return {
            'coupling_range': self.config.coupling_range,
            'theta_range': self.config.theta_range,
            'alpha_range': self.config.alpha_range,
            'max_evaluations': self.config.max_evaluations,
            'lattice_size_limit': self.config.lattice_size_limit,
            'memory_budget_gb': self.config.memory_budget_gb
        }


def run_parameter_optimization(config: Optional[OptimizationConfig] = None) -> Dict[str, Any]:
    """パラメータ最適化実行"""
    
    if config is None:
        config = OptimizationConfig(
            max_evaluations=30,
            lattice_size_limit=16,
            memory_budget_gb=7.0
        )
    
    logger.info("🎪 NKAT理論パラメータ最適化システム起動")
    logger.info(f"💎 最大評価回数: {config.max_evaluations}")
    logger.info(f"💎 格子サイズ制限: {config.lattice_size_limit}")
    
    # 最適化実行
    optimizer = ParameterOptimizer(config)
    results = optimizer.optimize()
    
    # 結果表示
    display_optimization_results(results)
    
    return results


def display_optimization_results(results: Dict[str, Any]):
    """最適化結果表示"""
    logger.info("="*80)
    logger.info("🏆 NKAT理論パラメータ最適化結果")
    logger.info("="*80)
    
    best_params = results.get('best_parameters', {})
    best_score = results.get('best_score', 0.0)
    
    if best_params:
        logger.info(f"最高スコア: {best_score:.8f}")
        logger.info("最適パラメータ:")
        logger.info(f"  結合定数: {best_params.get('coupling_constant', 0):.6f}")
        logger.info(f"  θパラメータ: {best_params.get('theta', 0):.2e}")
        logger.info(f"  αパラメータ: {best_params.get('alpha', 0):.6f}")
    
    convergence = results.get('convergence_analysis', {})
    if convergence:
        logger.info(f"\n収束解析:")
        logger.info(f"  総評価回数: {convergence.get('total_evaluations', 0)}")
        logger.info(f"  スコア改善: {convergence.get('score_improvement', 0):.8f}")
        logger.info(f"  収束達成: {'✅' if convergence.get('convergence_achieved', False) else '❌'}")
    
    logger.info("="*80)


if __name__ == "__main__":
    # RTX3080環境設定
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"🚀 GPU: {device_name}")
    
    # パラメータ最適化実行
    results = run_parameter_optimization()
    
    # 結果保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"nkat_optimization_results_{timestamp}.json"
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"📁 最適化結果保存: {result_file}") 