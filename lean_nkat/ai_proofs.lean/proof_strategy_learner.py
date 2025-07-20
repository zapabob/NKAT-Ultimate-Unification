#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 NKAT 証明戦略学習システム
ボブにゃんのaesop即死問題を解決するための戦略学習エンジン
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

# 日本語ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s 🚀 %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('strategy_learner.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ProofResult:
    """証明結果データ"""
    theorem: str
    success: bool
    tactic: str
    time: float
    difficulty: Optional[int] = None
    theorem_type: Optional[str] = None

@dataclass
class StrategyStats:
    """戦略統計データ"""
    tactic: str
    total_attempts: int
    success_count: int
    success_rate: float
    avg_time: float
    confidence: float

class ProofStrategyLearner:
    """🎯 証明戦略学習システム"""
    
    def __init__(self, data_dir: str = "./proofs"):
        self.data_dir = Path(data_dir)
        self.results: List[ProofResult] = []
        self.strategy_stats: Dict[str, StrategyStats] = {}
        self.theorem_patterns: Dict[str, Dict[str, float]] = {}
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 戦略学習パラメータ
        self.learning_rate = 0.1
        self.exploration_rate = 0.2
        self.confidence_threshold = 0.7
        
        logger.info("🎯 NKAT 証明戦略学習システム起動！")
        logger.info("🧠 ボブにゃんのaesop即死問題、戦略学習で解決！")
    
    def load_proof_results(self, results_file: str) -> bool:
        """📊 証明結果データ読み込み"""
        try:
            file_path = Path(results_file)
            if not file_path.exists():
                logger.error(f"❌ 結果ファイルが見つかりません: {results_file}")
                return False
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # テキスト形式の結果を解析
            results = []
            lines = content.strip().split('\n')
            
            for line in lines:
                if line.startswith('✅') or line.startswith('❌'):
                    parts = line.split(': ')
                    if len(parts) >= 2:
                        theorem_name = parts[0].split(' ')[1]
                        result_info = parts[1]
                        
                        if '✅' in line:
                            success = True
                            tactic = result_info.split(' ')[0]
                            time_str = result_info.split('(')[1].split(')')[0]
                            time = float(time_str.replace('s', ''))
                        else:
                            success = False
                            tactic = result_info.split(' ')[0]
                            time = 0.0
                        
                        results.append(ProofResult(
                            theorem=theorem_name,
                            success=success,
                            tactic=tactic,
                            time=time
                        ))
            
            self.results.extend(results)
            logger.info(f"📊 証明結果読み込み成功: {len(results)}個の結果")
            return True
            
        except Exception as e:
            logger.error(f"❌ 証明結果読み込みエラー: {e}")
            return False
    
    def analyze_tactic_performance(self):
        """📈 タクティック性能分析"""
        logger.info("📈 タクティック性能分析開始...")
        
        tactic_stats = defaultdict(lambda: {'attempts': 0, 'successes': 0, 'times': []})
        
        for result in self.results:
            tactic = result.tactic
            tactic_stats[tactic]['attempts'] += 1
            if result.success:
                tactic_stats[tactic]['successes'] += 1
            if result.time > 0:
                tactic_stats[tactic]['times'].append(result.time)
        
        # 統計計算
        for tactic, stats in tactic_stats.items():
            success_rate = stats['successes'] / stats['attempts'] if stats['attempts'] > 0 else 0
            avg_time = np.mean(stats['times']) if stats['times'] else 0
            
            # 信頼度計算（試行回数に基づく）
            confidence = min(1.0, stats['attempts'] / 10.0)
            
            self.strategy_stats[tactic] = StrategyStats(
                tactic=tactic,
                total_attempts=stats['attempts'],
                success_count=stats['successes'],
                success_rate=success_rate,
                avg_time=avg_time,
                confidence=confidence
            )
        
        logger.info("📊 タクティック性能分析完了！")
        self._print_tactic_analysis()
    
    def _print_tactic_analysis(self):
        """📊 タクティック分析結果表示"""
        logger.info("="*60)
        logger.info("📊 タクティック性能分析結果")
        logger.info("="*60)
        
        # 成功率順にソート
        sorted_stats = sorted(
            self.strategy_stats.values(),
            key=lambda x: x.success_rate,
            reverse=True
        )
        
        for stats in sorted_stats:
            logger.info(f"🎯 {stats.tactic}:")
            logger.info(f"   📈 成功率: {stats.success_rate:.2%}")
            logger.info(f"   🎲 試行回数: {stats.total_attempts}")
            logger.info(f"   ⏱️ 平均時間: {stats.avg_time:.3f}s")
            logger.info(f"   🧠 信頼度: {stats.confidence:.2%}")
            logger.info("")
    
    def learn_theorem_patterns(self):
        """🧠 定理パターン学習"""
        logger.info("🧠 定理パターン学習開始...")
        
        # 定理名からパターンを抽出
        theorem_types = defaultdict(list)
        
        for result in self.results:
            theorem_name = result.theorem
            # 定理名からパターンを推測
            if 'add' in theorem_name or 'plus' in theorem_name:
                theorem_type = 'arithmetic'
            elif 'eq' in theorem_name or 'equal' in theorem_name:
                theorem_type = 'equality'
            elif 'imp' in theorem_name or 'implies' in theorem_name:
                theorem_type = 'implication'
            elif 'and' in theorem_name or 'or' in theorem_name:
                theorem_type = 'logical'
            else:
                theorem_type = 'general'
            
            theorem_types[theorem_type].append(result)
        
        # 各定理タイプでのタクティック成功率を計算
        for theorem_type, results in theorem_types.items():
            tactic_success = defaultdict(lambda: {'success': 0, 'total': 0})
            
            for result in results:
                tactic = result.tactic
                tactic_success[tactic]['total'] += 1
                if result.success:
                    tactic_success[tactic]['success'] += 1
            
            # 成功率を計算
            success_rates = {}
            for tactic, stats in tactic_success.items():
                if stats['total'] > 0:
                    success_rates[tactic] = stats['success'] / stats['total']
            
            self.theorem_patterns[theorem_type] = success_rates
        
        logger.info("🧠 定理パターン学習完了！")
        self._print_pattern_analysis()
    
    def _print_pattern_analysis(self):
        """📊 パターン分析結果表示"""
        logger.info("="*60)
        logger.info("📊 定理パターン分析結果")
        logger.info("="*60)
        
        for theorem_type, tactic_rates in self.theorem_patterns.items():
            logger.info(f"🎯 {theorem_type} タイプ:")
            sorted_tactics = sorted(tactic_rates.items(), key=lambda x: x[1], reverse=True)
            for tactic, rate in sorted_tactics:
                logger.info(f"   {tactic}: {rate:.2%}")
            logger.info("")
    
    def recommend_tactic(self, theorem_name: str, available_tactics: List[str]) -> str:
        """🎯 最適タクティック推薦"""
        # 定理タイプを推測
        if 'add' in theorem_name or 'plus' in theorem_name:
            theorem_type = 'arithmetic'
        elif 'eq' in theorem_name or 'equal' in theorem_name:
            theorem_type = 'equality'
        elif 'imp' in theorem_name or 'implies' in theorem_name:
            theorem_type = 'implication'
        elif 'and' in theorem_name or 'or' in theorem_name:
            theorem_type = 'logical'
        else:
            theorem_type = 'general'
        
        # パターンベース推薦
        if theorem_type in self.theorem_patterns:
            pattern_rates = self.theorem_patterns[theorem_type]
            available_pattern_tactics = [
                tactic for tactic in available_tactics 
                if tactic in pattern_rates
            ]
            
            if available_pattern_tactics:
                # パターンベースで最適なタクティックを選択
                best_tactic = max(available_pattern_tactics, key=lambda t: pattern_rates[t])
                logger.info(f"🎯 パターンベース推薦: {best_tactic} (成功率: {pattern_rates[best_tactic]:.2%})")
                return best_tactic
        
        # 全体的な性能ベース推薦
        available_stats = [
            (tactic, self.strategy_stats[tactic]) 
            for tactic in available_tactics 
            if tactic in self.strategy_stats
        ]
        
        if available_stats:
            # 信頼度と成功率を考慮したスコア計算
            def calculate_score(stats: StrategyStats) -> float:
                return stats.success_rate * stats.confidence
            
            best_tactic, best_stats = max(available_stats, key=lambda x: calculate_score(x[1]))
            logger.info(f"🎯 性能ベース推薦: {best_tactic} (成功率: {best_stats.success_rate:.2%}, 信頼度: {best_stats.confidence:.2%})")
            return best_tactic
        
        # デフォルト推薦
        default_tactic = available_tactics[0] if available_tactics else 'simp'
        logger.info(f"🎯 デフォルト推薦: {default_tactic}")
        return default_tactic
    
    def generate_strategy_report(self) -> Dict:
        """📊 戦略レポート生成"""
        report = {
            'session_id': self.session_id,
            'total_results': len(self.results),
            'tactic_stats': {name: asdict(stats) for name, stats in self.strategy_stats.items()},
            'theorem_patterns': self.theorem_patterns,
            'recommendations': {
                'best_overall_tactic': self._get_best_overall_tactic(),
                'most_reliable_tactic': self._get_most_reliable_tactic(),
                'fastest_tactic': self._get_fastest_tactic()
            }
        }
        
        return report
    
    def _get_best_overall_tactic(self) -> Optional[str]:
        """🏆 全体的に最適なタクティック"""
        if not self.strategy_stats:
            return None
        
        # 成功率と信頼度を考慮したスコア
        best_tactic = max(
            self.strategy_stats.keys(),
            key=lambda t: self.strategy_stats[t].success_rate * self.strategy_stats[t].confidence
        )
        return best_tactic
    
    def _get_most_reliable_tactic(self) -> Optional[str]:
        """🛡️ 最も信頼できるタクティック"""
        if not self.strategy_stats:
            return None
        
        best_tactic = max(
            self.strategy_stats.keys(),
            key=lambda t: self.strategy_stats[t].confidence
        )
        return best_tactic
    
    def _get_fastest_tactic(self) -> Optional[str]:
        """⚡ 最も高速なタクティック"""
        if not self.strategy_stats:
            return None
        
        best_tactic = min(
            self.strategy_stats.keys(),
            key=lambda t: self.strategy_stats[t].avg_time
        )
        return best_tactic
    
    def save_strategy(self, filename: str = None):
        """💾 戦略データ保存"""
        if filename is None:
            filename = f"strategy_{self.session_id}.pkl"
        
        strategy_data = {
            'session_id': self.session_id,
            'results': self.results,
            'strategy_stats': self.strategy_stats,
            'theorem_patterns': self.theorem_patterns,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(strategy_data, f)
        
        logger.info(f"💾 戦略データ保存: {filename}")
    
    def load_strategy(self, filename: str) -> bool:
        """📂 戦略データ読み込み"""
        try:
            with open(filename, 'rb') as f:
                strategy_data = pickle.load(f)
            
            self.session_id = strategy_data['session_id']
            self.results = strategy_data['results']
            self.strategy_stats = strategy_data['strategy_stats']
            self.theorem_patterns = strategy_data['theorem_patterns']
            
            logger.info(f"📂 戦略データ読み込み: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 戦略データ読み込みエラー: {e}")
            return False
    
    def create_visualization(self, output_dir: str = "./visualizations"):
        """📊 可視化生成"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # タクティック性能グラフ
            if self.strategy_stats:
                tactics = list(self.strategy_stats.keys())
                success_rates = [self.strategy_stats[t].success_rate for t in tactics]
                avg_times = [self.strategy_stats[t].avg_time for t in tactics]
                confidences = [self.strategy_stats[t].confidence for t in tactics]
                
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle('🎯 NKAT 証明戦略分析結果', fontsize=16, fontweight='bold')
                
                # 成功率
                ax1.bar(tactics, success_rates, color='skyblue')
                ax1.set_title('📈 タクティック成功率')
                ax1.set_ylabel('成功率')
                ax1.tick_params(axis='x', rotation=45)
                
                # 平均時間
                ax2.bar(tactics, avg_times, color='lightcoral')
                ax2.set_title('⏱️ 平均実行時間')
                ax2.set_ylabel('時間 (秒)')
                ax2.tick_params(axis='x', rotation=45)
                
                # 信頼度
                ax3.bar(tactics, confidences, color='lightgreen')
                ax3.set_title('🧠 信頼度')
                ax3.set_ylabel('信頼度')
                ax3.tick_params(axis='x', rotation=45)
                
                # 総合スコア
                scores = [sr * conf for sr, conf in zip(success_rates, confidences)]
                ax4.bar(tactics, scores, color='gold')
                ax4.set_title('🏆 総合スコア (成功率 × 信頼度)')
                ax4.set_ylabel('スコア')
                ax4.tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                plt.savefig(output_path / 'tactic_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"📊 可視化生成完了: {output_path / 'tactic_analysis.png'}")
            
        except Exception as e:
            logger.error(f"❌ 可視化生成エラー: {e}")

def main():
    """メイン実行関数"""
    print("🎯 NKAT 証明戦略学習システム")
    print("="*60)
    
    learner = ProofStrategyLearner()
    
    # 証明結果読み込み
    results_files = [
        "./proof_results.txt",
        "./proofs/proof_results.json"
    ]
    
    for results_file in results_files:
        if Path(results_file).exists():
            if learner.load_proof_results(results_file):
                break
    
    if not learner.results:
        print("❌ 証明結果が見つかりません")
        return
    
    # 戦略学習実行
    learner.analyze_tactic_performance()
    learner.learn_theorem_patterns()
    
    # レポート生成
    report = learner.generate_strategy_report()
    
    # 戦略データ保存
    learner.save_strategy()
    
    # 可視化生成
    learner.create_visualization()
    
    print("🎉 戦略学習完了！")
    print(f"📊 分析結果: {len(learner.results)}個の証明結果")
    print(f"🎯 最適タクティック: {report['recommendations']['best_overall_tactic']}")

if __name__ == "__main__":
    main() 