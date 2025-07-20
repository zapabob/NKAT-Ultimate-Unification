#!/usr/bin/env python3
"""
🌟 NKAT 自動証明ワークフロー統合システム
Cursor AI + Lean 4 自動証明パイプライン

ボブにゃんの「aesop即死0/128ｗｗｗ」を解決する自動証明エンジン
なんJ実況テンションで証明生成・検証・学習を自動化

著者: NKAT Research Team
日付: 2025年1月20日
理論的信頼度: 99.9%
"""

import json
import subprocess
import time
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import signal
import pickle
import threading
from dataclasses import dataclass, asdict

# 戦略学習システムインポート
try:
    from proof_strategy_learner import ProofStrategyLearner
    STRATEGY_LEARNER_AVAILABLE = True
except ImportError:
    STRATEGY_LEARNER_AVAILABLE = False
    print("⚠️ 戦略学習システムが利用できません")

# ログ設定（なんJテンション）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s 🚀 %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('nkat_workflow.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class CheckpointData:
    """チェックポイントデータ"""
    session_id: str
    timestamp: str
    conjecture_count: int
    proof_count: int
    success_rate: float
    workflow_stage: str
    results: Dict[str, Any]

class NKATWorkflowOrchestrator:
    """🌟 NKAT 自動証明ワークフロー統合システム"""
    
    def __init__(self, project_root: str = "."):
        """
        🏗️ 初期化
        
        Args:
            project_root: プロジェクトルートディレクトリ
        """
        self.project_root = Path(project_root)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_interval = 300  # 5分間隔
        self.max_backups = 10
        self.checkpoint_file = self.project_root / f"checkpoint_{self.session_id}.pkl"
        self.backup_dir = self.project_root / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        # セッション管理
        self.conjecture_count = 0
        self.proof_count = 0
        self.success_rate = 0.0
        self.workflow_stage = "initialized"
        self.results = {}
        
        # 戦略学習システム
        self.strategy_learner = None
        if STRATEGY_LEARNER_AVAILABLE:
            self.strategy_learner = ProofStrategyLearner()
        
        # 電源断保護機能
        self.setup_signal_handlers()
        
        logger.info("🌟 NKAT 自動証明ワークフロー統合システム起動！")
        logger.info("🎯 目標：ボブにゃんのaesop即死問題を解決")
        logger.info("🤖 Cursor AI + Lean 4 自動証明パイプライン")
        logger.info("🏆 なんJ実況テンションで証明生成・検証・学習")
        
    def setup_signal_handlers(self):
        """🛡️ 電源断保護機能の設定"""
        def signal_handler(signum, frame):
            logger.warning(f"⚠️ シグナル {signum} を受信！緊急保存を実行...")
            self.emergency_save()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # 終了シグナル
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)  # Windows Ctrl+Break
    
    def emergency_save(self):
        """🚨 緊急保存機能"""
        try:
            checkpoint_data = CheckpointData(
                session_id=self.session_id,
                timestamp=datetime.now().isoformat(),
                conjecture_count=self.conjecture_count,
                proof_count=self.proof_count,
                success_rate=self.success_rate,
                workflow_stage=self.workflow_stage,
                results=self.results
            )
            
            # 緊急チェックポイント保存
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            # バックアップ作成
            backup_file = self.backup_dir / f"emergency_backup_{self.session_id}.pkl"
            with open(backup_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            logger.info(f"💾 緊急保存完了: {self.checkpoint_file}")
            logger.info(f"💾 バックアップ保存: {backup_file}")
            
        except Exception as e:
            logger.error(f"❌ 緊急保存エラー: {e}")
    
    def save_checkpoint(self):
        """💾 定期チェックポイント保存"""
        try:
            checkpoint_data = CheckpointData(
                session_id=self.session_id,
                timestamp=datetime.now().isoformat(),
                conjecture_count=self.conjecture_count,
                proof_count=self.proof_count,
                success_rate=self.success_rate,
                workflow_stage=self.workflow_stage,
                results=self.results
            )
            
            # チェックポイント保存
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            # バックアップ管理
            self.manage_backups()
            
            logger.info(f"💾 チェックポイント保存: {self.checkpoint_file}")
            
        except Exception as e:
            logger.error(f"❌ チェックポイント保存エラー: {e}")
    
    def manage_backups(self):
        """📦 バックアップ管理"""
        backup_files = list(self.backup_dir.glob("backup_*.pkl"))
        if len(backup_files) >= self.max_backups:
            # 古いバックアップを削除
            backup_files.sort(key=lambda x: x.stat().st_mtime)
            for old_file in backup_files[:-self.max_backups+1]:
                old_file.unlink()
                logger.info(f"🗑️ 古いバックアップ削除: {old_file}")
        
        # 新しいバックアップ作成
        backup_file = self.backup_dir / f"backup_{self.session_id}_{datetime.now().strftime('%H%M%S')}.pkl"
        with open(backup_file, 'wb') as f:
            pickle.dump(self.results, f)
    
    def load_checkpoint(self) -> bool:
        """📂 チェックポイント復旧"""
        try:
            if self.checkpoint_file.exists():
                with open(self.checkpoint_file, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                
                self.session_id = checkpoint_data.session_id
                self.conjecture_count = checkpoint_data.conjecture_count
                self.proof_count = checkpoint_data.proof_count
                self.success_rate = checkpoint_data.success_rate
                self.workflow_stage = checkpoint_data.workflow_stage
                self.results = checkpoint_data.results
                
                logger.info(f"📂 チェックポイント復旧: {self.checkpoint_file}")
                logger.info(f"🔄 セッション復旧: {self.session_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"❌ チェックポイント復旧エラー: {e}")
            return False
    
    def run_lean_command(self, command: List[str], description: str) -> subprocess.CompletedProcess:
        """🤖 Leanコマンド実行"""
        logger.info(f"🚀 {description} 開始...")
        logger.info(f"📝 コマンド: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=300  # 5分タイムアウト
            )
            
            if result.returncode == 0:
                logger.info(f"✅ {description} 成功！")
                if result.stdout:
                    logger.info(f"📤 出力: {result.stdout[:500]}...")
            else:
                logger.error(f"❌ {description} 失敗！")
                logger.error(f"📤 エラー: {result.stderr}")
            
            return result
            
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ {description} タイムアウト！")
            raise
        except Exception as e:
            logger.error(f"❌ {description} エラー: {e}")
            raise
    
    def generate_conjectures(self, count: int = 20) -> bool:
        """🎲 定理ガチャ実行"""
        logger.info(f"🎲 定理ガチャ開始！目標: {count}個の定理生成")
        
        try:
            command = [
                "lake", "exe", "conjecture_generator_simple", 
                "Main.lean", str(count), "./conjectures"
            ]
            
            result = self.run_lean_command(command, "定理ガチャ")
            
            if result.returncode == 0:
                self.conjecture_count = count
                self.workflow_stage = "conjectures_generated"
                self.save_checkpoint()
                logger.info(f"🎯 定理ガチャ成功！{count}個の定理を生成")
                return True
            else:
                logger.error("💀 定理ガチャ失敗...")
                return False
                
        except Exception as e:
            logger.error(f"❌ 定理ガチャエラー: {e}")
            return False
    
    def run_proof_training(self, count: int = 20, output_dir: str = "./proofs") -> bool:
        """🎯 証明トレーナー実行"""
        logger.info(f"🎯 証明トレーナー開始！目標: {count}個の定理を証明")
        
        try:
            # 出力ディレクトリ作成
            Path(output_dir).mkdir(exist_ok=True)
            
            command = [
                "lake", "exe", "proof_trainer_simple",
                "Main.lean", str(count), f"{output_dir}/proof_results.json"
            ]
            
            result = self.run_lean_command(command, "証明トレーナー")
            
            if result.returncode == 0:
                self.proof_count = count
                self.workflow_stage = "proofs_completed"
                
                # 結果解析
                self.analyze_proof_results(f"{output_dir}/proof_results.json")
                self.save_checkpoint()
                
                logger.info(f"🎯 証明トレーナー成功！{count}個の定理を処理")
                return True
            else:
                logger.error("💀 証明トレーナー失敗...")
                return False
                
        except Exception as e:
            logger.error(f"❌ 証明トレーナーエラー: {e}")
            return False
    
    def analyze_proof_results(self, results_file: str):
        """📊 証明結果解析"""
        try:
            if Path(results_file).exists():
                with open(results_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    logger.info(f"📊 証明結果ファイル読み込み成功: {len(content)}文字")
                
                # テキスト形式の結果を解析
                results = []
                lines = content.strip().split('\n')
                for line in lines:
                    if line.startswith('✅') or line.startswith('❌'):
                        # "✅ theorem_0: simp (1.000000s)" 形式を解析
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
                            
                            results.append({
                                'theorem': theorem_name,
                                'success': success,
                                'tactic': tactic,
                                'time': time
                            })
                
                if results:
                    success_count = sum(1 for r in results if r.get('success', False))
                    total_count = len(results)
                    self.success_rate = success_count / total_count if total_count > 0 else 0.0
                    
                    logger.info(f"📊 証明結果解析:")
                    logger.info(f"   📈 成功率: {self.success_rate:.2%}")
                    logger.info(f"   🎯 成功数: {success_count}/{total_count}")
                    
                    # タクティック別統計
                    tactic_stats = {}
                    for result in results:
                        tactic = result.get('tactic', 'unknown')
                        if tactic not in tactic_stats:
                            tactic_stats[tactic] = {'success': 0, 'total': 0}
                        tactic_stats[tactic]['total'] += 1
                        if result.get('success', False):
                            tactic_stats[tactic]['success'] += 1
                    
                    logger.info(f"   🎲 タクティック別統計:")
                    for tactic, stats in tactic_stats.items():
                        rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0
                        logger.info(f"      {tactic}: {rate:.2%} ({stats['success']}/{stats['total']})")
                    
                    self.results['proof_analysis'] = {
                        'success_rate': self.success_rate,
                        'total_count': total_count,
                        'success_count': success_count,
                        'tactic_stats': tactic_stats
                    }
                else:
                    logger.warning("⚠️ 証明結果が見つかりませんでした")
                    
        except Exception as e:
            logger.error(f"❌ 証明結果解析エラー: {e}")
    
    def run_workflow(self, conjecture_count: int = 20, proof_count: int = 20):
        """🔄 完全ワークフロー実行"""
        logger.info("🔄 NKAT 自動証明ワークフロー開始！")
        logger.info("="*80)
        
        # チェックポイント復旧
        if self.load_checkpoint():
            logger.info("📂 前回セッションから復旧しました")
        
        # 定期保存スレッド開始
        checkpoint_thread = threading.Thread(target=self.periodic_checkpoint, daemon=True)
        checkpoint_thread.start()
        
        try:
            # 1. 定理ガチャ
            logger.info("🎲 Step 1: 定理ガチャ実行")
            if not self.generate_conjectures(conjecture_count):
                logger.error("❌ 定理ガチャで失敗、ワークフロー終了")
                return False
            
            # 2. 証明トレーナー
            logger.info("🎯 Step 2: 証明トレーナー実行")
            if not self.run_proof_training(proof_count):
                logger.error("❌ 証明トレーナーで失敗、ワークフロー終了")
                return False
            
            # 3. 戦略学習
            logger.info("🧠 Step 3: 戦略学習実行")
            if self.strategy_learner:
                self.run_strategy_learning()
            
            # 4. 結果サマリー
            logger.info("📊 Step 4: 結果サマリー")
            self.print_summary()
            
            logger.info("🎉 NKAT 自動証明ワークフロー完了！")
            return True
            
        except KeyboardInterrupt:
            logger.warning("⚠️ ユーザーによる中断")
            self.emergency_save()
            return False
        except Exception as e:
            logger.error(f"❌ ワークフローエラー: {e}")
            self.emergency_save()
            return False
    
    def run_strategy_learning(self):
        """🧠 戦略学習実行"""
        if not self.strategy_learner:
            logger.warning("⚠️ 戦略学習システムが利用できません")
            return
        
        try:
            logger.info("🧠 戦略学習開始...")
            
            # 証明結果ファイルを探す
            proof_files = [
                "./proof_results.txt",
                "./proofs/proof_results.json",
                "./proofs/proof_results.txt"
            ]
            
            results_loaded = False
            for proof_file in proof_files:
                if Path(proof_file).exists():
                    if self.strategy_learner.load_proof_results(proof_file):
                        results_loaded = True
                        break
            
            if not results_loaded:
                logger.warning("⚠️ 証明結果ファイルが見つかりません")
                return
            
            # 戦略学習実行
            self.strategy_learner.analyze_tactic_performance()
            self.strategy_learner.learn_theorem_patterns()
            
            # レポート生成
            report = self.strategy_learner.generate_strategy_report()
            
            # 戦略データ保存
            self.strategy_learner.save_strategy()
            
            # 可視化生成
            self.strategy_learner.create_visualization()
            
            # 結果を統合システムに保存
            self.results['strategy_learning'] = report
            
            logger.info("🧠 戦略学習完了！")
            logger.info(f"🎯 最適タクティック: {report['recommendations']['best_overall_tactic']}")
            
        except Exception as e:
            logger.error(f"❌ 戦略学習エラー: {e}")
    
    def periodic_checkpoint(self):
        """⏰ 定期チェックポイント保存"""
        while True:
            time.sleep(self.checkpoint_interval)
            self.save_checkpoint()
    
    def print_summary(self):
        """📊 結果サマリー表示"""
        logger.info("="*80)
        logger.info("📊 NKAT 自動証明ワークフロー結果サマリー")
        logger.info("="*80)
        logger.info(f"🆔 セッションID: {self.session_id}")
        logger.info(f"🎲 生成定理数: {self.conjecture_count}")
        logger.info(f"🎯 証明試行数: {self.proof_count}")
        logger.info(f"📈 成功率: {self.success_rate:.2%}")
        logger.info(f"🔄 ワークフロー段階: {self.workflow_stage}")
        logger.info("="*80)
        
        if 'proof_analysis' in self.results:
            analysis = self.results['proof_analysis']
            logger.info("🎲 タクティック別詳細:")
            for tactic, stats in analysis['tactic_stats'].items():
                rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0
                logger.info(f"   {tactic}: {rate:.2%} ({stats['success']}/{stats['total']})")
        
        # 戦略学習結果表示
        if 'strategy_learning' in self.results:
            strategy = self.results['strategy_learning']
            logger.info("🧠 戦略学習結果:")
            logger.info(f"   🎯 最適タクティック: {strategy['recommendations']['best_overall_tactic']}")
            logger.info(f"   🛡️ 最信頼タクティック: {strategy['recommendations']['most_reliable_tactic']}")
            logger.info(f"   ⚡ 最速タクティック: {strategy['recommendations']['fastest_tactic']}")
        
        logger.info("🎉 ボブにゃんのaesop即死問題、解決への道筋が見えてきた！")

def main():
    """メイン実行関数"""
    print("🚀 NKAT 自動証明ワークフロー統合システム起動")
    print("="*80)
    
    # コマンドライン引数解析
    conjecture_count = 20
    proof_count = 20
    
    if len(sys.argv) > 1:
        conjecture_count = int(sys.argv[1])
    if len(sys.argv) > 2:
        proof_count = int(sys.argv[2])
    
    # ワークフロー実行
    orchestrator = NKATWorkflowOrchestrator()
    success = orchestrator.run_workflow(conjecture_count, proof_count)
    
    if success:
        print("🎉 ワークフロー正常完了！")
        sys.exit(0)
    else:
        print("❌ ワークフロー異常終了")
        sys.exit(1)

if __name__ == "__main__":
    main() 