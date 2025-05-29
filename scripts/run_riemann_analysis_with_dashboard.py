#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT リーマン予想解析 統合実行システム
Integrated Riemann Analysis System with Dashboard

非可換コルモゴロフアーノルド表現理論によるリーマン予想解析
- RTX3080フル活用による高速計算
- 電源断リカバリー機能
- Streamlitダッシュボードによるリアルタイム監視
- 長時間計算対応
"""

import sys
import os
import time
import threading
import subprocess
import signal
import logging
from pathlib import Path
from datetime import datetime
import argparse
import json

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.riemann_analysis.nkat_riemann_analyzer import (
        NKATRiemannConfig, RiemannZetaAnalyzer, GPUMonitor, RecoveryManager
    )
    import torch
    import psutil
except ImportError as e:
    print(f"❌ 必要なモジュールのインポートに失敗: {e}")
    print("💡 requirements.txtの依存関係を確認してください")
    sys.exit(1)

class IntegratedRiemannAnalysisSystem:
    """統合リーマン予想解析システム"""
    
    def __init__(self, config: NKATRiemannConfig):
        self.config = config
        self.gpu_monitor = GPUMonitor()
        self.recovery_manager = RecoveryManager()
        self.analyzer = None
        
        # プロセス管理
        self.dashboard_process = None
        self.analysis_running = False
        self.shutdown_requested = False
        
        # ログ設定
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # シグナルハンドラー設定
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("🌌 統合リーマン予想解析システム初期化完了")
    
    def setup_logging(self):
        """ログ設定"""
        log_dir = Path("logs/integrated_analysis")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"integrated_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    
    def _signal_handler(self, signum, frame):
        """シグナルハンドラー（Ctrl+C等）"""
        self.logger.info(f"🛑 シャットダウンシグナル受信: {signum}")
        self.shutdown_requested = True
        self.shutdown()
    
    def start_dashboard(self, port: int = 8501):
        """Streamlitダッシュボードの起動"""
        dashboard_script = project_root / "src" / "dashboard" / "streamlit_dashboard.py"
        
        if not dashboard_script.exists():
            self.logger.error(f"❌ ダッシュボードスクリプトが見つかりません: {dashboard_script}")
            return False
        
        try:
            cmd = [
                sys.executable, "-m", "streamlit", "run",
                str(dashboard_script),
                "--server.port", str(port),
                "--server.headless", "true",
                "--browser.gatherUsageStats", "false"
            ]
            
            self.dashboard_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(project_root)
            )
            
            self.logger.info(f"🌐 Streamlitダッシュボード起動: http://localhost:{port}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ ダッシュボード起動エラー: {e}")
            return False
    
    def stop_dashboard(self):
        """ダッシュボードの停止"""
        if self.dashboard_process:
            try:
                self.dashboard_process.terminate()
                self.dashboard_process.wait(timeout=10)
                self.logger.info("🛑 ダッシュボード停止完了")
            except subprocess.TimeoutExpired:
                self.dashboard_process.kill()
                self.logger.warning("⚠️ ダッシュボードを強制終了しました")
            except Exception as e:
                self.logger.error(f"❌ ダッシュボード停止エラー: {e}")
    
    def initialize_gpu(self):
        """GPU初期化と最適化"""
        self.logger.info("🎮 GPU初期化中...")
        
        gpu_status = self.gpu_monitor.get_gpu_status()
        
        if not gpu_status['available']:
            self.logger.warning("⚠️ CUDA GPU が利用できません - CPU で実行します")
            self.config.device = 'cpu'
            return False
        
        # GPU情報表示
        self.logger.info(f"🎮 GPU: {gpu_status['device_name']}")
        self.logger.info(f"💾 VRAM: {gpu_status['memory_total_mb']:.0f} MB")
        
        # GPU最適化
        self.gpu_monitor.optimize_gpu_settings(self.config)
        
        # メモリクリア
        torch.cuda.empty_cache()
        
        return True
    
    def check_recovery_state(self):
        """リカバリー状態の確認"""
        self.logger.info("💾 リカバリー状態確認中...")
        
        checkpoint_state, is_valid = self.recovery_manager.load_checkpoint()
        
        if checkpoint_state and is_valid:
            self.logger.info("✅ 有効なチェックポイントが見つかりました")
            
            if 'analysis_timestamp' in checkpoint_state:
                timestamp = checkpoint_state['analysis_timestamp']
                self.logger.info(f"📅 最終解析: {timestamp}")
            
            # 継続するかユーザーに確認
            response = input("前回の解析を継続しますか？ (y/n): ").lower().strip()
            
            if response == 'y':
                return checkpoint_state
        
        return None
    
    def run_long_term_analysis(self, max_dimension: int = None, resume_state: dict = None):
        """長時間リーマン予想解析の実行"""
        if max_dimension is None:
            max_dimension = self.config.max_dimension
        
        self.logger.info(f"🔍 長時間リーマン予想解析開始 (最大次元: {max_dimension})")
        
        # 解析器の初期化
        self.analyzer = RiemannZetaAnalyzer(self.config)
        self.analysis_running = True
        
        start_time = time.time()
        checkpoint_counter = 0
        
        try:
            # 継続解析の場合
            if resume_state:
                self.logger.info("🔄 前回の解析から継続します")
                # 継続ロジックは簡略化
            
            # 段階的解析実行
            results = {
                'config': self.config.__dict__,
                'start_time': datetime.now().isoformat(),
                'dimensions_analyzed': [],
                'convergence_data': [],
                'zero_verification': [],
                'critical_line_analysis': [],
                'superconvergence_evidence': [],
                'nkat_zeta_correspondence': [],
                'checkpoints': []
            }
            
            # 次元ごとの段階的解析
            for dim in range(self.config.critical_dimension, max_dimension + 1, 5):
                if self.shutdown_requested:
                    self.logger.info("🛑 シャットダウン要求により解析を中断")
                    break
                
                self.logger.info(f"📊 次元 {dim} 解析中...")
                
                # GPU状態監視
                gpu_status = self.gpu_monitor.get_gpu_status()
                if gpu_status['available']:
                    temp = gpu_status.get('temperature', 0)
                    if temp > 85:  # 温度警告
                        self.logger.warning(f"🌡️ GPU温度警告: {temp}°C - 一時停止")
                        time.sleep(30)  # 冷却待機
                
                # 単一次元解析
                dim_start = time.time()
                dim_results = self.analyzer._analyze_single_dimension(dim)
                dim_duration = time.time() - dim_start
                
                # 結果の記録
                results['dimensions_analyzed'].append(dim)
                results['convergence_data'].append(dim_results['convergence'])
                results['zero_verification'].append(dim_results['zero_verification'])
                results['critical_line_analysis'].append(dim_results['critical_line'])
                results['superconvergence_evidence'].append(dim_results['superconvergence'])
                results['nkat_zeta_correspondence'].append(dim_results['correspondence'])
                
                self.logger.info(f"✅ 次元 {dim} 完了 ({dim_duration:.2f}秒)")
                self.logger.info(f"🔗 収束スコア: {dim_results['convergence']:.6f}")
                
                # 定期チェックポイント保存
                checkpoint_counter += 1
                if checkpoint_counter % self.config.checkpoint_interval == 0:
                    checkpoint_file = self.recovery_manager.save_checkpoint(results)
                    results['checkpoints'].append(checkpoint_file)
                    self.logger.info(f"💾 チェックポイント保存: {checkpoint_file}")
                
                # 自動保存間隔チェック
                elapsed_time = time.time() - start_time
                if elapsed_time > self.config.auto_save_interval:
                    checkpoint_file = self.recovery_manager.save_checkpoint(results)
                    results['checkpoints'].append(checkpoint_file)
                    start_time = time.time()  # タイマーリセット
            
            # 最終評価
            if not self.shutdown_requested:
                self.logger.info("📈 最終評価計算中...")
                results['final_assessment'] = self.analyzer._assess_riemann_hypothesis(results)
            
            # 最終結果保存
            total_time = time.time() - start_time
            results['total_execution_time'] = total_time
            results['completion_time'] = datetime.now().isoformat()
            
            final_checkpoint = self.recovery_manager.save_checkpoint(results)
            self.logger.info(f"💾 最終結果保存: {final_checkpoint}")
            
            # 結果サマリー表示
            self.display_final_results(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ 解析中にエラー: {e}")
            
            # エラー時もチェックポイント保存
            error_results = results.copy()
            error_results['error'] = str(e)
            error_results['error_time'] = datetime.now().isoformat()
            
            self.recovery_manager.save_checkpoint(error_results)
            raise
            
        finally:
            self.analysis_running = False
    
    def display_final_results(self, results: dict):
        """最終結果の表示"""
        print("\n" + "="*80)
        print("🎉 NKAT リーマン予想解析 完了")
        print("="*80)
        
        if 'final_assessment' in results:
            assessment = results['final_assessment']
            print(f"🎯 総合評価: {assessment['assessment']}")
            print(f"🔍 信頼度: {assessment['confidence']}")
            print(f"📈 総合スコア: {assessment['overall_score']:.4f}")
            
            if 'component_scores' in assessment:
                scores = assessment['component_scores']
                print(f"📊 詳細スコア:")
                print(f"  • NKAT-ゼータ対応: {scores.get('nkat_zeta_correspondence', 0):.4f}")
                print(f"  • ゼロ点検証: {scores.get('zero_verification', 0):.4f}")
                print(f"  • 臨界線選好: {scores.get('critical_line_preference', 0):.4f}")
                print(f"  • 収束性: {scores.get('convergence', 0):.4f}")
                print(f"  • 超収束一致: {scores.get('superconvergence_agreement', 0):.4f}")
        
        if 'total_execution_time' in results:
            total_time = results['total_execution_time']
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = int(total_time % 60)
            print(f"⏱️ 総実行時間: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        dimensions = results.get('dimensions_analyzed', [])
        if dimensions:
            print(f"📏 解析次元数: {len(dimensions)} (最大: {max(dimensions)})")
        
        checkpoints = results.get('checkpoints', [])
        print(f"💾 保存チェックポイント数: {len(checkpoints)}")
        
        print("="*80)
    
    def run_system_monitor(self):
        """システム監視スレッド"""
        self.logger.info("📡 システム監視開始")
        
        while not self.shutdown_requested:
            try:
                # GPU状態監視
                gpu_status = self.gpu_monitor.get_gpu_status()
                
                if gpu_status['available']:
                    temp = gpu_status.get('temperature', 0)
                    utilization = gpu_status.get('gpu_utilization', 0)
                    memory_util = gpu_status.get('memory_utilization', 0)
                    
                    # 警告チェック
                    if temp > 85:
                        self.logger.warning(f"🌡️ GPU温度警告: {temp}°C")
                    
                    if memory_util > 95:
                        self.logger.warning(f"💾 VRAM使用率警告: {memory_util:.1f}%")
                    
                    # 定期ログ
                    if int(time.time()) % 300 == 0:  # 5分ごと
                        self.logger.info(f"📊 GPU状態 - 使用率: {utilization:.1f}%, 温度: {temp}°C, VRAM: {memory_util:.1f}%")
                
                # システム状態監視
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
                
                if cpu_percent > 90:
                    self.logger.warning(f"🖥️ CPU使用率警告: {cpu_percent:.1f}%")
                
                if memory_percent > 90:
                    self.logger.warning(f"💾 メモリ使用率警告: {memory_percent:.1f}%")
                
                time.sleep(10)  # 10秒間隔
                
            except Exception as e:
                self.logger.error(f"❌ システム監視エラー: {e}")
                time.sleep(30)
    
    def run(self, max_dimension: int = None, enable_dashboard: bool = True, dashboard_port: int = 8501):
        """統合システムの実行"""
        print("🌌 NKAT リーマン予想解析システム")
        print("=" * 80)
        print("非可換コルモゴロフアーノルド表現理論によるリーマン予想解析")
        print("- RTX3080フル活用による高速計算")
        print("- 電源断リカバリー機能")
        print("- Streamlitダッシュボードによるリアルタイム監視")
        print("=" * 80)
        
        try:
            # GPU初期化
            gpu_available = self.initialize_gpu()
            
            # ダッシュボード起動
            if enable_dashboard:
                if self.start_dashboard(dashboard_port):
                    print(f"🌐 ダッシュボード: http://localhost:{dashboard_port}")
                else:
                    print("⚠️ ダッシュボードの起動に失敗しました")
            
            # システム監視スレッド開始
            monitor_thread = threading.Thread(target=self.run_system_monitor, daemon=True)
            monitor_thread.start()
            
            # リカバリー状態確認
            resume_state = self.check_recovery_state()
            
            # 解析実行
            results = self.run_long_term_analysis(max_dimension, resume_state)
            
            return results
            
        except KeyboardInterrupt:
            self.logger.info("🛑 ユーザーによる中断")
        except Exception as e:
            self.logger.error(f"❌ システムエラー: {e}")
            raise
        finally:
            self.shutdown()
    
    def shutdown(self):
        """システムシャットダウン"""
        self.logger.info("🛑 システムシャットダウン開始")
        
        # 解析停止
        self.analysis_running = False
        
        # ダッシュボード停止
        self.stop_dashboard()
        
        # GPU メモリクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("✅ システムシャットダウン完了")

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="NKAT リーマン予想解析システム")
    parser.add_argument("--max-dimension", type=int, default=50, help="最大解析次元")
    parser.add_argument("--critical-dimension", type=int, default=15, help="臨界次元")
    parser.add_argument("--no-dashboard", action="store_true", help="ダッシュボードを無効化")
    parser.add_argument("--dashboard-port", type=int, default=8501, help="ダッシュボードポート")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.95, help="GPU メモリ使用率")
    parser.add_argument("--checkpoint-interval", type=int, default=5, help="チェックポイント間隔")
    
    args = parser.parse_args()
    
    # 設定
    config = NKATRiemannConfig(
        max_dimension=args.max_dimension,
        critical_dimension=args.critical_dimension,
        gpu_memory_fraction=args.gpu_memory_fraction,
        checkpoint_interval=args.checkpoint_interval,
        precision=torch.float64,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # システム初期化
    system = IntegratedRiemannAnalysisSystem(config)
    
    # 実行
    try:
        results = system.run(
            max_dimension=args.max_dimension,
            enable_dashboard=not args.no_dashboard,
            dashboard_port=args.dashboard_port
        )
        
        print("\n🎉 解析システム正常終了")
        return 0
        
    except Exception as e:
        print(f"\n❌ システムエラー: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 