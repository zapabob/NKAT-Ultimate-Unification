#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔥 RTX3080極限計算 オールインワン自動実行システム
All-in-One Automatic RTX3080 Extreme Computation System

機能:
- RTX3080極限計算の自動実行
- リアルタイム監視とチェックポイント管理
- 自動結果解析とレポート生成
- 電源断からの自動復旧
- 完全無人動作対応

Author: NKAT Research Team
Date: 2025-05-26
Version: v1.0 - All-in-One Edition
"""

import os
import sys
import time
import datetime
import threading
import subprocess
from pathlib import Path
import json
import psutil
import signal
from typing import Optional, Dict, Any
import argparse

# NKAT Research Modules
from checkpoint_manager import RTX3080CheckpointManager
from extreme_computation_analyzer import RTX3080ResultAnalyzer

class RTX3080AutoComputationManager:
    """RTX3080極限計算 オールインワン自動管理システム"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = self._load_config(config_file)
        self.checkpoint_manager = RTX3080CheckpointManager()
        self.result_analyzer = RTX3080ResultAnalyzer()
        
        # 状態管理
        self.computation_process = None
        self.monitoring_active = False
        self.auto_restart = True
        self.restart_count = 0
        self.max_restarts = 5
        
        # ログ設定
        self.log_file = Path("auto_computation.log")
        
        # シグナルハンドラー設定
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self, config_file: Optional[str]) -> Dict:
        """設定ファイルの読み込み"""
        default_config = {
            'gamma_count': 100,
            'checkpoint_interval': 10,
            'auto_analysis': True,
            'auto_restart': True,
            'max_restarts': 5,
            'monitoring_interval': 30,
            'analysis_interval': 3600,  # 1時間ごとに解析
            'email_notifications': False,
            'computation_timeout': 86400  # 24時間でタイムアウト
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                self._log(f"⚠️ 設定ファイル読み込みエラー: {e}")
        
        return default_config
    
    def _log(self, message: str):
        """ログ出力"""
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        
        print(log_message)
        
        # ファイルにも出力
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_message + '\n')
        except:
            pass
    
    def _signal_handler(self, signum, frame):
        """シグナルハンドラー（優雅な終了）"""
        self._log(f"🛑 シグナル受信 ({signum}): システム終了中...")
        self.auto_restart = False
        self.monitoring_active = False
        
        if self.computation_process:
            try:
                self.computation_process.terminate()
                self.computation_process.wait(timeout=30)
            except:
                try:
                    self.computation_process.kill()
                except:
                    pass
        
        self._log("✅ システム正常終了")
        sys.exit(0)
    
    def check_system_requirements(self) -> bool:
        """システム要件チェック"""
        self._log("🔍 システム要件チェック開始...")
        
        # GPU可用性チェック
        try:
            import torch
            if not torch.cuda.is_available():
                self._log("❌ CUDA対応GPUが見つかりません")
                return False
            
            gpu_name = torch.cuda.get_device_name()
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            self._log(f"✅ GPU検出: {gpu_name}")
            self._log(f"💾 VRAM: {total_memory:.1f} GB")
            
            if total_memory < 8.0:
                self._log("⚠️ VRAM容量が不足している可能性があります")
        
        except ImportError:
            self._log("❌ PyTorchが見つかりません")
            return False
        
        # メモリ使用量チェック
        memory = psutil.virtual_memory()
        available_ram = memory.available / 1e9
        
        if available_ram < 4.0:
            self._log(f"⚠️ 使用可能RAM容量が少ないです: {available_ram:.1f} GB")
        else:
            self._log(f"✅ 使用可能RAM: {available_ram:.1f} GB")
        
        # ディスク容量チェック
        disk = psutil.disk_usage('.')
        free_space = disk.free / 1e9
        
        if free_space < 10.0:
            self._log(f"⚠️ ディスク容量不足: {free_space:.1f} GB")
        else:
            self._log(f"✅ 使用可能ディスク容量: {free_space:.1f} GB")
        
        return True
    
    def start_computation(self) -> bool:
        """RTX3080極限計算開始"""
        self._log("🔥 RTX3080極限計算開始...")
        
        try:
            # 計算スクリプトの実行
            computation_script = Path("src/riemann_rtx3080_extreme_computation.py")
            if not computation_script.exists():
                self._log("❌ 計算スクリプトが見つかりません")
                return False
            
            cmd = [sys.executable, str(computation_script)]
            
            # 環境変数設定
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'  # リアルタイム出力
            
            self.computation_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=env,
                cwd=str(computation_script.parent)
            )
            
            self._log(f"🚀 計算プロセス開始 (PID: {self.computation_process.pid})")
            return True
            
        except Exception as e:
            self._log(f"❌ 計算開始エラー: {e}")
            return False
    
    def monitor_computation(self):
        """計算監視スレッド"""
        self._log("📊 計算監視開始...")
        last_analysis_time = 0
        
        while self.monitoring_active:
            try:
                # プロセス生存確認
                if self.computation_process:
                    poll_result = self.computation_process.poll()
                    
                    if poll_result is not None:
                        # プロセス終了
                        self._log(f"🏁 計算プロセス終了 (終了コード: {poll_result})")
                        
                        if poll_result == 0:
                            self._log("✅ 計算正常完了")
                            self._perform_final_analysis()
                            break
                        else:
                            self._log("⚠️ 計算異常終了")
                            if self.auto_restart and self.restart_count < self.max_restarts:
                                self._restart_computation()
                            else:
                                self._log("❌ 最大再起動回数に達しました")
                                break
                
                # システム状況確認
                self._check_system_health()
                
                # 定期的な解析実行
                current_time = time.time()
                if (self.config['auto_analysis'] and 
                    current_time - last_analysis_time > self.config['analysis_interval']):
                    self._perform_intermediate_analysis()
                    last_analysis_time = current_time
                
                # チェックポイント状況確認
                self._check_checkpoint_status()
                
                time.sleep(self.config['monitoring_interval'])
                
            except Exception as e:
                self._log(f"⚠️ 監視エラー: {e}")
                time.sleep(30)
    
    def _check_system_health(self):
        """システム健康状態チェック"""
        try:
            # CPU使用率チェック
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 95:
                self._log(f"⚠️ 高CPU使用率: {cpu_percent}%")
            
            # メモリ使用率チェック
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                self._log(f"⚠️ 高メモリ使用率: {memory.percent}%")
            
            # GPU温度チェック（nvidia-smi使用）
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'], 
                                       capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    temp = int(result.stdout.strip())
                    if temp > 85:
                        self._log(f"🔥 高GPU温度: {temp}°C")
                    elif temp > 80:
                        self._log(f"⚠️ GPU温度注意: {temp}°C")
            except:
                pass
                
        except Exception as e:
            self._log(f"⚠️ システム健康チェックエラー: {e}")
    
    def _check_checkpoint_status(self):
        """チェックポイント状況確認"""
        try:
            status = self.checkpoint_manager.get_current_status()
            if status and 'results_summary' in status:
                summary = status['results_summary']
                progress = summary.get('progress_percentage', 0)
                
                # 進捗が更新されているかチェック
                current_time = time.time()
                checkpoint_file = self.checkpoint_manager.latest_checkpoint_file
                
                if checkpoint_file.exists():
                    last_modified = checkpoint_file.stat().st_mtime
                    time_diff = current_time - last_modified
                    
                    # 長時間更新されていない場合の警告
                    if time_diff > 3600:  # 1時間
                        self._log(f"⚠️ チェックポイント更新停滞: {time_diff/3600:.1f}時間")
                    
                    # 進捗レポート
                    if progress > 0:
                        self._log(f"📈 計算進捗: {progress:.1f}%")
                        
        except Exception as e:
            self._log(f"⚠️ チェックポイント確認エラー: {e}")
    
    def _restart_computation(self):
        """計算の再起動"""
        self.restart_count += 1
        self._log(f"🔄 計算再起動 ({self.restart_count}/{self.max_restarts})")
        
        # 少し待機
        time.sleep(30)
        
        # プロセスのクリーンアップ
        if self.computation_process:
            try:
                self.computation_process.kill()
                self.computation_process.wait()
            except:
                pass
        
        # 再起動
        if self.start_computation():
            self._log("✅ 計算再起動成功")
        else:
            self._log("❌ 計算再起動失敗")
    
    def _perform_intermediate_analysis(self):
        """中間解析実行"""
        self._log("📊 中間解析実行中...")
        
        try:
            # 簡単な進捗解析
            results = self.result_analyzer.load_latest_results()
            if results:
                gamma_count = len(results.get('gamma_values', []))
                completed_count = len([x for x in results.get('convergence_to_half', []) 
                                     if x is not None and not (isinstance(x, float) and x != x)])
                
                if gamma_count > 0:
                    progress = completed_count / gamma_count * 100
                    self._log(f"📈 中間解析: {completed_count}/{gamma_count} ({progress:.1f}%)")
        
        except Exception as e:
            self._log(f"⚠️ 中間解析エラー: {e}")
    
    def _perform_final_analysis(self):
        """最終解析実行"""
        self._log("🎉 最終解析実行中...")
        
        try:
            report_file = self.result_analyzer.run_complete_analysis()
            if report_file:
                self._log(f"📄 最終レポート生成完了: {Path(report_file).name}")
            else:
                self._log("⚠️ 最終解析に失敗しました")
                
        except Exception as e:
            self._log(f"❌ 最終解析エラー: {e}")
    
    def run_auto_computation(self):
        """自動計算メイン実行"""
        self._log("🚀 RTX3080極限計算 オールインワンシステム開始")
        self._log("=" * 80)
        
        # システム要件チェック
        if not self.check_system_requirements():
            self._log("❌ システム要件を満たしていません")
            return False
        
        # 監視スレッド開始
        self.monitoring_active = True
        monitor_thread = threading.Thread(target=self.monitor_computation, daemon=True)
        monitor_thread.start()
        
        # チェックポイント監視開始
        self.checkpoint_manager.monitor_computation_status(self.config['monitoring_interval'])
        
        # 計算開始
        if not self.start_computation():
            self._log("❌ 計算開始に失敗しました")
            return False
        
        try:
            # メインループ
            while self.auto_restart and self.computation_process:
                time.sleep(60)  # 1分間隔でメインチェック
                
                # タイムアウトチェック
                if self.computation_process:
                    # プロセス開始からの経過時間をチェック
                    # （実装の詳細は省略）
                    pass
            
            self._log("🏁 自動計算システム終了")
            return True
            
        except KeyboardInterrupt:
            self._log("⏹️ ユーザーによる中断")
            return False
        
        finally:
            self.monitoring_active = False
            self.checkpoint_manager.stop_monitoring()
    
    def create_status_dashboard(self):
        """状況ダッシュボード作成"""
        dashboard_content = f"""
# 🔥 RTX3080極限計算 - リアルタイムダッシュボード

**更新時刻**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🚀 計算状況
- **プロセス状態**: {'実行中' if self.computation_process and self.computation_process.poll() is None else '停止'}
- **再起動回数**: {self.restart_count}/{self.max_restarts}
- **自動再起動**: {'有効' if self.auto_restart else '無効'}

## 📊 システム監視
"""
        
        # システム情報を追加
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            dashboard_content += f"""
- **CPU使用率**: {cpu_percent:.1f}%
- **メモリ使用率**: {memory.percent:.1f}%
- **使用可能メモリ**: {memory.available/1e9:.1f} GB
"""
        except:
            pass
        
        # チェックポイント情報
        try:
            status = self.checkpoint_manager.get_current_status()
            if status and 'results_summary' in status:
                summary = status['results_summary']
                dashboard_content += f"""
## 💾 チェックポイント状況
- **計算進捗**: {summary.get('progress_percentage', 0):.1f}%
- **完了γ値**: {summary.get('completed_gamma_values', 0)}/{summary.get('total_gamma_values', 0)}
"""
        except:
            pass
        
        dashboard_content += f"""
## ⚙️ 設定
- **γ値数**: {self.config['gamma_count']}
- **チェックポイント間隔**: {self.config['checkpoint_interval']}
- **監視間隔**: {self.config['monitoring_interval']}秒
- **自動解析**: {'有効' if self.config['auto_analysis'] else '無効'}

---
*最終更新: {datetime.datetime.now().isoformat()}*
"""
        
        # ダッシュボードファイル保存
        dashboard_file = Path("rtx3080_dashboard.md")
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write(dashboard_content)
        
        return str(dashboard_file)

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='RTX3080極限計算 オールインワンシステム')
    parser.add_argument('--config', '-c', help='設定ファイルパス')
    parser.add_argument('--gamma-count', '-g', type=int, default=100, help='γ値数')
    parser.add_argument('--no-auto-restart', action='store_true', help='自動再起動無効')
    parser.add_argument('--dashboard-only', action='store_true', help='ダッシュボードのみ作成')
    
    args = parser.parse_args()
    
    # 設定の作成/更新
    config = {
        'gamma_count': args.gamma_count,
        'auto_restart': not args.no_auto_restart
    }
    
    if args.config:
        # 設定ファイル保存
        with open(args.config, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"📄 設定ファイル作成: {args.config}")
    
    # システム初期化
    manager = RTX3080AutoComputationManager(args.config)
    
    if args.dashboard_only:
        # ダッシュボードのみ作成
        dashboard_file = manager.create_status_dashboard()
        print(f"📊 ダッシュボード作成: {dashboard_file}")
        return
    
    print("🔥 RTX3080極限計算 オールインワンシステム v1.0")
    print("=" * 60)
    print("💡 このシステムは自動でRTX3080極限計算を実行し、")
    print("   チェックポイント管理、監視、解析を統合的に行います。")
    print("   Ctrl+Cで安全に終了できます。")
    print("=" * 60)
    
    # 実行確認
    user_input = input("\n🚀 RTX3080極限計算を開始しますか？ (y/N): ")
    if user_input.lower() != 'y':
        print("❌ 実行がキャンセルされました")
        return
    
    # 自動計算実行
    success = manager.run_auto_computation()
    
    if success:
        print("🎉 RTX3080極限計算システム正常完了")
    else:
        print("❌ RTX3080極限計算システムエラー終了")

if __name__ == "__main__":
    main() 