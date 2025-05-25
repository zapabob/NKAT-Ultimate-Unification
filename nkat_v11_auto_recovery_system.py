#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT v11 自動リカバリーシステム
電源断対応・自動復旧・プロセス監視

作成者: NKAT Research Team
作成日: 2025年5月26日
バージョン: v11.0
"""

import os
import sys
import time
import json
import psutil
import subprocess
import threading
import signal
import hashlib
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import logging

class RecoveryState:
    """リカバリー状態管理クラス"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.last_backup = None
        self.backup_count = 0
        self.recovery_count = 0
        self.monitored_processes = {}
        self.system_health = {
            'cpu_usage': 0,
            'memory_usage': 0,
            'disk_usage': 0,
            'gpu_available': False
        }
        
    def to_dict(self):
        """辞書形式に変換"""
        return {
            'start_time': self.start_time.isoformat(),
            'last_backup': self.last_backup.isoformat() if self.last_backup else None,
            'backup_count': self.backup_count,
            'recovery_count': self.recovery_count,
            'monitored_processes': self.monitored_processes,
            'system_health': self.system_health
        }

class ProcessInfo:
    """プロセス情報クラス"""
    
    def __init__(self, name, command, working_dir=None, auto_restart=True):
        self.name = name
        self.command = command
        self.working_dir = working_dir or os.getcwd()
        self.auto_restart = auto_restart
        self.pid = None
        self.start_time = None
        self.restart_count = 0
        self.last_restart = None
        
    def to_dict(self):
        """辞書形式に変換"""
        return {
            'name': self.name,
            'command': self.command,
            'working_dir': self.working_dir,
            'auto_restart': self.auto_restart,
            'pid': self.pid,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'restart_count': self.restart_count,
            'last_restart': self.last_restart.isoformat() if self.last_restart else None
        }

class AutoRecoverySystem:
    """自動リカバリーシステム"""
    
    def __init__(self, backup_interval=300, health_check_interval=60):
        """
        初期化
        
        Args:
            backup_interval: バックアップ間隔（秒）デフォルト5分
            health_check_interval: ヘルスチェック間隔（秒）デフォルト1分
        """
        self.backup_interval = backup_interval
        self.health_check_interval = health_check_interval
        self.recovery_state = RecoveryState()
        self.monitored_processes = {}
        self.running = False
        
        # ディレクトリ設定
        self.recovery_dir = Path("recovery_data")
        self.backup_dir = self.recovery_dir / "backups"
        self.checkpoint_dir = self.recovery_dir / "checkpoints"
        self.log_dir = self.recovery_dir / "logs"
        
        # ディレクトリ作成
        for dir_path in [self.recovery_dir, self.backup_dir, self.checkpoint_dir, self.log_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # ログ設定
        self.setup_logging()
        
        # シグナルハンドラー設定
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.logger.info("自動リカバリーシステムを初期化しました")
    
    def setup_logging(self):
        """ログ設定"""
        log_file = self.log_dir / f"recovery_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def signal_handler(self, signum, frame):
        """シグナルハンドラー"""
        self.logger.warning(f"シグナル {signum} を受信しました。緊急バックアップを実行中...")
        self.create_emergency_backup()
        self.stop()
        sys.exit(0)
    
    def add_monitored_process(self, process_info):
        """監視対象プロセスを追加"""
        self.monitored_processes[process_info.name] = process_info
        self.logger.info(f"監視対象プロセスを追加: {process_info.name}")
    
    def create_backup(self):
        """バックアップを作成"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{timestamp}"
            backup_path = self.backup_dir / backup_name
            
            # バックアップデータ収集
            backup_data = {
                'timestamp': timestamp,
                'recovery_state': self.recovery_state.to_dict(),
                'monitored_processes': {
                    name: proc.to_dict() 
                    for name, proc in self.monitored_processes.items()
                },
                'system_info': self.get_system_info(),
                'file_checksums': self.calculate_file_checksums()
            }
            
            # バックアップファイル作成
            backup_file = backup_path.with_suffix('.json')
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            
            # 重要ファイルのコピー
            self.backup_important_files(backup_path)
            
            self.recovery_state.last_backup = datetime.now()
            self.recovery_state.backup_count += 1
            
            # 古いバックアップの削除（最新10個を保持）
            self.cleanup_old_backups()
            
            self.logger.info(f"バックアップを作成しました: {backup_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"バックアップ作成エラー: {e}")
            return False
    
    def backup_important_files(self, backup_path):
        """重要ファイルをバックアップ"""
        important_files = [
            'high_precision_riemann_results.json',
            'ultimate_mastery_riemann_results.json',
            'extended_riemann_results.json',
            'improved_riemann_results.json'
        ]
        
        backup_path.mkdir(exist_ok=True)
        
        for file_name in important_files:
            if os.path.exists(file_name):
                try:
                    shutil.copy2(file_name, backup_path / file_name)
                except Exception as e:
                    self.logger.warning(f"ファイルバックアップエラー {file_name}: {e}")
    
    def calculate_file_checksums(self):
        """ファイルのチェックサムを計算"""
        checksums = {}
        important_files = [
            'high_precision_riemann_results.json',
            'ultimate_mastery_riemann_results.json',
            'extended_riemann_results.json',
            'improved_riemann_results.json'
        ]
        
        for file_name in important_files:
            if os.path.exists(file_name):
                try:
                    with open(file_name, 'rb') as f:
                        content = f.read()
                        checksums[file_name] = hashlib.md5(content).hexdigest()
                except Exception as e:
                    self.logger.warning(f"チェックサム計算エラー {file_name}: {e}")
        
        return checksums
    
    def cleanup_old_backups(self):
        """古いバックアップを削除"""
        try:
            backup_files = list(self.backup_dir.glob("backup_*.json"))
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # 最新10個を保持
            for old_backup in backup_files[10:]:
                try:
                    old_backup.unlink()
                    # 対応するディレクトリも削除
                    backup_dir = old_backup.with_suffix('')
                    if backup_dir.exists():
                        shutil.rmtree(backup_dir)
                except Exception as e:
                    self.logger.warning(f"古いバックアップ削除エラー: {e}")
                    
        except Exception as e:
            self.logger.error(f"バックアップクリーンアップエラー: {e}")
    
    def create_emergency_backup(self):
        """緊急バックアップを作成"""
        self.logger.info("緊急バックアップを作成中...")
        
        emergency_backup = {
            'timestamp': datetime.now().isoformat(),
            'type': 'emergency',
            'recovery_state': self.recovery_state.to_dict(),
            'system_info': self.get_system_info(),
            'running_processes': self.get_running_python_processes()
        }
        
        emergency_file = self.recovery_dir / f"emergency_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(emergency_file, 'w', encoding='utf-8') as f:
                json.dump(emergency_backup, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"緊急バックアップを作成しました: {emergency_file}")
            
        except Exception as e:
            self.logger.error(f"緊急バックアップエラー: {e}")
    
    def get_system_info(self):
        """システム情報を取得"""
        try:
            # CPU情報
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # メモリ情報
            memory = psutil.virtual_memory()
            
            # ディスク情報
            disk = psutil.disk_usage('.')
            
            # GPU情報
            gpu_available = False
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                gpu_available = len(gpus) > 0
            except ImportError:
                pass
            
            system_info = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available': memory.available,
                'disk_percent': (disk.used / disk.total) * 100,
                'disk_free': disk.free,
                'gpu_available': gpu_available,
                'platform': sys.platform,
                'python_version': sys.version,
                'working_directory': os.getcwd()
            }
            
            # システムヘルス更新
            self.recovery_state.system_health.update({
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'disk_usage': (disk.used / disk.total) * 100,
                'gpu_available': gpu_available
            })
            
            return system_info
            
        except Exception as e:
            self.logger.error(f"システム情報取得エラー: {e}")
            return {}
    
    def get_running_python_processes(self):
        """実行中のPythonプロセスを取得"""
        processes = []
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
                try:
                    if 'python' in proc.info['name'].lower():
                        processes.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'cmdline': proc.info['cmdline'],
                            'create_time': proc.info['create_time']
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
        except Exception as e:
            self.logger.error(f"プロセス情報取得エラー: {e}")
        
        return processes
    
    def check_process_health(self):
        """プロセスヘルスチェック"""
        for name, proc_info in self.monitored_processes.items():
            try:
                if proc_info.pid:
                    # プロセスが生きているかチェック
                    if psutil.pid_exists(proc_info.pid):
                        process = psutil.Process(proc_info.pid)
                        if process.is_running():
                            continue
                    
                    # プロセスが死んでいる場合
                    self.logger.warning(f"プロセス {name} (PID: {proc_info.pid}) が停止しています")
                    proc_info.pid = None
                
                # 自動再起動が有効な場合
                if proc_info.auto_restart and not proc_info.pid:
                    self.restart_process(proc_info)
                    
            except Exception as e:
                self.logger.error(f"プロセスヘルスチェックエラー {name}: {e}")
    
    def restart_process(self, proc_info):
        """プロセスを再起動"""
        try:
            self.logger.info(f"プロセス {proc_info.name} を再起動中...")
            
            # 作業ディレクトリ変更
            original_cwd = os.getcwd()
            if proc_info.working_dir:
                os.chdir(proc_info.working_dir)
            
            # プロセス起動
            process = subprocess.Popen(
                proc_info.command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            proc_info.pid = process.pid
            proc_info.start_time = datetime.now()
            proc_info.restart_count += 1
            proc_info.last_restart = datetime.now()
            
            # 作業ディレクトリを戻す
            os.chdir(original_cwd)
            
            self.recovery_state.recovery_count += 1
            
            self.logger.info(f"プロセス {proc_info.name} を再起動しました (PID: {proc_info.pid})")
            
            return True
            
        except Exception as e:
            self.logger.error(f"プロセス再起動エラー {proc_info.name}: {e}")
            return False
    
    def create_checkpoint(self):
        """チェックポイントを作成"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_file = self.checkpoint_dir / f"checkpoint_{timestamp}.json"
            
            checkpoint_data = {
                'timestamp': timestamp,
                'recovery_state': self.recovery_state.to_dict(),
                'system_info': self.get_system_info(),
                'file_checksums': self.calculate_file_checksums(),
                'checkpoint_type': 'regular'
            }
            
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"チェックポイントを作成しました: {checkpoint_file}")
            return checkpoint_file
            
        except Exception as e:
            self.logger.error(f"チェックポイント作成エラー: {e}")
            return None
    
    def restore_from_checkpoint(self, checkpoint_file):
        """チェックポイントから復元"""
        try:
            self.logger.info(f"チェックポイントから復元中: {checkpoint_file}")
            
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            # ファイル整合性チェック
            if 'file_checksums' in checkpoint_data:
                self.verify_file_integrity(checkpoint_data['file_checksums'])
            
            self.logger.info("チェックポイントからの復元が完了しました")
            return True
            
        except Exception as e:
            self.logger.error(f"チェックポイント復元エラー: {e}")
            return False
    
    def verify_file_integrity(self, expected_checksums):
        """ファイル整合性を検証"""
        for file_name, expected_checksum in expected_checksums.items():
            if os.path.exists(file_name):
                try:
                    with open(file_name, 'rb') as f:
                        content = f.read()
                        actual_checksum = hashlib.md5(content).hexdigest()
                    
                    if actual_checksum != expected_checksum:
                        self.logger.warning(f"ファイル整合性エラー: {file_name}")
                    else:
                        self.logger.debug(f"ファイル整合性OK: {file_name}")
                        
                except Exception as e:
                    self.logger.error(f"ファイル整合性チェックエラー {file_name}: {e}")
            else:
                self.logger.warning(f"ファイルが見つかりません: {file_name}")
    
    def run_backup_loop(self):
        """バックアップループを実行"""
        while self.running:
            try:
                self.create_backup()
                time.sleep(self.backup_interval)
            except Exception as e:
                self.logger.error(f"バックアップループエラー: {e}")
                time.sleep(60)  # エラー時は1分待機
    
    def run_health_check_loop(self):
        """ヘルスチェックループを実行"""
        while self.running:
            try:
                self.check_process_health()
                self.get_system_info()  # システム情報更新
                time.sleep(self.health_check_interval)
            except Exception as e:
                self.logger.error(f"ヘルスチェックループエラー: {e}")
                time.sleep(30)  # エラー時は30秒待機
    
    def start(self):
        """システム開始"""
        if self.running:
            self.logger.warning("システムは既に実行中です")
            return
        
        self.running = True
        self.logger.info("自動リカバリーシステムを開始します")
        
        # 初期バックアップ
        self.create_backup()
        
        # バックアップスレッド開始
        backup_thread = threading.Thread(target=self.run_backup_loop, daemon=True)
        backup_thread.start()
        
        # ヘルスチェックスレッド開始
        health_thread = threading.Thread(target=self.run_health_check_loop, daemon=True)
        health_thread.start()
        
        self.logger.info("自動リカバリーシステムが開始されました")
    
    def stop(self):
        """システム停止"""
        if not self.running:
            return
        
        self.logger.info("自動リカバリーシステムを停止中...")
        self.running = False
        
        # 最終バックアップ
        self.create_backup()
        
        self.logger.info("自動リカバリーシステムが停止しました")
    
    def get_status(self):
        """システム状態を取得"""
        return {
            'running': self.running,
            'recovery_state': self.recovery_state.to_dict(),
            'monitored_processes': {
                name: proc.to_dict() 
                for name, proc in self.monitored_processes.items()
            },
            'system_info': self.get_system_info()
        }

def main():
    """メイン実行関数"""
    print("NKAT v11 自動リカバリーシステム")
    print("電源断対応・自動復旧・プロセス監視")
    print("=" * 50)
    
    # リカバリーシステム初期化
    recovery_system = AutoRecoverySystem(
        backup_interval=300,  # 5分間隔
        health_check_interval=60  # 1分間隔
    )
    
    # 監視対象プロセスの例
    # recovery_system.add_monitored_process(
    #     ProcessInfo(
    #         name="riemann_verification",
    #         command="py -3 riemann_high_precision.py",
    #         auto_restart=True
    #     )
    # )
    
    try:
        # システム開始
        recovery_system.start()
        
        print("🚀 自動リカバリーシステムが開始されました")
        print("📋 状態:")
        print(f"   バックアップ間隔: {recovery_system.backup_interval}秒")
        print(f"   ヘルスチェック間隔: {recovery_system.health_check_interval}秒")
        print("   Ctrl+C で停止")
        
        # メインループ
        while True:
            time.sleep(10)
            status = recovery_system.get_status()
            print(f"\r⏰ 稼働中... バックアップ数: {status['recovery_state']['backup_count']}, "
                  f"復旧数: {status['recovery_state']['recovery_count']}", end="")
            
    except KeyboardInterrupt:
        print("\n\n🛑 停止要求を受信しました")
        recovery_system.stop()
        print("✅ システムが正常に停止しました")
    
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        recovery_system.stop()

if __name__ == "__main__":
    main() 