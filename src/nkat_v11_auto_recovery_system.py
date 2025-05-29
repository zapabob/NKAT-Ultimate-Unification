#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛡️ NKAT v11 自動リカバリーシステム - 電源断対応
NKAT v11 Auto Recovery System - Power Failure Protection

Author: NKAT Research Consortium
Date: 2025-05-26
Version: 11.0 - Auto Recovery System
"""

import os
import sys
import json
import time
import pickle
import psutil
import subprocess
import threading
import signal
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, asdict
import shutil
import hashlib

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nkat_v11_recovery.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class RecoveryState:
    """リカバリー状態管理"""
    timestamp: str
    process_states: Dict[str, Any]
    system_metrics: Dict[str, float]
    checkpoint_info: Dict[str, str]
    verification_progress: Dict[str, Any]
    last_successful_operation: str
    recovery_count: int
    
@dataclass
class ProcessInfo:
    """プロセス情報"""
    name: str
    pid: int
    command: str
    start_time: str
    status: str
    memory_usage: float
    cpu_usage: float

class NKATAutoRecoverySystem:
    """NKAT v11 自動リカバリーシステム"""
    
    def __init__(self):
        self.base_path = Path(".")
        self.recovery_dir = Path("recovery_data")
        self.recovery_dir.mkdir(exist_ok=True)
        
        # 状態ファイル
        self.state_file = self.recovery_dir / "recovery_state.pkl"
        self.process_registry = self.recovery_dir / "process_registry.json"
        self.checkpoint_registry = self.recovery_dir / "checkpoint_registry.json"
        
        # 監視対象プロセス
        self.monitored_processes = [
            "nkat_v11_rigorous_mathematical_verification.py",
            "nkat_v11_enhanced_large_scale_verification.py",
            "riemann_high_precision.py",
            "nkat_v11_results_visualization.py"
        ]
        
        # 重要ディレクトリ
        self.critical_directories = [
            "rigorous_verification_results",
            "enhanced_verification_results",
            "10k_gamma_checkpoints_production",
            "test_checkpoints"
        ]
        
        # リカバリー設定
        self.recovery_config = {
            "auto_restart": True,
            "backup_interval": 300,  # 5分
            "health_check_interval": 60,  # 1分
            "max_recovery_attempts": 3,
            "process_timeout": 3600,  # 1時間
            "memory_threshold": 90,  # %
            "cpu_threshold": 95  # %
        }
        
        # 実行状態
        self.is_monitoring = False
        self.recovery_threads = []
        self.last_backup_time = datetime.now()
        
        # シグナルハンドラー設定
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info("🛡️ NKAT v11 自動リカバリーシステム初期化完了")
    
    def signal_handler(self, signum, frame):
        """シグナルハンドラー（緊急停止時の処理）"""
        logger.warning(f"⚠️ シグナル {signum} を受信、緊急バックアップ実行中...")
        self.emergency_backup()
        self.stop_monitoring()
        sys.exit(0)
    
    def save_recovery_state(self, state: RecoveryState):
        """リカバリー状態の保存"""
        try:
            with open(self.state_file, 'wb') as f:
                pickle.dump(asdict(state), f)
            logger.debug(f"リカバリー状態保存完了: {state.timestamp}")
        except Exception as e:
            logger.error(f"リカバリー状態保存エラー: {e}")
    
    def load_recovery_state(self) -> Optional[RecoveryState]:
        """リカバリー状態の読み込み"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'rb') as f:
                    state_dict = pickle.load(f)
                return RecoveryState(**state_dict)
        except Exception as e:
            logger.error(f"リカバリー状態読み込みエラー: {e}")
        return None
    
    def get_system_metrics(self) -> Dict[str, float]:
        """システムメトリクスの取得"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            metrics = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available": memory.available / 1e9,
                "disk_percent": disk.percent,
                "disk_free": disk.free / 1e9,
                "timestamp": time.time()
            }
            
            # GPU情報（利用可能な場合）
            try:
                import torch
                if torch.cuda.is_available():
                    metrics["gpu_memory_used"] = torch.cuda.memory_allocated() / 1e9
                    metrics["gpu_memory_total"] = torch.cuda.get_device_properties(0).total_memory / 1e9
            except ImportError:
                pass
            
            return metrics
        except Exception as e:
            logger.error(f"システムメトリクス取得エラー: {e}")
            return {}
    
    def get_process_info(self) -> List[ProcessInfo]:
        """監視対象プロセスの情報取得"""
        process_list = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'status', 'memory_percent', 'cpu_percent']):
            try:
                if proc.info['cmdline']:
                    cmdline_str = ' '.join(proc.info['cmdline'])
                    for monitored in self.monitored_processes:
                        if monitored in cmdline_str:
                            process_info = ProcessInfo(
                                name=monitored,
                                pid=proc.info['pid'],
                                command=cmdline_str,
                                start_time=datetime.fromtimestamp(proc.info['create_time']).isoformat(),
                                status=proc.info['status'],
                                memory_usage=proc.info['memory_percent'] or 0.0,
                                cpu_usage=proc.info['cpu_percent'] or 0.0
                            )
                            process_list.append(process_info)
                            break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return process_list
    
    def create_checkpoint(self, checkpoint_type: str = "auto") -> str:
        """チェックポイントの作成"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_id = f"{checkpoint_type}_{timestamp}"
            checkpoint_dir = self.recovery_dir / "checkpoints" / checkpoint_id
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # システム状態の保存
            system_metrics = self.get_system_metrics()
            process_info = self.get_process_info()
            
            checkpoint_data = {
                "checkpoint_id": checkpoint_id,
                "timestamp": timestamp,
                "checkpoint_type": checkpoint_type,
                "system_metrics": system_metrics,
                "process_info": [asdict(p) for p in process_info],
                "critical_files": []
            }
            
            # 重要ファイルのバックアップ
            for directory in self.critical_directories:
                dir_path = Path(directory)
                if dir_path.exists():
                    backup_dir = checkpoint_dir / directory
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 最新ファイルのみバックアップ（容量節約）
                    for file_pattern in ["*.json", "*.pkl", "*.log"]:
                        files = list(dir_path.glob(file_pattern))
                        if files:
                            latest_file = max(files, key=lambda x: x.stat().st_mtime)
                            backup_file = backup_dir / latest_file.name
                            shutil.copy2(latest_file, backup_file)
                            
                            # ファイルハッシュを記録
                            file_hash = self.calculate_file_hash(latest_file)
                            checkpoint_data["critical_files"].append({
                                "original_path": str(latest_file),
                                "backup_path": str(backup_file),
                                "hash": file_hash,
                                "size": latest_file.stat().st_size
                            })
            
            # チェックポイント情報の保存
            checkpoint_file = checkpoint_dir / "checkpoint_info.json"
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
            # レジストリ更新
            self.update_checkpoint_registry(checkpoint_id, checkpoint_data)
            
            logger.info(f"✅ チェックポイント作成完了: {checkpoint_id}")
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"❌ チェックポイント作成エラー: {e}")
            return ""
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """ファイルハッシュの計算"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"ファイルハッシュ計算エラー: {e}")
            return ""
    
    def update_checkpoint_registry(self, checkpoint_id: str, checkpoint_data: Dict):
        """チェックポイントレジストリの更新"""
        try:
            registry = {}
            if self.checkpoint_registry.exists():
                with open(self.checkpoint_registry, 'r', encoding='utf-8') as f:
                    registry = json.load(f)
            
            registry[checkpoint_id] = {
                "timestamp": checkpoint_data["timestamp"],
                "type": checkpoint_data["checkpoint_type"],
                "file_count": len(checkpoint_data["critical_files"]),
                "system_metrics": checkpoint_data["system_metrics"]
            }
            
            # 古いチェックポイントの削除（最新10個を保持）
            if len(registry) > 10:
                sorted_checkpoints = sorted(registry.items(), key=lambda x: x[1]["timestamp"])
                for old_checkpoint, _ in sorted_checkpoints[:-10]:
                    old_checkpoint_dir = self.recovery_dir / "checkpoints" / old_checkpoint
                    if old_checkpoint_dir.exists():
                        shutil.rmtree(old_checkpoint_dir)
                    del registry[old_checkpoint]
            
            with open(self.checkpoint_registry, 'w', encoding='utf-8') as f:
                json.dump(registry, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"チェックポイントレジストリ更新エラー: {e}")
    
    def emergency_backup(self):
        """緊急バックアップ"""
        logger.warning("🚨 緊急バックアップ実行中...")
        checkpoint_id = self.create_checkpoint("emergency")
        if checkpoint_id:
            logger.info(f"✅ 緊急バックアップ完了: {checkpoint_id}")
        else:
            logger.error("❌ 緊急バックアップ失敗")
    
    def restore_from_checkpoint(self, checkpoint_id: str) -> bool:
        """チェックポイントからの復元"""
        try:
            checkpoint_dir = self.recovery_dir / "checkpoints" / checkpoint_id
            checkpoint_file = checkpoint_dir / "checkpoint_info.json"
            
            if not checkpoint_file.exists():
                logger.error(f"チェックポイントが見つかりません: {checkpoint_id}")
                return False
            
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            logger.info(f"🔄 チェックポイントから復元中: {checkpoint_id}")
            
            # ファイルの復元
            for file_info in checkpoint_data["critical_files"]:
                backup_path = Path(file_info["backup_path"])
                original_path = Path(file_info["original_path"])
                
                if backup_path.exists():
                    # ディレクトリ作成
                    original_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # ファイル復元
                    shutil.copy2(backup_path, original_path)
                    
                    # ハッシュ検証
                    restored_hash = self.calculate_file_hash(original_path)
                    if restored_hash == file_info["hash"]:
                        logger.info(f"✅ ファイル復元成功: {original_path}")
                    else:
                        logger.warning(f"⚠️ ハッシュ不一致: {original_path}")
                else:
                    logger.warning(f"⚠️ バックアップファイルが見つかりません: {backup_path}")
            
            logger.info(f"✅ チェックポイント復元完了: {checkpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ チェックポイント復元エラー: {e}")
            return False
    
    def restart_process(self, process_name: str) -> bool:
        """プロセスの再起動"""
        try:
            logger.info(f"🔄 プロセス再起動: {process_name}")
            
            # 既存プロセスの終了
            for proc in psutil.process_iter(['pid', 'cmdline']):
                try:
                    if proc.info['cmdline']:
                        cmdline_str = ' '.join(proc.info['cmdline'])
                        if process_name in cmdline_str:
                            proc.terminate()
                            proc.wait(timeout=10)
                            logger.info(f"✅ プロセス終了: PID {proc.pid}")
                except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                    pass
            
            # 新しいプロセスの開始
            if Path(process_name).exists():
                subprocess.Popen([sys.executable, process_name], 
                               cwd=self.base_path,
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
                logger.info(f"✅ プロセス開始: {process_name}")
                return True
            else:
                logger.error(f"❌ プロセスファイルが見つかりません: {process_name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ プロセス再起動エラー: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """システムヘルスチェック"""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_health": "healthy",
            "issues": [],
            "recommendations": []
        }
        
        try:
            # システムメトリクスチェック
            metrics = self.get_system_metrics()
            
            if metrics.get("memory_percent", 0) > self.recovery_config["memory_threshold"]:
                health_status["issues"].append("高メモリ使用率")
                health_status["recommendations"].append("メモリ使用量の最適化")
                health_status["overall_health"] = "warning"
            
            if metrics.get("cpu_percent", 0) > self.recovery_config["cpu_threshold"]:
                health_status["issues"].append("高CPU使用率")
                health_status["recommendations"].append("CPU負荷の分散")
                health_status["overall_health"] = "warning"
            
            if metrics.get("disk_percent", 0) > 90:
                health_status["issues"].append("ディスク容量不足")
                health_status["recommendations"].append("不要ファイルの削除")
                health_status["overall_health"] = "critical"
            
            # プロセスチェック
            processes = self.get_process_info()
            active_processes = [p.name for p in processes]
            
            for monitored in self.monitored_processes:
                if monitored not in active_processes:
                    health_status["issues"].append(f"プロセス停止: {monitored}")
                    health_status["recommendations"].append(f"プロセス再起動: {monitored}")
                    if health_status["overall_health"] == "healthy":
                        health_status["overall_health"] = "warning"
            
            # 長時間実行プロセスのチェック
            for process in processes:
                start_time = datetime.fromisoformat(process.start_time)
                runtime = datetime.now() - start_time
                if runtime.total_seconds() > self.recovery_config["process_timeout"]:
                    health_status["issues"].append(f"長時間実行プロセス: {process.name}")
                    health_status["recommendations"].append(f"プロセス状態確認: {process.name}")
            
        except Exception as e:
            logger.error(f"ヘルスチェックエラー: {e}")
            health_status["overall_health"] = "error"
            health_status["issues"].append(f"ヘルスチェックエラー: {e}")
        
        return health_status
    
    def auto_recovery_loop(self):
        """自動リカバリーループ"""
        logger.info("🔄 自動リカバリーループ開始")
        
        while self.is_monitoring:
            try:
                # ヘルスチェック
                health_status = self.health_check()
                
                # 問題がある場合の対処
                if health_status["overall_health"] in ["warning", "critical"]:
                    logger.warning(f"⚠️ システム状態: {health_status['overall_health']}")
                    
                    # 停止プロセスの再起動
                    for issue in health_status["issues"]:
                        if "プロセス停止" in issue:
                            process_name = issue.split(": ")[1]
                            if self.recovery_config["auto_restart"]:
                                self.restart_process(process_name)
                
                # 定期バックアップ
                if (datetime.now() - self.last_backup_time).total_seconds() > self.recovery_config["backup_interval"]:
                    self.create_checkpoint("auto")
                    self.last_backup_time = datetime.now()
                
                # リカバリー状態の保存
                current_state = RecoveryState(
                    timestamp=datetime.now().isoformat(),
                    process_states={p.name: asdict(p) for p in self.get_process_info()},
                    system_metrics=self.get_system_metrics(),
                    checkpoint_info={"last_checkpoint": self.last_backup_time.isoformat()},
                    verification_progress={},
                    last_successful_operation="health_check",
                    recovery_count=0
                )
                self.save_recovery_state(current_state)
                
                # 待機
                time.sleep(self.recovery_config["health_check_interval"])
                
            except Exception as e:
                logger.error(f"❌ 自動リカバリーループエラー: {e}")
                time.sleep(30)  # エラー時は短い間隔で再試行
    
    def start_monitoring(self):
        """監視開始"""
        if self.is_monitoring:
            logger.warning("⚠️ 監視は既に開始されています")
            return
        
        self.is_monitoring = True
        
        # 初期チェックポイント作成
        self.create_checkpoint("startup")
        
        # 監視スレッド開始
        recovery_thread = threading.Thread(target=self.auto_recovery_loop, daemon=True)
        recovery_thread.start()
        self.recovery_threads.append(recovery_thread)
        
        logger.info("🚀 NKAT v11 自動リカバリー監視開始")
        print("🛡️ NKAT v11 自動リカバリーシステム開始")
        print("📊 監視対象プロセス:", self.monitored_processes)
        print("💾 チェックポイント間隔:", f"{self.recovery_config['backup_interval']}秒")
        print("🔍 ヘルスチェック間隔:", f"{self.recovery_config['health_check_interval']}秒")
    
    def stop_monitoring(self):
        """監視停止"""
        if not self.is_monitoring:
            return
        
        logger.info("🛑 自動リカバリー監視停止中...")
        self.is_monitoring = False
        
        # 最終チェックポイント作成
        self.create_checkpoint("shutdown")
        
        # スレッド終了待機
        for thread in self.recovery_threads:
            thread.join(timeout=5)
        
        logger.info("✅ 自動リカバリー監視停止完了")
    
    def get_recovery_status(self) -> Dict[str, Any]:
        """リカバリー状況の取得"""
        status = {
            "monitoring_active": self.is_monitoring,
            "last_backup": self.last_backup_time.isoformat(),
            "health_status": self.health_check(),
            "checkpoint_count": 0,
            "recovery_config": self.recovery_config
        }
        
        # チェックポイント数の取得
        if self.checkpoint_registry.exists():
            try:
                with open(self.checkpoint_registry, 'r', encoding='utf-8') as f:
                    registry = json.load(f)
                status["checkpoint_count"] = len(registry)
            except:
                pass
        
        return status

def main():
    """メイン実行関数"""
    recovery_system = NKATAutoRecoverySystem()
    
    try:
        # 監視開始
        recovery_system.start_monitoring()
        
        # メインループ（Ctrl+Cで終了）
        while True:
            time.sleep(10)
            status = recovery_system.get_recovery_status()
            print(f"\r🛡️ 監視中... ヘルス: {status['health_status']['overall_health']} | "
                  f"チェックポイント: {status['checkpoint_count']}個", end="", flush=True)
            
    except KeyboardInterrupt:
        print("\n🛑 監視停止要求を受信")
    finally:
        recovery_system.stop_monitoring()
        print("✅ NKAT v11 自動リカバリーシステム終了")

if __name__ == "__main__":
    main() 