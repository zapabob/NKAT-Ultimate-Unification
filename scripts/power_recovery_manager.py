#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT理論 RTX3080 電源断検出・復旧管理システム
"""

import os
import sys
import time
import json
import signal
import threading
from pathlib import Path
from datetime import datetime
import logging

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/power_recovery.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PowerRecoveryManager:
    """電源断検出・復旧管理システム"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.checkpoint_dir = self.project_root / "src" / "rtx3080_extreme_checkpoints"
        self.recovery_state_file = self.project_root / "logs" / "recovery_state.json"
        self.is_shutdown_requested = False
        self.monitoring_thread = None
        
        # シグナルハンドラー設定
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        if hasattr(signal, 'SIGBREAK'):  # Windows
            signal.signal(signal.SIGBREAK, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """シグナルハンドラー（緊急シャットダウン）"""
        logger.warning(f"⚠️ シグナル {signum} 受信 - 緊急シャットダウン開始")
        self.emergency_shutdown()
    
    def save_recovery_state(self, training_state=None):
        """復旧用状態保存"""
        recovery_state = {
            "timestamp": datetime.now().isoformat(),
            "shutdown_type": "planned" if not self.is_shutdown_requested else "emergency",
            "training_active": training_state is not None,
            "gpu_info": self.get_gpu_info(),
            "system_info": self.get_system_info(),
            "checkpoint_info": self.get_latest_checkpoint_info()
        }
        
        if training_state:
            recovery_state["training_state"] = training_state
        
        # ディレクトリ作成
        self.recovery_state_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 状態保存
        with open(self.recovery_state_file, 'w', encoding='utf-8') as f:
            json.dump(recovery_state, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 復旧状態保存: {self.recovery_state_file}")
        return recovery_state
    
    def load_recovery_state(self):
        """復旧用状態読み込み"""
        if not self.recovery_state_file.exists():
            logger.info("📭 復旧状態ファイルが存在しません")
            return None
        
        try:
            with open(self.recovery_state_file, 'r', encoding='utf-8') as f:
                recovery_state = json.load(f)
            
            logger.info("📄 復旧状態読み込み完了")
            logger.info(f"   前回シャットダウン: {recovery_state.get('timestamp', 'N/A')}")
            logger.info(f"   シャットダウン種別: {recovery_state.get('shutdown_type', 'N/A')}")
            logger.info(f"   学習状態: {'有効' if recovery_state.get('training_active') else '無効'}")
            
            return recovery_state
            
        except Exception as e:
            logger.error(f"❌ 復旧状態読み込みエラー: {e}")
            return None
    
    def get_gpu_info(self):
        """GPU情報取得"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                return {
                    "name": gpu.name,
                    "memory_used": gpu.memoryUsed,
                    "memory_total": gpu.memoryTotal,
                    "temperature": gpu.temperature,
                    "load": gpu.load
                }
        except:
            pass
        return None
    
    def get_system_info(self):
        """システム情報取得"""
        try:
            import psutil
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('.').percent
            }
        except:
            pass
        return None
    
    def get_latest_checkpoint_info(self):
        """最新チェックポイント情報取得"""
        if not self.checkpoint_dir.exists():
            return None
        
        try:
            # 最新のメタデータファイル検索
            metadata_files = list(self.checkpoint_dir.glob("metadata_*.json"))
            if not metadata_files:
                return None
            
            latest_metadata = max(metadata_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_metadata, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            return {
                "file": latest_metadata.name,
                "epoch": metadata.get('epoch'),
                "loss": metadata.get('loss'),
                "timestamp": datetime.fromtimestamp(latest_metadata.stat().st_mtime).isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ チェックポイント情報取得エラー: {e}")
            return None
    
    def monitor_system_health(self):
        """システム健全性監視"""
        logger.info("🔍 システム健全性監視開始")
        
        while not self.is_shutdown_requested:
            try:
                # GPU温度監視
                gpu_info = self.get_gpu_info()
                if gpu_info and gpu_info.get('temperature', 0) > 85:
                    logger.warning(f"⚠️ GPU温度警告: {gpu_info['temperature']}°C")
                
                # VRAM使用量監視
                if gpu_info:
                    vram_usage = (gpu_info['memory_used'] / gpu_info['memory_total']) * 100
                    if vram_usage > 95:
                        logger.warning(f"⚠️ VRAM使用量警告: {vram_usage:.1f}%")
                
                # システムメモリ監視
                system_info = self.get_system_info()
                if system_info and system_info.get('memory_percent', 0) > 90:
                    logger.warning(f"⚠️ システムメモリ警告: {system_info['memory_percent']}%")
                
                time.sleep(30)  # 30秒間隔で監視
                
            except Exception as e:
                logger.error(f"❌ システム監視エラー: {e}")
                time.sleep(60)  # エラー時は1分待機
    
    def emergency_shutdown(self):
        """緊急シャットダウン"""
        logger.warning("🚨 緊急シャットダウン開始")
        self.is_shutdown_requested = True
        
        # 現在の学習状態を保存
        training_state = self.get_current_training_state()
        self.save_recovery_state(training_state)
        
        # 学習プロセス終了
        self.terminate_training_processes()
        
        logger.info("✅ 緊急シャットダウン完了")
    
    def get_current_training_state(self):
        """現在の学習状態取得"""
        try:
            # 最新のメタデータから学習状態を取得
            checkpoint_info = self.get_latest_checkpoint_info()
            if checkpoint_info:
                return {
                    "last_epoch": checkpoint_info.get('epoch'),
                    "last_loss": checkpoint_info.get('loss'),
                    "checkpoint_file": checkpoint_info.get('file')
                }
        except Exception as e:
            logger.error(f"❌ 学習状態取得エラー: {e}")
        
        return None
    
    def terminate_training_processes(self):
        """学習プロセス終了"""
        try:
            import psutil
            
            terminated_count = 0
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info['cmdline']
                    if cmdline and any('nkat' in arg.lower() or 'rtx3080' in arg.lower() for arg in cmdline):
                        logger.info(f"🔄 プロセス終了: PID {proc.info['pid']}")
                        proc.terminate()
                        terminated_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            if terminated_count > 0:
                logger.info(f"✅ {terminated_count}個のプロセスを終了しました")
                time.sleep(5)  # プロセス終了待機
            
        except Exception as e:
            logger.error(f"❌ プロセス終了エラー: {e}")
    
    def start_monitoring(self):
        """監視開始"""
        logger.info("🚀 電源断検出・復旧管理システム開始")
        
        # 前回の復旧状態確認
        recovery_state = self.load_recovery_state()
        if recovery_state:
            if recovery_state.get('shutdown_type') == 'emergency':
                logger.warning("⚠️ 前回は緊急シャットダウンでした")
            
            if recovery_state.get('training_active'):
                logger.info("🔄 前回は学習が実行中でした - 自動復旧を推奨")
        
        # システム健全性監視開始
        self.monitoring_thread = threading.Thread(target=self.monitor_system_health, daemon=True)
        self.monitoring_thread.start()
        
        # メインループ
        try:
            while not self.is_shutdown_requested:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("🛑 ユーザーによる停止要求")
            self.emergency_shutdown()
    
    def create_recovery_script(self):
        """復旧スクリプト作成"""
        recovery_script = self.project_root / "scripts" / "quick_recovery.py"
        
        script_content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT理論 クイック復旧スクリプト
電源復旧後の即座実行用
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.auto_recovery_startup import AutoRecoverySystem

def main():
    print("🚀 NKAT理論 クイック復旧開始")
    
    recovery_system = AutoRecoverySystem()
    recovery_system.startup_delay = 5  # 短縮待機時間
    
    success = recovery_system.run_auto_recovery()
    
    if success:
        print("✅ 復旧完了！")
        print("📊 ダッシュボード: http://localhost:8501")
    else:
        print("❌ 復旧失敗")
        print("手動で確認してください")

if __name__ == "__main__":
    main()
'''
        
        with open(recovery_script, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        logger.info(f"📄 クイック復旧スクリプト作成: {recovery_script}")
        return recovery_script

def main():
    """メイン関数"""
    manager = PowerRecoveryManager()
    
    # コマンドライン引数処理
    if len(sys.argv) > 1:
        if sys.argv[1] == "--create-recovery-script":
            manager.create_recovery_script()
            return
        elif sys.argv[1] == "--check-state":
            manager.load_recovery_state()
            return
        elif sys.argv[1] == "--emergency-shutdown":
            manager.emergency_shutdown()
            return
    
    # 監視開始
    manager.start_monitoring()

if __name__ == "__main__":
    main() 