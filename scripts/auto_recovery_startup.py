#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT理論 RTX3080 電源復旧自動計算再開システム
Windows起動時自動実行対応
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime
import logging

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/auto_recovery.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutoRecoverySystem:
    """電源復旧自動計算再開システム"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.checkpoint_dir = self.project_root / "src" / "rtx3080_extreme_checkpoints"
        self.config_file = self.project_root / "config" / "auto_recovery_config.json"
        self.startup_delay = 30  # 起動後30秒待機
        
    def check_system_ready(self):
        """システム準備完了確認"""
        logger.info("🔍 システム準備状況確認中...")
        
        # GPU確認
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if not gpus:
                logger.error("❌ GPU が検出されません")
                return False
            
            gpu = gpus[0]
            logger.info(f"✅ GPU検出: {gpu.name}")
            logger.info(f"   VRAM: {gpu.memoryUsed}/{gpu.memoryTotal} MB")
            logger.info(f"   温度: {gpu.temperature}°C")
            
        except Exception as e:
            logger.error(f"❌ GPU確認エラー: {e}")
            return False
        
        # CUDA確認
        try:
            import torch
            if not torch.cuda.is_available():
                logger.error("❌ CUDA が利用できません")
                return False
            logger.info(f"✅ CUDA利用可能: {torch.version.cuda}")
        except Exception as e:
            logger.error(f"❌ CUDA確認エラー: {e}")
            return False
        
        # ディレクトリ確認
        required_dirs = [
            self.checkpoint_dir,
            self.project_root / "logs",
            self.project_root / "Results" / "rtx3080_extreme_checkpoints"
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                logger.info(f"📂 ディレクトリ作成: {dir_path}")
                dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ システム準備完了")
        return True
    
    def detect_previous_training(self):
        """前回の学習状況検出"""
        logger.info("🔍 前回の学習状況検出中...")
        
        if not self.checkpoint_dir.exists():
            logger.info("📭 チェックポイントディレクトリが存在しません")
            return None
        
        # 最新のメタデータファイル検索
        metadata_files = list(self.checkpoint_dir.glob("metadata_*.json"))
        if not metadata_files:
            logger.info("📭 メタデータファイルが見つかりません")
            return None
        
        latest_metadata = max(metadata_files, key=lambda x: x.stat().st_mtime)
        
        try:
            with open(latest_metadata, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            logger.info(f"📄 前回の学習状況検出:")
            logger.info(f"   エポック: {metadata.get('epoch', 'N/A')}")
            logger.info(f"   損失: {metadata.get('loss', 'N/A')}")
            logger.info(f"   最終更新: {datetime.fromtimestamp(latest_metadata.stat().st_mtime)}")
            
            return metadata
            
        except Exception as e:
            logger.error(f"❌ メタデータ読み込みエラー: {e}")
            return None
    
    def check_running_processes(self):
        """実行中プロセス確認"""
        try:
            import psutil
            
            nkat_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info['cmdline']
                    if cmdline and any('nkat' in arg.lower() or 'rtx3080' in arg.lower() for arg in cmdline):
                        nkat_processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            if nkat_processes:
                logger.info(f"🔄 NKAT関連プロセス {len(nkat_processes)}個が実行中")
                for proc in nkat_processes:
                    logger.info(f"   PID {proc['pid']}: {' '.join(proc['cmdline'][:3])}...")
                return True
            else:
                logger.info("🔴 NKAT関連プロセスは実行されていません")
                return False
                
        except ImportError:
            logger.warning("⚠️ psutil がインストールされていません")
            return False
        except Exception as e:
            logger.error(f"❌ プロセス確認エラー: {e}")
            return False
    
    def start_training_with_recovery(self, metadata=None):
        """リカバリー付き学習開始"""
        logger.info("🚀 学習プロセス開始中...")
        
        # 学習スクリプトパス
        training_script = self.project_root / "scripts" / "run_rtx3080_training.py"
        
        if not training_script.exists():
            logger.error(f"❌ 学習スクリプトが見つかりません: {training_script}")
            return False
        
        # 実行コマンド構築
        cmd = [
            sys.executable,
            str(training_script),
            "--mode", "both",  # 学習+ダッシュボード
            "--auto-recovery"  # 自動リカバリーモード
        ]
        
        if metadata:
            logger.info(f"🔄 エポック {metadata.get('epoch', 0)} から学習再開")
        
        try:
            # バックグラウンドで実行
            process = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
            )
            
            logger.info(f"✅ 学習プロセス開始 (PID: {process.pid})")
            logger.info("📊 ダッシュボード: http://localhost:8501")
            
            # プロセス情報保存
            process_info = {
                "pid": process.pid,
                "start_time": datetime.now().isoformat(),
                "command": cmd,
                "recovery_mode": True
            }
            
            with open(self.project_root / "logs" / "auto_recovery_process.json", 'w', encoding='utf-8') as f:
                json.dump(process_info, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 学習プロセス開始エラー: {e}")
            return False
    
    def create_startup_config(self):
        """スタートアップ設定作成"""
        config = {
            "auto_recovery_enabled": True,
            "startup_delay_seconds": self.startup_delay,
            "max_recovery_attempts": 3,
            "dashboard_auto_start": True,
            "gpu_temperature_threshold": 80,
            "vram_usage_threshold": 90,
            "created": datetime.now().isoformat()
        }
        
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📝 自動リカバリー設定作成: {self.config_file}")
        return config
    
    def run_auto_recovery(self):
        """自動リカバリー実行"""
        logger.info("🚀 NKAT理論 電源復旧自動計算再開システム開始")
        logger.info(f"⏰ 起動後 {self.startup_delay} 秒待機中...")
        
        # 起動待機
        time.sleep(self.startup_delay)
        
        # システム準備確認
        if not self.check_system_ready():
            logger.error("❌ システム準備未完了のため終了")
            return False
        
        # 既存プロセス確認
        if self.check_running_processes():
            logger.info("✅ 学習プロセスが既に実行中です")
            return True
        
        # 前回の学習状況検出
        metadata = self.detect_previous_training()
        
        # 設定ファイル確認/作成
        if not self.config_file.exists():
            self.create_startup_config()
        
        # 学習再開
        success = self.start_training_with_recovery(metadata)
        
        if success:
            logger.info("🎉 自動リカバリー完了！")
            logger.info("📊 ブラウザで http://localhost:8501 にアクセスしてください")
        else:
            logger.error("❌ 自動リカバリー失敗")
        
        return success

def main():
    """メイン関数"""
    recovery_system = AutoRecoverySystem()
    
    # コマンドライン引数処理
    if len(sys.argv) > 1:
        if sys.argv[1] == "--setup-startup":
            # Windowsスタートアップ設定
            setup_windows_startup()
            return
        elif sys.argv[1] == "--check-only":
            # 確認のみ
            recovery_system.check_system_ready()
            recovery_system.detect_previous_training()
            recovery_system.check_running_processes()
            return
    
    # 自動リカバリー実行
    recovery_system.run_auto_recovery()

def setup_windows_startup():
    """Windowsスタートアップ設定"""
    logger.info("🔧 Windowsスタートアップ設定中...")
    
    try:
        import winreg
        
        # スタートアップレジストリキー
        key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
        
        # 現在のスクリプトパス
        script_path = Path(__file__).absolute()
        python_exe = sys.executable
        
        # 実行コマンド
        command = f'"{python_exe}" "{script_path}"'
        
        # レジストリに登録
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE) as key:
            winreg.SetValueEx(key, "NKAT_AutoRecovery", 0, winreg.REG_SZ, command)
        
        logger.info("✅ Windowsスタートアップ設定完了")
        logger.info(f"   実行コマンド: {command}")
        logger.info("   次回Windows起動時から自動実行されます")
        
        # バッチファイル作成（代替手段）
        batch_file = Path(__file__).parent / "nkat_auto_recovery.bat"
        with open(batch_file, 'w', encoding='utf-8') as f:
            f.write(f'@echo off\n')
            f.write(f'cd /d "{Path(__file__).parent.parent}"\n')
            f.write(f'"{python_exe}" "{script_path}"\n')
            f.write(f'pause\n')
        
        logger.info(f"📄 バッチファイル作成: {batch_file}")
        
    except ImportError:
        logger.error("❌ winreg モジュールが利用できません（Windows以外の環境）")
    except Exception as e:
        logger.error(f"❌ スタートアップ設定エラー: {e}")

if __name__ == "__main__":
    main() 