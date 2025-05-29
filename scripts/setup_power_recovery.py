#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT理論 RTX3080 電源復旧システム一括セットアップ
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
import logging

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PowerRecoverySetup:
    """電源復旧システムセットアップ"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.scripts_dir = self.project_root / "scripts"
        self.config_dir = self.project_root / "config"
        self.logs_dir = self.project_root / "logs"
        
    def create_directories(self):
        """必要ディレクトリ作成"""
        logger.info("📂 ディレクトリ構造作成中...")
        
        directories = [
            self.config_dir,
            self.logs_dir,
            self.project_root / "src" / "rtx3080_extreme_checkpoints",
            self.project_root / "Results" / "rtx3080_extreme_checkpoints"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"   ✅ {directory}")
    
    def create_startup_batch_file(self):
        """Windows起動用バッチファイル作成"""
        logger.info("📄 Windows起動用バッチファイル作成中...")
        
        batch_file = self.scripts_dir / "nkat_auto_recovery.bat"
        python_exe = sys.executable
        auto_recovery_script = self.scripts_dir / "auto_recovery_startup.py"
        
        batch_content = f'''@echo off
title NKAT理論 RTX3080 自動復旧システム
echo 🚀 NKAT理論 RTX3080 自動復旧システム起動中...
echo.

REM プロジェクトディレクトリに移動
cd /d "{self.project_root}"

REM Python環境確認
echo 🔍 Python環境確認中...
"{python_exe}" --version
if errorlevel 1 (
    echo ❌ Python が見つかりません
    pause
    exit /b 1
)

REM GPU確認
echo 🎮 GPU確認中...
"{python_exe}" -c "import torch; print('CUDA利用可能:', torch.cuda.is_available())"
if errorlevel 1 (
    echo ⚠️ GPU確認でエラーが発生しました
)

REM 自動復旧スクリプト実行
echo 🚀 自動復旧スクリプト実行中...
"{python_exe}" "{auto_recovery_script}"

echo.
echo 📊 ダッシュボードアクセス: http://localhost:8501
echo 🛑 終了するには何かキーを押してください
pause
'''
        
        with open(batch_file, 'w', encoding='utf-8') as f:
            f.write(batch_content)
        
        logger.info(f"   ✅ {batch_file}")
        return batch_file
    
    def create_quick_recovery_batch(self):
        """クイック復旧用バッチファイル作成"""
        logger.info("⚡ クイック復旧用バッチファイル作成中...")
        
        batch_file = self.scripts_dir / "quick_recovery.bat"
        python_exe = sys.executable
        quick_recovery_script = self.scripts_dir / "quick_recovery.py"
        
        batch_content = f'''@echo off
title NKAT理論 クイック復旧
echo ⚡ NKAT理論 クイック復旧開始
echo.

cd /d "{self.project_root}"

echo 🚀 復旧スクリプト実行中...
"{python_exe}" "{quick_recovery_script}"

echo.
echo ✅ 復旧完了
echo 📊 ダッシュボード: http://localhost:8501
echo.
pause
'''
        
        with open(batch_file, 'w', encoding='utf-8') as f:
            f.write(batch_content)
        
        logger.info(f"   ✅ {batch_file}")
        return batch_file
    
    def setup_windows_startup(self):
        """Windowsスタートアップ設定"""
        logger.info("🔧 Windowsスタートアップ設定中...")
        
        try:
            import winreg
            
            # スタートアップレジストリキー
            key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
            
            # バッチファイルパス
            batch_file = self.scripts_dir / "nkat_auto_recovery.bat"
            
            # レジストリに登録
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE) as key:
                winreg.SetValueEx(key, "NKAT_AutoRecovery", 0, winreg.REG_SZ, str(batch_file))
            
            logger.info("   ✅ レジストリ登録完了")
            logger.info(f"   📄 実行ファイル: {batch_file}")
            
            return True
            
        except ImportError:
            logger.error("❌ winreg モジュールが利用できません（Windows以外の環境）")
            return False
        except Exception as e:
            logger.error(f"❌ スタートアップ設定エラー: {e}")
            return False
    
    def create_config_files(self):
        """設定ファイル作成"""
        logger.info("⚙️ 設定ファイル作成中...")
        
        # 自動復旧設定
        auto_recovery_config = {
            "auto_recovery_enabled": True,
            "startup_delay_seconds": 30,
            "max_recovery_attempts": 3,
            "dashboard_auto_start": True,
            "gpu_temperature_threshold": 80,
            "vram_usage_threshold": 90,
            "monitoring_interval_seconds": 30,
            "emergency_shutdown_timeout": 10,
            "created": datetime.now().isoformat(),
            "version": "1.0.0"
        }
        
        config_file = self.config_dir / "auto_recovery_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(auto_recovery_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"   ✅ {config_file}")
        
        # 電源管理設定
        power_config = {
            "power_monitoring_enabled": True,
            "battery_threshold_percent": 20,
            "ups_monitoring_enabled": False,
            "safe_shutdown_delay_seconds": 60,
            "checkpoint_save_interval_seconds": 300,
            "system_health_check_interval": 30
        }
        
        power_config_file = self.config_dir / "power_management_config.json"
        with open(power_config_file, 'w', encoding='utf-8') as f:
            json.dump(power_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"   ✅ {power_config_file}")
        
        return auto_recovery_config, power_config
    
    def create_desktop_shortcuts(self):
        """デスクトップショートカット作成"""
        logger.info("🖥️ デスクトップショートカット作成中...")
        
        try:
            import winshell
            from win32com.client import Dispatch
            
            desktop = winshell.desktop()
            
            # 自動復旧ショートカット
            auto_recovery_shortcut = Path(desktop) / "NKAT自動復旧.lnk"
            shell = Dispatch('WScript.Shell')
            shortcut = shell.CreateShortCut(str(auto_recovery_shortcut))
            shortcut.Targetpath = str(self.scripts_dir / "nkat_auto_recovery.bat")
            shortcut.WorkingDirectory = str(self.project_root)
            shortcut.IconLocation = str(self.scripts_dir / "nkat_auto_recovery.bat")
            shortcut.save()
            
            logger.info(f"   ✅ {auto_recovery_shortcut}")
            
            # クイック復旧ショートカット
            quick_recovery_shortcut = Path(desktop) / "NKATクイック復旧.lnk"
            shortcut = shell.CreateShortCut(str(quick_recovery_shortcut))
            shortcut.Targetpath = str(self.scripts_dir / "quick_recovery.bat")
            shortcut.WorkingDirectory = str(self.project_root)
            shortcut.IconLocation = str(self.scripts_dir / "quick_recovery.bat")
            shortcut.save()
            
            logger.info(f"   ✅ {quick_recovery_shortcut}")
            
            return True
            
        except ImportError:
            logger.warning("⚠️ winshell または win32com が利用できません")
            logger.info("   手動でショートカットを作成してください")
            return False
        except Exception as e:
            logger.error(f"❌ ショートカット作成エラー: {e}")
            return False
    
    def test_recovery_system(self):
        """復旧システムテスト"""
        logger.info("🧪 復旧システムテスト中...")
        
        try:
            # GPU確認
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                logger.info(f"   ✅ GPU検出: {gpu.name}")
            else:
                logger.warning("   ⚠️ GPU が検出されません")
            
            # CUDA確認
            import torch
            if torch.cuda.is_available():
                logger.info(f"   ✅ CUDA利用可能: {torch.version.cuda}")
            else:
                logger.warning("   ⚠️ CUDA が利用できません")
            
            # チェックポイントディレクトリ確認
            checkpoint_dir = self.project_root / "src" / "rtx3080_extreme_checkpoints"
            if checkpoint_dir.exists():
                checkpoint_files = list(checkpoint_dir.glob("*.pth"))
                logger.info(f"   ✅ チェックポイント: {len(checkpoint_files)}個")
            else:
                logger.info("   📭 チェックポイントなし（初回実行）")
            
            logger.info("   ✅ システムテスト完了")
            return True
            
        except Exception as e:
            logger.error(f"❌ システムテストエラー: {e}")
            return False
    
    def create_usage_guide(self):
        """使用方法ガイド作成"""
        logger.info("📖 使用方法ガイド作成中...")
        
        guide_content = """# 🚀 NKAT理論 RTX3080 電源復旧システム 使用方法

## 📋 セットアップ完了項目

✅ 自動復旧スクリプト
✅ Windowsスタートアップ登録
✅ バッチファイル作成
✅ 設定ファイル生成
✅ デスクトップショートカット

## 🔄 電源復旧の流れ

### 1. 自動復旧（推奨）
- Windows起動時に自動実行
- 30秒待機後、前回の学習状況を検出
- 自動的に学習を再開

### 2. 手動復旧
- デスクトップの「NKATクイック復旧」をダブルクリック
- または以下のコマンドを実行:
  ```
  py -3 scripts/quick_recovery.py
  ```

### 3. 完全手動復旧
- 以下のコマンドで学習+ダッシュボード起動:
  ```
  py -3 scripts/run_rtx3080_training.py --mode both
  ```

## 📊 監視とアクセス

### ダッシュボード
- URL: http://localhost:8501
- リアルタイム学習状況監視
- GPU使用量・温度確認

### ログファイル
- 自動復旧ログ: logs/auto_recovery.log
- 電源管理ログ: logs/power_recovery.log
- 学習ログ: logs/rtx3080_training/

## ⚙️ 設定変更

### 自動復旧設定
ファイル: config/auto_recovery_config.json
- startup_delay_seconds: 起動待機時間
- gpu_temperature_threshold: GPU温度閾値
- vram_usage_threshold: VRAM使用量閾値

### 電源管理設定
ファイル: config/power_management_config.json
- safe_shutdown_delay_seconds: 安全シャットダウン待機時間
- checkpoint_save_interval_seconds: チェックポイント保存間隔

## 🛠️ トラブルシューティング

### 自動復旧が動作しない
1. Windowsスタートアップ確認:
   ```
   py -3 scripts/auto_recovery_startup.py --setup-startup
   ```

2. 手動テスト:
   ```
   py -3 scripts/auto_recovery_startup.py --check-only
   ```

### 学習が再開されない
1. チェックポイント確認:
   ```
   py -3 check_training_status.py
   ```

2. GPU状況確認:
   ```
   py -3 -c "import GPUtil; print(GPUtil.getGPUs()[0].name)"
   ```

## 📞 サポート

問題が発生した場合は、以下のログを確認してください:
- logs/auto_recovery.log
- logs/power_recovery.log
- logs/rtx3080_training/training.log

---
NKAT理論 RTX3080 電源復旧システム v1.0.0
"""
        
        guide_file = self.project_root / "POWER_RECOVERY_GUIDE.md"
        with open(guide_file, 'w', encoding='utf-8') as f:
            f.write(guide_content)
        
        logger.info(f"   ✅ {guide_file}")
        return guide_file
    
    def run_setup(self):
        """セットアップ実行"""
        logger.info("🚀 NKAT理論 RTX3080 電源復旧システム セットアップ開始")
        logger.info("=" * 60)
        
        # ディレクトリ作成
        self.create_directories()
        
        # バッチファイル作成
        self.create_startup_batch_file()
        self.create_quick_recovery_batch()
        
        # 設定ファイル作成
        self.create_config_files()
        
        # Windowsスタートアップ設定
        startup_success = self.setup_windows_startup()
        
        # デスクトップショートカット作成
        shortcut_success = self.create_desktop_shortcuts()
        
        # システムテスト
        test_success = self.test_recovery_system()
        
        # 使用方法ガイド作成
        self.create_usage_guide()
        
        # セットアップ完了レポート
        logger.info("=" * 60)
        logger.info("🎉 セットアップ完了！")
        logger.info("")
        logger.info("📋 セットアップ結果:")
        logger.info(f"   ✅ ディレクトリ構造: 完了")
        logger.info(f"   ✅ バッチファイル: 完了")
        logger.info(f"   ✅ 設定ファイル: 完了")
        logger.info(f"   {'✅' if startup_success else '❌'} Windowsスタートアップ: {'完了' if startup_success else '失敗'}")
        logger.info(f"   {'✅' if shortcut_success else '⚠️'} デスクトップショートカット: {'完了' if shortcut_success else '手動作成が必要'}")
        logger.info(f"   {'✅' if test_success else '❌'} システムテスト: {'完了' if test_success else '要確認'}")
        logger.info("")
        logger.info("🔄 電源復旧機能:")
        logger.info("   • Windows起動時に自動実行")
        logger.info("   • 前回の学習状況を自動検出")
        logger.info("   • 学習を自動再開")
        logger.info("")
        logger.info("📊 ダッシュボード: http://localhost:8501")
        logger.info("📖 使用方法: POWER_RECOVERY_GUIDE.md を参照")
        logger.info("")
        logger.info("⚡ クイック復旧テスト:")
        logger.info("   py -3 scripts/quick_recovery.py")

def main():
    """メイン関数"""
    setup = PowerRecoverySetup()
    setup.run_setup()

if __name__ == "__main__":
    main() 