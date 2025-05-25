#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 NKAT v11 統合ランチャー - 包括的システム起動
NKAT v11 Integrated Launcher - Comprehensive System Startup

Author: NKAT Research Consortium
Date: 2025-05-26
Version: 11.0 - Integrated Launcher
"""

import os
import sys
import time
import subprocess
import threading
import webbrowser
from pathlib import Path
from datetime import datetime
import logging
import json
from typing import Dict, List, Optional

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nkat_v11_launcher.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NKATIntegratedLauncher:
    """NKAT v11 統合ランチャー"""
    
    def __init__(self):
        self.base_path = Path(".")
        self.processes = {}
        self.launch_config = {
            "auto_recovery": True,
            "streamlit_dashboard": True,
            "detailed_analysis": True,
            "verification_systems": True,
            "browser_auto_open": True,
            "streamlit_port": 8501
        }
        
        # 起動対象システム
        self.systems = {
            "recovery": {
                "script": "nkat_v11_auto_recovery_system.py",
                "description": "自動リカバリーシステム",
                "priority": 1,
                "required": True
            },
            "dashboard": {
                "script": "nkat_v11_comprehensive_recovery_dashboard.py",
                "description": "包括的リカバリーダッシュボード",
                "priority": 2,
                "required": True,
                "streamlit": True
            },
            "analysis": {
                "script": "nkat_v11_detailed_convergence_analyzer.py",
                "description": "詳細収束分析システム",
                "priority": 3,
                "required": False
            },
            "verification": {
                "script": "nkat_v11_rigorous_mathematical_verification.py",
                "description": "厳密数学検証システム",
                "priority": 4,
                "required": False
            },
            "large_scale": {
                "script": "nkat_v11_enhanced_large_scale_verification.py",
                "description": "大規模検証システム",
                "priority": 5,
                "required": False
            }
        }
        
        logger.info("🚀 NKAT v11 統合ランチャー初期化完了")
    
    def check_dependencies(self) -> Dict[str, bool]:
        """依存関係チェック"""
        logger.info("🔍 依存関係チェック開始...")
        
        dependencies = {
            "python": True,
            "streamlit": False,
            "torch": False,
            "numpy": False,
            "matplotlib": False,
            "psutil": False
        }
        
        # Python基本モジュール
        try:
            import streamlit
            dependencies["streamlit"] = True
        except ImportError:
            logger.warning("⚠️ Streamlitが見つかりません")
        
        try:
            import torch
            dependencies["torch"] = True
        except ImportError:
            logger.warning("⚠️ PyTorchが見つかりません")
        
        try:
            import numpy
            dependencies["numpy"] = True
        except ImportError:
            logger.warning("⚠️ NumPyが見つかりません")
        
        try:
            import matplotlib
            dependencies["matplotlib"] = True
        except ImportError:
            logger.warning("⚠️ Matplotlibが見つかりません")
        
        try:
            import psutil
            dependencies["psutil"] = True
        except ImportError:
            logger.warning("⚠️ psutilが見つかりません")
        
        # ファイル存在チェック
        missing_files = []
        for system_name, system_info in self.systems.items():
            script_path = self.base_path / system_info["script"]
            if not script_path.exists():
                missing_files.append(system_info["script"])
                logger.warning(f"⚠️ スクリプトが見つかりません: {system_info['script']}")
        
        if missing_files:
            dependencies["scripts"] = False
            logger.error(f"❌ 不足スクリプト: {missing_files}")
        else:
            dependencies["scripts"] = True
        
        logger.info(f"✅ 依存関係チェック完了: {dependencies}")
        return dependencies
    
    def install_missing_dependencies(self):
        """不足依存関係のインストール"""
        logger.info("📦 不足依存関係のインストール開始...")
        
        required_packages = [
            "streamlit",
            "torch",
            "numpy",
            "matplotlib",
            "psutil",
            "pandas",
            "plotly",
            "scipy",
            "seaborn"
        ]
        
        for package in required_packages:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", package], 
                             check=True, capture_output=True)
                logger.info(f"✅ インストール完了: {package}")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ インストール失敗: {package} - {e}")
    
    def launch_system(self, system_name: str, system_info: Dict) -> Optional[subprocess.Popen]:
        """個別システムの起動"""
        try:
            script_path = self.base_path / system_info["script"]
            if not script_path.exists():
                logger.error(f"❌ スクリプトが見つかりません: {system_info['script']}")
                return None
            
            logger.info(f"🚀 起動中: {system_info['description']}")
            
            if system_info.get("streamlit", False):
                # Streamlitアプリケーションの起動
                cmd = [sys.executable, "-m", "streamlit", "run", system_info["script"], 
                       "--server.port", str(self.launch_config["streamlit_port"])]
                process = subprocess.Popen(cmd, cwd=self.base_path)
                
                # ブラウザ自動オープン
                if self.launch_config["browser_auto_open"]:
                    time.sleep(3)  # Streamlitの起動を待機
                    webbrowser.open(f"http://localhost:{self.launch_config['streamlit_port']}")
            else:
                # 通常のPythonスクリプトの起動
                cmd = [sys.executable, system_info["script"]]
                process = subprocess.Popen(cmd, cwd=self.base_path)
            
            self.processes[system_name] = process
            logger.info(f"✅ 起動完了: {system_info['description']} (PID: {process.pid})")
            return process
            
        except Exception as e:
            logger.error(f"❌ 起動エラー: {system_info['description']} - {e}")
            return None
    
    def launch_all_systems(self):
        """全システムの起動"""
        logger.info("🚀 NKAT v11 全システム起動開始...")
        print("=" * 80)
        print("🚀 NKAT v11 統合システム起動")
        print("=" * 80)
        print(f"📅 起動時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🔬 電源断対応リカバリーシステム統合版")
        print("=" * 80)
        
        # 依存関係チェック
        dependencies = self.check_dependencies()
        if not all(dependencies.values()):
            print("⚠️ 依存関係に問題があります。インストールを実行しますか？ (y/n)")
            if input().lower() == 'y':
                self.install_missing_dependencies()
        
        # 優先順位順にシステムを起動
        sorted_systems = sorted(self.systems.items(), key=lambda x: x[1]["priority"])
        
        for system_name, system_info in sorted_systems:
            if system_info["required"] or self.launch_config.get(system_name.replace("_", ""), True):
                print(f"\n🚀 {system_info['description']} を起動中...")
                process = self.launch_system(system_name, system_info)
                
                if process:
                    print(f"✅ 起動成功: PID {process.pid}")
                    time.sleep(2)  # 起動間隔
                else:
                    print(f"❌ 起動失敗: {system_info['description']}")
                    if system_info["required"]:
                        print("⚠️ 必須システムの起動に失敗しました")
        
        print("\n" + "=" * 80)
        print("🎉 NKAT v11 統合システム起動完了！")
        print("=" * 80)
        
        # 起動状況表示
        self.display_system_status()
        
        # 使用方法の表示
        self.display_usage_instructions()
    
    def display_system_status(self):
        """システム状況表示"""
        print("\n📊 システム状況:")
        print("-" * 50)
        
        for system_name, process in self.processes.items():
            system_info = self.systems[system_name]
            if process and process.poll() is None:
                status = "🟢 実行中"
                pid_info = f"(PID: {process.pid})"
            else:
                status = "🔴 停止"
                pid_info = ""
            
            print(f"{status} {system_info['description']} {pid_info}")
        
        print("-" * 50)
    
    def display_usage_instructions(self):
        """使用方法の表示"""
        print("\n📖 使用方法:")
        print("-" * 50)
        print("🌐 ダッシュボード: http://localhost:8501")
        print("📊 リアルタイム監視とリカバリー機能が利用可能")
        print("🛡️ 電源断時は自動的にバックアップが作成されます")
        print("🔄 プロセス停止時は自動再起動されます")
        print("\n⌨️ 制御コマンド:")
        print("  Ctrl+C: 全システム停止")
        print("  's': システム状況表示")
        print("  'r': システム再起動")
        print("  'q': 終了")
        print("-" * 50)
    
    def monitor_systems(self):
        """システム監視"""
        logger.info("👁️ システム監視開始")
        
        try:
            while True:
                command = input("\n> ").strip().lower()
                
                if command == 'q':
                    break
                elif command == 's':
                    self.display_system_status()
                elif command == 'r':
                    self.restart_failed_systems()
                elif command == 'help':
                    self.display_usage_instructions()
                else:
                    print("❓ 不明なコマンド。'help'で使用方法を確認してください。")
                    
        except KeyboardInterrupt:
            print("\n🛑 停止要求を受信")
        finally:
            self.shutdown_all_systems()
    
    def restart_failed_systems(self):
        """失敗したシステムの再起動"""
        logger.info("🔄 失敗システムの再起動開始...")
        
        for system_name, process in list(self.processes.items()):
            if process.poll() is not None:  # プロセスが終了している
                system_info = self.systems[system_name]
                print(f"🔄 再起動中: {system_info['description']}")
                new_process = self.launch_system(system_name, system_info)
                if new_process:
                    print(f"✅ 再起動成功: {system_info['description']}")
                else:
                    print(f"❌ 再起動失敗: {system_info['description']}")
    
    def shutdown_all_systems(self):
        """全システムの停止"""
        logger.info("🛑 全システム停止開始...")
        print("\n🛑 全システムを停止しています...")
        
        for system_name, process in self.processes.items():
            if process and process.poll() is None:
                system_info = self.systems[system_name]
                print(f"🛑 停止中: {system_info['description']}")
                
                try:
                    process.terminate()
                    process.wait(timeout=10)
                    print(f"✅ 停止完了: {system_info['description']}")
                except subprocess.TimeoutExpired:
                    print(f"⚠️ 強制終了: {system_info['description']}")
                    process.kill()
                except Exception as e:
                    print(f"❌ 停止エラー: {system_info['description']} - {e}")
        
        print("✅ 全システム停止完了")
        logger.info("✅ 全システム停止完了")
    
    def create_startup_summary(self):
        """起動サマリーの作成"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "launched_systems": [],
            "failed_systems": [],
            "configuration": self.launch_config
        }
        
        for system_name, process in self.processes.items():
            system_info = self.systems[system_name]
            if process and process.poll() is None:
                summary["launched_systems"].append({
                    "name": system_name,
                    "description": system_info["description"],
                    "pid": process.pid,
                    "script": system_info["script"]
                })
            else:
                summary["failed_systems"].append({
                    "name": system_name,
                    "description": system_info["description"],
                    "script": system_info["script"]
                })
        
        # サマリー保存
        summary_file = Path("nkat_v11_startup_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 起動サマリー保存: {summary_file}")
        return summary

def main():
    """メイン実行関数"""
    launcher = NKATIntegratedLauncher()
    
    try:
        # 全システム起動
        launcher.launch_all_systems()
        
        # 起動サマリー作成
        launcher.create_startup_summary()
        
        # システム監視開始
        launcher.monitor_systems()
        
    except Exception as e:
        logger.error(f"❌ 統合ランチャーエラー: {e}")
        print(f"❌ エラーが発生しました: {e}")
    finally:
        launcher.shutdown_all_systems()

if __name__ == "__main__":
    main() 