#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 NKAT v8.0 統合ダッシュボード起動システム
Dashboard Launcher for NKAT v8.0 RTX3080 Extreme Computation System

機能:
- 複数のダッシュボードオプション
- 自動環境設定
- エラーハンドリング
- バックアップオプション

Author: NKAT Research Team
Date: 2025-05-26
Version: v1.0 - Dashboard Launcher Edition
"""

import subprocess
import sys
import os
import time
from pathlib import Path

class DashboardLauncher:
    """ダッシュボード起動クラス"""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.project_root = self.script_dir.parent
        
    def check_dependencies(self):
        """依存関係チェック"""
        try:
            import streamlit
            import plotly
            import psutil
            return True
        except ImportError as e:
            print(f"⚠️ 依存関係が不足しています: {e}")
            return False
    
    def install_dependencies(self):
        """依存関係インストール"""
        print("📦 依存関係をインストール中...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 
                          'streamlit', 'plotly', 'psutil'], check=True)
            print("✅ 依存関係インストール完了")
            return True
        except subprocess.CalledProcessError:
            print("❌ 依存関係インストール失敗")
            return False
    
    def launch_streamlit_dashboard(self):
        """Streamlitダッシュボード起動"""
        dashboard_file = self.script_dir / "integrated_dashboard.py"
        
        if not dashboard_file.exists():
            print(f"❌ ダッシュボードファイルが見つかりません: {dashboard_file}")
            return False
        
        print("🚀 Streamlitダッシュボードを起動中...")
        print("📍 ブラウザで http://localhost:8501 にアクセスしてください")
        
        try:
            # Streamlitプロセス起動
            process = subprocess.Popen([
                'streamlit', 'run', str(dashboard_file),
                '--server.port', '8501',
                '--server.address', 'localhost',
                '--browser.gatherUsageStats', 'false'
            ], cwd=self.script_dir)
            
            print(f"✅ Streamlitダッシュボード起動成功 (PID: {process.pid})")
            print("⚠️ 終了するにはCtrl+Cを押してください")
            
            # プロセス監視
            try:
                process.wait()
            except KeyboardInterrupt:
                print("\n🛑 ダッシュボード終了中...")
                process.terminate()
                process.wait()
                print("✅ ダッシュボード終了完了")
            
            return True
            
        except FileNotFoundError:
            print("❌ Streamlitが見つかりません。インストールしてください:")
            print("   pip install streamlit")
            return False
        except Exception as e:
            print(f"❌ Streamlitダッシュボード起動エラー: {e}")
            return False
    
    def launch_simplified_dashboard(self):
        """シンプル版ダッシュボード起動"""
        dashboard_file = self.script_dir / "simplified_dashboard.py"
        
        if not dashboard_file.exists():
            print(f"❌ シンプル版ダッシュボードが見つかりません: {dashboard_file}")
            return False
        
        print("🔥 シンプル版ダッシュボードを起動中...")
        
        try:
            subprocess.run([sys.executable, str(dashboard_file)], cwd=self.script_dir)
            return True
        except Exception as e:
            print(f"❌ シンプル版ダッシュボード起動エラー: {e}")
            return False
    
    def launch_terminal_dashboard(self):
        """ターミナル版ダッシュボード起動"""
        print("💻 ターミナル版ダッシュボード")
        print("=" * 50)
        
        # GPU状況の簡易表示
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                data = result.stdout.strip().split(', ')
                print(f"🎮 GPU: {data[0]}")
                print(f"⚡ 使用率: {data[1]}%")
                print(f"💾 VRAM: {data[2]}/{data[3]} MB")
                print(f"🌡️ 温度: {data[4]}°C")
            else:
                print("❌ GPU情報取得失敗")
        except Exception as e:
            print(f"❌ GPU情報取得エラー: {e}")
        
        # NKAT関連プロセス確認
        print("\n🔄 NKAT関連プロセス:")
        try:
            import psutil
            found_processes = False
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    cmdline = ' '.join(proc.cmdline()) if hasattr(proc, 'cmdline') else ''
                    if any(script in cmdline for script in ['riemann', 'rtx3080', 'checkpoint', 'auto_']):
                        print(f"  - PID {proc.info['pid']}: {proc.info['name']}")
                        found_processes = True
                except:
                    continue
            
            if not found_processes:
                print("  ❌ NKAT関連プロセスが見つかりません")
        except ImportError:
            print("  ⚠️ psutilが必要です: pip install psutil")
        
        return True
    
    def show_menu(self):
        """メニュー表示"""
        print("🔥 NKAT v8.0 RTX3080極限計算システム - 統合ダッシュボード起動")
        print("=" * 70)
        print("利用可能なダッシュボード:")
        print("1. 🌐 Streamlit統合ダッシュボード (推奨)")
        print("2. 🔥 シンプル版ダッシュボード")
        print("3. 💻 ターミナル版ダッシュボード")
        print("4. 📦 依存関係インストール")
        print("5. 🚪 終了")
        print("=" * 70)
    
    def run(self):
        """起動システム実行"""
        while True:
            try:
                self.show_menu()
                choice = input("\n選択 (1-5): ").strip()
                
                if choice == '1':
                    if not self.check_dependencies():
                        install = input("依存関係をインストールしますか？ (y/N): ").strip().lower()
                        if install == 'y':
                            if not self.install_dependencies():
                                continue
                        else:
                            continue
                    
                    if not self.launch_streamlit_dashboard():
                        print("⚠️ Streamlitダッシュボード起動に失敗しました")
                        print("代替案: シンプル版ダッシュボードを試してください")
                
                elif choice == '2':
                    self.launch_simplified_dashboard()
                
                elif choice == '3':
                    self.launch_terminal_dashboard()
                    input("\nEnterキーで続行...")
                
                elif choice == '4':
                    self.install_dependencies()
                    input("\nEnterキーで続行...")
                
                elif choice == '5':
                    print("🚪 起動システム終了")
                    break
                
                else:
                    print("❌ 無効な選択です")
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                print("\n🛑 起動システム終了")
                break
            except Exception as e:
                print(f"❌ エラー: {e}")
                time.sleep(2)

def main():
    """メイン関数"""
    try:
        launcher = DashboardLauncher()
        launcher.run()
    except Exception as e:
        print(f"❌ 起動システムエラー: {e}")

if __name__ == "__main__":
    main() 