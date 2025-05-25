#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT v11 統合ランチャー
全システム統合起動・監視・制御

作成者: NKAT Research Team
作成日: 2025年5月26日
バージョン: v11.0
"""

import os
import sys
import time
import subprocess
import threading
import webbrowser
from datetime import datetime
import json
import psutil

class IntegratedLauncher:
    """統合ランチャークラス"""
    
    def __init__(self):
        """初期化"""
        self.systems = {}
        self.running_processes = {}
        self.start_time = datetime.now()
        
        # システム定義
        self.system_definitions = {
            'recovery_dashboard': {
                'name': 'リカバリーダッシュボード',
                'command': 'streamlit run nkat_v11_comprehensive_recovery_dashboard.py --server.port 8501',
                'port': 8501,
                'url': 'http://localhost:8501',
                'priority': 1,
                'auto_start': True,
                'description': '包括的システム監視・リカバリーダッシュボード'
            },
            'convergence_analyzer': {
                'name': '詳細収束分析',
                'command': 'py -3 nkat_v11_detailed_convergence_analyzer.py',
                'priority': 2,
                'auto_start': True,
                'description': '0.497762収束結果の詳細分析システム'
            },
            'auto_recovery': {
                'name': '自動リカバリー',
                'command': 'py -3 nkat_v11_auto_recovery_system.py',
                'priority': 3,
                'auto_start': True,
                'description': '電源断対応自動リカバリーシステム'
            }
        }
    
    def check_dependencies(self):
        """依存関係をチェック"""
        print("🔍 依存関係をチェック中...")
        
        required_packages = [
            'streamlit', 'pandas', 'numpy', 'plotly', 'psutil',
            'matplotlib', 'scipy', 'seaborn'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"✅ {package}")
            except ImportError:
                print(f"❌ {package} (未インストール)")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\n⚠️ 不足パッケージ: {', '.join(missing_packages)}")
            print("以下のコマンドでインストールしてください:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
        
        print("✅ 全ての依存関係が満たされています")
        return True
    
    def check_system_files(self):
        """システムファイルの存在確認"""
        print("\n📁 システムファイルをチェック中...")
        
        required_files = [
            'nkat_v11_comprehensive_recovery_dashboard.py',
            'nkat_v11_detailed_convergence_analyzer.py',
            'nkat_v11_auto_recovery_system.py'
        ]
        
        missing_files = []
        
        for file_name in required_files:
            if os.path.exists(file_name):
                print(f"✅ {file_name}")
            else:
                print(f"❌ {file_name} (見つかりません)")
                missing_files.append(file_name)
        
        if missing_files:
            print(f"\n⚠️ 不足ファイル: {', '.join(missing_files)}")
            return False
        
        print("✅ 全てのシステムファイルが存在します")
        return True
    
    def check_data_files(self):
        """データファイルの存在確認"""
        print("\n📊 データファイルをチェック中...")
        
        data_files = [
            'high_precision_riemann_results.json',
            'ultimate_mastery_riemann_results.json',
            'extended_riemann_results.json',
            'improved_riemann_results.json'
        ]
        
        found_files = []
        
        for file_name in data_files:
            if os.path.exists(file_name):
                print(f"✅ {file_name}")
                found_files.append(file_name)
            else:
                print(f"⚠️ {file_name} (オプション)")
        
        if found_files:
            print(f"✅ {len(found_files)}個のデータファイルが利用可能です")
        else:
            print("⚠️ データファイルが見つかりません（一部機能が制限される可能性があります）")
        
        return True
    
    def start_system(self, system_key):
        """システムを起動"""
        if system_key not in self.system_definitions:
            print(f"❌ 未知のシステム: {system_key}")
            return False
        
        system = self.system_definitions[system_key]
        
        try:
            print(f"🚀 {system['name']} を起動中...")
            
            # プロセス起動
            process = subprocess.Popen(
                system['command'],
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0
            )
            
            self.running_processes[system_key] = {
                'process': process,
                'start_time': datetime.now(),
                'system': system
            }
            
            print(f"✅ {system['name']} が起動しました (PID: {process.pid})")
            
            # Streamlitの場合は起動待機
            if 'streamlit' in system['command']:
                print(f"⏳ {system['name']} の起動を待機中...")
                time.sleep(5)  # Streamlit起動待機
            
            return True
            
        except Exception as e:
            print(f"❌ {system['name']} の起動に失敗: {e}")
            return False
    
    def stop_system(self, system_key):
        """システムを停止"""
        if system_key not in self.running_processes:
            print(f"⚠️ {system_key} は実行されていません")
            return False
        
        try:
            process_info = self.running_processes[system_key]
            process = process_info['process']
            system = process_info['system']
            
            print(f"🛑 {system['name']} を停止中...")
            
            # プロセス終了
            process.terminate()
            
            # 強制終了が必要な場合
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            
            del self.running_processes[system_key]
            print(f"✅ {system['name']} が停止しました")
            
            return True
            
        except Exception as e:
            print(f"❌ {system_key} の停止に失敗: {e}")
            return False
    
    def check_system_status(self):
        """システム状態をチェック"""
        status = {}
        
        for system_key, process_info in self.running_processes.items():
            process = process_info['process']
            system = process_info['system']
            
            try:
                # プロセス状態確認
                if process.poll() is None:
                    status[system_key] = {
                        'status': 'running',
                        'pid': process.pid,
                        'uptime': datetime.now() - process_info['start_time']
                    }
                else:
                    status[system_key] = {
                        'status': 'stopped',
                        'exit_code': process.returncode
                    }
            except Exception as e:
                status[system_key] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return status
    
    def open_dashboards(self):
        """ダッシュボードをブラウザで開く"""
        print("\n🌐 ダッシュボードを開いています...")
        
        for system_key, system in self.system_definitions.items():
            if 'url' in system and system_key in self.running_processes:
                try:
                    print(f"🔗 {system['name']}: {system['url']}")
                    webbrowser.open(system['url'])
                    time.sleep(2)  # ブラウザ起動間隔
                except Exception as e:
                    print(f"❌ ブラウザ起動エラー {system['name']}: {e}")
    
    def start_all_systems(self):
        """全システムを起動"""
        print("\n🚀 NKAT v11 統合システムを起動中...")
        print("=" * 60)
        
        # 優先順位でソート
        sorted_systems = sorted(
            self.system_definitions.items(),
            key=lambda x: x[1]['priority']
        )
        
        success_count = 0
        
        for system_key, system in sorted_systems:
            if system.get('auto_start', False):
                if self.start_system(system_key):
                    success_count += 1
                    time.sleep(2)  # システム間起動間隔
        
        print(f"\n✅ {success_count}/{len([s for s in self.system_definitions.values() if s.get('auto_start')])} システムが起動しました")
        
        # ダッシュボード自動オープン
        if success_count > 0:
            time.sleep(3)
            self.open_dashboards()
        
        return success_count > 0
    
    def stop_all_systems(self):
        """全システムを停止"""
        print("\n🛑 全システムを停止中...")
        
        for system_key in list(self.running_processes.keys()):
            self.stop_system(system_key)
        
        print("✅ 全システムが停止しました")
    
    def monitor_systems(self):
        """システム監視"""
        print("\n📊 システム監視を開始...")
        print("Ctrl+C で停止")
        
        try:
            while True:
                status = self.check_system_status()
                
                # ステータス表示
                running_count = sum(1 for s in status.values() if s['status'] == 'running')
                total_count = len(self.system_definitions)
                
                print(f"\r⏰ 監視中... 実行中: {running_count}/{total_count} システム", end="")
                
                # 停止したシステムの再起動
                for system_key, system_status in status.items():
                    if system_status['status'] == 'stopped':
                        system = self.system_definitions[system_key]
                        if system.get('auto_start', False):
                            print(f"\n🔄 {system['name']} を再起動中...")
                            self.start_system(system_key)
                
                time.sleep(10)  # 10秒間隔で監視
                
        except KeyboardInterrupt:
            print("\n\n🛑 監視を停止します...")
            self.stop_all_systems()
    
    def show_status(self):
        """システム状態を表示"""
        print("\n📊 システム状態:")
        print("=" * 60)
        
        status = self.check_system_status()
        
        for system_key, system in self.system_definitions.items():
            print(f"\n🔧 {system['name']}")
            print(f"   説明: {system['description']}")
            
            if system_key in status:
                sys_status = status[system_key]
                if sys_status['status'] == 'running':
                    uptime = sys_status['uptime']
                    print(f"   状態: ✅ 実行中 (PID: {sys_status['pid']})")
                    print(f"   稼働時間: {uptime}")
                else:
                    print(f"   状態: ❌ 停止")
            else:
                print(f"   状態: ⚪ 未起動")
            
            if 'url' in system:
                print(f"   URL: {system['url']}")
    
    def create_startup_summary(self):
        """起動サマリーを作成"""
        summary = {
            'launcher_version': 'v11.0',
            'start_time': self.start_time.isoformat(),
            'systems': {},
            'system_info': {
                'platform': sys.platform,
                'python_version': sys.version,
                'working_directory': os.getcwd(),
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total
            }
        }
        
        # システム情報追加
        for system_key, system in self.system_definitions.items():
            summary['systems'][system_key] = {
                'name': system['name'],
                'description': system['description'],
                'auto_start': system.get('auto_start', False),
                'running': system_key in self.running_processes
            }
        
        # サマリー保存
        summary_file = f"nkat_v11_startup_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print(f"📄 起動サマリーを保存: {summary_file}")
            
        except Exception as e:
            print(f"⚠️ サマリー保存エラー: {e}")
        
        return summary
    
    def interactive_menu(self):
        """インタラクティブメニュー"""
        while True:
            print("\n" + "=" * 60)
            print("🎛️ NKAT v11 統合制御メニュー")
            print("=" * 60)
            print("1. 全システム起動")
            print("2. 全システム停止")
            print("3. システム状態表示")
            print("4. システム監視開始")
            print("5. ダッシュボードを開く")
            print("6. 個別システム制御")
            print("0. 終了")
            print("=" * 60)
            
            try:
                choice = input("選択してください (0-6): ").strip()
                
                if choice == '1':
                    self.start_all_systems()
                elif choice == '2':
                    self.stop_all_systems()
                elif choice == '3':
                    self.show_status()
                elif choice == '4':
                    self.monitor_systems()
                elif choice == '5':
                    self.open_dashboards()
                elif choice == '6':
                    self.individual_system_control()
                elif choice == '0':
                    print("🛑 システムを終了します...")
                    self.stop_all_systems()
                    break
                else:
                    print("❌ 無効な選択です")
                    
            except KeyboardInterrupt:
                print("\n🛑 終了します...")
                self.stop_all_systems()
                break
            except Exception as e:
                print(f"❌ エラー: {e}")
    
    def individual_system_control(self):
        """個別システム制御"""
        print("\n🔧 個別システム制御")
        print("-" * 30)
        
        for i, (system_key, system) in enumerate(self.system_definitions.items(), 1):
            status = "実行中" if system_key in self.running_processes else "停止中"
            print(f"{i}. {system['name']} ({status})")
        
        try:
            choice = int(input("システム番号を選択 (0で戻る): "))
            
            if choice == 0:
                return
            
            if 1 <= choice <= len(self.system_definitions):
                system_key = list(self.system_definitions.keys())[choice - 1]
                system = self.system_definitions[system_key]
                
                print(f"\n{system['name']} の制御:")
                print("1. 起動")
                print("2. 停止")
                print("3. 再起動")
                
                action = input("アクション選択 (1-3): ").strip()
                
                if action == '1':
                    self.start_system(system_key)
                elif action == '2':
                    self.stop_system(system_key)
                elif action == '3':
                    self.stop_system(system_key)
                    time.sleep(2)
                    self.start_system(system_key)
                else:
                    print("❌ 無効な選択です")
            else:
                print("❌ 無効なシステム番号です")
                
        except ValueError:
            print("❌ 数値を入力してください")
        except Exception as e:
            print(f"❌ エラー: {e}")

def main():
    """メイン実行関数"""
    print("🚀 NKAT v11 統合ランチャー")
    print("全システム統合起動・監視・制御")
    print("=" * 50)
    
    # ランチャー初期化
    launcher = IntegratedLauncher()
    
    # 事前チェック
    print("🔍 システム事前チェック...")
    
    if not launcher.check_dependencies():
        print("❌ 依存関係の問題により起動できません")
        return
    
    if not launcher.check_system_files():
        print("❌ システムファイルの問題により起動できません")
        return
    
    launcher.check_data_files()
    
    print("\n✅ 事前チェック完了")
    
    # 起動サマリー作成
    launcher.create_startup_summary()
    
    # コマンドライン引数チェック
    if len(sys.argv) > 1:
        if sys.argv[1] == '--auto':
            # 自動起動モード
            print("\n🤖 自動起動モード")
            if launcher.start_all_systems():
                launcher.monitor_systems()
        elif sys.argv[1] == '--status':
            # ステータス表示のみ
            launcher.show_status()
        else:
            print(f"❌ 未知のオプション: {sys.argv[1]}")
    else:
        # インタラクティブモード
        launcher.interactive_menu()
    
    print("👋 NKAT v11 統合ランチャーを終了しました")

if __name__ == "__main__":
    main() 