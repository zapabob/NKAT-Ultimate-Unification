#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔥 NKAT v8.0 RTX3080極限計算システム - シンプル版ダッシュボード
Simplified Dashboard for NKAT v8.0 RTX3080 Extreme Computation System

機能:
- GPU監視
- プロセス監視  
- ログ監視
- 簡易制御

Author: NKAT Research Team
Date: 2025-05-26
Version: v1.0 - Simplified Edition
"""

import subprocess
import time
import json
import psutil
from pathlib import Path
import datetime
import os
import threading

class SimplifiedDashboard:
    """シンプル版ダッシュボードクラス"""
    
    def __init__(self):
        self.running = True
        self.clear_screen = lambda: os.system('cls' if os.name == 'nt' else 'clear')
    
    def get_gpu_stats(self):
        """GPU統計取得"""
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                data = result.stdout.strip().split(', ')
                return {
                    'name': data[0],
                    'gpu_utilization': int(data[1]),
                    'memory_used': int(data[2]),
                    'memory_total': int(data[3]),
                    'temperature': int(data[4]),
                    'power_draw': float(data[5]),
                    'memory_percent': int(data[2]) / int(data[3]) * 100
                }
        except Exception as e:
            print(f"GPU統計取得エラー: {e}")
        return None
    
    def get_nkat_processes(self):
        """NKAT関連プロセス取得"""
        processes = []
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
                try:
                    cmdline = ' '.join(proc.cmdline()) if hasattr(proc, 'cmdline') else ''
                    if any(script in cmdline for script in ['riemann', 'rtx3080', 'checkpoint', 'auto_', 'nkat']):
                        processes.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'], 
                            'cpu_percent': proc.info['cpu_percent'],
                            'cmdline': cmdline
                        })
                except:
                    continue
        except:
            pass
        return processes
    
    def get_latest_log(self, n_lines=10):
        """最新ログ取得"""
        log_files = [
            "auto_computation.log",
            "rtx3080_optimization.log", 
            "../rtx3080_optimization.log",
            "../auto_computation.log"
        ]
        
        for log_file in log_files:
            try:
                if Path(log_file).exists():
                    with open(log_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        return ''.join(lines[-n_lines:])
            except:
                continue
        return "ログファイルが見つかりません"
    
    def display_header(self):
        """ヘッダー表示"""
        print("🔥" * 60)
        print(" " * 15 + "NKAT v8.0 RTX3080極限計算ダッシュボード")
        print(" " * 20 + f"最終更新: {datetime.datetime.now().strftime('%H:%M:%S')}")
        print("🔥" * 60)
    
    def display_gpu_info(self, gpu_stats):
        """GPU情報表示"""
        if not gpu_stats:
            print("❌ GPU情報取得できません")
            return
        
        print("\n🎮 GPU状況")
        print("=" * 50)
        print(f"名前: {gpu_stats['name']}")
        print(f"使用率: {gpu_stats['gpu_utilization']}%")
        print(f"VRAM: {gpu_stats['memory_used']}/{gpu_stats['memory_total']} MB ({gpu_stats['memory_percent']:.1f}%)")
        print(f"温度: {gpu_stats['temperature']}°C")
        print(f"電力: {gpu_stats['power_draw']} W")
        
        # 状態判定
        if gpu_stats['gpu_utilization'] > 90:
            print("✅ GPU極限活用中！")
        elif gpu_stats['gpu_utilization'] > 70:
            print("⚡ GPU高使用率")
        else:
            print("🔍 GPU低使用率")
        
        if gpu_stats['temperature'] > 85:
            print("🔥 高温警告！")
        elif gpu_stats['temperature'] > 80:
            print("⚠️ 温度注意")
        else:
            print("❄️ 温度正常")
    
    def display_processes(self, processes):
        """プロセス情報表示"""
        print("\n🔄 NKAT関連プロセス")
        print("=" * 50)
        if not processes:
            print("❌ NKAT関連プロセスが検出されません")
            return
        
        for proc in processes:
            script_type = "Unknown"
            if 'riemann' in proc['cmdline'] or 'auto_' in proc['cmdline']:
                script_type = "🔥 極限計算"
            elif 'checkpoint' in proc['cmdline']:
                script_type = "💾 チェックポイント"
            elif 'optimizer' in proc['cmdline']:
                script_type = "⚡ 性能最適化"
            
            print(f"{script_type} | PID: {proc['pid']} | CPU: {proc['cpu_percent']:.1f}%")
    
    def display_log(self, log_content):
        """ログ表示"""
        print("\n📋 最新ログ")
        print("=" * 50)
        log_lines = log_content.split('\n')[-8:]  # 最新8行
        for line in log_lines:
            if line.strip():
                print(line.strip())
    
    def display_controls(self):
        """制御オプション表示"""
        print("\n⚙️ 制御オプション")
        print("=" * 50)
        print("[R] 手動更新 | [Q] 終了 | [L] 詳細ログ | [P] プロセス詳細")
    
    def run_interactive(self):
        """インタラクティブモード実行"""
        print("🚀 インタラクティブモード開始")
        print("制御コマンド:")
        print("R: 手動更新")
        print("Q: 終了")
        print("L: 詳細ログ表示")
        print("P: プロセス詳細表示")
        print("S: GPU統計詳細")
        print("-" * 50)
        
        while self.running:
            try:
                user_input = input("\nコマンド入力 (Enter=更新): ").strip().upper()
                
                if user_input == 'Q':
                    self.running = False
                    print("🛑 ダッシュボード終了")
                    break
                elif user_input == 'L':
                    log_content = self.get_latest_log(30)
                    print("\n📋 詳細ログ (最新30行)")
                    print("-" * 50)
                    print(log_content)
                elif user_input == 'P':
                    processes = self.get_nkat_processes()
                    print("\n🔄 プロセス詳細")
                    print("-" * 50)
                    for proc in processes:
                        print(f"PID: {proc['pid']}")
                        print(f"名前: {proc['name']}")
                        print(f"CPU: {proc['cpu_percent']:.1f}%")
                        print(f"コマンド: {proc['cmdline'][:80]}...")
                        print("-" * 30)
                elif user_input == 'S':
                    gpu_stats = self.get_gpu_stats()
                    if gpu_stats:
                        print("\n🎮 GPU統計詳細")
                        print("-" * 50)
                        for key, value in gpu_stats.items():
                            print(f"{key}: {value}")
                else:
                    # 通常更新
                    self.update_display()
                    
            except KeyboardInterrupt:
                self.running = False
                print("\n🛑 ダッシュボード終了")
                break
            except Exception as e:
                print(f"⚠️ エラー: {e}")
    
    def update_display(self):
        """表示更新"""
        self.clear_screen()
        
        # ヘッダー
        self.display_header()
        
        # GPU情報
        gpu_stats = self.get_gpu_stats()
        self.display_gpu_info(gpu_stats)
        
        # プロセス情報
        processes = self.get_nkat_processes()
        self.display_processes(processes)
        
        # ログ
        log_content = self.get_latest_log()
        self.display_log(log_content)
        
        # 制御オプション
        self.display_controls()
    
    def run_auto_update(self, interval=15):
        """自動更新モード"""
        print(f"🔄 自動更新モード開始 (間隔: {interval}秒)")
        print("Ctrl+C で終了")
        
        try:
            while self.running:
                self.update_display()
                time.sleep(interval)
        except KeyboardInterrupt:
            self.running = False
            print("\n🛑 自動更新停止")

def main():
    """メイン関数"""
    print("🔥 NKAT v8.0 RTX3080極限計算システム - シンプル版ダッシュボード")
    print("=" * 70)
    
    dashboard = SimplifiedDashboard()
    
    # モード選択
    print("モード選択:")
    print("1. 自動更新モード (15秒間隔)")
    print("2. インタラクティブモード")
    print("3. ワンショット表示")
    
    try:
        choice = input("\n選択 (1-3): ").strip()
        
        if choice == '1':
            dashboard.run_auto_update()
        elif choice == '2':
            dashboard.run_interactive()
        elif choice == '3':
            dashboard.update_display()
        else:
            print("無効な選択です。自動更新モードを開始します。")
            dashboard.run_auto_update()
            
    except KeyboardInterrupt:
        print("\n🛑 ダッシュボード終了")
    except Exception as e:
        print(f"❌ エラー: {e}")

if __name__ == "__main__":
    main() 