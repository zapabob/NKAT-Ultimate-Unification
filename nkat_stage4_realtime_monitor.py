#!/usr/bin/env python3
"""
🔥 NKAT Stage4 リアルタイムモニタリングダッシュボード
================================================
100万ゼロ点計算の進行状況をリアルタイムで監視
"""

import os
import sys
import json
import time
import psutil
import pickle
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import threading
import subprocess

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (16, 10)

class NKAT_Stage4_Monitor:
    def __init__(self):
        """モニター初期化"""
        self.start_time = datetime.now()
        self.data_history = {
            'timestamps': deque(maxlen=100),
            'cpu_usage': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'python_processes': deque(maxlen=100),
            'zero_count': deque(maxlen=100),
            'processing_rate': deque(maxlen=100)
        }
        
        # CUDA最適化版のディレクトリを検索
        self.output_dirs = []
        for dir_name in os.listdir('.'):
            if dir_name.startswith('nkat_stage4_1M_CUDA_'):
                self.output_dirs.append(Path(dir_name))
        
        if self.output_dirs:
            self.current_dir = max(self.output_dirs, key=lambda x: x.stat().st_mtime)
            print(f"🎯 監視対象: {self.current_dir}")
        else:
            self.current_dir = None
            print("⚠️ Stage4実行ディレクトリが見つかりません")
        
        self.last_zero_count = 0
        
    def get_system_stats(self):
        """システム統計取得"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # メモリ使用率
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Pythonプロセス数とメモリ使用量
            python_processes = []
            total_python_memory = 0
            
            for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
                try:
                    if 'python' in proc.info['name'].lower():
                        mem_mb = proc.info['memory_info'].rss / 1024 / 1024
                        if mem_mb > 10:  # 10MB以上のプロセスのみ
                            python_processes.append(proc.info['pid'])
                            total_python_memory += mem_mb
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'python_count': len(python_processes),
                'python_memory_mb': total_python_memory,
                'timestamp': datetime.now()
            }
        except Exception as e:
            print(f"⚠️ システム統計取得エラー: {e}")
            return None
    
    def get_zero_progress(self):
        """ゼロ点計算進行状況取得"""
        if not self.current_dir:
            return 0
        
        checkpoint_dir = self.current_dir / "checkpoints"
        if not checkpoint_dir.exists():
            return 0
        
        # 最新のチェックポイントファイルを検索
        checkpoint_files = list(checkpoint_dir.glob("cuda_checkpoint_*.pkl"))
        if not checkpoint_files:
            return 0
        
        latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
        
        try:
            with open(latest_checkpoint, 'rb') as f:
                data = pickle.load(f)
                return data['metadata']['zeros_count']
        except Exception as e:
            print(f"⚠️ チェックポイント読み込みエラー: {e}")
            return 0
    
    def update_data(self):
        """データ更新"""
        stats = self.get_system_stats()
        if not stats:
            return
        
        current_time = stats['timestamp']
        
        # ゼロ点進行状況
        zero_count = self.get_zero_progress()
        
        # 処理速度計算
        time_diff = (current_time - self.data_history['timestamps'][-1]).total_seconds() if self.data_history['timestamps'] else 1.0
        zero_diff = zero_count - self.last_zero_count
        processing_rate = zero_diff / time_diff if time_diff > 0 else 0
        
        # データ追加
        self.data_history['timestamps'].append(current_time)
        self.data_history['cpu_usage'].append(stats['cpu_percent'])
        self.data_history['memory_usage'].append(stats['memory_percent'])
        self.data_history['python_processes'].append(stats['python_count'])
        self.data_history['zero_count'].append(zero_count)
        self.data_history['processing_rate'].append(processing_rate)
        
        self.last_zero_count = zero_count
    
    def create_dashboard(self):
        """ダッシュボード作成"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('🚀 NKAT Stage4: 1,000,000ゼロ点計算 リアルタイムモニター', fontsize=16, fontweight='bold')
        
        def animate(frame):
            # データ更新
            self.update_data()
            
            if not self.data_history['timestamps']:
                return
            
            # 時間軸
            times = [t.strftime('%H:%M:%S') for t in self.data_history['timestamps']]
            
            # CPU & メモリ使用率
            ax1.clear()
            ax1.plot(times, self.data_history['cpu_usage'], 'b-', label='CPU %', linewidth=2)
            ax1.plot(times, self.data_history['memory_usage'], 'r-', label='Memory %', linewidth=2)
            ax1.set_title('🖥️ システムリソース使用率', fontweight='bold')
            ax1.set_ylabel('使用率 (%)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 100)
            
            # Pythonプロセス数
            ax2.clear()
            ax2.plot(times, self.data_history['python_processes'], 'g-', marker='o', linewidth=2)
            ax2.set_title('🐍 Python並列プロセス数', fontweight='bold')
            ax2.set_ylabel('プロセス数')
            ax2.grid(True, alpha=0.3)
            
            # ゼロ点計算進行状況
            ax3.clear()
            if self.data_history['zero_count']:
                current_zeros = list(self.data_history['zero_count'])[-1]
                progress_percent = (current_zeros / 1000000) * 100
                
                ax3.barh([0], [progress_percent], color='orange', alpha=0.7)
                ax3.barh([0], [100-progress_percent], left=[progress_percent], color='lightgray', alpha=0.3)
                ax3.set_xlim(0, 100)
                ax3.set_title(f'📊 ゼロ点計算進行状況: {current_zeros:,}/1,000,000 ({progress_percent:.2f}%)', fontweight='bold')
                ax3.set_xlabel('進行率 (%)')
                ax3.text(50, 0, f'{current_zeros:,}', ha='center', va='center', fontweight='bold', fontsize=12)
            
            # 処理速度
            ax4.clear()
            if len(self.data_history['processing_rate']) > 1:
                ax4.plot(times, self.data_history['processing_rate'], 'purple', linewidth=2, marker='.')
                ax4.set_title('⚡ ゼロ点処理速度', fontweight='bold')
                ax4.set_ylabel('ゼロ点/秒')
                ax4.grid(True, alpha=0.3)
                
                # 平均速度表示
                avg_rate = np.mean(list(self.data_history['processing_rate'])[-10:])
                ax4.axhline(y=avg_rate, color='red', linestyle='--', alpha=0.7, label=f'平均: {avg_rate:.1f}/秒')
                ax4.legend()
            
            # X軸ラベル回転
            for ax in [ax1, ax2, ax4]:
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # ステータス表示
            if self.data_history['timestamps']:
                elapsed = datetime.now() - self.start_time
                current_zeros = list(self.data_history['zero_count'])[-1] if self.data_history['zero_count'] else 0
                current_rate = list(self.data_history['processing_rate'])[-1] if self.data_history['processing_rate'] else 0
                
                # 推定残り時間
                if current_rate > 0:
                    remaining_zeros = 1000000 - current_zeros
                    eta_seconds = remaining_zeros / current_rate
                    eta_hours = eta_seconds / 3600
                    eta_str = f"{eta_hours:.1f}時間"
                else:
                    eta_str = "計算中..."
                
                status_text = (
                    f"🕐 経過時間: {elapsed}\n"
                    f"🔢 処理済み: {current_zeros:,}/1,000,000\n"
                    f"⚡ 処理速度: {current_rate:.1f}ゼロ点/秒\n"
                    f"⏱️ 推定残り時間: {eta_str}\n"
                    f"🐍 並列プロセス: {list(self.data_history['python_processes'])[-1] if self.data_history['python_processes'] else 0}個"
                )
                
                fig.text(0.02, 0.02, status_text, fontsize=10, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        # アニメーション設定
        ani = animation.FuncAnimation(fig, animate, interval=2000, blit=False)
        
        # 保存オプション
        save_path = "nkat_stage4_monitor_dashboard.png"
        
        plt.show()
        return ani
    
    def console_monitor(self):
        """コンソールモニター"""
        print("🚀 NKAT Stage4 コンソールモニター開始...")
        print("=" * 80)
        
        while True:
            try:
                self.update_data()
                
                if not self.data_history['timestamps']:
                    time.sleep(5)
                    continue
                
                elapsed = datetime.now() - self.start_time
                current_zeros = list(self.data_history['zero_count'])[-1] if self.data_history['zero_count'] else 0
                current_rate = list(self.data_history['processing_rate'])[-1] if self.data_history['processing_rate'] else 0
                cpu_usage = list(self.data_history['cpu_usage'])[-1] if self.data_history['cpu_usage'] else 0
                memory_usage = list(self.data_history['memory_usage'])[-1] if self.data_history['memory_usage'] else 0
                python_count = list(self.data_history['python_processes'])[-1] if self.data_history['python_processes'] else 0
                
                progress_percent = (current_zeros / 1000000) * 100
                
                # ETAの計算
                if current_rate > 0:
                    remaining_zeros = 1000000 - current_zeros
                    eta_seconds = remaining_zeros / current_rate
                    eta_hours = eta_seconds / 3600
                    eta_str = f"{eta_hours:.1f}時間"
                else:
                    eta_str = "計算中..."
                
                # プログレスバー
                bar_length = 50
                filled_length = int(bar_length * progress_percent / 100)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                
                # ステータス表示
                os.system('cls' if os.name == 'nt' else 'clear')
                print("🚀 NKAT Stage4: 1,000,000ゼロ点計算 リアルタイムモニター")
                print("=" * 80)
                print(f"📊 進行状況: [{bar}] {progress_percent:.2f}%")
                print(f"🔢 処理済みゼロ点: {current_zeros:,} / 1,000,000")
                print(f"⚡ 処理速度: {current_rate:.1f} ゼロ点/秒")
                print(f"⏱️ 推定残り時間: {eta_str}")
                print(f"🕐 経過時間: {elapsed}")
                print()
                print(f"🖥️ CPU使用率: {cpu_usage:.1f}%")
                print(f"💾 メモリ使用率: {memory_usage:.1f}%")
                print(f"🐍 Python並列プロセス: {python_count}個")
                print()
                print(f"📁 監視ディレクトリ: {self.current_dir}")
                print("=" * 80)
                print("🔄 更新間隔: 5秒 | Ctrl+C で終了")
                
                time.sleep(5)
                
            except KeyboardInterrupt:
                print("\n👋 モニター終了")
                break
            except Exception as e:
                print(f"⚠️ モニターエラー: {e}")
                time.sleep(5)


def main():
    """メイン実行"""
    monitor = NKAT_Stage4_Monitor()
    
    print("🚀 NKAT Stage4 リアルタイムモニターシステム")
    print("選択してください:")
    print("1. グラフィカルダッシュボード（推奨）")
    print("2. コンソールモニター")
    
    try:
        choice = input("選択 (1/2): ").strip()
        
        if choice == "1":
            print("🎨 グラフィカルダッシュボード起動中...")
            monitor.create_dashboard()
        elif choice == "2":
            print("📟 コンソールモニター起動中...")
            monitor.console_monitor()
        else:
            print("📟 デフォルトでコンソールモニター起動...")
            monitor.console_monitor()
            
    except KeyboardInterrupt:
        print("\n👋 モニター終了")


if __name__ == "__main__":
    main() 