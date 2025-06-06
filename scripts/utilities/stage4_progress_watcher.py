#!/usr/bin/env python3
"""
🔥 NKAT Stage4 進行状況ウォッチャー
定期的に進行状況をチェックして報告
"""

import os
import time
import psutil
import pickle
from pathlib import Path
from datetime import datetime, timedelta

class Stage4ProgressWatcher:
    def __init__(self, check_interval=30):
        self.check_interval = check_interval
        self.start_time = datetime.now()
        self.last_zeros = 0
        self.checkpoints_seen = set()
        
    def get_stage4_status(self):
        """Stage4のステータス取得"""
        # ディレクトリチェック
        output_dirs = [Path(d) for d in os.listdir('.') if d.startswith('nkat_stage4_1M_CUDA_')]
        if not output_dirs:
            return None
        
        latest_dir = max(output_dirs, key=lambda x: x.stat().st_mtime)
        checkpoint_dir = latest_dir / "checkpoints"
        
        status = {
            'directory': latest_dir,
            'zeros_count': 0,
            'last_checkpoint': None,
            'checkpoint_count': 0
        }
        
        if checkpoint_dir.exists():
            checkpoint_files = list(checkpoint_dir.glob("cuda_checkpoint_*.pkl"))
            status['checkpoint_count'] = len(checkpoint_files)
            
            if checkpoint_files:
                latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
                status['last_checkpoint'] = latest_checkpoint
                
                try:
                    with open(latest_checkpoint, 'rb') as f:
                        data = pickle.load(f)
                        status['zeros_count'] = data['metadata']['zeros_count']
                        status['checkpoint_time'] = data['metadata']['timestamp']
                except Exception as e:
                    print(f"⚠️ チェックポイント読み込みエラー: {e}")
        
        return status
    
    def get_python_status(self):
        """Python プロセス状況"""
        python_procs = []
        total_memory = 0
        
        for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
            try:
                if 'python' in proc.info['name'].lower():
                    mem_mb = proc.info['memory_info'].rss / 1024 / 1024
                    if mem_mb > 10:
                        python_procs.append(mem_mb)
                        total_memory += mem_mb
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return len(python_procs), total_memory
    
    def calculate_eta(self, current_zeros, rate):
        """完了予想時間計算"""
        if rate <= 0:
            return "計算中..."
        
        remaining = 1000000 - current_zeros
        eta_seconds = remaining / rate
        eta_delta = timedelta(seconds=int(eta_seconds))
        completion_time = datetime.now() + eta_delta
        
        return f"{eta_delta} (完了予定: {completion_time.strftime('%H:%M')})"
    
    def run_watcher(self):
        """ウォッチャー実行"""
        print("🚀 NKAT Stage4 進行状況ウォッチャー開始")
        print(f"🔄 チェック間隔: {self.check_interval}秒")
        print("=" * 80)
        
        iteration = 0
        
        while True:
            try:
                iteration += 1
                elapsed = datetime.now() - self.start_time
                
                # Stage4ステータス
                stage4_status = self.get_stage4_status()
                
                if not stage4_status:
                    print(f"⏰ {datetime.now().strftime('%H:%M:%S')} - Stage4プロセスが見つかりません")
                    time.sleep(self.check_interval)
                    continue
                
                # Pythonプロセス状況
                proc_count, total_memory = self.get_python_status()
                
                # システム情報
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_percent = psutil.virtual_memory().percent
                
                # 進行率計算
                current_zeros = stage4_status['zeros_count']
                progress_percent = (current_zeros / 1000000) * 100
                
                # 処理速度計算
                if iteration > 1:
                    time_diff = self.check_interval
                    zeros_diff = current_zeros - self.last_zeros
                    rate = zeros_diff / time_diff
                else:
                    rate = 0
                
                # 新しいチェックポイント検出
                new_checkpoint = False
                if stage4_status['last_checkpoint'] and stage4_status['last_checkpoint'] not in self.checkpoints_seen:
                    self.checkpoints_seen.add(stage4_status['last_checkpoint'])
                    new_checkpoint = True
                
                # レポート表示
                print(f"⏰ {datetime.now().strftime('%H:%M:%S')} - チェック #{iteration}")
                print(f"📊 進行状況: {current_zeros:,}/1,000,000 ({progress_percent:.3f}%)")
                print(f"⚡ 処理速度: {rate:.1f} ゼロ点/秒")
                if rate > 0:
                    eta = self.calculate_eta(current_zeros, rate)
                    print(f"⏱️ 完了予想: {eta}")
                print(f"💾 チェックポイント数: {stage4_status['checkpoint_count']}")
                print(f"🐍 Pythonプロセス: {proc_count}個 ({total_memory:.1f}MB)")
                print(f"🖥️ システム: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%")
                print(f"🕐 経過時間: {elapsed}")
                
                if new_checkpoint:
                    print(f"🎉 新しいチェックポイント作成: {stage4_status['last_checkpoint'].name}")
                
                print("-" * 80)
                
                # マイルストーン通知
                if current_zeros > 0:
                    milestones = [50000, 100000, 200000, 500000, 750000, 900000]
                    for milestone in milestones:
                        if self.last_zeros < milestone <= current_zeros:
                            print(f"🎊 マイルストーン達成: {milestone:,}ゼロ点!")
                            print("-" * 80)
                
                self.last_zeros = current_zeros
                
                # 完了チェック
                if current_zeros >= 1000000:
                    print("🎉🎉🎉 100万ゼロ点計算完了！！！ 🎉🎉🎉")
                    break
                
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                print("\n👋 ウォッチャー終了")
                break
            except Exception as e:
                print(f"⚠️ ウォッチャーエラー: {e}")
                time.sleep(5)

def main():
    watcher = Stage4ProgressWatcher(check_interval=60)  # 1分間隔
    watcher.run_watcher()

if __name__ == "__main__":
    main() 