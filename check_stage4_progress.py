#!/usr/bin/env python3
"""
🚀 NKAT Stage4 進行状況チェッカー
"""

import os
import psutil
import pickle
from pathlib import Path
from datetime import datetime

def check_python_processes():
    """Pythonプロセスをチェック"""
    python_procs = []
    total_memory = 0
    
    for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cpu_percent']):
        try:
            if 'python' in proc.info['name'].lower():
                mem_mb = proc.info['memory_info'].rss / 1024 / 1024
                if mem_mb > 50:  # 50MB以上のプロセス
                    python_procs.append({
                        'pid': proc.info['pid'],
                        'memory_mb': mem_mb,
                        'cpu_percent': proc.info['cpu_percent']
                    })
                    total_memory += mem_mb
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    return python_procs, total_memory

def check_stage4_progress():
    """Stage4進行状況チェック"""
    # 出力ディレクトリを検索
    output_dirs = []
    for dir_name in os.listdir('.'):
        if dir_name.startswith('nkat_stage4_1M_CUDA_'):
            output_dirs.append(Path(dir_name))
    
    if not output_dirs:
        return None, "Stage4ディレクトリが見つかりません"
    
    latest_dir = max(output_dirs, key=lambda x: x.stat().st_mtime)
    checkpoint_dir = latest_dir / "checkpoints"
    
    if not checkpoint_dir.exists():
        return latest_dir, "チェックポイントなし（計算中）"
    
    # 最新チェックポイント
    checkpoint_files = list(checkpoint_dir.glob("cuda_checkpoint_*.pkl"))
    if not checkpoint_files:
        return latest_dir, "チェックポイントなし（計算中）"
    
    latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
    
    try:
        with open(latest_checkpoint, 'rb') as f:
            data = pickle.load(f)
            zeros_count = data['metadata']['zeros_count']
            timestamp = data['metadata']['timestamp']
            return latest_dir, f"ゼロ点: {zeros_count:,}/1,000,000 ({zeros_count/10000:.1f}%) - {timestamp}"
    except Exception as e:
        return latest_dir, f"チェックポイント読み込みエラー: {e}"

def main():
    print("🚀 NKAT Stage4 進行状況チェック")
    print("=" * 60)
    
    # Python プロセス状況
    python_procs, total_memory = check_python_processes()
    print(f"🐍 Python プロセス数: {len(python_procs)}")
    print(f"💾 総メモリ使用量: {total_memory:.1f} MB")
    
    if python_procs:
        print("\n主要プロセス:")
        for i, proc in enumerate(python_procs[:5]):  # 上位5個
            print(f"  PID {proc['pid']}: {proc['memory_mb']:.1f}MB, CPU {proc['cpu_percent']:.1f}%")
    
    print()
    
    # Stage4 進行状況
    stage4_dir, progress_info = check_stage4_progress()
    if stage4_dir:
        print(f"📁 Stage4ディレクトリ: {stage4_dir}")
        print(f"📊 進行状況: {progress_info}")
    else:
        print(f"❌ Stage4: {progress_info}")
    
    print()
    
    # システム統計
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    print(f"🖥️ CPU使用率: {cpu_percent:.1f}%")
    print(f"💾 メモリ使用率: {memory.percent:.1f}%")
    print(f"🔄 現在時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 