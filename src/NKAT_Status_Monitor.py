# -*- coding: utf-8 -*-
"""
NKAT リアルタイム進捗モニター
GPU訓練、LoI更新、κ-Minkowskiテストの状況を監視
"""

import os
import time
import json
import psutil
import datetime
from pathlib import Path

def check_gpu_usage():
    """GPU使用状況チェック（簡易版）"""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for i, line in enumerate(lines):
                parts = line.split(', ')
                if len(parts) >= 3:
                    gpu_info.append({
                        'gpu_id': i,
                        'utilization': int(parts[0]),
                        'memory_used': int(parts[1]),
                        'memory_total': int(parts[2])
                    })
            return gpu_info
        else:
            return [{'status': 'nvidia-smi failed'}]
    except:
        return [{'status': 'GPU info unavailable'}]

def check_training_status():
    """訓練状況チェック"""
    status = {
        'training_active': False,
        'latest_log': None,
        'current_epoch': 'Unknown',
        'latest_metrics': {}
    }
    
    # ログファイル検索
    log_files = list(Path('.').glob('*train*.log')) + list(Path('.').glob('*train*.txt'))
    
    if log_files:
        latest_log = max(log_files, key=os.path.getmtime)
        status['latest_log'] = str(latest_log)
        
        try:
            with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                if lines:
                    status['latest_entry'] = lines[-1].strip()
                    
                    # エポック情報抽出（簡易）
                    for line in reversed(lines[-10:]):
                        if 'epoch' in line.lower() or 'エポック' in line:
                            status['current_epoch'] = line.strip()
                            break
        except:
            pass
    
    # プロセス確認
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] and 'python' in proc.info['name'].lower():
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'nkat' in cmdline.lower() and ('train' in cmdline.lower() or 'colab' in cmdline.lower()):
                    status['training_active'] = True
                    status['training_pid'] = proc.info['pid']
                    break
        except:
            continue
    
    return status

def check_file_updates():
    """ファイル更新状況チェック"""
    files_to_check = [
        'NKAT_LoI_Final_Japanese_Updated_*.md',
        'nkat_ultimate_convergence_*.png',
        'kappa_moyal_comparison_*.png',
        'NKAT_Kappa_Minkowski_Test.py'
    ]
    
    updates = {}
    
    for pattern in files_to_check:
        matching_files = list(Path('.').glob(pattern))
        if matching_files:
            latest_file = max(matching_files, key=os.path.getmtime)
            mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(latest_file))
            updates[pattern] = {
                'file': str(latest_file),
                'modified': mod_time.strftime('%H:%M:%S'),
                'size': os.path.getsize(latest_file)
            }
    
    return updates

def display_status():
    """ステータス表示"""
    print("\n" + "="*80)
    print(f"🚀 NKAT 並列処理ステータス - {datetime.datetime.now().strftime('%H:%M:%S')}")
    print("="*80)
    
    # GPU状況
    gpu_info = check_gpu_usage()
    print("\n🔥 GPU状況:")
    for gpu in gpu_info:
        if 'utilization' in gpu:
            print(f"  GPU {gpu['gpu_id']}: {gpu['utilization']}% 使用中, "
                  f"メモリ {gpu['memory_used']}/{gpu['memory_total']}MB")
        else:
            print(f"  GPU情報: {gpu.get('status', 'Unknown')}")
    
    # 訓練状況
    training = check_training_status()
    print(f"\n🧠 訓練状況:")
    print(f"  アクティブ: {'🟢 YES' if training['training_active'] else '🔴 NO'}")
    if training.get('training_pid'):
        print(f"  PID: {training['training_pid']}")
    if training.get('current_epoch'):
        print(f"  現在エポック: {training['current_epoch']}")
    if training.get('latest_log'):
        print(f"  最新ログ: {training['latest_log']}")
    
    # ファイル更新
    updates = check_file_updates()
    print(f"\n📁 ファイル更新:")
    if updates:
        for pattern, info in updates.items():
            print(f"  {pattern}: {info['file']} ({info['modified']}, {info['size']} bytes)")
    else:
        print("  更新ファイルなし")
    
    # システムリソース
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    print(f"\n💻 システム:")
    print(f"  CPU: {cpu_percent}%")
    print(f"  メモリ: {memory.percent}% ({memory.used//1024//1024}MB/{memory.total//1024//1024}MB)")
    
    print("\n" + "="*80)

def main():
    """メインモニターループ"""
    print("🔍 NKAT 並列処理モニター開始")
    print("Ctrl+C で終了")
    
    try:
        while True:
            display_status()
            time.sleep(30)  # 30秒間隔
            
    except KeyboardInterrupt:
        print("\n👋 モニター終了")

if __name__ == "__main__":
    main() 