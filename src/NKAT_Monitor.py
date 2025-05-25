#!/usr/bin/env python3
"""
🔍 NKAT長期訓練リアルタイム監視システム
進捗・GPU使用量・収束状況をリアルタイム表示
"""

import os
import time
import json
import glob
import matplotlib.pyplot as plt
from datetime import datetime
import psutil

def monitor_training_progress():
    """長期訓練進捗監視"""
    
    print("🔍 NKAT長期訓練監視システム開始")
    print("=" * 60)
    
    checkpoint_dir = "./nkat_longterm_checkpoints"
    log_pattern = "nkat_training_log_*.txt"
    
    last_update = time.time()
    
    while True:
        try:
            # チェックポイント確認
            if os.path.exists(checkpoint_dir):
                checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
                latest_checkpoint = max(checkpoints, key=os.path.getctime) if checkpoints else None
                
                if latest_checkpoint:
                    checkpoint_time = os.path.getctime(latest_checkpoint)
                    checkpoint_name = os.path.basename(latest_checkpoint)
                    
                    print(f"📂 最新チェックポイント: {checkpoint_name}")
                    print(f"⏰ 更新時刻: {datetime.fromtimestamp(checkpoint_time).strftime('%H:%M:%S')}")
            
            # ログファイル確認
            log_files = glob.glob(log_pattern)
            if log_files:
                latest_log = max(log_files, key=os.path.getctime)
                
                try:
                    with open(latest_log, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        if lines:
                            last_line = lines[-1].strip()
                            print(f"📝 最新ログ: {last_line}")
                except Exception as e:
                    print(f"⚠️ ログ読み込みエラー: {e}")
            
            # GPU使用量確認
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / 1e9
                    gpu_max_memory = torch.cuda.max_memory_allocated() / 1e9
                    print(f"🔥 GPU使用量: {gpu_memory:.2f}GB / 最大: {gpu_max_memory:.2f}GB")
                    
                    # GPU温度（可能な場合）
                    try:
                        import nvidia_ml_py3 as nvml
                        nvml.nvmlInit()
                        handle = nvml.nvmlDeviceGetHandleByIndex(0)
                        temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                        print(f"🌡️ GPU温度: {temp}°C")
                    except:
                        pass
            except Exception as e:
                print(f"⚠️ GPU監視エラー: {e}")
            
            # CPU・メモリ使用量
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            print(f"💻 CPU使用率: {cpu_percent:.1f}%")
            print(f"💾 メモリ使用率: {memory.percent:.1f}% ({memory.used/1e9:.1f}GB/{memory.total/1e9:.1f}GB)")
            
            # 訓練継続確認
            python_processes = [p for p in psutil.process_iter(['pid', 'name', 'cmdline']) 
                              if 'python' in p.info['name'].lower() and 
                              any('NKAT' in str(cmd) for cmd in (p.info['cmdline'] or []))]
            
            if python_processes:
                print(f"🚀 NKAT訓練プロセス: {len(python_processes)}個実行中")
                for proc in python_processes:
                    try:
                        cpu_usage = proc.cpu_percent()
                        memory_usage = proc.memory_info().rss / 1e6  # MB
                        print(f"   PID {proc.pid}: CPU {cpu_usage:.1f}%, メモリ {memory_usage:.0f}MB")
                    except:
                        pass
            else:
                print("⚠️ NKAT訓練プロセスが見つかりません")
            
            print("-" * 60)
            print(f"🕐 監視時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)
            
            # 30秒間隔で更新
            time.sleep(30)
            
        except KeyboardInterrupt:
            print("\n🛑 監視終了")
            break
        except Exception as e:
            print(f"⚠️ 監視エラー: {e}")
            time.sleep(10)

def plot_training_progress():
    """訓練進捗グラフ作成"""
    
    print("📊 訓練進捗グラフ作成中...")
    
    # メトリクスファイル検索
    metrics_files = glob.glob("nkat_metrics_*.json")
    
    if not metrics_files:
        print("⚠️ メトリクスファイルが見つかりません")
        return
    
    latest_metrics = max(metrics_files, key=os.path.getctime)
    
    try:
        with open(latest_metrics, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        
        # グラフ作成
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('🔍 NKAT長期訓練進捗監視', fontsize=16, fontweight='bold')
        
        epochs = metrics.get('epoch', [])
        
        # 損失推移
        if 'total_loss' in metrics:
            axes[0, 0].plot(epochs, metrics['total_loss'], 'b-', linewidth=2)
            axes[0, 0].set_title('📉 総損失推移')
            axes[0, 0].set_xlabel('エポック')
            axes[0, 0].set_ylabel('損失')
            axes[0, 0].grid(True, alpha=0.3)
        
        # スペクトラル次元
        if 'spectral_dim_loss' in metrics:
            axes[0, 1].plot(epochs, metrics['spectral_dim_loss'], 'r-', linewidth=2)
            axes[0, 1].set_title('🎯 スペクトラル次元損失')
            axes[0, 1].set_xlabel('エポック')
            axes[0, 1].set_ylabel('スペクトラル次元損失')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 学習率
        if 'learning_rate' in metrics:
            axes[1, 0].plot(epochs, metrics['learning_rate'], 'g-', linewidth=2)
            axes[1, 0].set_title('📈 学習率推移')
            axes[1, 0].set_xlabel('エポック')
            axes[1, 0].set_ylabel('学習率')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        
        # GPU使用量
        if 'gpu_memory_mb' in metrics:
            axes[1, 1].plot(epochs, [m/1000 for m in metrics['gpu_memory_mb']], 'm-', linewidth=2)
            axes[1, 1].set_title('💾 GPU使用量')
            axes[1, 1].set_xlabel('エポック')
            axes[1, 1].set_ylabel('GPU使用量 (GB)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_file = f"nkat_progress_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 進捗グラフ保存: {plot_file}")
        
    except Exception as e:
        print(f"⚠️ グラフ作成エラー: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot_training_progress()
    else:
        monitor_training_progress() 