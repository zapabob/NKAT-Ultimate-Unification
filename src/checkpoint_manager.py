#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
💾 RTX3080極限計算チェックポイント管理システム
Advanced Checkpoint Management System for RTX3080 Extreme Computation

機能:
- チェックポイントの詳細管理
- 計算進捗の監視
- 電源断からの完全復旧
- リアルタイム状況表示
- バックアップ・復元機能

Author: NKAT Research Team
Date: 2025-05-26
Version: v1.0 - Checkpoint Management Edition
"""

import json
import pickle
import datetime
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import psutil
import os
import shutil
import glob
import threading
import subprocess
import sys

class RTX3080CheckpointManager:
    """RTX3080極限計算用チェックポイント管理システム"""
    
    def __init__(self, checkpoint_dir: str = "rtx3080_extreme_checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # 重要なファイルパス
        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        self.latest_checkpoint_file = self.checkpoint_dir / "latest_checkpoint.json"
        self.status_file = self.checkpoint_dir / "computation_status.json"
        self.backup_dir = self.checkpoint_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        # 監視スレッド
        self.monitoring_active = False
        self.monitor_thread = None
        
    def monitor_computation_status(self, interval: int = 30):
        """計算状況のリアルタイム監視"""
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    status = self.get_current_status()
                    if status:
                        self.display_status(status)
                        self.save_status(status)
                    time.sleep(interval)
                except Exception as e:
                    print(f"⚠️ 監視エラー: {e}")
                    time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"📊 計算監視開始（{interval}秒間隔）")
    
    def stop_monitoring(self):
        """監視停止"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("⏹️ 計算監視停止")
    
    def get_current_status(self) -> Optional[Dict]:
        """現在の計算状況を取得"""
        try:
            # 最新チェックポイントから状況を読み取り
            if self.latest_checkpoint_file.exists():
                with open(self.latest_checkpoint_file, 'r') as f:
                    metadata = json.load(f)
                
                checkpoint_file = metadata['checkpoint_file']
                if Path(checkpoint_file).exists():
                    with open(checkpoint_file, 'rb') as f:
                        checkpoint_data = pickle.load(f)
                    
                    # システム情報を追加
                    current_status = {
                        'timestamp': datetime.datetime.now().isoformat(),
                        'checkpoint_info': metadata,
                        'computation_state': checkpoint_data.get('computation_state', {}),
                        'results_summary': self._summarize_results(checkpoint_data.get('results_so_far', {})),
                        'system_status': self._get_system_status(),
                        'gpu_status': self._get_gpu_status()
                    }
                    
                    return current_status
            
            return None
            
        except Exception as e:
            print(f"❌ 状況取得エラー: {e}")
            return None
    
    def _summarize_results(self, results: Dict) -> Dict:
        """結果の要約作成"""
        if not results:
            return {}
        
        try:
            total_gammas = len(results.get('gamma_values', []))
            completed_gammas = len(results.get('spectral_dimensions', []))
            
            # 成功分類の統計
            classifications = results.get('success_classifications', [])
            classification_counts = {}
            for cls in classifications:
                classification_counts[cls] = classification_counts.get(cls, 0) + 1
            
            # 収束統計
            convergences = results.get('convergence_to_half', [])
            valid_convergences = [c for c in convergences if c is not None and not (isinstance(c, float) and c != c)]  # NaN除外
            
            summary = {
                'total_gamma_values': total_gammas,
                'completed_gamma_values': completed_gammas,
                'progress_percentage': (completed_gammas / total_gammas * 100) if total_gammas > 0 else 0,
                'classification_counts': classification_counts,
                'convergence_stats': {
                    'count': len(valid_convergences),
                    'best_convergence': min(valid_convergences) if valid_convergences else None,
                    'average_convergence': sum(valid_convergences) / len(valid_convergences) if valid_convergences else None
                }
            }
            
            return summary
            
        except Exception as e:
            return {'error': str(e)}
    
    def _get_system_status(self) -> Dict:
        """システム状況取得"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            return {
                'cpu_usage_percent': cpu_percent,
                'memory_total_gb': memory.total / 1e9,
                'memory_available_gb': memory.available / 1e9,
                'memory_usage_percent': memory.percent,
                'disk_total_gb': disk.total / 1e9,
                'disk_free_gb': disk.free / 1e9,
                'disk_usage_percent': (disk.used / disk.total) * 100
            }
        except:
            return {}
    
    def _get_gpu_status(self) -> Dict:
        """GPU状況取得"""
        try:
            # nvidia-smiコマンドを使ってGPU情報を取得
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu', '--format=csv,noheader,nounits'], 
                                   capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                gpu_info = result.stdout.strip().split(', ')
                return {
                    'gpu_available': True,
                    'memory_total_mb': int(gpu_info[0]),
                    'memory_used_mb': int(gpu_info[1]),
                    'memory_free_mb': int(gpu_info[2]),
                    'utilization_percent': int(gpu_info[3]),
                    'temperature_c': int(gpu_info[4]),
                    'memory_usage_percent': (int(gpu_info[1]) / int(gpu_info[0])) * 100
                }
            else:
                return {'gpu_available': False, 'error': 'nvidia-smi failed'}
                
        except Exception as e:
            return {'gpu_available': False, 'error': str(e)}
    
    def display_status(self, status: Dict):
        """状況の表示"""
        print("\n" + "=" * 100)
        print("🔥 RTX3080極限計算 - リアルタイム状況")
        print("=" * 100)
        print(f"📅 更新時刻: {status['timestamp']}")
        
        # 計算進捗
        if 'results_summary' in status:
            summary = status['results_summary']
            if summary:
                print(f"\n📊 計算進捗:")
                print(f"  完了γ値: {summary.get('completed_gamma_values', 0)}/{summary.get('total_gamma_values', 0)}")
                print(f"  進捗率: {summary.get('progress_percentage', 0):.1f}%")
                
                if 'classification_counts' in summary:
                    print(f"  成功分類:")
                    for cls, count in summary['classification_counts'].items():
                        print(f"    {cls}: {count}個")
                
                conv_stats = summary.get('convergence_stats', {})
                if conv_stats.get('best_convergence') is not None:
                    print(f"  最良収束: {conv_stats['best_convergence']:.2e}")
        
        # システム状況
        if 'system_status' in status:
            sys_status = status['system_status']
            print(f"\n💻 システム状況:")
            print(f"  CPU使用率: {sys_status.get('cpu_usage_percent', 0):.1f}%")
            print(f"  RAM使用率: {sys_status.get('memory_usage_percent', 0):.1f}%")
            print(f"  空きRAM: {sys_status.get('memory_available_gb', 0):.1f}GB")
            print(f"  ディスク使用率: {sys_status.get('disk_usage_percent', 0):.1f}%")
        
        # GPU状況
        if 'gpu_status' in status:
            gpu_status = status['gpu_status']
            if gpu_status.get('gpu_available', False):
                print(f"\n🎮 GPU状況 (RTX3080):")
                print(f"  GPU使用率: {gpu_status.get('utilization_percent', 0)}%")
                print(f"  VRAM使用率: {gpu_status.get('memory_usage_percent', 0):.1f}%")
                print(f"  VRAM使用量: {gpu_status.get('memory_used_mb', 0)/1024:.1f}GB/{gpu_status.get('memory_total_mb', 0)/1024:.1f}GB")
                print(f"  GPU温度: {gpu_status.get('temperature_c', 0)}°C")
            else:
                print(f"\n⚠️ GPU情報取得失敗: {gpu_status.get('error', 'Unknown')}")
        
        print("=" * 100)
    
    def save_status(self, status: Dict):
        """状況をファイルに保存"""
        try:
            with open(self.status_file, 'w') as f:
                json.dump(status, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"⚠️ 状況保存エラー: {e}")
    
    def list_checkpoints(self) -> List[Dict]:
        """利用可能なチェックポイントのリスト"""
        checkpoints = []
        
        for checkpoint_file in self.checkpoint_dir.glob("*.pkl"):
            try:
                with open(checkpoint_file, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                
                info = {
                    'file': str(checkpoint_file),
                    'name': checkpoint_file.stem,
                    'timestamp': checkpoint_data.get('timestamp', 'Unknown'),
                    'gamma_index': checkpoint_data.get('gamma_index', 0),
                    'file_size_mb': checkpoint_file.stat().st_size / 1e6
                }
                checkpoints.append(info)
                
            except Exception as e:
                print(f"⚠️ チェックポイント読み込みエラー {checkpoint_file}: {e}")
        
        return sorted(checkpoints, key=lambda x: x['gamma_index'], reverse=True)
    
    def backup_checkpoint(self, checkpoint_name: str) -> bool:
        """チェックポイントのバックアップ"""
        try:
            source_file = self.checkpoint_dir / f"{checkpoint_name}.pkl"
            if not source_file.exists():
                print(f"❌ チェックポイントファイルが見つかりません: {checkpoint_name}")
                return False
            
            backup_name = f"{checkpoint_name}_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            backup_file = self.backup_dir / backup_name
            
            shutil.copy2(source_file, backup_file)
            print(f"💾 バックアップ作成: {backup_name}")
            return True
            
        except Exception as e:
            print(f"❌ バックアップエラー: {e}")
            return False
    
    def restore_checkpoint(self, backup_name: str) -> bool:
        """バックアップからの復元"""
        try:
            backup_file = self.backup_dir / backup_name
            if not backup_file.exists():
                print(f"❌ バックアップファイルが見つかりません: {backup_name}")
                return False
            
            # 元のチェックポイント名を推定
            original_name = backup_name.replace('_backup_', '_').split('_')[:-2]
            original_name = '_'.join(original_name) + '.pkl'
            restore_file = self.checkpoint_dir / original_name
            
            shutil.copy2(backup_file, restore_file)
            print(f"🔄 復元完了: {original_name}")
            return True
            
        except Exception as e:
            print(f"❌ 復元エラー: {e}")
            return False
    
    def cleanup_old_files(self, keep_last_n: int = 10):
        """古いファイルの清理"""
        try:
            # チェックポイントファイルの清理
            checkpoint_files = sorted(self.checkpoint_dir.glob("*.pkl"), 
                                    key=lambda x: x.stat().st_mtime, reverse=True)
            
            deleted_count = 0
            if len(checkpoint_files) > keep_last_n:
                for old_file in checkpoint_files[keep_last_n:]:
                    old_file.unlink()
                    deleted_count += 1
            
            # バックアップファイルの清理（30日以上古いもの）
            cutoff_time = time.time() - (30 * 24 * 3600)  # 30日前
            backup_files = list(self.backup_dir.glob("*.pkl"))
            backup_deleted = 0
            
            for backup_file in backup_files:
                if backup_file.stat().st_mtime < cutoff_time:
                    backup_file.unlink()
                    backup_deleted += 1
            
            print(f"🗑️ 清理完了: チェックポイント{deleted_count}個、バックアップ{backup_deleted}個削除")
            
        except Exception as e:
            print(f"⚠️ 清理エラー: {e}")
    
    def generate_report(self) -> str:
        """計算状況レポート生成"""
        try:
            status = self.get_current_status()
            if not status:
                return "状況情報が取得できませんでした。"
            
            report_lines = []
            report_lines.append("# RTX3080極限計算 状況レポート")
            report_lines.append(f"生成日時: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")
            
            # 計算進捗
            if 'results_summary' in status:
                summary = status['results_summary']
                report_lines.append("## 計算進捗")
                report_lines.append(f"- 総γ値数: {summary.get('total_gamma_values', 0)}")
                report_lines.append(f"- 完了γ値数: {summary.get('completed_gamma_values', 0)}")
                report_lines.append(f"- 進捗率: {summary.get('progress_percentage', 0):.2f}%")
                report_lines.append("")
                
                if 'classification_counts' in summary:
                    report_lines.append("### 成功分類統計")
                    for cls, count in summary['classification_counts'].items():
                        report_lines.append(f"- {cls}: {count}個")
                    report_lines.append("")
            
            # システム状況
            if 'system_status' in status:
                sys_status = status['system_status']
                report_lines.append("## システム状況")
                report_lines.append(f"- CPU使用率: {sys_status.get('cpu_usage_percent', 0):.1f}%")
                report_lines.append(f"- RAM使用率: {sys_status.get('memory_usage_percent', 0):.1f}%")
                report_lines.append(f"- 空きRAM: {sys_status.get('memory_available_gb', 0):.1f}GB")
                report_lines.append("")
            
            # GPU状況
            if 'gpu_status' in status:
                gpu_status = status['gpu_status']
                report_lines.append("## GPU状況 (RTX3080)")
                if gpu_status.get('gpu_available', False):
                    report_lines.append(f"- GPU使用率: {gpu_status.get('utilization_percent', 0)}%")
                    report_lines.append(f"- VRAM使用率: {gpu_status.get('memory_usage_percent', 0):.1f}%")
                    report_lines.append(f"- GPU温度: {gpu_status.get('temperature_c', 0)}°C")
                else:
                    report_lines.append(f"- GPU情報取得失敗: {gpu_status.get('error', 'Unknown')}")
                report_lines.append("")
            
            # チェックポイント情報
            checkpoints = self.list_checkpoints()
            report_lines.append("## チェックポイント情報")
            report_lines.append(f"- 利用可能チェックポイント数: {len(checkpoints)}")
            if checkpoints:
                latest = checkpoints[0]
                report_lines.append(f"- 最新チェックポイント: {latest['name']}")
                report_lines.append(f"- 最新チェックポイント時刻: {latest['timestamp']}")
            report_lines.append("")
            
            return "\n".join(report_lines)
            
        except Exception as e:
            return f"レポート生成エラー: {e}"

def interactive_checkpoint_manager():
    """対話式チェックポイント管理"""
    manager = RTX3080CheckpointManager()
    
    print("🔥 RTX3080極限計算チェックポイント管理システム")
    print("=" * 60)
    
    while True:
        print("\n利用可能なコマンド:")
        print("1. 現在の状況表示 (status)")
        print("2. チェックポイント一覧 (list)")
        print("3. 監視開始 (monitor)")
        print("4. 監視停止 (stop)")
        print("5. レポート生成 (report)")
        print("6. バックアップ作成 (backup)")
        print("7. 清理実行 (cleanup)")
        print("8. 終了 (exit)")
        
        command = input("\nコマンドを入力: ").strip().lower()
        
        if command in ['1', 'status']:
            status = manager.get_current_status()
            if status:
                manager.display_status(status)
            else:
                print("❌ 状況情報が取得できませんでした")
        
        elif command in ['2', 'list']:
            checkpoints = manager.list_checkpoints()
            print(f"\n📂 利用可能なチェックポイント ({len(checkpoints)}個):")
            for i, cp in enumerate(checkpoints):
                print(f"  {i+1}. {cp['name']} (γ={cp['gamma_index']}, {cp['file_size_mb']:.1f}MB)")
        
        elif command in ['3', 'monitor']:
            interval = input("監視間隔（秒、デフォルト30）: ").strip()
            try:
                interval = int(interval) if interval else 30
                manager.monitor_computation_status(interval)
                print("📊 監視開始しました。'stop'コマンドで停止できます。")
            except ValueError:
                print("❌ 無効な間隔値です")
        
        elif command in ['4', 'stop']:
            manager.stop_monitoring()
        
        elif command in ['5', 'report']:
            report = manager.generate_report()
            print("\n" + report)
            
            save_report = input("\nレポートをファイルに保存しますか？ (y/N): ").strip().lower()
            if save_report == 'y':
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                report_file = f"rtx3080_computation_report_{timestamp}.md"
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(report)
                print(f"💾 レポート保存: {report_file}")
        
        elif command in ['6', 'backup']:
            checkpoints = manager.list_checkpoints()
            if not checkpoints:
                print("❌ バックアップ可能なチェックポイントがありません")
                continue
            
            print("\nバックアップ対象を選択:")
            for i, cp in enumerate(checkpoints):
                print(f"  {i+1}. {cp['name']}")
            
            try:
                choice = int(input("番号を入力: ")) - 1
                if 0 <= choice < len(checkpoints):
                    checkpoint_name = checkpoints[choice]['name']
                    manager.backup_checkpoint(checkpoint_name)
                else:
                    print("❌ 無効な選択です")
            except ValueError:
                print("❌ 無効な入力です")
        
        elif command in ['7', 'cleanup']:
            keep_count = input("保持するチェックポイント数（デフォルト10）: ").strip()
            try:
                keep_count = int(keep_count) if keep_count else 10
                manager.cleanup_old_files(keep_count)
            except ValueError:
                print("❌ 無効な数値です")
        
        elif command in ['8', 'exit']:
            if manager.monitoring_active:
                manager.stop_monitoring()
            print("👋 チェックポイント管理システムを終了します")
            break
        
        else:
            print("❌ 無効なコマンドです")

if __name__ == "__main__":
    interactive_checkpoint_manager() 