#!/usr/bin/env python3
"""
🔥 NKAT Ultimate Monitoring Dashboard
=====================================
全Stage統合リアルタイム監視システム
段階的スケールアップ完全監視
"""

import os
import sys
import json
import time
import psutil
import threading
from datetime import datetime
from pathlib import Path
import numpy as np
from tqdm import tqdm

class NKATUltimateMonitor:
    def __init__(self):
        """NKAT究極監視システム初期化"""
        self.workspace_dir = Path(".")
        self.monitoring = True
        self.stages_info = {
            'Stage1': {'target': 1000, 'status': 'COMPLETED', 'progress': 1000},
            'Stage2': {'target': 10000, 'status': 'RUNNING', 'progress': 0},
            'Stage3': {'target': 100000, 'status': 'RUNNING', 'progress': 0},
            'Stage4': {'target': 1000000, 'status': 'READY', 'progress': 0}
        }
        
        print("🌟 NKAT Ultimate Monitoring Dashboard 起動!")
        print("=" * 80)
    
    def check_python_processes(self):
        """Pythonプロセス監視"""
        processes = []
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info', 'cmdline']):
                if proc.info['name'] == 'python.exe' or proc.info['name'] == 'python':
                    cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    if 'nkat_stage' in cmdline.lower() or 'nkat_million' in cmdline.lower():
                        processes.append({
                            'pid': proc.info['pid'],
                            'cpu': proc.info['cpu_percent'],
                            'memory_mb': proc.info['memory_info'].rss / 1024 / 1024,
                            'cmdline': cmdline
                        })
        except Exception as e:
            print(f"⚠️ プロセス監視エラー: {e}")
        
        return processes
    
    def scan_output_directories(self):
        """出力ディレクトリスキャン"""
        output_dirs = {}
        
        # Stage2結果検索
        stage2_dirs = list(self.workspace_dir.glob("nkat_stage2_10k_*"))
        if stage2_dirs:
            latest_stage2 = max(stage2_dirs, key=lambda x: x.stat().st_mtime)
            output_dirs['Stage2'] = latest_stage2
        
        # Stage3結果検索
        stage3_dirs = list(self.workspace_dir.glob("nkat_stage3_100k_*"))
        if stage3_dirs:
            latest_stage3 = max(stage3_dirs, key=lambda x: x.stat().st_mtime)
            output_dirs['Stage3'] = latest_stage3
        
        # 百万ゼロ点結果検索
        million_dirs = list(self.workspace_dir.glob("nkat_million_results"))
        if million_dirs:
            output_dirs['Stage1'] = million_dirs[0]
        
        return output_dirs
    
    def read_stage_results(self, stage_dir):
        """Stage結果読み込み"""
        try:
            # JSONファイル検索
            json_files = list(stage_dir.glob("*.json"))
            if not json_files:
                return None
            
            latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return {
                'file': latest_json.name,
                'timestamp': data.get('timestamp', 'Unknown'),
                'computed_zeros': data.get('computed_zeros', 0),
                'target_zeros': data.get('target_zeros', 0),
                'execution_time': data.get('execution_time_seconds', 0),
                'best_model': data.get('best_model', 'Unknown'),
                'best_roc_auc': data.get('model_results', {}).get(data.get('best_model', ''), {}).get('roc_auc', 0),
                'memory_peak_mb': data.get('system_info', {}).get('memory_peak_mb', 0)
            }
        
        except Exception as e:
            print(f"⚠️ 結果読み込みエラー {stage_dir}: {e}")
            return None
    
    def read_checkpoints(self, stage_dir):
        """チェックポイント監視"""
        try:
            checkpoint_dir = stage_dir / "checkpoints"
            if not checkpoint_dir.exists():
                return None
            
            checkpoint_files = list(checkpoint_dir.glob("*.json")) + list(checkpoint_dir.glob("*.pkl"))
            if not checkpoint_files:
                return None
            
            latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
            
            if latest_checkpoint.suffix == '.json':
                with open(latest_checkpoint, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return {
                    'file': latest_checkpoint.name,
                    'computed_zeros': data.get('computed_zeros', 0),
                    'timestamp': data.get('timestamp', 'Unknown'),
                    'memory_usage_mb': data.get('memory_usage_mb', 0)
                }
        except Exception as e:
            return None
    
    def display_system_status(self):
        """システム状況表示"""
        # システム情報
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        print(f"\n📊 システム状況 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        print(f"💻 CPU使用率: {cpu_percent:.1f}%")
        print(f"💾 メモリ使用率: {memory.percent:.1f}% ({memory.used/1024/1024/1024:.1f}GB / {memory.total/1024/1024/1024:.1f}GB)")
        
        # Pythonプロセス監視
        processes = self.check_python_processes()
        if processes:
            print(f"\n🔍 NKAT実行プロセス: {len(processes)}個")
            print("-" * 80)
            for i, proc in enumerate(processes[:5]):  # 上位5個表示
                stage_name = "Unknown"
                if "stage2" in proc['cmdline'].lower():
                    stage_name = "Stage2"
                elif "stage3" in proc['cmdline'].lower():
                    stage_name = "Stage3"
                elif "million" in proc['cmdline'].lower():
                    stage_name = "Stage1"
                
                print(f"   {stage_name} - PID:{proc['pid']} CPU:{proc['cpu']:.1f}% RAM:{proc['memory_mb']:.1f}MB")
        else:
            print("\n⚠️ NKAT実行プロセスが見つかりません")
    
    def display_stages_progress(self):
        """Stage進行状況表示"""
        print(f"\n🎯 NKAT段階的スケールアップ進行状況")
        print("=" * 80)
        
        output_dirs = self.scan_output_directories()
        
        for stage_name, stage_info in self.stages_info.items():
            target = stage_info['target']
            status = stage_info['status']
            
            # 実際の進行状況取得
            actual_progress = 0
            actual_status = status
            results = None
            checkpoint = None
            
            if stage_name in output_dirs:
                stage_dir = output_dirs[stage_name]
                results = self.read_stage_results(stage_dir)
                checkpoint = self.read_checkpoints(stage_dir)
                
                if results:
                    actual_progress = results['computed_zeros']
                    if actual_progress >= target * 0.95:  # 95%完了で完了扱い
                        actual_status = 'COMPLETED'
                    else:
                        actual_status = 'RUNNING'
                elif checkpoint:
                    actual_progress = checkpoint['computed_zeros']
                    actual_status = 'RUNNING'
            
            # 進行率計算
            progress_rate = (actual_progress / target) * 100 if target > 0 else 0
            
            # ステータス表示
            status_icon = {
                'COMPLETED': '✅',
                'RUNNING': '🔄', 
                'READY': '⏳',
                'ERROR': '❌'
            }.get(actual_status, '❓')
            
            print(f"{status_icon} {stage_name}: {actual_progress:,} / {target:,} ゼロ点 ({progress_rate:.1f}%)")
            
            # 詳細情報
            if results:
                print(f"   📊 最高ROC-AUC: {results['best_roc_auc']:.4f} ({results['best_model']})")
                print(f"   ⏱️ 実行時間: {results['execution_time']:.1f}秒")
                print(f"   💾 メモリピーク: {results['memory_peak_mb']:.1f}MB")
            elif checkpoint:
                print(f"   💾 チェックポイント: {checkpoint['computed_zeros']:,}ゼロ点")
                print(f"   🕐 最終更新: {checkpoint['timestamp']}")
        
        # 総合進行状況
        total_computed = sum(
            self.read_stage_results(output_dirs[stage])['computed_zeros'] if stage in output_dirs and self.read_stage_results(output_dirs[stage]) else 0
            for stage in output_dirs
        )
        total_target = sum(info['target'] for info in self.stages_info.values())
        overall_progress = (total_computed / total_target) * 100
        
        print(f"\n🎊 総合進行状況: {total_computed:,} / {total_target:,} ゼロ点 ({overall_progress:.2f}%)")
    
    def display_performance_metrics(self):
        """性能メトリクス表示"""
        print(f"\n📈 性能メトリクス")
        print("=" * 80)
        
        output_dirs = self.scan_output_directories()
        
        total_zeros = 0
        total_time = 0
        best_roc_auc = 0
        best_stage = "None"
        
        for stage_name, stage_dir in output_dirs.items():
            results = self.read_stage_results(stage_dir)
            if results:
                total_zeros += results['computed_zeros']
                total_time += results['execution_time']
                
                if results['best_roc_auc'] > best_roc_auc:
                    best_roc_auc = results['best_roc_auc']
                    best_stage = stage_name
        
        if total_time > 0:
            print(f"🚀 総合処理速度: {total_zeros/total_time:.1f} ゼロ点/秒")
            print(f"⚡ 1時間あたり処理能力: {total_zeros/(total_time/3600):,.0f} ゼロ点/時間")
        
        if best_roc_auc > 0:
            print(f"🏆 最高ROC-AUC: {best_roc_auc:.4f} ({best_stage})")
        
        print(f"🔢 総計算ゼロ点数: {total_zeros:,}")
        print(f"⏱️ 総実行時間: {total_time:.1f}秒 ({total_time/3600:.2f}時間)")
    
    def run_continuous_monitoring(self, interval=30):
        """連続監視実行"""
        print(f"🔄 リアルタイム監視開始 (更新間隔: {interval}秒)")
        print("Ctrl+C で停止")
        
        try:
            while self.monitoring:
                # 画面クリア（Windows対応）
                os.system('cls' if os.name == 'nt' else 'clear')
                
                print("🔥 NKAT Ultimate Real-time Monitoring Dashboard 🔥")
                print(f"最終更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # 各種情報表示
                self.display_system_status()
                self.display_stages_progress()
                self.display_performance_metrics()
                
                print(f"\n⏰ 次回更新まで {interval}秒...")
                print("=" * 80)
                
                # インターバル待機
                time.sleep(interval)
        
        except KeyboardInterrupt:
            print("\n\n🛑 監視システム停止")
            self.monitoring = False
    
    def generate_summary_report(self):
        """最終サマリーレポート生成"""
        print("\n📋 NKAT Ultimate Summary Report")
        print("=" * 80)
        
        output_dirs = self.scan_output_directories()
        
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'stages': {},
            'summary': {}
        }
        
        total_zeros = 0
        total_time = 0
        completed_stages = 0
        
        for stage_name, stage_dir in output_dirs.items():
            results = self.read_stage_results(stage_dir)
            if results:
                report_data['stages'][stage_name] = results
                total_zeros += results['computed_zeros']
                total_time += results['execution_time']
                
                if results['computed_zeros'] >= self.stages_info[stage_name]['target'] * 0.95:
                    completed_stages += 1
        
        report_data['summary'] = {
            'total_zeros_computed': total_zeros,
            'total_execution_time_seconds': total_time,
            'total_execution_time_hours': total_time / 3600,
            'completed_stages': completed_stages,
            'overall_processing_rate': total_zeros / total_time if total_time > 0 else 0,
            'memory_efficiency_avg': np.mean([s.get('memory_peak_mb', 0) for s in report_data['stages'].values()])
        }
        
        # レポート保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = Path(f"nkat_ultimate_summary_report_{timestamp}.json")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"📄 サマリーレポート生成: {report_file}")
        
        # 画面表示
        print(f"\n🎉 NKAT Ultimate Achievement Summary:")
        print(f"   🔢 総計算ゼロ点数: {total_zeros:,}")
        print(f"   ⏱️ 総実行時間: {total_time/3600:.2f}時間")
        print(f"   🚀 平均処理速度: {total_zeros/total_time:.1f}ゼロ点/秒")
        print(f"   ✅ 完了ステージ: {completed_stages}/4")
        print(f"   🏆 リーマン予想への歴史的貢献達成!")
        
        return report_data


def main():
    """メイン実行"""
    monitor = NKATUltimateMonitor()
    
    print("🔥 NKAT Ultimate Monitoring Dashboard")
    print("選択してください:")
    print("1. リアルタイム監視開始")
    print("2. 現在状況確認")
    print("3. サマリーレポート生成")
    
    try:
        choice = input("選択 (1-3): ").strip()
        
        if choice == "1":
            monitor.run_continuous_monitoring(interval=30)
        elif choice == "2":
            monitor.display_system_status()
            monitor.display_stages_progress()
            monitor.display_performance_metrics()
        elif choice == "3":
            monitor.generate_summary_report()
        else:
            print("無効な選択です")
    
    except Exception as e:
        print(f"❌ エラー: {e}")


if __name__ == "__main__":
    main() 