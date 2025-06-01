#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔋 NKAT 電源断復旧状況レポートシステム
現在の復旧進捗、実行中プロセス、結果統合
"""
import os
import json
import glob
import time
import psutil
from datetime import datetime
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional

# 英語表記設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class NKATPowerOutageRecoveryStatus:
    """電源断復旧状況分析システム"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_dir = f"recovery_status_{self.timestamp}"
        os.makedirs(self.report_dir, exist_ok=True)
        
        print("🔋 NKAT Power Outage Recovery Status System")
        print("="*60)
        print(f"📅 Check Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Report Dir: {self.report_dir}")
        print("="*60)
    
    def check_running_processes(self) -> Dict[str, List[Dict]]:
        """実行中のNKATプロセス確認"""
        
        print("🔍 Scanning Running NKAT Processes...")
        
        nkat_processes = []
        python_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'memory_info']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline:
                    cmdline_str = ' '.join(cmdline)
                    
                    # NKATスクリプト検出
                    if 'nkat' in cmdline_str.lower() and '.py' in cmdline_str:
                        process_info = {
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'cmdline': cmdline_str,
                            'runtime_minutes': (time.time() - proc.info['create_time']) / 60,
                            'memory_mb': proc.info['memory_info'].rss / (1024*1024) if proc.info['memory_info'] else 0
                        }
                        
                        # スクリプトタイプ識別
                        if 'emergency_recovery' in cmdline_str:
                            process_info['type'] = 'Emergency Recovery'
                        elif 'stage5' in cmdline_str:
                            process_info['type'] = 'Stage5 Interpretability'
                        elif 'enhanced_transformer_v2' in cmdline_str:
                            process_info['type'] = 'Enhanced Training'
                        elif 'mission_recovery' in cmdline_str:
                            process_info['type'] = 'Mission Recovery'
                        else:
                            process_info['type'] = 'Other NKAT'
                        
                        nkat_processes.append(process_info)
                        
                    # Python/GPU関連プロセス
                    elif 'python' in proc.info['name'].lower():
                        python_processes.append({
                            'pid': proc.info['pid'],
                            'cmdline': cmdline_str[:100] + '...' if len(cmdline_str) > 100 else cmdline_str,
                            'memory_mb': proc.info['memory_info'].rss / (1024*1024) if proc.info['memory_info'] else 0
                        })
            
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        print(f"✅ Found {len(nkat_processes)} NKAT processes")
        for proc in nkat_processes:
            print(f"  🔄 {proc['type']}: PID {proc['pid']}, {proc['runtime_minutes']:.1f}min, {proc['memory_mb']:.0f}MB")
        
        return {
            'nkat_processes': nkat_processes,
            'python_processes': python_processes[:5]  # 上位5個
        }
    
    def scan_recovery_outputs(self) -> Dict[str, Any]:
        """復旧関連の出力ファイルスキャン"""
        
        print("\n📂 Scanning Recovery Output Files...")
        
        output_files = {
            'emergency_recovery': [],
            'stage_results': [],
            'checkpoints': [],
            'visualizations': [],
            'logs': [],
            'reports': []
        }
        
        # Emergency Recovery関連
        emergency_patterns = [
            "recovery_data/emergency_*/*.pth",
            "recovery_data/emergency_*/*.json",
            "recovery_data/emergency_*/*.png"
        ]
        
        for pattern in emergency_patterns:
            files = glob.glob(pattern)
            for file in files:
                output_files['emergency_recovery'].append({
                    'path': file,
                    'size_mb': os.path.getsize(file) / (1024*1024),
                    'modified': os.path.getmtime(file),
                    'type': 'checkpoint' if '.pth' in file else 'json' if '.json' in file else 'visualization'
                })
        
        # Stage結果
        stage_patterns = [
            "logs/*stage*20250601*.json",
            "*stage*20250601*.png",
            "nkat_comprehensive_analysis_*.png",
            "NKAT_Comprehensive_Analysis_Report_*.md"
        ]
        
        for pattern in stage_patterns:
            files = glob.glob(pattern)
            for file in files:
                output_files['stage_results'].append({
                    'path': file,
                    'size_mb': os.path.getsize(file) / (1024*1024),
                    'modified': os.path.getmtime(file)
                })
        
        # チェックポイント
        checkpoint_patterns = [
            "checkpoints/*.pth",
            "checkpoints/*/*.pth"
        ]
        
        for pattern in checkpoint_patterns:
            files = glob.glob(pattern)
            for file in files:
                if os.path.getsize(file) > 1024*1024:  # 1MB以上
                    output_files['checkpoints'].append({
                        'path': file,
                        'size_mb': os.path.getsize(file) / (1024*1024),
                        'modified': os.path.getmtime(file)
                    })
        
        # 可視化ファイル
        viz_patterns = ["*.png", "*.jpg"]
        for pattern in viz_patterns:
            files = glob.glob(pattern)
            for file in files:
                if 'nkat' in file.lower() and os.path.getmtime(file) > time.time() - 3600:  # 1時間以内
                    output_files['visualizations'].append({
                        'path': file,
                        'size_mb': os.path.getsize(file) / (1024*1024),
                        'modified': os.path.getmtime(file)
                    })
        
        # ログファイル
        log_patterns = ["logs/*.json", "logs/*.log"]
        for pattern in log_patterns:
            files = glob.glob(pattern)
            for file in files:
                if os.path.getmtime(file) > time.time() - 3600:  # 1時間以内
                    output_files['logs'].append({
                        'path': file,
                        'size_mb': os.path.getsize(file) / (1024*1024),
                        'modified': os.path.getmtime(file)
                    })
        
        # レポート
        report_patterns = ["*.md", "*report*.json"]
        for pattern in report_patterns:
            files = glob.glob(pattern)
            for file in files:
                if 'nkat' in file.lower() and os.path.getmtime(file) > time.time() - 3600:  # 1時間以内
                    output_files['reports'].append({
                        'path': file,
                        'size_mb': os.path.getsize(file) / (1024*1024),
                        'modified': os.path.getmtime(file)
                    })
        
        # サマリー表示
        total_files = sum(len(files) for files in output_files.values())
        print(f"✅ Found {total_files} recovery-related files:")
        
        for category, files in output_files.items():
            if files:
                print(f"  📋 {category.replace('_', ' ').title()}: {len(files)} files")
                # 最新2ファイル表示
                sorted_files = sorted(files, key=lambda x: x['modified'], reverse=True)
                for file in sorted_files[:2]:
                    print(f"    - {file['path']} ({file['size_mb']:.1f}MB)")
        
        return output_files
    
    def analyze_latest_results(self, output_files: Dict[str, Any]) -> Dict[str, Any]:
        """最新結果分析"""
        
        print("\n🔬 Analyzing Latest Results...")
        
        analysis = {
            'recovery_status': 'Unknown',
            'latest_accuracy': None,
            'stage_progress': {},
            'emergency_training_status': 'Unknown',
            'key_findings': []
        }
        
        # Emergency Recovery結果確認
        emergency_jsons = [f for f in output_files['emergency_recovery'] if f['type'] == 'json']
        if emergency_jsons:
            latest_emergency = max(emergency_jsons, key=lambda x: x['modified'])
            try:
                with open(latest_emergency['path'], 'r', encoding='utf-8') as f:
                    emergency_data = json.load(f)
                
                analysis['latest_accuracy'] = emergency_data.get('final_accuracy', None)
                analysis['recovery_status'] = emergency_data.get('recovery_status', 'Unknown')
                analysis['emergency_training_status'] = 'Completed'
                
                if emergency_data.get('final_accuracy', 0) > 0.8:
                    analysis['key_findings'].append("🟢 Emergency recovery achieved >80% accuracy")
                elif emergency_data.get('final_accuracy', 0) > 0.5:
                    analysis['key_findings'].append("🟡 Emergency recovery achieved >50% accuracy")
                else:
                    analysis['key_findings'].append("🔴 Emergency recovery needs improvement")
                
                print(f"📊 Emergency Recovery Status: {analysis['recovery_status']}")
                if analysis['latest_accuracy']:
                    print(f"🎯 Latest Accuracy: {analysis['latest_accuracy']:.3f}")
                
            except Exception as e:
                print(f"⚠️ Could not read emergency recovery data: {e}")
        
        # Stage結果確認
        stage_jsons = [f for f in output_files['stage_results'] if '.json' in f['path']]
        if stage_jsons:
            latest_stage = max(stage_jsons, key=lambda x: x['modified'])
            try:
                with open(latest_stage['path'], 'r', encoding='utf-8') as f:
                    stage_data = json.load(f)
                
                # Stage2データ
                if 'stage2' in stage_data:
                    s2_data = stage_data['stage2']
                    analysis['stage_progress']['Stage2'] = {
                        'global_tpe': s2_data.get('global_tpe', 0),
                        'status': '✅ Complete' if s2_data.get('global_tpe', 0) > 0.5 else '⚠️ Needs Improvement'
                    }
                
                # Stage3データ
                if 'stage3' in stage_data:
                    s3_data = stage_data['stage3']
                    analysis['stage_progress']['Stage3'] = {
                        'robustness_score': s3_data.get('robustness_score', 0),
                        'status': '✅ Complete' if s3_data.get('robustness_score', 0) > 50 else '⚠️ Needs Improvement'
                    }
                
                print(f"📋 Stage Progress: {len(analysis['stage_progress'])} stages analyzed")
                
            except Exception as e:
                print(f"⚠️ Could not read stage data: {e}")
        
        # プロセス状況確認
        if output_files.get('emergency_recovery'):
            latest_emergency_files = sorted(output_files['emergency_recovery'], 
                                          key=lambda x: x['modified'], reverse=True)
            if latest_emergency_files:
                latest_time = latest_emergency_files[0]['modified']
                if time.time() - latest_time < 300:  # 5分以内
                    analysis['key_findings'].append("🔄 Emergency recovery recently active")
        
        return analysis
    
    def create_status_visualization(self, processes: Dict, output_files: Dict, analysis: Dict) -> str:
        """復旧状況可視化"""
        
        print("\n📈 Creating Recovery Status Visualization...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('NKAT Power Outage Recovery Status', fontsize=16, fontweight='bold')
        
        # 1. 実行中プロセス
        nkat_procs = processes['nkat_processes']
        if nkat_procs:
            proc_types = [p['type'] for p in nkat_procs]
            proc_memories = [p['memory_mb'] for p in nkat_procs]
            
            bars = ax1.bar(range(len(proc_types)), proc_memories, 
                          color=['green', 'blue', 'orange', 'red'][:len(proc_types)])
            ax1.set_xlabel('Process Type')
            ax1.set_ylabel('Memory Usage (MB)')
            ax1.set_title('Active NKAT Processes')
            ax1.set_xticks(range(len(proc_types)))
            ax1.set_xticklabels(proc_types, rotation=45, ha='right')
            
            for bar, memory in zip(bars, proc_memories):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                        f'{memory:.0f}MB', ha='center', va='bottom')
        else:
            ax1.text(0.5, 0.5, 'No Active NKAT Processes', ha='center', va='center',
                    transform=ax1.transAxes, fontsize=14)
            ax1.set_title('Active NKAT Processes')
        
        # 2. 出力ファイル統計
        file_categories = list(output_files.keys())
        file_counts = [len(files) for files in output_files.values()]
        
        if any(file_counts):
            bars = ax2.bar(file_categories, file_counts, color='skyblue')
            ax2.set_xlabel('File Category')
            ax2.set_ylabel('Number of Files')
            ax2.set_title('Recovery Output Files')
            ax2.tick_params(axis='x', rotation=45)
            
            for bar, count in zip(bars, file_counts):
                if count > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            str(count), ha='center', va='bottom')
        else:
            ax2.text(0.5, 0.5, 'No Output Files Found', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=14)
            ax2.set_title('Recovery Output Files')
        
        # 3. 復旧状況メトリクス
        metrics = []
        values = []
        colors = []
        
        if analysis['latest_accuracy'] is not None:
            metrics.append('Latest Accuracy')
            values.append(analysis['latest_accuracy'] * 100)
            colors.append('green' if analysis['latest_accuracy'] > 0.8 else 'orange' if analysis['latest_accuracy'] > 0.5 else 'red')
        
        if 'Stage2' in analysis['stage_progress']:
            metrics.append('Stage2 TPE')
            values.append(analysis['stage_progress']['Stage2']['global_tpe'] * 100)
            colors.append('green' if analysis['stage_progress']['Stage2']['global_tpe'] > 0.5 else 'red')
        
        if 'Stage3' in analysis['stage_progress']:
            metrics.append('Stage3 Robustness')
            values.append(analysis['stage_progress']['Stage3']['robustness_score'])
            colors.append('green' if analysis['stage_progress']['Stage3']['robustness_score'] > 50 else 'red')
        
        if metrics:
            bars = ax3.barh(metrics, values, color=colors)
            ax3.set_xlabel('Score (%)')
            ax3.set_title('Recovery Performance Metrics')
            ax3.set_xlim(0, 100)
            
            for bar, value in zip(bars, values):
                ax3.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                        f'{value:.1f}%', ha='left', va='center')
        else:
            ax3.text(0.5, 0.5, 'No Metrics Available', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=14)
            ax3.set_title('Recovery Performance Metrics')
        
        # 4. 復旧タイムライン
        ax4.axis('off')
        
        timeline_text = f"""Recovery Timeline Status
        
🔋 Power Outage Recovery Initiated
📅 Current Time: {datetime.now().strftime('%H:%M:%S')}

📊 Current Status: {analysis['recovery_status']}
🔄 Emergency Training: {analysis['emergency_training_status']}

🔍 Key Findings:
"""
        
        for finding in analysis['key_findings'][:5]:  # 最大5個
            timeline_text += f"\n• {finding}"
        
        if len(analysis['key_findings']) > 5:
            timeline_text += f"\n• ... and {len(analysis['key_findings']) - 5} more"
        
        timeline_text += f"""

📁 Total Output Files: {sum(len(files) for files in output_files.values())}
🔄 Active Processes: {len(processes['nkat_processes'])}"""
        
        ax4.text(0.1, 0.9, timeline_text, transform=ax4.transAxes, 
                fontsize=11, verticalalignment='top', family='monospace')
        
        plt.tight_layout()
        
        # 保存
        viz_path = os.path.join(self.report_dir, f"nkat_recovery_status_{self.timestamp}.png")
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Status visualization saved: {viz_path}")
        return viz_path

    def generate_recovery_report(self, processes: Dict, output_files: Dict, analysis: Dict, viz_path: str) -> str:
        """復旧レポート生成"""
        
        print("\n📝 Generating Recovery Status Report...")
        
        # 精度の表示値を事前に計算
        accuracy_str = f"{analysis['latest_accuracy']:.3f}" if analysis['latest_accuracy'] is not None else 'N/A'
        
        report_content = f"""# 🔋 NKAT 電源断復旧状況レポート

**生成日時**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}
**レポートID**: {self.timestamp}

## 📊 Executive Summary

### 🎯 復旧状況
- **総合ステータス**: {analysis['recovery_status']}
- **緊急訓練**: {analysis['emergency_training_status']}
- **最新精度**: {accuracy_str}

### 🔄 実行中プロセス
- **アクティブNKATプロセス**: {len(processes['nkat_processes'])}個
"""
        
        for proc in processes['nkat_processes']:
            report_content += f"  - {proc['type']}: PID {proc['pid']}, 実行時間 {proc['runtime_minutes']:.1f}分\n"
        
        report_content += f"""
### 📂 出力ファイル統計
"""
        
        total_files = 0
        for category, files in output_files.items():
            if files:
                total_files += len(files)
                report_content += f"- **{category.replace('_', ' ').title()}**: {len(files)}個\n"
        
        report_content += f"\n**総ファイル数**: {total_files}個\n"
        
        report_content += f"""
## 🔍 詳細分析

### Stage別進捗
"""
        
        for stage, data in analysis['stage_progress'].items():
            report_content += f"- **{stage}**: {data['status']}\n"
            for key, value in data.items():
                if key != 'status':
                    if isinstance(value, float):
                        report_content += f"  - {key}: {value:.3f}\n"
                    else:
                        report_content += f"  - {key}: {value}\n"
        
        report_content += f"""
### 🔍 主要発見事項
"""
        
        for finding in analysis['key_findings']:
            report_content += f"- {finding}\n"
        
        report_content += f"""
## 🚀 次のアクション

### 即座に実行すべき項目
"""
        
        if analysis['latest_accuracy'] and analysis['latest_accuracy'] < 0.8:
            report_content += "1. **緊急訓練の延長**: 現在の精度が80%未満のため、追加訓練が必要\n"
        
        if not analysis['stage_progress']:
            report_content += "1. **Stageテストの実行**: Stage2-5の包括的テストを実行\n"
        
        if len(processes['nkat_processes']) == 0:
            report_content += "1. **プロセス再開**: アクティブなNKATプロセスが検出されていません\n"
        
        report_content += f"""
### 中期的な改善
1. **チェックポイント管理**: 定期的なチェックポイント保存体制の強化
2. **自動復旧システム**: 電源断対応の自動化
3. **モニタリング強化**: リアルタイム状況監視システムの導入

## 📈 復旧完了判定基準

✅ **完全復旧**:
- 緊急訓練精度 ≥ 80%
- Stage2 TPE ≥ 0.50
- Stage3 Robustness ≥ 50%
- 全Stageテスト完了

🟡 **部分復旧**:
- 緊急訓練精度 ≥ 50%
- 主要機能が動作

🟢 **復旧成功**:
- 緊急訓練精度 ≥ 80%
- Stage2 TPE ≥ 0.50
- Stage3 Robustness ≥ 50%
- 全Stageテスト完了

🔴 **復旧失敗**:
- 緊急訓練精度 < 50%
- 重要なプロセスが停止

## 📊 関連ファイル

- **状況可視化**: {viz_path}
- **レポート生成**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
*Generated by NKAT Power Outage Recovery Status System*
"""
        
        # レポート保存
        report_path = os.path.join(self.report_dir, f"nkat_recovery_status_report_{self.timestamp}.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"📝 Recovery report saved: {report_path}")
        return report_path

    def run_full_status_check(self) -> Dict[str, str]:
        """完全状況確認実行"""
        
        print("🔋 Starting Full Recovery Status Check...")
        
        # 1. プロセス確認
        processes = self.check_running_processes()
        
        # 2. 出力ファイルスキャン
        output_files = self.scan_recovery_outputs()
        
        # 3. 結果分析
        analysis = self.analyze_latest_results(output_files)
        
        # 4. 可視化作成
        viz_path = self.create_status_visualization(processes, output_files, analysis)
        
        # 5. レポート生成
        report_path = self.generate_recovery_report(processes, output_files, analysis, viz_path)
        
        # 6. JSON保存
        status_data = {
            'timestamp': self.timestamp,
            'processes': processes,
            'output_files': {k: len(v) for k, v in output_files.items()},  # サイズ削減
            'analysis': analysis,
            'files': {
                'visualization': viz_path,
                'report': report_path
            }
        }
        
        json_path = os.path.join(self.report_dir, f"nkat_recovery_status_{self.timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(status_data, f, indent=2, ensure_ascii=False)
        
        # サマリー表示
        print("\n" + "="*60)
        print("🔋 POWER OUTAGE RECOVERY STATUS SUMMARY")
        print("="*60)
        print(f"🎯 Recovery Status: {analysis['recovery_status']}")
        print(f"⚡ Emergency Training: {analysis['emergency_training_status']}")
        if analysis['latest_accuracy']:
            print(f"📊 Latest Accuracy: {analysis['latest_accuracy']:.1%}")
        print(f"🔄 Active Processes: {len(processes['nkat_processes'])}")
        print(f"📂 Output Files: {sum(len(files) for files in output_files.values())}")
        print(f"📋 Key Findings: {len(analysis['key_findings'])}")
        print()
        print(f"📊 Visualization: {viz_path}")
        print(f"📝 Report: {report_path}")
        print(f"💾 Data: {json_path}")
        print("="*60)
        
        return {
            'visualization': viz_path,
            'report': report_path,
            'data': json_path
        }

def main():
    """メイン実行関数"""
    
    # 復旧状況チェックシステム起動
    status_checker = NKATPowerOutageRecoveryStatus()
    
    # 完全状況確認実行
    results = status_checker.run_full_status_check()
    
    print("\n🎉 Recovery status check completed successfully!")
    print("📋 Check the generated files for detailed information.")

if __name__ == "__main__":
    main() 