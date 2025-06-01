#!/usr/bin/env python3
# nkat_mission_recovery_fixed.py
"""
NKAT ミッション復旧・継続スクリプト
Unicodeエラー修正版（絵文字削除、Windows cp932対応）

現在の状況:
- Stage2結果: nkat_stage2_mnist_results_20250601_180958.png
- チェックポイント: nkat_enhanced_v2_best.pth
- ミッション: mission_99_percent_20250601_185503

目標:
- 既存の進捗を活用
- 99%精度達成の確認
- 最終レポート生成
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import glob

# 英語グラフ設定（文字化け防止）
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


class NKATMissionRecovery:
    """NKAT ミッション復旧システム"""
    
    def __init__(self):
        self.mission_start_time = time.time()
        self.results = {}
        self.mission_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 復旧設定
        self.config = {
            'target_accuracy': 0.99,
            'target_global_tpe': 0.70,
            'device': 'cuda',
            'output_dir': f'logs/mission_recovery_{self.mission_id}',
            'existing_checkpoint': 'checkpoints/nkat_enhanced_v2_best.pth',
            'existing_mission_dir': 'logs/mission_99_percent_20250601_185503'
        }
        
        # 出力ディレクトリ作成
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        print("NKAT MISSION RECOVERY SYSTEM")
        print("="*50)
        print(f"Recovery ID: {self.mission_id}")
        print(f"Target: ValAcc >= {self.config['target_accuracy']*100:.1f}%")
        print(f"Target: Global TPE >= {self.config['target_global_tpe']:.2f}")
        print(f"Device: {self.config['device']}")
        print("="*50)
    
    def analyze_existing_progress(self) -> Dict[str, Any]:
        """既存の進捗分析"""
        print("\nSTEP 1: Analyzing Existing Progress")
        print("-" * 30)
        
        progress = {
            'checkpoint_found': False,
            'stage2_results_found': False,
            'mission_dir_found': False,
            'checkpoint_path': None,
            'stage2_results': [],
            'mission_results': None
        }
        
        # チェックポイント確認
        if os.path.exists(self.config['existing_checkpoint']):
            progress['checkpoint_found'] = True
            progress['checkpoint_path'] = self.config['existing_checkpoint']
            size_mb = os.path.getsize(self.config['existing_checkpoint']) / (1024*1024)
            print(f"Found checkpoint: {self.config['existing_checkpoint']} ({size_mb:.1f}MB)")
        
        # Stage2結果確認
        stage2_files = glob.glob('*stage2*20250601*.png')
        if stage2_files:
            progress['stage2_results_found'] = True
            progress['stage2_results'] = stage2_files
            print(f"Found Stage2 results: {len(stage2_files)} files")
            for f in stage2_files:
                print(f"  - {f}")
        
        # ミッションディレクトリ確認
        if os.path.exists(self.config['existing_mission_dir']):
            progress['mission_dir_found'] = True
            print(f"Found mission directory: {self.config['existing_mission_dir']}")
            
            # ミッション結果ファイル確認
            mission_json = os.path.join(self.config['existing_mission_dir'], 'mission_report.json')
            if os.path.exists(mission_json):
                try:
                    with open(mission_json, 'r', encoding='utf-8') as f:
                        progress['mission_results'] = json.load(f)
                    print("Found mission report JSON")
                except Exception as e:
                    print(f"WARNING: Could not read mission report: {e}")
        
        return progress
    
    def run_quick_evaluation(self, checkpoint_path: str) -> Dict[str, Any]:
        """クイック評価実行"""
        print("\nSTEP 2: Quick Model Evaluation")
        print("-" * 30)
        
        try:
            # クイック評価スクリプト作成・実行
            eval_script = self._create_quick_eval_script()
            
            cmd = [
                sys.executable, eval_script,
                '--checkpoint', checkpoint_path,
                '--device', self.config['device']
            ]
            
            print(f"Running quick evaluation...")
            start_time = time.time()
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                print("Quick evaluation completed successfully")
                
                # 結果解析
                eval_results = self._parse_eval_output(result.stdout)
                eval_results['execution_time'] = execution_time
                
                return eval_results
            else:
                print(f"ERROR: Quick evaluation failed: {result.stderr}")
                return {'error': result.stderr, 'execution_time': execution_time}
                
        except Exception as e:
            print(f"ERROR: Quick evaluation error: {e}")
            return {'error': str(e), 'execution_time': 0}
    
    def _create_quick_eval_script(self) -> str:
        """クイック評価スクリプト作成"""
        script_path = 'nkat_quick_eval_fixed.py'
        
        script_content = '''#!/usr/bin/env python3
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
import sys
import os

# パス追加
sys.path.append('.')

from nkat_transformer.model import NKATVisionTransformer
from utils.metrics import tpe_metric, count_nkat_parameters

def quick_eval(checkpoint_path, device='cuda'):
    """クイック評価"""
    print("Starting quick evaluation...")
    
    # データローダー
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)
    
    # モデル作成
    model = NKATVisionTransformer(
        img_size=28,
        patch_size=4,
        num_classes=10,
        embed_dim=384,
        depth=5,
        num_heads=8
    ).to(device)
    
    # チェックポイントロード
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint: {checkpoint_path}")
    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")
        return
    
    # 評価
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device).long()
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = correct / total
    
    # パラメータ分析
    param_analysis = count_nkat_parameters(model)
    tpe_score = tpe_metric(accuracy, param_analysis['nkat_params'])
    
    # 結果出力
    print(f"EVAL_RESULT_START")
    print(f"accuracy={accuracy:.6f}")
    print(f"tpe_score={tpe_score:.6f}")
    print(f"lambda_theory={param_analysis['nkat_params']}")
    print(f"nkat_ratio={param_analysis['nkat_ratio']:.6f}")
    print(f"total_params={param_analysis['total_params']}")
    print(f"target_99_achieved={accuracy >= 0.99}")
    print(f"EVAL_RESULT_END")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    quick_eval(args.checkpoint, args.device)
'''
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        return script_path
    
    def _parse_eval_output(self, output: str) -> Dict[str, Any]:
        """評価出力解析"""
        results = {}
        
        lines = output.split('\n')
        in_result_section = False
        
        for line in lines:
            line = line.strip()
            
            if line == "EVAL_RESULT_START":
                in_result_section = True
                continue
            elif line == "EVAL_RESULT_END":
                in_result_section = False
                break
            
            if in_result_section and '=' in line:
                key, value = line.split('=', 1)
                
                if key in ['accuracy', 'tpe_score', 'nkat_ratio']:
                    results[key] = float(value)
                elif key in ['lambda_theory', 'total_params']:
                    results[key] = int(value)
                elif key == 'target_99_achieved':
                    results[key] = value.lower() == 'true'
        
        return results
    
    def run_final_stage2_test(self, checkpoint_path: str) -> Dict[str, Any]:
        """最終Stage2テスト実行"""
        print("\nSTEP 3: Final Stage2 Generalization Test")
        print("-" * 30)
        
        try:
            # Stage2テストスクリプト実行（絵文字削除版）
            cmd = [
                sys.executable, 'nkat_stage2_optimized_generalization.py',
                '--pretrained_model', checkpoint_path,
                '--epochs', '5',  # 高速化
                '--device', self.config['device'],
                '--output_dir', self.config['output_dir']
            ]
            
            print(f"Running final Stage2 test...")
            start_time = time.time()
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                print("Final Stage2 test completed successfully")
                
                # 結果ファイル検索
                result_files = glob.glob(f'{self.config["output_dir"]}/stage2_optimized_*.json')
                if not result_files:
                    result_files = glob.glob('logs/stage2_optimized_*.json')
                
                if result_files:
                    latest_result = max(result_files, key=os.path.getctime)
                    with open(latest_result, 'r', encoding='utf-8') as f:
                        stage2_results = json.load(f)
                    
                    stage2_results['execution_time'] = execution_time
                    return stage2_results
                else:
                    # 標準出力から抽出
                    return self._extract_stage2_from_stdout(result.stdout, execution_time)
            else:
                print(f"ERROR: Stage2 test failed: {result.stderr}")
                return {'error': result.stderr, 'execution_time': execution_time}
                
        except Exception as e:
            print(f"ERROR: Stage2 test error: {e}")
            return {'error': str(e), 'execution_time': 0}
    
    def _extract_stage2_from_stdout(self, stdout: str, execution_time: float) -> Dict[str, Any]:
        """Stage2結果を標準出力から抽出"""
        import re
        
        result = {
            'execution_time': execution_time,
            'extracted_from_stdout': True,
            'global_metrics': {
                'global_tpe': 0.0,
                'global_accuracy': 0.0,
                'generalization_score': 0.0,
                'consistency_score': 1.0,
                'robustness_score': 1.0
            }
        }
        
        # TPE情報を抽出
        tpe_matches = re.findall(r'TPE=([\d.]+)', stdout)
        acc_matches = re.findall(r'Acc=([\d.]+)', stdout)
        
        if tpe_matches:
            tpe_scores = [float(t) for t in tpe_matches]
            result['global_metrics']['global_tpe'] = np.mean(tpe_scores)
            result['global_metrics']['generalization_score'] = min(tpe_scores)
        
        if acc_matches:
            accuracies = [float(a) for a in acc_matches]
            result['global_metrics']['global_accuracy'] = np.mean(accuracies)
        
        return result
    
    def generate_final_report(self, progress: Dict[str, Any], 
                            eval_results: Dict[str, Any],
                            stage2_results: Dict[str, Any]) -> Dict[str, Any]:
        """最終レポート生成"""
        print("\nSTEP 4: Final Report Generation")
        print("-" * 30)
        
        mission_end_time = time.time()
        total_time = mission_end_time - self.mission_start_time
        
        # 目標達成評価
        accuracy_achieved = eval_results.get('target_99_achieved', False)
        tpe_achieved = stage2_results.get('global_metrics', {}).get('global_tpe', 0) >= self.config['target_global_tpe']
        
        # 最終分析
        final_analysis = {
            'recovery_id': self.mission_id,
            'timestamp': datetime.now().isoformat(),
            'total_recovery_time': total_time,
            'config': self.config,
            'progress_analysis': progress,
            'evaluation_results': eval_results,
            'stage2_results': stage2_results,
            'mission_success': accuracy_achieved and tpe_achieved,
            'achievements': {
                '99_percent_accuracy': accuracy_achieved,
                'global_tpe_target': tpe_achieved,
                'best_accuracy': eval_results.get('accuracy', 0),
                'final_tpe': eval_results.get('tpe_score', 0),
                'global_tpe': stage2_results.get('global_metrics', {}).get('global_tpe', 0)
            },
            'mission_grade': self._calculate_final_grade(accuracy_achieved, tpe_achieved, eval_results, stage2_results)
        }
        
        # レポート保存
        report_path = os.path.join(self.config['output_dir'], 'final_mission_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(final_analysis, f, indent=2, ensure_ascii=False)
        
        # Markdown レポート
        md_path = os.path.join(self.config['output_dir'], 'final_mission_report.md')
        self._create_markdown_report(final_analysis, md_path)
        
        # 可視化
        viz_path = os.path.join(self.config['output_dir'], 'final_mission_visualization.png')
        self._create_final_visualization(final_analysis, viz_path)
        
        return final_analysis
    
    def _calculate_final_grade(self, accuracy_achieved: bool, tpe_achieved: bool,
                              eval_results: Dict[str, Any], stage2_results: Dict[str, Any]) -> str:
        """最終成績計算"""
        if not accuracy_achieved and not tpe_achieved:
            return "D - Needs Improvement"
        
        score = 0
        accuracy = eval_results.get('accuracy', 0)
        global_tpe = stage2_results.get('global_metrics', {}).get('global_tpe', 0)
        
        # 精度評価 (50点)
        if accuracy >= 0.995:
            score += 50
        elif accuracy >= 0.99:
            score += 45
        elif accuracy >= 0.985:
            score += 40
        else:
            score += max(0, int((accuracy - 0.95) * 1000))
        
        # TPE評価 (50点)
        if global_tpe >= 0.75:
            score += 50
        elif global_tpe >= 0.70:
            score += 45
        elif global_tpe >= 0.65:
            score += 40
        else:
            score += max(0, int(global_tpe * 50))
        
        # グレード判定
        if score >= 95:
            return "S+ - Perfect Achievement"
        elif score >= 90:
            return "S - Excellent Achievement"
        elif score >= 85:
            return "A+ - Outstanding Achievement"
        elif score >= 80:
            return "A - Great Achievement"
        elif score >= 70:
            return "B - Good Achievement"
        else:
            return "C - Satisfactory Achievement"
    
    def _create_markdown_report(self, analysis: Dict[str, Any], output_path: str):
        """Markdown レポート作成"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# NKAT Mission Recovery Report\n\n")
            f.write(f"**Recovery ID:** {analysis['recovery_id']}\n")
            f.write(f"**Timestamp:** {analysis['timestamp']}\n")
            f.write(f"**Total Time:** {analysis['total_recovery_time']/60:.1f} minutes\n")
            f.write(f"**Mission Grade:** {analysis['mission_grade']}\n\n")
            
            f.write(f"## Mission Status\n\n")
            f.write(f"- Mission Success: {'ACHIEVED' if analysis['mission_success'] else 'PARTIAL'}\n")
            f.write(f"- 99% Accuracy: {'ACHIEVED' if analysis['achievements']['99_percent_accuracy'] else 'NOT ACHIEVED'}\n")
            f.write(f"- Global TPE Target: {'ACHIEVED' if analysis['achievements']['global_tpe_target'] else 'NOT ACHIEVED'}\n\n")
            
            f.write(f"## Key Metrics\n\n")
            f.write(f"- Best Accuracy: {analysis['achievements']['best_accuracy']:.4f} ({analysis['achievements']['best_accuracy']*100:.2f}%)\n")
            f.write(f"- Final TPE: {analysis['achievements']['final_tpe']:.6f}\n")
            f.write(f"- Global TPE: {analysis['achievements']['global_tpe']:.6f}\n\n")
            
            f.write(f"## Progress Analysis\n\n")
            progress = analysis['progress_analysis']
            f.write(f"- Checkpoint Found: {'YES' if progress['checkpoint_found'] else 'NO'}\n")
            f.write(f"- Stage2 Results Found: {'YES' if progress['stage2_results_found'] else 'NO'}\n")
            f.write(f"- Mission Directory Found: {'YES' if progress['mission_dir_found'] else 'NO'}\n\n")
            
            if analysis['mission_success']:
                f.write(f"## Conclusion\n\n")
                f.write(f"**MISSION ACCOMPLISHED!** The NKAT-Transformer has successfully achieved the 99% precision target with excellent generalization capabilities.\n\n")
            else:
                f.write(f"## Next Steps\n\n")
                f.write(f"Continue optimization to achieve remaining targets. Significant progress has been made.\n\n")
    
    def _create_final_visualization(self, analysis: Dict[str, Any], output_path: str):
        """最終可視化作成"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # 目標達成状況
        targets = ['99% Accuracy', 'Global TPE >= 0.70', 'Mission Success']
        achievements = [
            analysis['achievements']['99_percent_accuracy'],
            analysis['achievements']['global_tpe_target'],
            analysis['mission_success']
        ]
        colors = ['green' if a else 'red' for a in achievements]
        
        ax1.bar(targets, [1 if a else 0 for a in achievements], color=colors, alpha=0.7)
        ax1.set_title('Target Achievement Status', fontweight='bold')
        ax1.set_ylabel('Achieved (1) / Not Achieved (0)')
        ax1.set_ylim([0, 1.2])
        ax1.tick_params(axis='x', rotation=45)
        
        # キーメトリクス
        metrics = ['Best Accuracy', 'Final TPE', 'Global TPE']
        values = [
            analysis['achievements']['best_accuracy'],
            analysis['achievements']['final_tpe'],
            analysis['achievements']['global_tpe']
        ]
        
        bars = ax2.bar(metrics, values, color=['blue', 'purple', 'orange'], alpha=0.7)
        ax2.set_title('Key Performance Metrics', fontweight='bold')
        ax2.set_ylabel('Score')
        ax2.tick_params(axis='x', rotation=45)
        
        # 値をバーの上に表示
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 進捗分析
        progress = analysis['progress_analysis']
        progress_items = ['Checkpoint', 'Stage2 Results', 'Mission Dir']
        progress_status = [
            progress['checkpoint_found'],
            progress['stage2_results_found'],
            progress['mission_dir_found']
        ]
        colors = ['green' if s else 'red' for s in progress_status]
        
        ax3.bar(progress_items, [1 if s else 0 for s in progress_status], color=colors, alpha=0.7)
        ax3.set_title('Recovery Progress Analysis', fontweight='bold')
        ax3.set_ylabel('Found (1) / Not Found (0)')
        ax3.set_ylim([0, 1.2])
        ax3.tick_params(axis='x', rotation=45)
        
        # 成績表示
        grade_text = analysis['mission_grade']
        ax4.text(0.5, 0.5, grade_text, ha='center', va='center', 
                fontsize=16, fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax4.set_xlim([0, 1])
        ax4.set_ylim([0, 1])
        ax4.set_title('Mission Grade', fontweight='bold')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_final_summary(self, analysis: Dict[str, Any]):
        """最終サマリー表示"""
        print("\n" + "="*50)
        print("NKAT MISSION RECOVERY SUMMARY")
        print("="*50)
        print(f"Recovery ID: {analysis['recovery_id']}")
        print(f"Total Time: {analysis['total_recovery_time']/60:.1f} minutes")
        print(f"Mission Grade: {analysis['mission_grade']}")
        print(f"Mission Success: {'ACHIEVED' if analysis['mission_success'] else 'PARTIAL'}")
        
        print(f"\nKEY ACHIEVEMENTS:")
        achievements = analysis['achievements']
        print(f"  99% Accuracy: {'ACHIEVED' if achievements['99_percent_accuracy'] else 'NOT ACHIEVED'}")
        print(f"  Global TPE Target: {'ACHIEVED' if achievements['global_tpe_target'] else 'NOT ACHIEVED'}")
        print(f"  Best Accuracy: {achievements['best_accuracy']:.4f} ({achievements['best_accuracy']*100:.2f}%)")
        print(f"  Final TPE: {achievements['final_tpe']:.6f}")
        print(f"  Global TPE: {achievements['global_tpe']:.6f}")
        
        if analysis['mission_success']:
            print(f"\nMISSION ACCOMPLISHED!")
            print(f"The NKAT-Transformer has achieved 99% precision!")
        else:
            print(f"\nSIGNIFICANT PROGRESS MADE!")
            print(f"Continue optimization for full mission success.")
        
        print("="*50)
    
    def run_recovery_mission(self):
        """復旧ミッション実行"""
        try:
            # Step 1: 既存進捗分析
            progress = self.analyze_existing_progress()
            
            # Step 2: クイック評価
            if progress['checkpoint_found']:
                eval_results = self.run_quick_evaluation(progress['checkpoint_path'])
            else:
                print("WARNING: No checkpoint found, skipping evaluation")
                eval_results = {'error': 'No checkpoint found'}
            
            # Step 3: 最終Stage2テスト
            if progress['checkpoint_found'] and 'error' not in eval_results:
                stage2_results = self.run_final_stage2_test(progress['checkpoint_path'])
            else:
                print("WARNING: Skipping Stage2 test due to missing checkpoint or eval error")
                stage2_results = {'error': 'Skipped due to missing checkpoint'}
            
            # Step 4: 最終レポート生成
            final_analysis = self.generate_final_report(progress, eval_results, stage2_results)
            
            # 結果表示
            self.print_final_summary(final_analysis)
            
            return final_analysis
            
        except Exception as e:
            print(f"ERROR: Recovery mission failed: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="NKAT Mission Recovery")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    args = parser.parse_args()
    
    # 復旧ミッション実行
    recovery = NKATMissionRecovery()
    recovery.config['device'] = args.device
    
    # 復旧開始
    analysis = recovery.run_recovery_mission()
    
    if analysis and analysis['mission_success']:
        print(f"\nRECOVERY SUCCESSFUL! Check results in {recovery.config['output_dir']}")
        return 0
    else:
        print(f"\nRecovery completed. Check results in {recovery.config['output_dir']}")
        return 1


if __name__ == "__main__":
    exit(main()) 