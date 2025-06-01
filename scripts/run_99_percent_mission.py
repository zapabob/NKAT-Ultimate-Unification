#!/usr/bin/env python3
# run_99_percent_mission.py
"""
NKAT 99%精度達成ミッション統合実行スクリプト
TPE=0.7113ベストパラメータで99%圏突破を目指す

実行フロー:
1. フル30epoch訓練（99%精度目標）
2. Stage2汎化テスト（4データセット）
3. 統合結果分析・レポート生成

目標:
- MNIST ValAcc ≥ 99.0%
- Global TPE ≥ 0.70
- 汎化性能の確保
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np

# 英語グラフ設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


class NKAT99PercentMission:
    """NKAT 99%精度達成ミッション管理"""
    
    def __init__(self):
        self.mission_start_time = time.time()
        self.results = {}
        self.mission_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # ミッション設定
        self.config = {
            'target_accuracy': 0.99,
            'target_global_tpe': 0.70,
            'full_training_epochs': 30,
            'stage2_epochs': 8,
            'device': 'cuda',
            'output_dir': f'logs/mission_99_percent_{self.mission_id}'
        }
        
        # 出力ディレクトリ作成
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        print("NKAT 99% PRECISION MISSION INITIATED")
        print("="*60)
        print(f"Mission ID: {self.mission_id}")
        print(f"Target: ValAcc >= {self.config['target_accuracy']*100:.1f}%")
        print(f"Target: Global TPE >= {self.config['target_global_tpe']:.2f}")
        print(f"Device: {self.config['device']}")
        print("="*60)
    
    def run_full_training(self) -> Dict[str, Any]:
        """フル訓練実行"""
        print("\nPHASE 1: Full Training (99% Push)")
        print("-" * 40)
        
        try:
            # フル訓練スクリプト実行
            cmd = [
                sys.executable, 'nkat_full_training_99_percent_push.py',
                '--epochs', str(self.config['full_training_epochs']),
                '--device', self.config['device'],
                '--output_dir', self.config['output_dir']
            ]
            
            print(f"Executing: {' '.join(cmd)}")
            start_time = time.time()
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2時間タイムアウト
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                print("Full training completed successfully")
                
                # 結果ファイル検索（複数の場所を確認）
                search_dirs = ['logs', self.config['output_dir']]
                result_files = []
                
                for search_dir in search_dirs:
                    if os.path.exists(search_dir):
                        files = [f for f in os.listdir(search_dir) 
                               if f.startswith('full_training_99_percent_') and f.endswith('.json')]
                        result_files.extend([(search_dir, f) for f in files])
                
                if result_files:
                    # 最新のファイルを選択
                    latest_result = max(result_files, 
                                      key=lambda x: os.path.getctime(os.path.join(x[0], x[1])))
                    result_path = os.path.join(latest_result[0], latest_result[1])
                    
                    with open(result_path, 'r', encoding='utf-8') as f:
                        training_results = json.load(f)
                    
                    training_results['execution_time'] = execution_time
                    training_results['phase'] = 'full_training'
                    
                    print(f"Found result file: {result_path}")
                    return training_results
                else:
                    print("WARNING: No result file found")
                    # 標準出力から基本情報を抽出を試みる
                    return self._extract_results_from_stdout(result.stdout, execution_time, 'full_training')
            else:
                print(f"ERROR: Full training failed: {result.stderr}")
                return {'error': result.stderr, 'execution_time': execution_time}
                
        except subprocess.TimeoutExpired:
            print("ERROR: Full training timed out")
            return {'error': 'Training timed out', 'execution_time': 7200}
        except Exception as e:
            print(f"ERROR: Full training error: {e}")
            return {'error': str(e), 'execution_time': 0}
    
    def run_stage2_generalization(self, pretrained_model_path: str = None) -> Dict[str, Any]:
        """Stage2汎化テスト実行"""
        print("\nPHASE 2: Stage2 Generalization Test")
        print("-" * 40)
        
        try:
            # Stage2テストスクリプト実行
            cmd = [
                sys.executable, 'nkat_stage2_optimized_generalization.py',
                '--epochs', str(self.config['stage2_epochs']),
                '--device', self.config['device'],
                '--output_dir', self.config['output_dir']
            ]
            
            if pretrained_model_path:
                cmd.extend(['--pretrained_model', pretrained_model_path])
            
            print(f"Executing: {' '.join(cmd)}")
            start_time = time.time()
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1時間タイムアウト
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                print("Stage2 generalization test completed successfully")
                
                # 結果ファイル検索（複数の場所を確認）
                search_dirs = ['logs', self.config['output_dir']]
                result_files = []
                
                for search_dir in search_dirs:
                    if os.path.exists(search_dir):
                        files = [f for f in os.listdir(search_dir) 
                               if f.startswith('stage2_optimized_generalization_') and f.endswith('.json')]
                        result_files.extend([(search_dir, f) for f in files])
                
                if result_files:
                    # 最新のファイルを選択
                    latest_result = max(result_files, 
                                      key=lambda x: os.path.getctime(os.path.join(x[0], x[1])))
                    result_path = os.path.join(latest_result[0], latest_result[1])
                    
                    with open(result_path, 'r', encoding='utf-8') as f:
                        stage2_results = json.load(f)
                    
                    stage2_results['execution_time'] = execution_time
                    stage2_results['phase'] = 'stage2_generalization'
                    
                    print(f"Found result file: {result_path}")
                    return stage2_results
                else:
                    print("WARNING: No result file found")
                    # 標準出力から基本情報を抽出を試みる
                    return self._extract_results_from_stdout(result.stdout, execution_time, 'stage2_generalization')
            else:
                print(f"ERROR: Stage2 test failed: {result.stderr}")
                return {'error': result.stderr, 'execution_time': execution_time}
                
        except subprocess.TimeoutExpired:
            print("ERROR: Stage2 test timed out")
            return {'error': 'Stage2 test timed out', 'execution_time': 3600}
        except Exception as e:
            print(f"ERROR: Stage2 test error: {e}")
            return {'error': str(e), 'execution_time': 0}
    
    def _extract_results_from_stdout(self, stdout: str, execution_time: float, phase: str) -> Dict[str, Any]:
        """標準出力から基本的な結果を抽出"""
        result = {
            'execution_time': execution_time,
            'phase': phase,
            'extracted_from_stdout': True
        }
        
        if phase == 'full_training':
            # 精度情報を抽出
            import re
            acc_matches = re.findall(r'ValAcc:\s*([\d.]+)', stdout)
            if acc_matches:
                result['best_accuracy'] = float(acc_matches[-1])
                result['target_achieved'] = result['best_accuracy'] >= 0.99
            else:
                result['best_accuracy'] = 0.0
                result['target_achieved'] = False
            
            result['final_tpe_score'] = 0.0
            result['lambda_theory'] = 0
            
        elif phase == 'stage2_generalization':
            # TPE情報を抽出
            import re
            tpe_matches = re.findall(r'TPE=([\d.]+)', stdout)
            if tpe_matches:
                tpe_scores = [float(t) for t in tpe_matches]
                result['global_metrics'] = {
                    'global_tpe': np.mean(tpe_scores),
                    'global_accuracy': 0.0,
                    'generalization_score': min(tpe_scores) if tpe_scores else 0.0,
                    'consistency_score': 1.0,
                    'robustness_score': 1.0
                }
            else:
                result['global_metrics'] = {
                    'global_tpe': 0.0,
                    'global_accuracy': 0.0,
                    'generalization_score': 0.0,
                    'consistency_score': 0.0,
                    'robustness_score': 0.0
                }
        
        return result
    
    def analyze_mission_results(self, full_training_results: Dict[str, Any], 
                               stage2_results: Dict[str, Any]) -> Dict[str, Any]:
        """ミッション結果統合分析"""
        print("\nPHASE 3: Mission Results Analysis")
        print("-" * 40)
        
        mission_end_time = time.time()
        total_mission_time = mission_end_time - self.mission_start_time
        
        # 目標達成評価
        full_training_success = (
            'error' not in full_training_results and 
            full_training_results.get('target_achieved', False)
        )
        
        stage2_success = (
            'error' not in stage2_results and 
            stage2_results.get('global_metrics', {}).get('global_tpe', 0) >= self.config['target_global_tpe']
        )
        
        # 統合メトリクス計算
        if full_training_success:
            best_accuracy = full_training_results.get('best_accuracy', 0)
            final_tpe = full_training_results.get('final_tpe_score', 0)
            lambda_theory = full_training_results.get('lambda_theory', 0)
        else:
            best_accuracy = 0
            final_tpe = 0
            lambda_theory = 0
        
        if stage2_success:
            global_tpe = stage2_results.get('global_metrics', {}).get('global_tpe', 0)
            global_accuracy = stage2_results.get('global_metrics', {}).get('global_accuracy', 0)
            generalization_score = stage2_results.get('global_metrics', {}).get('generalization_score', 0)
        else:
            global_tpe = 0
            global_accuracy = 0
            generalization_score = 0
        
        # ミッション成功判定
        mission_success = full_training_success and stage2_success
        
        # 統合分析結果
        analysis = {
            'mission_id': self.mission_id,
            'timestamp': datetime.now().isoformat(),
            'total_mission_time': total_mission_time,
            'config': self.config,
            'mission_success': mission_success,
            'phase_results': {
                'full_training': {
                    'success': full_training_success,
                    'best_accuracy': best_accuracy,
                    'final_tpe': final_tpe,
                    'lambda_theory': lambda_theory,
                    'target_achieved': best_accuracy >= self.config['target_accuracy'],
                    'results': full_training_results
                },
                'stage2_generalization': {
                    'success': stage2_success,
                    'global_tpe': global_tpe,
                    'global_accuracy': global_accuracy,
                    'generalization_score': generalization_score,
                    'target_achieved': global_tpe >= self.config['target_global_tpe'],
                    'results': stage2_results
                }
            },
            'integrated_metrics': {
                'overall_tpe': (final_tpe + global_tpe) / 2 if final_tpe > 0 and global_tpe > 0 else max(final_tpe, global_tpe),
                'accuracy_consistency': abs(best_accuracy - global_accuracy) if best_accuracy > 0 and global_accuracy > 0 else 1.0,
                'performance_stability': min(final_tpe, global_tpe) / max(final_tpe, global_tpe) if final_tpe > 0 and global_tpe > 0 else 0.0
            },
            'mission_grade': self._calculate_mission_grade(mission_success, best_accuracy, global_tpe, generalization_score)
        }
        
        return analysis
    
    def _calculate_mission_grade(self, success: bool, accuracy: float, global_tpe: float, generalization: float) -> str:
        """ミッション成績評価"""
        if not success:
            return "F - Mission Failed"
        
        score = 0
        
        # 精度評価 (40点)
        if accuracy >= 0.995:
            score += 40
        elif accuracy >= 0.99:
            score += 35
        elif accuracy >= 0.985:
            score += 30
        elif accuracy >= 0.98:
            score += 25
        else:
            score += max(0, int((accuracy - 0.95) * 400))
        
        # TPE評価 (40点)
        if global_tpe >= 0.75:
            score += 40
        elif global_tpe >= 0.70:
            score += 35
        elif global_tpe >= 0.65:
            score += 30
        elif global_tpe >= 0.60:
            score += 25
        else:
            score += max(0, int((global_tpe - 0.5) * 200))
        
        # 汎化性評価 (20点)
        if generalization >= 0.70:
            score += 20
        elif generalization >= 0.65:
            score += 18
        elif generalization >= 0.60:
            score += 15
        else:
            score += max(0, int(generalization * 25))
        
        # グレード判定
        if score >= 95:
            return "S+ - Perfect Mission"
        elif score >= 90:
            return "S - Excellent Mission"
        elif score >= 85:
            return "A+ - Outstanding Mission"
        elif score >= 80:
            return "A - Great Mission"
        elif score >= 75:
            return "B+ - Good Mission"
        elif score >= 70:
            return "B - Satisfactory Mission"
        elif score >= 60:
            return "C - Acceptable Mission"
        else:
            return "D - Poor Mission"
    
    def create_mission_report(self, analysis: Dict[str, Any]):
        """ミッション報告書作成"""
        print("\nPHASE 4: Mission Report Generation")
        print("-" * 40)
        
        # JSON報告書
        report_path = os.path.join(self.config['output_dir'], 'mission_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        # Markdown報告書
        md_report_path = os.path.join(self.config['output_dir'], 'mission_report.md')
        self._create_markdown_report(analysis, md_report_path)
        
        # 可視化
        viz_path = os.path.join(self.config['output_dir'], 'mission_visualization.png')
        self._create_mission_visualization(analysis, viz_path)
        
        print(f"Mission report saved to {self.config['output_dir']}")
    
    def _create_markdown_report(self, analysis: Dict[str, Any], output_path: str):
        """Markdown報告書作成"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# NKAT 99% Precision Mission Report\n\n")
            f.write(f"**Mission ID:** {analysis['mission_id']}\n")
            f.write(f"**Timestamp:** {analysis['timestamp']}\n")
            f.write(f"**Total Time:** {analysis['total_mission_time']/3600:.2f} hours\n")
            f.write(f"**Mission Grade:** {analysis['mission_grade']}\n\n")
            
            f.write(f"## Mission Objectives\n\n")
            f.write(f"- Target Accuracy: >= {analysis['config']['target_accuracy']*100:.1f}%\n")
            f.write(f"- Target Global TPE: >= {analysis['config']['target_global_tpe']:.2f}\n")
            f.write(f"- Mission Success: {'ACHIEVED' if analysis['mission_success'] else 'FAILED'}\n\n")
            
            f.write(f"## Phase Results\n\n")
            
            # Phase 1
            phase1 = analysis['phase_results']['full_training']
            f.write(f"### Phase 1: Full Training\n")
            f.write(f"- Status: {'SUCCESS' if phase1['success'] else 'FAILED'}\n")
            f.write(f"- Best Accuracy: {phase1['best_accuracy']:.4f} ({phase1['best_accuracy']*100:.2f}%)\n")
            f.write(f"- Final TPE: {phase1['final_tpe']:.6f}\n")
            f.write(f"- Lambda Theory: {phase1['lambda_theory']}\n")
            f.write(f"- Target Achieved: {'YES' if phase1['target_achieved'] else 'NO'}\n\n")
            
            # Phase 2
            phase2 = analysis['phase_results']['stage2_generalization']
            f.write(f"### Phase 2: Stage2 Generalization\n")
            f.write(f"- Status: {'SUCCESS' if phase2['success'] else 'FAILED'}\n")
            f.write(f"- Global TPE: {phase2['global_tpe']:.6f}\n")
            f.write(f"- Global Accuracy: {phase2['global_accuracy']:.4f}\n")
            f.write(f"- Generalization Score: {phase2['generalization_score']:.6f}\n")
            f.write(f"- Target Achieved: {'YES' if phase2['target_achieved'] else 'NO'}\n\n")
            
            # 統合メトリクス
            integrated = analysis['integrated_metrics']
            f.write(f"## Integrated Metrics\n\n")
            f.write(f"- Overall TPE: {integrated['overall_tpe']:.6f}\n")
            f.write(f"- Accuracy Consistency: {integrated['accuracy_consistency']:.4f}\n")
            f.write(f"- Performance Stability: {integrated['performance_stability']:.4f}\n\n")
            
            f.write(f"## Mission Conclusion\n\n")
            if analysis['mission_success']:
                f.write(f"**MISSION ACCOMPLISHED!** The NKAT-Transformer has successfully achieved 99% precision with excellent generalization capabilities.\n\n")
            else:
                f.write(f"**MISSION PROGRESS:** Significant progress made towards 99% precision. Continue optimization for full success.\n\n")
    
    def _create_mission_visualization(self, analysis: Dict[str, Any], output_path: str):
        """ミッション可視化作成"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Phase別成功状況
        phases = ['Full Training', 'Stage2 Generalization']
        successes = [
            analysis['phase_results']['full_training']['success'],
            analysis['phase_results']['stage2_generalization']['success']
        ]
        colors = ['green' if s else 'red' for s in successes]
        
        ax1.bar(phases, [1 if s else 0 for s in successes], color=colors, alpha=0.7)
        ax1.set_title('Mission Phase Success', fontweight='bold')
        ax1.set_ylabel('Success (1) / Failure (0)')
        ax1.set_ylim([0, 1.2])
        
        # メトリクス比較
        metrics = ['Best Accuracy', 'Final TPE', 'Global TPE', 'Generalization']
        values = [
            analysis['phase_results']['full_training']['best_accuracy'],
            analysis['phase_results']['full_training']['final_tpe'],
            analysis['phase_results']['stage2_generalization']['global_tpe'],
            analysis['phase_results']['stage2_generalization']['generalization_score']
        ]
        
        bars = ax2.bar(metrics, values, color=['blue', 'purple', 'orange', 'green'], alpha=0.7)
        ax2.set_title('Key Metrics', fontweight='bold')
        ax2.set_ylabel('Score')
        ax2.tick_params(axis='x', rotation=45)
        
        # 値をバーの上に表示
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 目標達成状況
        targets = ['99% Accuracy', 'TPE >= 0.70', 'Mission Success']
        achievements = [
            analysis['phase_results']['full_training']['target_achieved'],
            analysis['phase_results']['stage2_generalization']['target_achieved'],
            analysis['mission_success']
        ]
        colors = ['green' if a else 'red' for a in achievements]
        
        ax3.bar(targets, [1 if a else 0 for a in achievements], color=colors, alpha=0.7)
        ax3.set_title('Target Achievement', fontweight='bold')
        ax3.set_ylabel('Achieved (1) / Not Achieved (0)')
        ax3.set_ylim([0, 1.2])
        ax3.tick_params(axis='x', rotation=45)
        
        # 統合評価
        integrated = analysis['integrated_metrics']
        int_metrics = ['Overall TPE', 'Accuracy Consistency', 'Performance Stability']
        int_values = [
            integrated['overall_tpe'],
            1.0 - integrated['accuracy_consistency'],  # 一貫性として表示
            integrated['performance_stability']
        ]
        
        bars4 = ax4.bar(int_metrics, int_values, color=['purple', 'blue', 'green'], alpha=0.7)
        ax4.set_title('Integrated Performance', fontweight='bold')
        ax4.set_ylabel('Score')
        ax4.tick_params(axis='x', rotation=45)
        
        # 値をバーの上に表示
        for bar, value in zip(bars4, int_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_mission_summary(self, analysis: Dict[str, Any]):
        """ミッション結果サマリー表示"""
        print("\n" + "="*60)
        print("NKAT 99% PRECISION MISSION SUMMARY")
        print("="*60)
        print(f"Mission ID: {analysis['mission_id']}")
        print(f"Total Time: {analysis['total_mission_time']/3600:.2f} hours")
        print(f"Mission Grade: {analysis['mission_grade']}")
        print(f"Mission Success: {'ACHIEVED' if analysis['mission_success'] else 'FAILED'}")
        
        print(f"\nPHASE RESULTS:")
        phase1 = analysis['phase_results']['full_training']
        phase2 = analysis['phase_results']['stage2_generalization']
        
        print(f"  Phase 1 (Full Training):")
        print(f"    Status: {'SUCCESS' if phase1['success'] else 'FAILED'}")
        print(f"    Best Accuracy: {phase1['best_accuracy']:.4f} ({phase1['best_accuracy']*100:.2f}%)")
        print(f"    Final TPE: {phase1['final_tpe']:.6f}")
        
        print(f"  Phase 2 (Stage2 Generalization):")
        print(f"    Status: {'SUCCESS' if phase2['success'] else 'FAILED'}")
        print(f"    Global TPE: {phase2['global_tpe']:.6f}")
        print(f"    Global Accuracy: {phase2['global_accuracy']:.4f}")
        
        integrated = analysis['integrated_metrics']
        print(f"\nINTEGRATED METRICS:")
        print(f"  Overall TPE: {integrated['overall_tpe']:.6f}")
        print(f"  Accuracy Consistency: {integrated['accuracy_consistency']:.4f}")
        print(f"  Performance Stability: {integrated['performance_stability']:.4f}")
        
        if analysis['mission_success']:
            print(f"\nMISSION ACCOMPLISHED!")
            print(f"The NKAT-Transformer has achieved 99% precision with excellent generalization!")
        else:
            print(f"\nMISSION PROGRESS:")
            print(f"Significant progress made. Continue optimization for full success.")
        
        print("="*60)
    
    def run_mission(self):
        """ミッション実行"""
        try:
            # Phase 1: フル訓練
            full_training_results = self.run_full_training()
            
            # Phase 2: Stage2汎化テスト
            pretrained_model_path = 'checkpoints/best_99_percent_model.pt' if 'error' not in full_training_results else None
            stage2_results = self.run_stage2_generalization(pretrained_model_path)
            
            # Phase 3: 結果分析
            analysis = self.analyze_mission_results(full_training_results, stage2_results)
            
            # Phase 4: 報告書作成
            self.create_mission_report(analysis)
            
            # 結果表示
            self.print_mission_summary(analysis)
            
            return analysis
            
        except Exception as e:
            print(f"ERROR: Mission failed with error: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="NKAT 99% Precision Mission")
    parser.add_argument("--epochs", type=int, default=30, help="Full training epochs")
    parser.add_argument("--stage2_epochs", type=int, default=8, help="Stage2 epochs per dataset")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    args = parser.parse_args()
    
    # ミッション実行
    mission = NKAT99PercentMission()
    
    # 設定上書き
    mission.config['full_training_epochs'] = args.epochs
    mission.config['stage2_epochs'] = args.stage2_epochs
    mission.config['device'] = args.device
    
    # ミッション開始
    analysis = mission.run_mission()
    
    if analysis and analysis['mission_success']:
        print(f"\nMISSION ACCOMPLISHED! Check results in {mission.config['output_dir']}")
        return 0
    else:
        print(f"\nMission completed with partial success. Check results in {mission.config['output_dir']}")
        return 1


if __name__ == "__main__":
    exit(main()) 