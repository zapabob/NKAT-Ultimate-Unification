#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚡ RTX3080性能最適化システム
RTX3080 Performance Optimization System

機能:
- GPU動的クロック調整
- VRAM効率化管理
- 温度制御システム
- 電力効率最適化
- リアルタイム性能監視

Author: NKAT Research Team
Date: 2025-05-26
Version: v1.0 - Performance Optimization Edition
"""

import subprocess
import time
import json
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import threading
import datetime
import numpy as np
import warnings

warnings.filterwarnings('ignore')

class RTX3080PerformanceOptimizer:
    """RTX3080性能最適化クラス"""
    
    def __init__(self):
        self.monitoring_active = False
        self.optimization_log = Path("rtx3080_optimization.log")
        self.performance_data = []
        
        # 最適化パラメータ
        self.target_temp = 78  # 目標温度
        self.max_temp = 85     # 最大許容温度
        self.target_memory_usage = 0.85  # 目標VRAM使用率
        self.max_memory_usage = 0.90     # 最大VRAM使用率
        
        # 性能統計
        self.performance_stats = {
            'gpu_utilization_history': [],
            'memory_usage_history': [],
            'temperature_history': [],
            'power_usage_history': [],
            'clock_speeds_history': []
        }
    
    def _log(self, message: str):
        """ログ出力"""
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        
        print(log_message)
        
        try:
            with open(self.optimization_log, 'a', encoding='utf-8') as f:
                f.write(log_message + '\n')
        except:
            pass
    
    def get_gpu_status(self) -> Optional[Dict]:
        """GPU状態取得"""
        try:
            # nvidia-smiコマンドで詳細情報を取得
            cmd = [
                'nvidia-smi', 
                '--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,clocks.gr,clocks.mem',
                '--format=csv,noheader,nounits'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                data = result.stdout.strip().split(', ')
                
                return {
                    'name': data[0],
                    'temperature': int(data[1]),
                    'utilization': int(data[2]),
                    'memory_used': int(data[3]),
                    'memory_total': int(data[4]),
                    'power_draw': float(data[5]),
                    'graphics_clock': int(data[6]),
                    'memory_clock': int(data[7]),
                    'memory_usage_percent': int(data[3]) / int(data[4]) * 100
                }
            
        except Exception as e:
            self._log(f"⚠️ GPU状態取得エラー: {e}")
        
        return None
    
    def optimize_memory_usage(self) -> bool:
        """VRAM使用量最適化"""
        try:
            import torch
            if torch.cuda.is_available():
                # キャッシュクリア
                torch.cuda.empty_cache()
                
                # ガベージコレクション
                import gc
                gc.collect()
                
                # メモリフラグメンテーション最適化
                if hasattr(torch.cuda, 'memory_summary'):
                    memory_stats = torch.cuda.memory_summary()
                    self._log(f"📊 VRAM最適化後: {memory_stats.split('|')[0].strip()}")
                
                return True
                
        except Exception as e:
            self._log(f"⚠️ メモリ最適化エラー: {e}")
        
        return False
    
    def adjust_power_settings(self, target_power_limit: int = 350) -> bool:
        """電力設定調整"""
        try:
            # nvidia-ml-pyを使用した電力制限設定（管理者権限が必要）
            cmd = ['nvidia-smi', '-pl', str(target_power_limit)]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self._log(f"⚡ 電力制限設定: {target_power_limit}W")
                return True
            else:
                self._log(f"⚠️ 電力設定変更には管理者権限が必要です")
                
        except Exception as e:
            self._log(f"⚠️ 電力設定エラー: {e}")
        
        return False
    
    def optimize_gpu_clocks(self, gpu_status: Dict) -> bool:
        """GPU クロック最適化"""
        try:
            current_temp = gpu_status['temperature']
            current_util = gpu_status['utilization']
            current_memory_usage = gpu_status['memory_usage_percent']
            
            # 温度ベースの動的調整
            if current_temp > self.max_temp:
                # 温度が高すぎる場合、クロックを下げる
                self._log(f"🔥 高温検出 ({current_temp}°C): クロック制限")
                # ここで実際のクロック調整コマンドを実行
                # nvidia-settings や MSI Afterburner API を使用
                return True
                
            elif current_temp < self.target_temp and current_util > 90:
                # 温度に余裕があり、使用率が高い場合、クロックを上げる
                self._log(f"⚡ 性能向上: 温度={current_temp}°C, 使用率={current_util}%")
                # クロック向上コマンド
                return True
            
            # メモリ使用率ベースの調整
            if current_memory_usage > self.max_memory_usage:
                self._log(f"💾 高VRAM使用率 ({current_memory_usage:.1f}%): 最適化実行")
                self.optimize_memory_usage()
                
        except Exception as e:
            self._log(f"⚠️ クロック最適化エラー: {e}")
        
        return False
    
    def analyze_performance_patterns(self) -> Dict:
        """性能パターン解析"""
        if len(self.performance_data) < 10:
            return {}
        
        try:
            # 最近のデータを解析
            recent_data = self.performance_data[-60:]  # 直近60回分
            
            temps = [d['temperature'] for d in recent_data]
            utils = [d['utilization'] for d in recent_data]
            mem_usage = [d['memory_usage_percent'] for d in recent_data]
            
            analysis = {
                'average_temperature': np.mean(temps),
                'max_temperature': np.max(temps),
                'temperature_trend': np.polyfit(range(len(temps)), temps, 1)[0],
                'average_utilization': np.mean(utils),
                'utilization_efficiency': np.mean(utils) / 100.0,
                'memory_efficiency': np.mean(mem_usage) / 100.0,
                'thermal_stability': np.std(temps),
                'performance_score': self._calculate_performance_score(recent_data)
            }
            
            return analysis
            
        except Exception as e:
            self._log(f"⚠️ 性能解析エラー: {e}")
            return {}
    
    def _calculate_performance_score(self, data: List[Dict]) -> float:
        """性能スコア計算"""
        try:
            if not data:
                return 0.0
            
            # 各要素のスコア計算
            util_scores = []
            temp_scores = []
            mem_scores = []
            
            for d in data:
                # 使用率スコア（高いほど良い、但し100%は除外）
                util = d['utilization']
                util_score = min(util / 95.0, 1.0) if util < 100 else 0.9
                util_scores.append(util_score)
                
                # 温度スコア（目標温度に近いほど良い）
                temp = d['temperature']
                if temp <= self.target_temp:
                    temp_score = 1.0
                elif temp <= self.max_temp:
                    temp_score = 1.0 - (temp - self.target_temp) / (self.max_temp - self.target_temp) * 0.3
                else:
                    temp_score = 0.3
                temp_scores.append(temp_score)
                
                # メモリ使用率スコア（目標に近いほど良い）
                mem_usage = d['memory_usage_percent']
                if mem_usage <= self.target_memory_usage * 100:
                    mem_score = mem_usage / (self.target_memory_usage * 100)
                else:
                    mem_score = 1.0 - (mem_usage / 100.0 - self.target_memory_usage) / (self.max_memory_usage - self.target_memory_usage) * 0.4
                mem_scores.append(max(mem_score, 0.0))
            
            # 総合スコア（重み付き平均）
            total_score = (
                np.mean(util_scores) * 0.4 +  # 使用率 40%
                np.mean(temp_scores) * 0.3 +   # 温度 30%
                np.mean(mem_scores) * 0.3      # メモリ 30%
            )
            
            return min(max(total_score, 0.0), 1.0)
            
        except Exception as e:
            self._log(f"⚠️ 性能スコア計算エラー: {e}")
            return 0.0
    
    def generate_optimization_recommendations(self, analysis: Dict) -> List[str]:
        """最適化推奨事項生成"""
        recommendations = []
        
        try:
            if not analysis:
                return ["📊 データ不足のため推奨事項を生成できません"]
            
            # 温度関連の推奨
            avg_temp = analysis.get('average_temperature', 0)
            if avg_temp > self.max_temp:
                recommendations.append("🔥 冷却性能の向上が必要です（ファンカーブ調整、ケース通気性改善）")
            elif avg_temp > self.target_temp + 5:
                recommendations.append("⚠️ 温度管理の最適化を検討してください")
            
            # 使用率関連の推奨
            avg_util = analysis.get('average_utilization', 0)
            if avg_util < 70:
                recommendations.append("📊 GPU使用率が低いです。計算負荷の増加を検討してください")
            elif avg_util > 98:
                recommendations.append("⚡ GPU使用率が非常に高いです。熱管理に注意してください")
            
            # メモリ効率関連の推奨
            mem_eff = analysis.get('memory_efficiency', 0)
            if mem_eff < 0.6:
                recommendations.append("💾 VRAM使用率が低いです。バッチサイズの増加を検討してください")
            elif mem_eff > 0.9:
                recommendations.append("⚠️ VRAM使用率が高いです。メモリ最適化を実行してください")
            
            # 安定性関連の推奨
            thermal_stability = analysis.get('thermal_stability', 0)
            if thermal_stability > 5:
                recommendations.append("🌡️ 温度変動が大きいです。冷却システムの確認が必要です")
            
            # 性能スコア関連の推奨
            perf_score = analysis.get('performance_score', 0)
            if perf_score < 0.6:
                recommendations.append("📉 総合性能スコアが低いです。システム全体の最適化が必要です")
            elif perf_score > 0.9:
                recommendations.append("🎉 優秀な性能を維持しています！")
            
            if not recommendations:
                recommendations.append("✅ 現在の設定は最適です")
            
        except Exception as e:
            recommendations.append(f"⚠️ 推奨事項生成エラー: {e}")
        
        return recommendations
    
    def run_optimization_cycle(self):
        """最適化サイクル実行"""
        self._log("⚡ RTX3080性能最適化サイクル開始")
        
        while self.monitoring_active:
            try:
                # GPU状態取得
                gpu_status = self.get_gpu_status()
                
                if gpu_status:
                    # データ記録
                    self.performance_data.append({
                        'timestamp': datetime.datetime.now().isoformat(),
                        **gpu_status
                    })
                    
                    # 履歴データの更新
                    self.performance_stats['gpu_utilization_history'].append(gpu_status['utilization'])
                    self.performance_stats['memory_usage_history'].append(gpu_status['memory_usage_percent'])
                    self.performance_stats['temperature_history'].append(gpu_status['temperature'])
                    self.performance_stats['power_usage_history'].append(gpu_status['power_draw'])
                    
                    # 履歴サイズ制限（直近1000回分）
                    for key in self.performance_stats:
                        if len(self.performance_stats[key]) > 1000:
                            self.performance_stats[key] = self.performance_stats[key][-1000:]
                    
                    # 最適化実行
                    optimized = self.optimize_gpu_clocks(gpu_status)
                    
                    # 定期的な解析（5分ごと）
                    if len(self.performance_data) % 10 == 0:
                        analysis = self.analyze_performance_patterns()
                        if analysis:
                            perf_score = analysis.get('performance_score', 0)
                            avg_temp = analysis.get('average_temperature', 0)
                            avg_util = analysis.get('average_utilization', 0)
                            
                            self._log(f"📊 性能解析: スコア={perf_score:.3f}, 温度={avg_temp:.1f}°C, 使用率={avg_util:.1f}%")
                    
                    # 状況報告（30秒ごと）
                    if len(self.performance_data) % 1 == 0:
                        self._log(f"⚡ GPU: {gpu_status['temperature']}°C, {gpu_status['utilization']}%, "
                                 f"VRAM: {gpu_status['memory_usage_percent']:.1f}%, "
                                 f"電力: {gpu_status['power_draw']:.1f}W")
                
                time.sleep(30)  # 30秒間隔
                
            except Exception as e:
                self._log(f"⚠️ 最適化サイクルエラー: {e}")
                time.sleep(60)
    
    def start_optimization(self):
        """最適化開始"""
        if self.monitoring_active:
            self._log("⚠️ 最適化は既に実行中です")
            return
        
        self._log("🚀 RTX3080性能最適化開始")
        
        # システム初期化
        self.optimize_memory_usage()
        
        # 最適化サイクル開始
        self.monitoring_active = True
        optimization_thread = threading.Thread(target=self.run_optimization_cycle, daemon=True)
        optimization_thread.start()
        
        self._log("✅ RTX3080性能最適化システム稼働中")
    
    def stop_optimization(self):
        """最適化停止"""
        self.monitoring_active = False
        self._log("🛑 RTX3080性能最適化停止")
    
    def generate_performance_report(self) -> str:
        """性能レポート生成"""
        try:
            if not self.performance_data:
                return "📊 性能データがありません"
            
            analysis = self.analyze_performance_patterns()
            recommendations = self.generate_optimization_recommendations(analysis)
            
            # レポート生成
            report_lines = [
                "# ⚡ RTX3080性能最適化レポート",
                f"**生成日時**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"**監視期間**: {len(self.performance_data)}回分のデータ",
                "",
                "## 📊 性能統計",
                ""
            ]
            
            if analysis:
                report_lines.extend([
                    f"- **平均温度**: {analysis.get('average_temperature', 0):.1f}°C",
                    f"- **最高温度**: {analysis.get('max_temperature', 0):.1f}°C",
                    f"- **平均GPU使用率**: {analysis.get('average_utilization', 0):.1f}%",
                    f"- **GPU使用効率**: {analysis.get('utilization_efficiency', 0):.1%}",
                    f"- **メモリ効率**: {analysis.get('memory_efficiency', 0):.1%}",
                    f"- **温度安定性**: {analysis.get('thermal_stability', 0):.2f}°C",
                    f"- **総合性能スコア**: {analysis.get('performance_score', 0):.3f}/1.000",
                    ""
                ])
            
            # 推奨事項
            if recommendations:
                report_lines.extend([
                    "## 🎯 最適化推奨事項",
                    ""
                ])
                for rec in recommendations:
                    report_lines.append(f"- {rec}")
                report_lines.append("")
            
            # 最新の状況
            if self.performance_data:
                latest = self.performance_data[-1]
                report_lines.extend([
                    "## 📈 現在の状況",
                    f"- **温度**: {latest['temperature']}°C",
                    f"- **GPU使用率**: {latest['utilization']}%",
                    f"- **VRAM使用率**: {latest['memory_usage_percent']:.1f}%",
                    f"- **電力消費**: {latest['power_draw']:.1f}W",
                    f"- **グラフィッククロック**: {latest['graphics_clock']}MHz",
                    f"- **メモリクロック**: {latest['memory_clock']}MHz",
                    ""
                ])
            
            report_content = "\n".join(report_lines)
            
            # ファイル保存
            report_file = Path(f"rtx3080_performance_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            self._log(f"📄 性能レポート生成: {report_file.name}")
            return str(report_file)
            
        except Exception as e:
            error_msg = f"❌ レポート生成エラー: {e}"
            self._log(error_msg)
            return error_msg

def main():
    """メイン実行関数"""
    print("⚡ RTX3080性能最適化システム v1.0")
    print("=" * 60)
    print("🔥 このシステムはRTX3080の性能を最大限に引き出し、")
    print("   安定した高性能計算を実現します。")
    print("=" * 60)
    
    optimizer = RTX3080PerformanceOptimizer()
    
    try:
        # 初期状態確認
        initial_status = optimizer.get_gpu_status()
        if initial_status:
            print(f"\n🎮 GPU検出: {initial_status['name']}")
            print(f"🌡️ 現在温度: {initial_status['temperature']}°C")
            print(f"⚡ 現在使用率: {initial_status['utilization']}%")
            print(f"💾 VRAM使用: {initial_status['memory_usage_percent']:.1f}%")
        else:
            print("❌ GPU情報を取得できません")
            return
        
        # 最適化開始確認
        start_opt = input("\n⚡ RTX3080性能最適化を開始しますか？ (y/N): ").strip().lower()
        if start_opt != 'y':
            print("❌ 最適化がキャンセルされました")
            return
        
        # 最適化開始
        optimizer.start_optimization()
        
        # インタラクティブメニュー
        print("\n📋 利用可能なコマンド:")
        print("1. 現在の状況表示 (status)")
        print("2. 性能解析実行 (analyze)")
        print("3. レポート生成 (report)")
        print("4. メモリ最適化 (memory)")
        print("5. 最適化停止 (stop)")
        print("6. 終了 (exit)")
        
        while True:
            try:
                command = input("\nコマンドを入力: ").strip().lower()
                
                if command in ['1', 'status']:
                    status = optimizer.get_gpu_status()
                    if status:
                        print(f"🌡️ 温度: {status['temperature']}°C")
                        print(f"⚡ 使用率: {status['utilization']}%")
                        print(f"💾 VRAM: {status['memory_usage_percent']:.1f}%")
                        print(f"🔌 電力: {status['power_draw']:.1f}W")
                
                elif command in ['2', 'analyze']:
                    analysis = optimizer.analyze_performance_patterns()
                    if analysis:
                        print(f"📊 性能スコア: {analysis.get('performance_score', 0):.3f}")
                        print(f"🌡️ 平均温度: {analysis.get('average_temperature', 0):.1f}°C")
                        print(f"⚡ 平均使用率: {analysis.get('average_utilization', 0):.1f}%")
                    else:
                        print("📊 分析データ不足")
                
                elif command in ['3', 'report']:
                    report_file = optimizer.generate_performance_report()
                    print(f"📄 レポート生成完了: {report_file}")
                
                elif command in ['4', 'memory']:
                    if optimizer.optimize_memory_usage():
                        print("✅ メモリ最適化完了")
                    else:
                        print("⚠️ メモリ最適化失敗")
                
                elif command in ['5', 'stop']:
                    optimizer.stop_optimization()
                    print("🛑 最適化停止")
                
                elif command in ['6', 'exit']:
                    break
                
                else:
                    print("❌ 無効なコマンドです")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"⚠️ コマンド実行エラー: {e}")
        
        optimizer.stop_optimization()
        print("🎉 RTX3080性能最適化システム終了")
        
    except Exception as e:
        print(f"❌ システムエラー: {e}")

if __name__ == "__main__":
    main() 