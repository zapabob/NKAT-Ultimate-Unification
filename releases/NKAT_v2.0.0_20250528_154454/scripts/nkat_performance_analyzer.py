#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 NKAT Performance Analyzer
NKATシステムパフォーマンス分析・ベンチマークツール

機能:
- GPU/CPU パフォーマンス測定
- メモリ使用量分析
- 計算速度ベンチマーク
- 統計解析性能評価
- レポート生成
"""

import time
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass, asdict

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# オプションモジュール
GPU_AVAILABLE = False
TORCH_AVAILABLE = False

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    logger.warning("GPUtil未インストール")

try:
    import torch
    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        logger.info(f"PyTorch CUDA利用可能: {torch.cuda.get_device_name()}")
    else:
        logger.info("PyTorch CPU版")
except ImportError:
    logger.warning("PyTorch未インストール")

@dataclass
class PerformanceMetrics:
    """パフォーマンス指標"""
    timestamp: str
    test_name: str
    duration: float
    cpu_usage_avg: float
    cpu_usage_max: float
    memory_usage_avg: float
    memory_usage_max: float
    gpu_usage_avg: float = 0.0
    gpu_usage_max: float = 0.0
    gpu_memory_avg: float = 0.0
    gpu_memory_max: float = 0.0
    gpu_temperature_avg: float = 0.0
    gpu_temperature_max: float = 0.0
    operations_per_second: float = 0.0
    memory_efficiency: float = 0.0
    error_count: int = 0
    success_rate: float = 100.0

class SystemMonitor:
    """システム監視クラス"""
    
    def __init__(self):
        self.monitoring = False
        self.data = {
            'timestamps': [],
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_usage': [],
            'gpu_memory': [],
            'gpu_temperature': []
        }
    
    def start_monitoring(self):
        """監視開始"""
        self.monitoring = True
        self.data = {key: [] for key in self.data.keys()}
    
    def stop_monitoring(self):
        """監視停止"""
        self.monitoring = False
    
    def collect_metrics(self):
        """メトリクス収集"""
        if not self.monitoring:
            return
        
        timestamp = datetime.now()
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        self.data['timestamps'].append(timestamp)
        self.data['cpu_usage'].append(cpu_usage)
        self.data['memory_usage'].append(memory_usage)
        
        # GPU情報
        gpu_usage = 0.0
        gpu_memory = 0.0
        gpu_temperature = 0.0
        
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_usage = gpu.load * 100
                    gpu_memory = gpu.memoryUtil * 100
                    gpu_temperature = gpu.temperature
            except Exception as e:
                logger.warning(f"GPU情報取得エラー: {e}")
        
        self.data['gpu_usage'].append(gpu_usage)
        self.data['gpu_memory'].append(gpu_memory)
        self.data['gpu_temperature'].append(gpu_temperature)
    
    def get_summary(self) -> Dict[str, float]:
        """サマリー統計取得"""
        if not self.data['timestamps']:
            return {}
        
        summary = {}
        for key in ['cpu_usage', 'memory_usage', 'gpu_usage', 'gpu_memory', 'gpu_temperature']:
            if self.data[key]:
                summary[f'{key}_avg'] = np.mean(self.data[key])
                summary[f'{key}_max'] = np.max(self.data[key])
                summary[f'{key}_min'] = np.min(self.data[key])
                summary[f'{key}_std'] = np.std(self.data[key])
        
        return summary

class NKATPerformanceAnalyzer:
    """NKATパフォーマンス分析器"""
    
    def __init__(self):
        self.monitor = SystemMonitor()
        self.results = []
        self.project_root = Path(__file__).parent.parent
        self.results_dir = self.project_root / "Results" / "performance"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def benchmark_basic_operations(self) -> PerformanceMetrics:
        """基本演算ベンチマーク"""
        logger.info("基本演算ベンチマーク開始")
        
        self.monitor.start_monitoring()
        start_time = time.time()
        
        operations = 0
        errors = 0
        
        try:
            # 数値計算ベンチマーク
            for i in range(10000):
                # 複素数計算
                s = 0.5 + 1j * (i * 0.01)
                
                # ゼータ関数近似
                zeta_sum = 0.0
                for n in range(1, 100):
                    try:
                        term = 1.0 / (n ** s)
                        zeta_sum += term
                        operations += 1
                    except Exception:
                        errors += 1
                
                # 監視データ収集
                if i % 1000 == 0:
                    self.monitor.collect_metrics()
        
        except Exception as e:
            logger.error(f"ベンチマークエラー: {e}")
            errors += 1
        
        duration = time.time() - start_time
        self.monitor.stop_monitoring()
        
        # 統計計算
        summary = self.monitor.get_summary()
        ops_per_second = operations / duration if duration > 0 else 0
        success_rate = (operations / (operations + errors)) * 100 if (operations + errors) > 0 else 0
        
        metrics = PerformanceMetrics(
            timestamp=datetime.now().isoformat(),
            test_name="基本演算ベンチマーク",
            duration=duration,
            cpu_usage_avg=summary.get('cpu_usage_avg', 0),
            cpu_usage_max=summary.get('cpu_usage_max', 0),
            memory_usage_avg=summary.get('memory_usage_avg', 0),
            memory_usage_max=summary.get('memory_usage_max', 0),
            gpu_usage_avg=summary.get('gpu_usage_avg', 0),
            gpu_usage_max=summary.get('gpu_usage_max', 0),
            gpu_memory_avg=summary.get('gpu_memory_avg', 0),
            gpu_memory_max=summary.get('gpu_memory_max', 0),
            gpu_temperature_avg=summary.get('gpu_temperature_avg', 0),
            gpu_temperature_max=summary.get('gpu_temperature_max', 0),
            operations_per_second=ops_per_second,
            memory_efficiency=summary.get('memory_usage_avg', 0),
            error_count=errors,
            success_rate=success_rate
        )
        
        logger.info(f"基本演算ベンチマーク完了: {ops_per_second:.2f} ops/sec")
        return metrics
    
    def benchmark_gpu_operations(self) -> PerformanceMetrics:
        """GPU演算ベンチマーク"""
        logger.info("GPU演算ベンチマーク開始")
        
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            logger.warning("GPU演算ベンチマークをスキップ（CUDA利用不可）")
            return PerformanceMetrics(
                timestamp=datetime.now().isoformat(),
                test_name="GPU演算ベンチマーク",
                duration=0,
                cpu_usage_avg=0,
                cpu_usage_max=0,
                memory_usage_avg=0,
                memory_usage_max=0,
                operations_per_second=0,
                memory_efficiency=0,
                error_count=1,
                success_rate=0
            )
        
        self.monitor.start_monitoring()
        start_time = time.time()
        
        operations = 0
        errors = 0
        
        try:
            device = torch.device('cuda')
            
            # GPU上で行列演算
            for i in range(1000):
                try:
                    # 大きな行列作成
                    a = torch.randn(1000, 1000, device=device)
                    b = torch.randn(1000, 1000, device=device)
                    
                    # 行列積
                    c = torch.matmul(a, b)
                    
                    # 複素数演算
                    complex_tensor = torch.complex(a, b)
                    result = torch.abs(complex_tensor)
                    
                    operations += 3  # 3つの演算
                    
                    # メモリクリア
                    del a, b, c, complex_tensor, result
                    torch.cuda.empty_cache()
                    
                    # 監視データ収集
                    if i % 100 == 0:
                        self.monitor.collect_metrics()
                
                except Exception as e:
                    logger.warning(f"GPU演算エラー: {e}")
                    errors += 1
        
        except Exception as e:
            logger.error(f"GPUベンチマークエラー: {e}")
            errors += 1
        
        duration = time.time() - start_time
        self.monitor.stop_monitoring()
        
        # 統計計算
        summary = self.monitor.get_summary()
        ops_per_second = operations / duration if duration > 0 else 0
        success_rate = (operations / (operations + errors)) * 100 if (operations + errors) > 0 else 0
        
        metrics = PerformanceMetrics(
            timestamp=datetime.now().isoformat(),
            test_name="GPU演算ベンチマーク",
            duration=duration,
            cpu_usage_avg=summary.get('cpu_usage_avg', 0),
            cpu_usage_max=summary.get('cpu_usage_max', 0),
            memory_usage_avg=summary.get('memory_usage_avg', 0),
            memory_usage_max=summary.get('memory_usage_max', 0),
            gpu_usage_avg=summary.get('gpu_usage_avg', 0),
            gpu_usage_max=summary.get('gpu_usage_max', 0),
            gpu_memory_avg=summary.get('gpu_memory_avg', 0),
            gpu_memory_max=summary.get('gpu_memory_max', 0),
            gpu_temperature_avg=summary.get('gpu_temperature_avg', 0),
            gpu_temperature_max=summary.get('gpu_temperature_max', 0),
            operations_per_second=ops_per_second,
            memory_efficiency=summary.get('gpu_memory_avg', 0),
            error_count=errors,
            success_rate=success_rate
        )
        
        logger.info(f"GPU演算ベンチマーク完了: {ops_per_second:.2f} ops/sec")
        return metrics
    
    def benchmark_memory_operations(self) -> PerformanceMetrics:
        """メモリ操作ベンチマーク"""
        logger.info("メモリ操作ベンチマーク開始")
        
        self.monitor.start_monitoring()
        start_time = time.time()
        
        operations = 0
        errors = 0
        
        try:
            # 大きなデータ構造の作成・操作
            data_arrays = []
            
            for i in range(100):
                try:
                    # 大きな配列作成
                    arr = np.random.random((1000, 1000))
                    data_arrays.append(arr)
                    
                    # 数値計算
                    result = np.fft.fft2(arr)
                    inverse = np.fft.ifft2(result)
                    
                    # 統計計算
                    mean_val = np.mean(arr)
                    std_val = np.std(arr)
                    
                    operations += 4  # 4つの演算
                    
                    # 監視データ収集
                    if i % 10 == 0:
                        self.monitor.collect_metrics()
                
                except Exception as e:
                    logger.warning(f"メモリ操作エラー: {e}")
                    errors += 1
            
            # メモリクリア
            del data_arrays
        
        except Exception as e:
            logger.error(f"メモリベンチマークエラー: {e}")
            errors += 1
        
        duration = time.time() - start_time
        self.monitor.stop_monitoring()
        
        # 統計計算
        summary = self.monitor.get_summary()
        ops_per_second = operations / duration if duration > 0 else 0
        success_rate = (operations / (operations + errors)) * 100 if (operations + errors) > 0 else 0
        
        metrics = PerformanceMetrics(
            timestamp=datetime.now().isoformat(),
            test_name="メモリ操作ベンチマーク",
            duration=duration,
            cpu_usage_avg=summary.get('cpu_usage_avg', 0),
            cpu_usage_max=summary.get('cpu_usage_max', 0),
            memory_usage_avg=summary.get('memory_usage_avg', 0),
            memory_usage_max=summary.get('memory_usage_max', 0),
            gpu_usage_avg=summary.get('gpu_usage_avg', 0),
            gpu_usage_max=summary.get('gpu_usage_max', 0),
            gpu_memory_avg=summary.get('gpu_memory_avg', 0),
            gpu_memory_max=summary.get('gpu_memory_max', 0),
            gpu_temperature_avg=summary.get('gpu_temperature_avg', 0),
            gpu_temperature_max=summary.get('gpu_temperature_max', 0),
            operations_per_second=ops_per_second,
            memory_efficiency=summary.get('memory_usage_avg', 0),
            error_count=errors,
            success_rate=success_rate
        )
        
        logger.info(f"メモリ操作ベンチマーク完了: {ops_per_second:.2f} ops/sec")
        return metrics
    
    def benchmark_statistical_analysis(self) -> PerformanceMetrics:
        """統計解析ベンチマーク"""
        logger.info("統計解析ベンチマーク開始")
        
        self.monitor.start_monitoring()
        start_time = time.time()
        
        operations = 0
        errors = 0
        
        try:
            # 拡張統計解析モジュールのテスト
            try:
                from riemann_zeros_extended import RiemannZerosDatabase, RiemannZerosStatistics
                
                # データベース初期化
                zeros_db = RiemannZerosDatabase()
                stats_analyzer = RiemannZerosStatistics(zeros_db)
                
                # 統計解析実行
                for n_zeros in [100, 500, 1000, 2000]:
                    try:
                        # 基本統計
                        basic_stats = stats_analyzer.compute_basic_statistics(n_zeros)
                        operations += 1
                        
                        # 間隔分布解析
                        spacing_analysis = stats_analyzer.analyze_spacing_distribution(n_zeros)
                        operations += 1
                        
                        # Montgomery-Odlyzko解析
                        mo_analysis = stats_analyzer.montgomery_odlyzko_analysis(n_zeros)
                        operations += 1
                        
                        # 監視データ収集
                        self.monitor.collect_metrics()
                        
                    except Exception as e:
                        logger.warning(f"統計解析エラー (n={n_zeros}): {e}")
                        errors += 1
                
            except ImportError:
                logger.warning("拡張統計解析モジュール未利用")
                # 基本統計計算
                for i in range(100):
                    try:
                        data = np.random.random(1000)
                        mean_val = np.mean(data)
                        std_val = np.std(data)
                        skew_val = np.mean(((data - mean_val) / std_val) ** 3)
                        operations += 3
                    except Exception as e:
                        errors += 1
        
        except Exception as e:
            logger.error(f"統計解析ベンチマークエラー: {e}")
            errors += 1
        
        duration = time.time() - start_time
        self.monitor.stop_monitoring()
        
        # 統計計算
        summary = self.monitor.get_summary()
        ops_per_second = operations / duration if duration > 0 else 0
        success_rate = (operations / (operations + errors)) * 100 if (operations + errors) > 0 else 0
        
        metrics = PerformanceMetrics(
            timestamp=datetime.now().isoformat(),
            test_name="統計解析ベンチマーク",
            duration=duration,
            cpu_usage_avg=summary.get('cpu_usage_avg', 0),
            cpu_usage_max=summary.get('cpu_usage_max', 0),
            memory_usage_avg=summary.get('memory_usage_avg', 0),
            memory_usage_max=summary.get('memory_usage_max', 0),
            gpu_usage_avg=summary.get('gpu_usage_avg', 0),
            gpu_usage_max=summary.get('gpu_usage_max', 0),
            gpu_memory_avg=summary.get('gpu_memory_avg', 0),
            gpu_memory_max=summary.get('gpu_memory_max', 0),
            gpu_temperature_avg=summary.get('gpu_temperature_avg', 0),
            gpu_temperature_max=summary.get('gpu_temperature_max', 0),
            operations_per_second=ops_per_second,
            memory_efficiency=summary.get('memory_usage_avg', 0),
            error_count=errors,
            success_rate=success_rate
        )
        
        logger.info(f"統計解析ベンチマーク完了: {ops_per_second:.2f} ops/sec")
        return metrics
    
    def run_full_benchmark(self) -> List[PerformanceMetrics]:
        """完全ベンチマーク実行"""
        logger.info("🚀 NKAT完全パフォーマンスベンチマーク開始")
        
        benchmarks = [
            self.benchmark_basic_operations,
            self.benchmark_memory_operations,
            self.benchmark_statistical_analysis,
            self.benchmark_gpu_operations
        ]
        
        results = []
        
        for i, benchmark in enumerate(benchmarks, 1):
            logger.info(f"ベンチマーク {i}/{len(benchmarks)} 実行中...")
            try:
                result = benchmark()
                results.append(result)
                self.results.append(result)
                
                # 間隔を空ける
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"ベンチマーク {i} エラー: {e}")
        
        logger.info("✅ 完全ベンチマーク完了")
        return results
    
    def generate_performance_report(self, results: List[PerformanceMetrics]) -> str:
        """パフォーマンスレポート生成"""
        if not results:
            return "ベンチマーク結果がありません"
        
        report = []
        report.append("# 🌌 NKAT パフォーマンス分析レポート")
        report.append(f"**生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # システム情報
        report.append("## 🖥️ システム情報")
        report.append(f"- **CPU**: {psutil.cpu_count()}コア")
        
        memory = psutil.virtual_memory()
        report.append(f"- **メモリ**: {memory.total / (1024**3):.1f}GB")
        
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    report.append(f"- **GPU**: {gpu.name} ({gpu.memoryTotal}MB)")
                else:
                    report.append("- **GPU**: 検出されませんでした")
            except:
                report.append("- **GPU**: 情報取得エラー")
        else:
            report.append("- **GPU**: GPUtil未インストール")
        
        report.append(f"- **PyTorch CUDA**: {'✅ 利用可能' if TORCH_AVAILABLE and torch.cuda.is_available() else '❌ 利用不可'}")
        report.append("")
        
        # ベンチマーク結果サマリー
        report.append("## 📊 ベンチマーク結果サマリー")
        report.append("")
        
        total_duration = sum(r.duration for r in results)
        avg_cpu_usage = np.mean([r.cpu_usage_avg for r in results])
        avg_memory_usage = np.mean([r.memory_usage_avg for r in results])
        total_operations = sum(r.operations_per_second * r.duration for r in results)
        total_errors = sum(r.error_count for r in results)
        avg_success_rate = np.mean([r.success_rate for r in results])
        
        report.append(f"- **総実行時間**: {total_duration:.2f}秒")
        report.append(f"- **平均CPU使用率**: {avg_cpu_usage:.1f}%")
        report.append(f"- **平均メモリ使用率**: {avg_memory_usage:.1f}%")
        report.append(f"- **総演算数**: {total_operations:.0f}")
        report.append(f"- **総エラー数**: {total_errors}")
        report.append(f"- **平均成功率**: {avg_success_rate:.1f}%")
        report.append("")
        
        # 詳細結果
        report.append("## 📈 詳細ベンチマーク結果")
        report.append("")
        
        for result in results:
            report.append(f"### {result.test_name}")
            report.append(f"- **実行時間**: {result.duration:.2f}秒")
            report.append(f"- **演算速度**: {result.operations_per_second:.2f} ops/sec")
            report.append(f"- **CPU使用率**: 平均 {result.cpu_usage_avg:.1f}% / 最大 {result.cpu_usage_max:.1f}%")
            report.append(f"- **メモリ使用率**: 平均 {result.memory_usage_avg:.1f}% / 最大 {result.memory_usage_max:.1f}%")
            
            if result.gpu_usage_avg > 0:
                report.append(f"- **GPU使用率**: 平均 {result.gpu_usage_avg:.1f}% / 最大 {result.gpu_usage_max:.1f}%")
                report.append(f"- **GPUメモリ**: 平均 {result.gpu_memory_avg:.1f}% / 最大 {result.gpu_memory_max:.1f}%")
                report.append(f"- **GPU温度**: 平均 {result.gpu_temperature_avg:.1f}°C / 最大 {result.gpu_temperature_max:.1f}°C")
            
            report.append(f"- **エラー数**: {result.error_count}")
            report.append(f"- **成功率**: {result.success_rate:.1f}%")
            report.append("")
        
        # パフォーマンス評価
        report.append("## 🎯 パフォーマンス評価")
        report.append("")
        
        # CPU評価
        if avg_cpu_usage < 50:
            cpu_rating = "優秀"
        elif avg_cpu_usage < 80:
            cpu_rating = "良好"
        else:
            cpu_rating = "要改善"
        
        report.append(f"- **CPU効率**: {cpu_rating} (平均使用率: {avg_cpu_usage:.1f}%)")
        
        # メモリ評価
        if avg_memory_usage < 60:
            memory_rating = "優秀"
        elif avg_memory_usage < 85:
            memory_rating = "良好"
        else:
            memory_rating = "要改善"
        
        report.append(f"- **メモリ効率**: {memory_rating} (平均使用率: {avg_memory_usage:.1f}%)")
        
        # 総合評価
        if avg_success_rate > 95 and avg_cpu_usage < 70 and avg_memory_usage < 80:
            overall_rating = "優秀"
        elif avg_success_rate > 90 and avg_cpu_usage < 85:
            overall_rating = "良好"
        else:
            overall_rating = "要改善"
        
        report.append(f"- **総合評価**: {overall_rating}")
        report.append("")
        
        # 推奨事項
        report.append("## 💡 推奨事項")
        report.append("")
        
        if avg_cpu_usage > 80:
            report.append("- CPU使用率が高いため、バッチサイズの調整を検討してください")
        
        if avg_memory_usage > 85:
            report.append("- メモリ使用率が高いため、データサイズの最適化を検討してください")
        
        if total_errors > 0:
            report.append("- エラーが発生しています。ログを確認して原因を調査してください")
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_results = [r for r in results if r.gpu_usage_avg > 0]
            if not gpu_results:
                report.append("- GPU加速が利用されていません。GPU最適化の有効化を検討してください")
        
        report.append("")
        report.append("---")
        report.append("**NKAT Performance Analyzer** | 自動生成レポート")
        
        return "\n".join(report)
    
    def create_performance_plots(self, results: List[PerformanceMetrics]):
        """パフォーマンスプロット作成"""
        if not results:
            return
        
        # データ準備
        test_names = [r.test_name for r in results]
        durations = [r.duration for r in results]
        cpu_usage = [r.cpu_usage_avg for r in results]
        memory_usage = [r.memory_usage_avg for r in results]
        ops_per_sec = [r.operations_per_second for r in results]
        success_rates = [r.success_rate for r in results]
        
        # プロット作成
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('NKAT パフォーマンス分析結果', fontsize=16, fontweight='bold')
        
        # 実行時間
        axes[0, 0].bar(test_names, durations, color='skyblue')
        axes[0, 0].set_title('実行時間 (秒)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # CPU使用率
        axes[0, 1].bar(test_names, cpu_usage, color='lightcoral')
        axes[0, 1].set_title('平均CPU使用率 (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # メモリ使用率
        axes[0, 2].bar(test_names, memory_usage, color='lightgreen')
        axes[0, 2].set_title('平均メモリ使用率 (%)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 演算速度
        axes[1, 0].bar(test_names, ops_per_sec, color='gold')
        axes[1, 0].set_title('演算速度 (ops/sec)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 成功率
        axes[1, 1].bar(test_names, success_rates, color='mediumpurple')
        axes[1, 1].set_title('成功率 (%)')
        axes[1, 1].set_ylim(0, 100)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 総合スコア（正規化）
        normalized_scores = []
        for r in results:
            score = (r.success_rate / 100) * (100 - r.cpu_usage_avg) / 100 * (100 - r.memory_usage_avg) / 100
            normalized_scores.append(score * 100)
        
        axes[1, 2].bar(test_names, normalized_scores, color='orange')
        axes[1, 2].set_title('総合効率スコア')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 保存
        plot_file = self.results_dir / f"performance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"パフォーマンスプロット保存: {plot_file}")
        
        plt.show()
    
    def save_results(self, results: List[PerformanceMetrics]):
        """結果保存"""
        if not results:
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # JSON保存
        json_data = [asdict(r) for r in results]
        json_file = self.results_dir / f"performance_results_{timestamp}.json"
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"結果JSON保存: {json_file}")
        
        # CSV保存
        df = pd.DataFrame([asdict(r) for r in results])
        csv_file = self.results_dir / f"performance_results_{timestamp}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        logger.info(f"結果CSV保存: {csv_file}")
        
        # レポート保存
        report = self.generate_performance_report(results)
        report_file = self.results_dir / f"performance_report_{timestamp}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"レポート保存: {report_file}")
        
        return {
            'json_file': json_file,
            'csv_file': csv_file,
            'report_file': report_file
        }

def main():
    """メイン関数"""
    print("📊 NKAT Performance Analyzer")
    print("=" * 50)
    
    analyzer = NKATPerformanceAnalyzer()
    
    try:
        # 完全ベンチマーク実行
        results = analyzer.run_full_benchmark()
        
        if results:
            # レポート生成・表示
            report = analyzer.generate_performance_report(results)
            print("\n" + report)
            
            # 結果保存
            saved_files = analyzer.save_results(results)
            print(f"\n📁 結果ファイル:")
            for file_type, file_path in saved_files.items():
                print(f"  - {file_type}: {file_path}")
            
            # プロット作成
            try:
                analyzer.create_performance_plots(results)
            except Exception as e:
                logger.warning(f"プロット作成エラー: {e}")
            
            print("\n✅ パフォーマンス分析完了")
        else:
            print("❌ ベンチマーク結果がありません")
    
    except KeyboardInterrupt:
        print("\n⏹️ ユーザーによる中断")
    except Exception as e:
        logger.error(f"分析エラー: {e}")
        print(f"❌ エラーが発生しました: {e}")

if __name__ == "__main__":
    main() 