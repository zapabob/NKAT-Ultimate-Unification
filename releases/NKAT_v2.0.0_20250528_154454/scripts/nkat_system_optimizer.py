#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 NKAT System Optimizer
NKATシステム最適化ツール

機能:
- GPU最適化設定
- メモリ使用量最適化
- パフォーマンス調整
- システム設定最適化
- 自動調整機能
"""

import os
import sys
import json
import time
import psutil
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

# GPU関連
try:
    import torch
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# 数値計算
import numpy as np
from tqdm import tqdm

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/nkat_optimizer.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """最適化設定"""
    # GPU設定
    gpu_memory_fraction: float = 0.8
    mixed_precision: bool = True
    cuda_optimization: bool = True
    
    # CPU設定
    cpu_threads: int = -1  # -1で自動設定
    cpu_affinity: Optional[List[int]] = None
    
    # メモリ設定
    memory_limit_gb: float = 16.0
    swap_usage: bool = False
    memory_mapping: bool = True
    
    # 計算設定
    batch_size: int = 1000
    precision: str = "float32"  # float16, float32, float64
    vectorization: bool = True
    
    # I/O設定
    async_io: bool = True
    buffer_size: int = 8192
    compression: bool = True
    
    # 監視設定
    monitoring_enabled: bool = True
    auto_adjustment: bool = True
    performance_threshold: float = 0.7

@dataclass
class SystemProfile:
    """システムプロファイル"""
    cpu_count: int
    cpu_freq: float
    memory_total: float
    memory_available: float
    gpu_count: int
    gpu_memory_total: float
    gpu_memory_available: float
    gpu_compute_capability: str
    storage_type: str
    network_speed: float

class GPUOptimizer:
    """GPU最適化クラス"""
    
    def __init__(self):
        self.device = None
        self.gpu_info = {}
        self._initialize_gpu()
    
    def _initialize_gpu(self):
        """GPU初期化"""
        if not GPU_AVAILABLE:
            logger.warning("GPU関連ライブラリが利用できません")
            return
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.gpu_info = self._get_gpu_info()
            logger.info(f"GPU初期化完了: {self.gpu_info['name']}")
        else:
            logger.warning("CUDA GPUが利用できません")
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """GPU情報取得"""
        if not torch.cuda.is_available():
            return {}
        
        gpu = GPUtil.getGPUs()[0]
        return {
            'name': gpu.name,
            'memory_total': gpu.memoryTotal,
            'memory_free': gpu.memoryFree,
            'memory_used': gpu.memoryUsed,
            'temperature': gpu.temperature,
            'load': gpu.load,
            'compute_capability': torch.cuda.get_device_capability()
        }
    
    def optimize_memory_settings(self, config: OptimizationConfig):
        """GPU メモリ設定最適化"""
        if not torch.cuda.is_available():
            return
        
        try:
            # メモリ使用量制限
            torch.cuda.set_per_process_memory_fraction(config.gpu_memory_fraction)
            
            # メモリプール設定
            torch.cuda.empty_cache()
            
            # 混合精度設定
            if config.mixed_precision:
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cuda.matmul.allow_tf32 = True
            
            logger.info(f"GPU メモリ設定最適化完了: {config.gpu_memory_fraction*100:.1f}%")
            
        except Exception as e:
            logger.error(f"GPU メモリ最適化エラー: {e}")
    
    def benchmark_gpu_performance(self) -> Dict[str, float]:
        """GPU パフォーマンステスト"""
        if not torch.cuda.is_available():
            return {}
        
        results = {}
        
        try:
            # メモリ帯域幅テスト
            size = 1024 * 1024 * 100  # 100MB
            data = torch.randn(size, device=self.device)
            
            start_time = time.time()
            for _ in range(10):
                data = data * 2.0
            torch.cuda.synchronize()
            memory_bandwidth = (size * 4 * 10) / (time.time() - start_time) / 1e9
            results['memory_bandwidth_gb_s'] = memory_bandwidth
            
            # 計算性能テスト
            matrix_size = 2048
            a = torch.randn(matrix_size, matrix_size, device=self.device)
            b = torch.randn(matrix_size, matrix_size, device=self.device)
            
            start_time = time.time()
            for _ in range(10):
                c = torch.matmul(a, b)
            torch.cuda.synchronize()
            compute_time = time.time() - start_time
            
            flops = 2 * matrix_size**3 * 10  # 行列乗算のFLOPS
            results['compute_tflops'] = flops / compute_time / 1e12
            
            logger.info(f"GPU ベンチマーク完了: {results}")
            
        except Exception as e:
            logger.error(f"GPU ベンチマークエラー: {e}")
        
        return results

class CPUOptimizer:
    """CPU最適化クラス"""
    
    def __init__(self):
        self.cpu_info = self._get_cpu_info()
    
    def _get_cpu_info(self) -> Dict[str, Any]:
        """CPU情報取得"""
        return {
            'physical_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
            'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            'architecture': os.uname().machine if hasattr(os, 'uname') else 'unknown'
        }
    
    def optimize_cpu_settings(self, config: OptimizationConfig):
        """CPU設定最適化"""
        try:
            # スレッド数設定
            if config.cpu_threads == -1:
                optimal_threads = min(self.cpu_info['physical_cores'], 8)
            else:
                optimal_threads = config.cpu_threads
            
            os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
            os.environ['MKL_NUM_THREADS'] = str(optimal_threads)
            os.environ['NUMEXPR_NUM_THREADS'] = str(optimal_threads)
            
            # CPU親和性設定
            if config.cpu_affinity:
                try:
                    psutil.Process().cpu_affinity(config.cpu_affinity)
                except:
                    logger.warning("CPU親和性設定に失敗しました")
            
            logger.info(f"CPU設定最適化完了: {optimal_threads}スレッド")
            
        except Exception as e:
            logger.error(f"CPU最適化エラー: {e}")
    
    def benchmark_cpu_performance(self) -> Dict[str, float]:
        """CPU パフォーマンステスト"""
        results = {}
        
        try:
            # 整数演算テスト
            start_time = time.time()
            total = 0
            for i in range(1000000):
                total += i * i
            int_time = time.time() - start_time
            results['integer_ops_per_sec'] = 1000000 / int_time
            
            # 浮動小数点演算テスト
            start_time = time.time()
            data = np.random.randn(10000, 1000)
            result = np.sum(data * data)
            float_time = time.time() - start_time
            results['float_ops_per_sec'] = (10000 * 1000) / float_time
            
            # メモリアクセステスト
            start_time = time.time()
            large_array = np.random.randn(1000000)
            np.sum(large_array)
            memory_time = time.time() - start_time
            results['memory_bandwidth_mb_s'] = (1000000 * 8) / memory_time / 1e6
            
            logger.info(f"CPU ベンチマーク完了: {results}")
            
        except Exception as e:
            logger.error(f"CPU ベンチマークエラー: {e}")
        
        return results

class MemoryOptimizer:
    """メモリ最適化クラス"""
    
    def __init__(self):
        self.memory_info = self._get_memory_info()
    
    def _get_memory_info(self) -> Dict[str, float]:
        """メモリ情報取得"""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / 1e9,
            'available_gb': memory.available / 1e9,
            'used_gb': memory.used / 1e9,
            'percent': memory.percent
        }
    
    def optimize_memory_settings(self, config: OptimizationConfig):
        """メモリ設定最適化"""
        try:
            # メモリ制限設定
            if config.memory_limit_gb > 0:
                # プロセスメモリ制限（概算）
                max_memory = min(config.memory_limit_gb * 1e9, 
                               self.memory_info['available_gb'] * 0.8 * 1e9)
                
                # NumPy メモリ設定
                os.environ['NPY_MEM_OVERLAP'] = '1'
                
            # スワップ使用設定
            if not config.swap_usage:
                try:
                    # スワップ使用を最小化
                    with open('/proc/sys/vm/swappiness', 'w') as f:
                        f.write('1')
                except:
                    pass  # 権限がない場合は無視
            
            logger.info(f"メモリ設定最適化完了: 制限 {config.memory_limit_gb:.1f}GB")
            
        except Exception as e:
            logger.error(f"メモリ最適化エラー: {e}")
    
    def monitor_memory_usage(self) -> Dict[str, float]:
        """メモリ使用量監視"""
        memory = psutil.virtual_memory()
        process = psutil.Process()
        
        return {
            'system_used_percent': memory.percent,
            'system_available_gb': memory.available / 1e9,
            'process_memory_mb': process.memory_info().rss / 1e6,
            'process_memory_percent': process.memory_percent()
        }

class NKATSystemOptimizer:
    """NKAT システム最適化メインクラス"""
    
    def __init__(self):
        self.config = OptimizationConfig()
        self.gpu_optimizer = GPUOptimizer()
        self.cpu_optimizer = CPUOptimizer()
        self.memory_optimizer = MemoryOptimizer()
        self.system_profile = self._create_system_profile()
        
        # 結果保存ディレクトリ
        self.results_dir = Path("Results/optimization")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_system_profile(self) -> SystemProfile:
        """システムプロファイル作成"""
        memory = psutil.virtual_memory()
        
        # GPU情報
        gpu_count = 0
        gpu_memory_total = 0
        gpu_memory_available = 0
        gpu_compute_capability = "N/A"
        
        if GPU_AVAILABLE and torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                gpu_memory_available = (torch.cuda.get_device_properties(0).total_memory - 
                                      torch.cuda.memory_allocated(0)) / 1e9
                gpu_compute_capability = f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}"
        
        return SystemProfile(
            cpu_count=psutil.cpu_count(logical=False),
            cpu_freq=psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            memory_total=memory.total / 1e9,
            memory_available=memory.available / 1e9,
            gpu_count=gpu_count,
            gpu_memory_total=gpu_memory_total,
            gpu_memory_available=gpu_memory_available,
            gpu_compute_capability=gpu_compute_capability,
            storage_type="SSD",  # 簡略化
            network_speed=1000.0  # 簡略化
        )
    
    def auto_optimize(self) -> Dict[str, Any]:
        """自動最適化"""
        logger.info("🔧 NKAT システム自動最適化開始")
        
        optimization_results = {
            'timestamp': datetime.now().isoformat(),
            'system_profile': asdict(self.system_profile),
            'original_config': asdict(self.config),
            'optimizations_applied': [],
            'performance_improvements': {}
        }
        
        try:
            # ベースライン性能測定
            baseline_performance = self._measure_baseline_performance()
            optimization_results['baseline_performance'] = baseline_performance
            
            # GPU最適化
            if self.system_profile.gpu_count > 0:
                self._optimize_gpu_settings()
                optimization_results['optimizations_applied'].append('GPU最適化')
            
            # CPU最適化
            self._optimize_cpu_settings()
            optimization_results['optimizations_applied'].append('CPU最適化')
            
            # メモリ最適化
            self._optimize_memory_settings()
            optimization_results['optimizations_applied'].append('メモリ最適化')
            
            # 最適化後性能測定
            optimized_performance = self._measure_baseline_performance()
            optimization_results['optimized_performance'] = optimized_performance
            
            # 改善率計算
            improvements = self._calculate_improvements(
                baseline_performance, optimized_performance
            )
            optimization_results['performance_improvements'] = improvements
            
            # 結果保存
            self._save_optimization_results(optimization_results)
            
            logger.info("✅ NKAT システム自動最適化完了")
            
        except Exception as e:
            logger.error(f"自動最適化エラー: {e}")
            optimization_results['error'] = str(e)
        
        return optimization_results
    
    def _optimize_gpu_settings(self):
        """GPU設定最適化"""
        # GPU メモリ使用量を動的調整
        available_memory = self.system_profile.gpu_memory_available
        if available_memory > 8:
            self.config.gpu_memory_fraction = 0.9
        elif available_memory > 4:
            self.config.gpu_memory_fraction = 0.8
        else:
            self.config.gpu_memory_fraction = 0.7
        
        self.gpu_optimizer.optimize_memory_settings(self.config)
    
    def _optimize_cpu_settings(self):
        """CPU設定最適化"""
        # CPU コア数に基づく最適化
        if self.system_profile.cpu_count >= 8:
            self.config.cpu_threads = min(8, self.system_profile.cpu_count)
        else:
            self.config.cpu_threads = self.system_profile.cpu_count
        
        self.cpu_optimizer.optimize_cpu_settings(self.config)
    
    def _optimize_memory_settings(self):
        """メモリ設定最適化"""
        # 利用可能メモリに基づく制限設定
        available_memory = self.system_profile.memory_available
        self.config.memory_limit_gb = min(available_memory * 0.8, 16.0)
        
        self.memory_optimizer.optimize_memory_settings(self.config)
    
    def _measure_baseline_performance(self) -> Dict[str, float]:
        """ベースライン性能測定"""
        performance = {}
        
        # CPU性能
        cpu_perf = self.cpu_optimizer.benchmark_cpu_performance()
        performance.update({f"cpu_{k}": v for k, v in cpu_perf.items()})
        
        # GPU性能
        if self.system_profile.gpu_count > 0:
            gpu_perf = self.gpu_optimizer.benchmark_gpu_performance()
            performance.update({f"gpu_{k}": v for k, v in gpu_perf.items()})
        
        # メモリ使用量
        memory_info = self.memory_optimizer.monitor_memory_usage()
        performance.update({f"memory_{k}": v for k, v in memory_info.items()})
        
        return performance
    
    def _calculate_improvements(self, baseline: Dict[str, float], 
                              optimized: Dict[str, float]) -> Dict[str, float]:
        """改善率計算"""
        improvements = {}
        
        for key in baseline:
            if key in optimized and baseline[key] > 0:
                improvement = ((optimized[key] - baseline[key]) / baseline[key]) * 100
                improvements[key] = improvement
        
        return improvements
    
    def _save_optimization_results(self, results: Dict[str, Any]):
        """最適化結果保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.results_dir / f"optimization_results_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"最適化結果保存: {filename}")
    
    def generate_optimization_report(self) -> str:
        """最適化レポート生成"""
        report = f"""
🔧 NKAT システム最適化レポート
{'='*50}

📊 システム情報:
- CPU: {self.system_profile.cpu_count}コア @ {self.system_profile.cpu_freq:.1f}MHz
- メモリ: {self.system_profile.memory_total:.1f}GB (利用可能: {self.system_profile.memory_available:.1f}GB)
- GPU: {self.system_profile.gpu_count}基 ({self.system_profile.gpu_memory_total:.1f}GB VRAM)
- GPU計算能力: {self.system_profile.gpu_compute_capability}

⚙️ 最適化設定:
- GPU メモリ使用率: {self.config.gpu_memory_fraction*100:.1f}%
- CPU スレッド数: {self.config.cpu_threads}
- メモリ制限: {self.config.memory_limit_gb:.1f}GB
- 混合精度: {'有効' if self.config.mixed_precision else '無効'}
- ベクトル化: {'有効' if self.config.vectorization else '無効'}

🚀 推奨設定:
"""
        
        # 推奨設定生成
        recommendations = self._generate_recommendations()
        for rec in recommendations:
            report += f"- {rec}\n"
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """推奨設定生成"""
        recommendations = []
        
        # GPU推奨設定
        if self.system_profile.gpu_count > 0:
            if self.system_profile.gpu_memory_total >= 10:
                recommendations.append("RTX3080以上: バッチサイズ2000、混合精度有効")
            elif self.system_profile.gpu_memory_total >= 6:
                recommendations.append("RTX3060以上: バッチサイズ1000、混合精度有効")
            else:
                recommendations.append("GPU メモリ不足: バッチサイズ500、混合精度有効")
        
        # CPU推奨設定
        if self.system_profile.cpu_count >= 8:
            recommendations.append("高性能CPU: 並列処理最大活用、8スレッド")
        else:
            recommendations.append(f"標準CPU: {self.system_profile.cpu_count}スレッド使用")
        
        # メモリ推奨設定
        if self.system_profile.memory_total >= 32:
            recommendations.append("大容量メモリ: 大規模データセット対応")
        elif self.system_profile.memory_total >= 16:
            recommendations.append("標準メモリ: 中規模データセット推奨")
        else:
            recommendations.append("メモリ制限: 小規模データセットのみ")
        
        return recommendations

def main():
    """メイン関数"""
    print("🔧 NKAT システム最適化ツール")
    print("=" * 50)
    
    try:
        # 最適化実行
        optimizer = NKATSystemOptimizer()
        
        print("\n📊 システム情報:")
        profile = optimizer.system_profile
        print(f"CPU: {profile.cpu_count}コア @ {profile.cpu_freq:.1f}MHz")
        print(f"メモリ: {profile.memory_total:.1f}GB")
        print(f"GPU: {profile.gpu_count}基 ({profile.gpu_memory_total:.1f}GB)")
        
        print("\n🚀 自動最適化実行中...")
        results = optimizer.auto_optimize()
        
        print("\n✅ 最適化完了!")
        
        # 改善結果表示
        if 'performance_improvements' in results:
            improvements = results['performance_improvements']
            print("\n📈 性能改善:")
            for key, improvement in improvements.items():
                if abs(improvement) > 1:  # 1%以上の変化のみ表示
                    print(f"  {key}: {improvement:+.1f}%")
        
        # レポート生成
        print("\n📋 最適化レポート:")
        report = optimizer.generate_optimization_report()
        print(report)
        
        print(f"\n💾 詳細結果: Results/optimization/")
        
    except KeyboardInterrupt:
        print("\n⚠️ 最適化が中断されました")
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        logger.error(f"最適化エラー: {e}")

if __name__ == "__main__":
    main() 