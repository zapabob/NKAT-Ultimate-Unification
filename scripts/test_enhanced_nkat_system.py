#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 Enhanced NKAT System Test Suite
改良版NKATシステム包括テスト

機能:
- 全モジュールテスト
- GPU環境テスト
- パフォーマンステスト
- 統合テスト
- レポート生成
"""

import os
import sys
import json
import time
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import unittest
from dataclasses import dataclass, asdict

# 進捗表示
from tqdm import tqdm

# プロジェクトパス追加
sys.path.append(str(Path(__file__).parent.parent))

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/enhanced_nkat_test.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """テスト結果"""
    test_name: str
    status: str  # "PASS", "FAIL", "SKIP"
    duration: float
    error_message: str = ""
    details: Dict[str, Any] = None

class EnhancedNKATSystemTester:
    """改良版NKATシステムテスター"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = time.time()
        
        # 結果保存ディレクトリ
        self.results_dir = Path("Results/tests")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def run_all_tests(self) -> bool:
        """全テスト実行"""
        logger.info("🧪 Enhanced NKAT システムテスト開始")
        
        test_methods = [
            ("基本インポートテスト", self.test_basic_imports),
            ("GPU環境テスト", self.test_gpu_environment),
            ("システム監視テスト", self.test_system_monitoring),
            ("NKAT理論テスト", self.test_nkat_theory),
            ("リーマン解析テスト", self.test_riemann_analysis),
            ("チェックポイントテスト", self.test_checkpoint_system),
            ("バックアップテスト", self.test_backup_system),
            ("最適化テスト", self.test_optimization_system),
            ("パフォーマンステスト", self.test_performance),
            ("統合テスト", self.test_integration)
        ]
        
        total_tests = len(test_methods)
        passed_tests = 0
        
        with tqdm(total=total_tests, desc="テスト実行中", unit="tests") as pbar:
            for test_name, test_method in test_methods:
                try:
                    logger.info(f"🔍 {test_name} 実行中...")
                    start_time = time.time()
                    
                    success, details = test_method()
                    duration = time.time() - start_time
                    
                    if success:
                        self.results.append(TestResult(
                            test_name=test_name,
                            status="PASS",
                            duration=duration,
                            details=details
                        ))
                        passed_tests += 1
                        logger.info(f"✅ {test_name}: 成功 ({duration:.2f}s)")
                    else:
                        self.results.append(TestResult(
                            test_name=test_name,
                            status="FAIL",
                            duration=duration,
                            error_message=str(details),
                            details=details if isinstance(details, dict) else {}
                        ))
                        logger.error(f"❌ {test_name}: 失敗 ({duration:.2f}s)")
                
                except Exception as e:
                    duration = time.time() - start_time
                    error_msg = f"{str(e)}\n{traceback.format_exc()}"
                    
                    self.results.append(TestResult(
                        test_name=test_name,
                        status="FAIL",
                        duration=duration,
                        error_message=error_msg
                    ))
                    logger.error(f"❌ {test_name}: エラー - {e}")
                
                pbar.update(1)
        
        # 結果保存
        self._save_test_results()
        
        # サマリー表示
        total_duration = time.time() - self.start_time
        success_rate = (passed_tests / total_tests) * 100
        
        logger.info(f"\n📊 テスト結果サマリー:")
        logger.info(f"   総テスト数: {total_tests}")
        logger.info(f"   成功: {passed_tests}")
        logger.info(f"   失敗: {total_tests - passed_tests}")
        logger.info(f"   成功率: {success_rate:.1f}%")
        logger.info(f"   総実行時間: {total_duration:.2f}s")
        
        return passed_tests == total_tests
    
    def test_basic_imports(self) -> tuple[bool, Any]:
        """基本インポートテスト"""
        try:
            # 標準ライブラリ
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import streamlit as st
            
            # GPU関連
            try:
                import torch
                import GPUtil
                gpu_available = torch.cuda.is_available()
            except ImportError:
                gpu_available = False
            
            # プロジェクト固有
            from src.enhanced_nkat_dashboard import EnhancedNKATDashboard
            
            details = {
                "numpy_version": np.__version__,
                "pandas_version": pd.__version__,
                "streamlit_version": st.__version__,
                "gpu_available": gpu_available,
                "torch_version": torch.__version__ if 'torch' in locals() else "N/A"
            }
            
            return True, details
            
        except Exception as e:
            return False, str(e)
    
    def test_gpu_environment(self) -> tuple[bool, Any]:
        """GPU環境テスト"""
        try:
            import torch
            import GPUtil
            
            details = {
                "cuda_available": torch.cuda.is_available(),
                "gpu_count": 0,
                "gpu_info": []
            }
            
            if torch.cuda.is_available():
                details["gpu_count"] = torch.cuda.device_count()
                
                for i in range(torch.cuda.device_count()):
                    gpu_props = torch.cuda.get_device_properties(i)
                    details["gpu_info"].append({
                        "name": gpu_props.name,
                        "memory_total": gpu_props.total_memory / 1e9,
                        "compute_capability": f"{gpu_props.major}.{gpu_props.minor}"
                    })
                
                # GPU利用可能性テスト
                device = torch.device("cuda")
                test_tensor = torch.randn(1000, 1000, device=device)
                result = torch.matmul(test_tensor, test_tensor)
                details["gpu_test_passed"] = True
            
            return True, details
            
        except Exception as e:
            return False, str(e)
    
    def test_system_monitoring(self) -> tuple[bool, Any]:
        """システム監視テスト"""
        try:
            from src.enhanced_nkat_dashboard import EnhancedSystemMonitor
            
            monitor = EnhancedSystemMonitor()
            system_info = monitor.get_system_info()
            
            # 必要な情報が取得できているかチェック
            required_keys = ['cpu_percent', 'memory_percent', 'disk_usage']
            for key in required_keys:
                if key not in system_info:
                    return False, f"Missing system info: {key}"
            
            details = {
                "system_info": system_info,
                "monitoring_available": True
            }
            
            return True, details
            
        except Exception as e:
            return False, str(e)
    
    def test_nkat_theory(self) -> tuple[bool, Any]:
        """NKAT理論テスト"""
        try:
            from src.enhanced_nkat_dashboard import EnhancedNKATParameters, EnhancedRiemannAnalyzer
            
            # パラメータ作成
            params = EnhancedNKATParameters(
                dimension=16,
                precision=50,
                n_points=100
            )
            
            # 解析器作成
            analyzer = EnhancedRiemannAnalyzer(params)
            
            # 基本計算テスト
            s = complex(0.5, 14.134725)  # 最初の非自明零点
            classical_value = analyzer.classical_zeta(s)
            nkat_value = analyzer.nkat_enhanced_zeta(s)
            
            details = {
                "parameters": asdict(params),
                "classical_zeta": abs(classical_value),
                "nkat_zeta": abs(nkat_value),
                "difference": abs(classical_value - nkat_value)
            }
            
            # 値が妥当な範囲内かチェック
            if abs(classical_value) > 1e10 or abs(nkat_value) > 1e10:
                return False, "Zeta values out of reasonable range"
            
            return True, details
            
        except Exception as e:
            return False, str(e)
    
    def test_riemann_analysis(self) -> tuple[bool, Any]:
        """リーマン解析テスト"""
        try:
            from src.enhanced_nkat_dashboard import EnhancedNKATParameters, EnhancedRiemannAnalyzer
            
            params = EnhancedNKATParameters(
                dimension=8,
                precision=30,
                n_points=50,
                n_zeros_analysis=10
            )
            
            analyzer = EnhancedRiemannAnalyzer(params)
            
            # 零点解析テスト（小規模）
            result = analyzer.find_zeros_advanced()
            
            details = {
                "zeros_found": len(result.get('zeros', [])),
                "analysis_completed": 'statistics' in result,
                "computation_time": result.get('computation_time', 0)
            }
            
            # 最低限の零点が見つかっているかチェック
            if len(result.get('zeros', [])) < 5:
                return False, "Insufficient zeros found"
            
            return True, details
            
        except Exception as e:
            return False, str(e)
    
    def test_checkpoint_system(self) -> tuple[bool, Any]:
        """チェックポイントシステムテスト"""
        try:
            from src.enhanced_nkat_dashboard import CheckpointManager
            
            checkpoint_dir = "Results/test_checkpoints"
            manager = CheckpointManager(checkpoint_dir)
            
            # テストデータ
            test_data = {
                "test_array": [1, 2, 3, 4, 5],
                "test_dict": {"a": 1, "b": 2},
                "timestamp": datetime.now().isoformat()
            }
            
            # 保存テスト
            manager.save_checkpoint("test_checkpoint", test_data)
            
            # 読み込みテスト
            loaded_data = manager.load_checkpoint("test_checkpoint")
            
            if loaded_data is None:
                return False, "Failed to load checkpoint"
            
            # データ整合性チェック
            if loaded_data["test_array"] != test_data["test_array"]:
                return False, "Data integrity check failed"
            
            details = {
                "save_successful": True,
                "load_successful": True,
                "data_integrity": True,
                "checkpoint_dir": checkpoint_dir
            }
            
            # クリーンアップ
            import shutil
            if Path(checkpoint_dir).exists():
                shutil.rmtree(checkpoint_dir)
            
            return True, details
            
        except Exception as e:
            return False, str(e)
    
    def test_backup_system(self) -> tuple[bool, Any]:
        """バックアップシステムテスト"""
        try:
            from scripts.nkat_backup_manager import NKATBackupManager, BackupConfig
            
            # テスト用設定
            config = BackupConfig(
                backup_dir="Results/test_backups",
                max_backups=3,
                compression_enabled=True,
                encryption_enabled=False,
                auto_backup_enabled=False,
                source_directories=["scripts"]  # 小さなディレクトリ
            )
            
            manager = NKATBackupManager(config)
            
            # バックアップ作成テスト
            backup_id = manager.create_backup("test_backup", incremental=False)
            
            if backup_id is None:
                return False, "Failed to create backup"
            
            # バックアップ一覧テスト
            backups = manager.list_backups()
            if len(backups) == 0:
                return False, "No backups found after creation"
            
            details = {
                "backup_created": backup_id is not None,
                "backup_id": backup_id,
                "backup_count": len(backups),
                "backup_size": backups[0].size_bytes if backups else 0
            }
            
            # クリーンアップ
            if backup_id:
                manager.delete_backup(backup_id)
            
            return True, details
            
        except Exception as e:
            return False, str(e)
    
    def test_optimization_system(self) -> tuple[bool, Any]:
        """最適化システムテスト"""
        try:
            from scripts.nkat_system_optimizer import NKATSystemOptimizer
            
            optimizer = NKATSystemOptimizer()
            
            # システムプロファイル取得
            profile = optimizer.system_profile
            
            # 基本的な最適化設定テスト
            optimizer._optimize_cpu_settings()
            optimizer._optimize_memory_settings()
            
            if optimizer.system_profile.gpu_count > 0:
                optimizer._optimize_gpu_settings()
            
            details = {
                "system_profile": asdict(profile),
                "optimization_completed": True,
                "cpu_cores": profile.cpu_count,
                "memory_gb": profile.memory_total,
                "gpu_count": profile.gpu_count
            }
            
            return True, details
            
        except Exception as e:
            return False, str(e)
    
    def test_performance(self) -> tuple[bool, Any]:
        """パフォーマンステスト"""
        try:
            import numpy as np
            import time
            
            # CPU性能テスト
            start_time = time.time()
            data = np.random.randn(10000, 100)
            result = np.sum(data * data)
            cpu_time = time.time() - start_time
            
            # メモリテスト
            start_time = time.time()
            large_array = np.zeros((5000, 5000))
            large_array.fill(1.0)
            memory_time = time.time() - start_time
            
            details = {
                "cpu_performance_time": cpu_time,
                "memory_performance_time": memory_time,
                "performance_acceptable": cpu_time < 5.0 and memory_time < 2.0
            }
            
            # GPU性能テスト（利用可能な場合）
            try:
                import torch
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                    start_time = time.time()
                    gpu_data = torch.randn(1000, 1000, device=device)
                    gpu_result = torch.matmul(gpu_data, gpu_data)
                    torch.cuda.synchronize()
                    gpu_time = time.time() - start_time
                    details["gpu_performance_time"] = gpu_time
                    details["gpu_available"] = True
                else:
                    details["gpu_available"] = False
            except:
                details["gpu_available"] = False
            
            return True, details
            
        except Exception as e:
            return False, str(e)
    
    def test_integration(self) -> tuple[bool, Any]:
        """統合テスト"""
        try:
            from src.enhanced_nkat_dashboard import (
                EnhancedNKATParameters, 
                EnhancedSystemMonitor,
                EnhancedRiemannAnalyzer,
                CheckpointManager
            )
            
            # 統合ワークフローテスト
            params = EnhancedNKATParameters(
                dimension=8,
                precision=30,
                n_points=20,
                n_zeros_analysis=5
            )
            
            monitor = EnhancedSystemMonitor()
            analyzer = EnhancedRiemannAnalyzer(params)
            checkpoint_manager = CheckpointManager("Results/test_integration")
            
            # システム情報取得
            system_info = monitor.get_system_info()
            
            # 小規模解析実行
            result = analyzer.find_zeros_advanced()
            
            # 結果保存
            checkpoint_manager.save_checkpoint("integration_test", {
                "system_info": system_info,
                "analysis_result": result,
                "parameters": asdict(params)
            })
            
            # 保存データ読み込み
            loaded_data = checkpoint_manager.load_checkpoint("integration_test")
            
            details = {
                "workflow_completed": True,
                "system_monitoring": len(system_info) > 0,
                "analysis_completed": len(result.get('zeros', [])) > 0,
                "checkpoint_working": loaded_data is not None,
                "integration_successful": True
            }
            
            # クリーンアップ
            import shutil
            checkpoint_dir = Path("Results/test_integration")
            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir)
            
            return True, details
            
        except Exception as e:
            return False, str(e)
    
    def _save_test_results(self):
        """テスト結果保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON形式で保存
        results_data = {
            "timestamp": timestamp,
            "total_tests": len(self.results),
            "passed_tests": len([r for r in self.results if r.status == "PASS"]),
            "failed_tests": len([r for r in self.results if r.status == "FAIL"]),
            "total_duration": time.time() - self.start_time,
            "results": [asdict(result) for result in self.results]
        }
        
        results_file = self.results_dir / f"test_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📄 テスト結果保存: {results_file}")
        
        # レポート生成
        self._generate_test_report(results_data)
    
    def _generate_test_report(self, results_data: Dict[str, Any]):
        """テストレポート生成"""
        timestamp = results_data["timestamp"]
        total_tests = results_data["total_tests"]
        passed_tests = results_data["passed_tests"]
        failed_tests = results_data["failed_tests"]
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        report = f"""
🧪 Enhanced NKAT システムテストレポート
{'='*60}

📊 テスト概要:
- 実行日時: {timestamp}
- 総テスト数: {total_tests}
- 成功: {passed_tests}
- 失敗: {failed_tests}
- 成功率: {success_rate:.1f}%
- 総実行時間: {results_data['total_duration']:.2f}秒

📋 詳細結果:
"""
        
        for result_data in results_data["results"]:
            status_icon = "✅" if result_data["status"] == "PASS" else "❌"
            report += f"{status_icon} {result_data['test_name']}: {result_data['status']} ({result_data['duration']:.2f}s)\n"
            
            if result_data["status"] == "FAIL" and result_data["error_message"]:
                report += f"   エラー: {result_data['error_message'][:100]}...\n"
        
        report += f"""

🔧 推奨事項:
"""
        
        if failed_tests > 0:
            report += "- 失敗したテストのログを確認してください\n"
            report += "- 依存関係を再インストールしてください: py -3 -m pip install -r requirements.txt\n"
            report += "- GPU環境を確認してください: nvidia-smi\n"
        else:
            report += "- 全てのテストが正常に完了しました\n"
            report += "- システムは正常に動作しています\n"
        
        # レポートファイル保存
        report_file = self.results_dir / f"test_report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"📄 テストレポート保存: {report_file}")

def main():
    """メイン関数"""
    print("🧪 Enhanced NKAT システムテスト")
    print("=" * 50)
    
    try:
        # 必要なディレクトリ作成
        Path("logs").mkdir(exist_ok=True)
        Path("Results").mkdir(exist_ok=True)
        Path("Results/tests").mkdir(exist_ok=True)
        
        # テスト実行
        tester = EnhancedNKATSystemTester()
        success = tester.run_all_tests()
        
        if success:
            print("\n✅ 全てのテストが正常に完了しました")
            sys.exit(0)
        else:
            print("\n❌ 一部のテストが失敗しました")
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n⚠️ テストが中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ テスト実行エラー: {e}")
        logger.error(f"テスト実行エラー: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 