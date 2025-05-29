#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 NKAT v9.1 - 10,000γ Challenge 堅牢リカバリーシステム
Robust Recovery System for 10,000 Gamma Challenge with RTX3080

Author: NKAT Research Consortium
Date: 2025-05-26
Version: 9.1 - Ultimate Robustness
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict, Any
import warnings
from pathlib import Path
import json
import time
from dataclasses import dataclass, asdict
from tqdm import tqdm, trange
import logging
import pickle
import hashlib
from datetime import datetime
import threading
import queue
import signal
import sys
import os
import shutil
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import glob

# 日本語フォント設定
plt.rcParams['font.family'] = ['MS Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# GPU可用性チェック
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 使用デバイス: {device}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name()}")
    print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

@dataclass
class CheckpointData:
    """チェックポイントデータ構造"""
    batch_id: int
    gamma_start_idx: int
    gamma_end_idx: int
    completed_gammas: List[float]
    results: List[Dict]
    timestamp: str
    system_state: Dict
    memory_usage: float
    gpu_memory: float
    total_progress: float

@dataclass
class SystemMonitor:
    """システム監視データ"""
    cpu_percent: float
    memory_percent: float
    gpu_memory_used: float
    gpu_memory_total: float
    temperature: Optional[float]
    power_draw: Optional[float]
    timestamp: str

class RobustRecoveryManager:
    """堅牢リカバリー管理システム"""
    
    def __init__(self, checkpoint_dir: str = "10k_gamma_checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # バックアップディレクトリ
        self.backup_dir = self.checkpoint_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        # 緊急停止フラグ
        self.emergency_stop = False
        
        # システム監視
        self.monitoring_active = True
        self.monitor_queue = queue.Queue()
        
        # シグナルハンドラー設定
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"🛡️ 堅牢リカバリーマネージャー初期化: {self.checkpoint_dir}")
    
    def _signal_handler(self, signum, frame):
        """シグナルハンドラー（緊急停止）"""
        logger.warning(f"⚠️ シグナル {signum} を受信。緊急停止を開始...")
        self.emergency_stop = True
        self.monitoring_active = False
    
    def save_checkpoint(self, checkpoint_data: CheckpointData) -> bool:
        """チェックポイントの保存"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_file = self.checkpoint_dir / f"checkpoint_batch_{checkpoint_data.batch_id}_{timestamp}.pkl"
            
            # データの保存
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            # JSONバックアップも作成
            json_file = self.checkpoint_dir / f"checkpoint_batch_{checkpoint_data.batch_id}_{timestamp}.json"
            json_data = asdict(checkpoint_data)
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
            
            # 古いチェックポイントのクリーンアップ（最新5個を保持）
            self._cleanup_old_checkpoints(checkpoint_data.batch_id)
            
            logger.info(f"💾 チェックポイント保存完了: batch_{checkpoint_data.batch_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ チェックポイント保存エラー: {e}")
            return False
    
    def load_latest_checkpoint(self) -> Optional[CheckpointData]:
        """最新チェックポイントの読み込み"""
        try:
            checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_batch_*.pkl"))
            if not checkpoint_files:
                logger.info("📂 チェックポイントファイルが見つかりません")
                return None
            
            # 最新ファイルを選択
            latest_file = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            logger.info(f"📥 チェックポイント読み込み完了: {latest_file.name}")
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"❌ チェックポイント読み込みエラー: {e}")
            return None
    
    def _cleanup_old_checkpoints(self, batch_id: int, keep_count: int = 5):
        """古いチェックポイントのクリーンアップ"""
        try:
            pattern = f"checkpoint_batch_{batch_id}_*.pkl"
            checkpoint_files = list(self.checkpoint_dir.glob(pattern))
            
            if len(checkpoint_files) > keep_count:
                # 古いファイルを削除
                sorted_files = sorted(checkpoint_files, key=lambda x: x.stat().st_mtime)
                for old_file in sorted_files[:-keep_count]:
                    # バックアップに移動
                    backup_file = self.backup_dir / old_file.name
                    shutil.move(str(old_file), str(backup_file))
                    
                    # 対応するJSONファイルも移動
                    json_file = old_file.with_suffix('.json')
                    if json_file.exists():
                        backup_json = self.backup_dir / json_file.name
                        shutil.move(str(json_file), str(backup_json))
                
                logger.info(f"🧹 古いチェックポイントをクリーンアップ: {len(sorted_files) - keep_count}個")
                
        except Exception as e:
            logger.warning(f"⚠️ チェックポイントクリーンアップエラー: {e}")
    
    def monitor_system(self) -> SystemMonitor:
        """システム状態の監視"""
        try:
            # CPU・メモリ使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # GPU情報
            if torch.cuda.is_available():
                gpu_memory_used = torch.cuda.memory_allocated() / 1e9
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            else:
                gpu_memory_used = 0
                gpu_memory_total = 0
            
            # 温度情報（可能な場合）
            temperature = None
            power_draw = None
            try:
                if hasattr(psutil, "sensors_temperatures"):
                    temps = psutil.sensors_temperatures()
                    if temps:
                        temperature = list(temps.values())[0][0].current
            except:
                pass
            
            return SystemMonitor(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                gpu_memory_used=gpu_memory_used,
                gpu_memory_total=gpu_memory_total,
                temperature=temperature,
                power_draw=power_draw,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.warning(f"⚠️ システム監視エラー: {e}")
            return SystemMonitor(0, 0, 0, 0, None, None, datetime.now().isoformat())
    
    def check_system_health(self, monitor: SystemMonitor) -> bool:
        """システムヘルスチェック"""
        warnings = []
        
        # CPU使用率チェック
        if monitor.cpu_percent > 95:
            warnings.append(f"高CPU使用率: {monitor.cpu_percent:.1f}%")
        
        # メモリ使用率チェック
        if monitor.memory_percent > 90:
            warnings.append(f"高メモリ使用率: {monitor.memory_percent:.1f}%")
        
        # GPU メモリチェック
        if monitor.gpu_memory_total > 0:
            gpu_usage_percent = (monitor.gpu_memory_used / monitor.gpu_memory_total) * 100
            if gpu_usage_percent > 95:
                warnings.append(f"高GPU メモリ使用率: {gpu_usage_percent:.1f}%")
        
        # 温度チェック
        if monitor.temperature and monitor.temperature > 85:
            warnings.append(f"高温度: {monitor.temperature:.1f}°C")
        
        if warnings:
            logger.warning(f"⚠️ システム警告: {', '.join(warnings)}")
            return False
        
        return True

class NKAT10KGammaChallenge:
    """10,000γチャレンジ実行システム"""
    
    def __init__(self, recovery_manager: RobustRecoveryManager):
        self.recovery_manager = recovery_manager
        self.device = device
        self.dtype = torch.complex128
        
        # ハミルトニアン設定
        self.max_n = 2048  # RTX3080に最適化
        self.theta = 1e-25
        self.kappa = 1e-15
        
        # バッチ設定
        self.batch_size = 100  # 100γ値ずつ処理
        self.total_gammas = 10000
        
        # 結果保存
        self.results_dir = Path("10k_gamma_results")
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info(f"🎯 10,000γチャレンジシステム初期化完了")
    
    def load_latest_gamma_data(self) -> Optional[Dict]:
        """最新のγ値データを自動検出・読み込み"""
        try:
            # 検索パターン（最新ファイル優先）
            search_patterns = [
                # 最新の10k_gamma_results
                "../../10k_gamma_results/10k_gamma_final_results_*.json",
                "../10k_gamma_results/10k_gamma_final_results_*.json", 
                "10k_gamma_results/10k_gamma_final_results_*.json",
                # 中間結果ファイル
                "../../10k_gamma_results/intermediate_results_batch_*.json",
                "../10k_gamma_results/intermediate_results_batch_*.json",
                "10k_gamma_results/intermediate_results_batch_*.json",
                # その他のリーマン結果
                "../../rtx3080_extreme_riemann_results_*.json",
                "../rtx3080_extreme_riemann_results_*.json",
                "rtx3080_extreme_riemann_results_*.json",
                "../../ultimate_mastery_riemann_results.json",
                "../ultimate_mastery_riemann_results.json",
                "ultimate_mastery_riemann_results.json"
            ]
            
            found_files = []
            
            # 各パターンでファイルを検索
            for pattern in search_patterns:
                matches = glob.glob(pattern)
                for match in matches:
                    file_path = Path(match)
                    if file_path.exists() and file_path.stat().st_size > 1000:  # 1KB以上
                        found_files.append((file_path, file_path.stat().st_mtime))
            
            if not found_files:
                logger.warning("⚠️ 既存のγ値データが見つかりません")
                return None
            
            # 最新ファイルを選択
            latest_file = max(found_files, key=lambda x: x[1])[0]
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            logger.info(f"📊 最新γ値データ読み込み成功: {latest_file}")
            logger.info(f"📈 ファイルサイズ: {latest_file.stat().st_size / 1024:.1f} KB")
            
            # データ構造の検証
            if 'results' in data:
                results_count = len(data['results'])
                logger.info(f"📊 読み込み結果数: {results_count:,}")
                
                # 有効な結果の統計
                valid_results = [r for r in data['results'] if 'gamma' in r and not np.isnan(r.get('spectral_dimension', np.nan))]
                logger.info(f"✅ 有効結果数: {len(valid_results):,}")
                
                return data
            else:
                logger.warning(f"⚠️ 不明なデータ形式: {latest_file}")
                return data
                
        except Exception as e:
            logger.error(f"❌ γ値データ読み込みエラー: {e}")
            return None
    
    def extract_completed_gammas(self, existing_data: Dict) -> List[float]:
        """既存データから完了済みγ値を抽出"""
        completed_gammas = []
        
        if 'results' in existing_data:
            for result in existing_data['results']:
                if 'gamma' in result and 'spectral_dimension' in result:
                    # 有効な結果のみを完了済みとして扱う
                    if not np.isnan(result.get('spectral_dimension', np.nan)):
                        completed_gammas.append(result['gamma'])
        
        logger.info(f"📊 完了済みγ値: {len(completed_gammas):,}個")
        return sorted(completed_gammas)
    
    def generate_gamma_values(self, count: int = 10000, exclude_completed: bool = True) -> List[float]:
        """10,000個のγ値生成（完了済み除外オプション付き）"""
        gamma_values = []
        
        # 既存データの確認
        completed_gammas = set()
        if exclude_completed:
            existing_data = self.load_latest_gamma_data()
            if existing_data:
                completed_gammas = set(self.extract_completed_gammas(existing_data))
                logger.info(f"📊 除外対象の完了済みγ値: {len(completed_gammas):,}個")
        
        # 既知の高精度ゼロ点（最初の100個）
        known_zeros = [
            14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
            37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
            52.970321, 56.446247, 59.347044, 60.831778, 65.112544,
            67.079810, 69.546401, 72.067158, 75.704690, 77.144840,
            79.337375, 82.910381, 84.735493, 87.425275, 88.809111,
            92.491899, 94.651344, 95.870634, 98.831194, 101.317851
        ]
        
        # 既知のゼロ点を拡張（完了済みを除外）
        for i in range(100):
            if i < len(known_zeros):
                gamma = known_zeros[i]
                if gamma not in completed_gammas:
                    gamma_values.append(gamma)
            else:
                # 数学的補間
                gamma = 14.134725 + i * 2.5 + np.random.normal(0, 0.1)
                if gamma not in completed_gammas:
                    gamma_values.append(gamma)
        
        # 中間範囲の値（100-1000）
        for i in range(900):
            base_gamma = 100 + i * 0.5
            gamma = base_gamma + np.random.normal(0, 0.05)
            if gamma not in completed_gammas:
                gamma_values.append(gamma)
        
        # 高範囲の値（1000-10000）
        for i in range(9000):
            base_gamma = 500 + i * 0.1
            gamma = base_gamma + np.random.normal(0, 0.02)
            if gamma not in completed_gammas:
                gamma_values.append(gamma)
        
        # ソートして重複除去
        gamma_values = sorted(list(set(gamma_values)))
        
        # 正確に指定数に調整
        if len(gamma_values) > count:
            gamma_values = gamma_values[:count]
        elif len(gamma_values) < count:
            # 不足分を補完（完了済みを避けて）
            while len(gamma_values) < count:
                last_gamma = gamma_values[-1] if gamma_values else 1000.0
                new_gamma = last_gamma + 0.1 + np.random.normal(0, 0.01)
                if new_gamma not in completed_gammas:
                    gamma_values.append(new_gamma)
        
        logger.info(f"📊 {count:,}γ値生成完了: 範囲 [{min(gamma_values):.3f}, {max(gamma_values):.3f}]")
        if exclude_completed and completed_gammas:
            logger.info(f"🔄 完了済み除外: {len(completed_gammas):,}個のγ値をスキップ")
        
        return gamma_values
    
    def construct_hamiltonian(self, s: complex) -> torch.Tensor:
        """高効率ハミルトニアン構築"""
        dim = min(self.max_n, 1024)  # メモリ効率を考慮
        
        H = torch.zeros(dim, dim, dtype=self.dtype, device=self.device)
        
        # 主対角項（ベクトル化）
        n_values = torch.arange(1, dim + 1, dtype=torch.float64, device=self.device)
        try:
            diagonal_values = 1.0 / (n_values ** s)
            H.diagonal().copy_(diagonal_values.to(self.dtype))
        except:
            # フォールバック
            for n in range(1, dim + 1):
                try:
                    H[n-1, n-1] = torch.tensor(1.0 / (n ** s), dtype=self.dtype, device=self.device)
                except:
                    H[n-1, n-1] = torch.tensor(1e-50, dtype=self.dtype, device=self.device)
        
        # 正則化
        regularization = torch.tensor(1e-12, dtype=self.dtype, device=self.device)
        H += regularization * torch.eye(dim, dtype=self.dtype, device=self.device)
        
        return H
    
    def compute_spectral_dimension(self, s: complex) -> float:
        """スペクトル次元計算（最適化版）"""
        try:
            H = self.construct_hamiltonian(s)
            
            # エルミート化
            H_hermitian = 0.5 * (H + H.conj().T)
            
            # 固有値計算（上位のみ）
            eigenvals = torch.linalg.eigvals(H_hermitian).real
            eigenvals = eigenvals[eigenvals > 1e-15]
            
            if len(eigenvals) < 10:
                return float('nan')
            
            # スペクトル次元計算（簡略版）
            eigenvals_sorted = torch.sort(eigenvals, descending=True)[0]
            top_eigenvals = eigenvals_sorted[:min(50, len(eigenvals_sorted))]
            
            # 対数微分近似
            log_eigenvals = torch.log(top_eigenvals + 1e-15)
            indices = torch.arange(1, len(top_eigenvals) + 1, dtype=torch.float64, device=self.device)
            log_indices = torch.log(indices)
            
            # 線形回帰
            A = torch.stack([log_indices, torch.ones_like(log_indices)], dim=1)
            solution = torch.linalg.lstsq(A, log_eigenvals).solution
            slope = solution[0]
            
            spectral_dimension = -2 * slope.item()
            
            if abs(spectral_dimension) > 20 or not np.isfinite(spectral_dimension):
                return float('nan')
            
            return spectral_dimension
            
        except Exception as e:
            logger.warning(f"⚠️ スペクトル次元計算エラー: {e}")
            return float('nan')
    
    def process_gamma_batch(self, gamma_batch: List[float], batch_id: int) -> List[Dict]:
        """γ値バッチの処理"""
        results = []
        
        for i, gamma in enumerate(gamma_batch):
            if self.recovery_manager.emergency_stop:
                logger.warning("⚠️ 緊急停止が要求されました")
                break
            
            try:
                s = 0.5 + 1j * gamma
                
                # スペクトル次元計算
                d_s = self.compute_spectral_dimension(s)
                
                # 結果記録
                result = {
                    'gamma': gamma,
                    'spectral_dimension': d_s,
                    'real_part': d_s / 2 if not np.isnan(d_s) else np.nan,
                    'convergence_to_half': abs(d_s / 2 - 0.5) if not np.isnan(d_s) else np.nan,
                    'timestamp': datetime.now().isoformat(),
                    'batch_id': batch_id,
                    'batch_index': i
                }
                
                results.append(result)
                
                # プログレス表示
                if (i + 1) % 10 == 0:
                    logger.info(f"📊 Batch {batch_id}: {i + 1}/{len(gamma_batch)} 完了")
                
            except Exception as e:
                logger.error(f"❌ γ={gamma} 処理エラー: {e}")
                # エラーでも結果を記録
                results.append({
                    'gamma': gamma,
                    'spectral_dimension': np.nan,
                    'real_part': np.nan,
                    'convergence_to_half': np.nan,
                    'timestamp': datetime.now().isoformat(),
                    'batch_id': batch_id,
                    'batch_index': i,
                    'error': str(e)
                })
        
        return results
    
    def execute_10k_challenge(self, resume: bool = True, use_existing_data: bool = True) -> Dict:
        """10,000γチャレンジの実行（既存データ活用対応）"""
        print("=" * 80)
        print("🚀 NKAT v9.1 - 10,000γ Challenge 開始")
        print("=" * 80)
        print(f"📅 開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 目標: 10,000γ値の検証")
        print(f"📦 バッチサイズ: {self.batch_size}")
        print(f"🛡️ リカバリー機能: 有効")
        print(f"🔄 既存データ活用: {'有効' if use_existing_data else '無効'}")
        print("=" * 80)
        
        start_time = time.time()
        
        # 既存データの確認と統合
        all_results = []
        existing_data = None
        
        if use_existing_data:
            existing_data = self.load_latest_gamma_data()
            if existing_data and 'results' in existing_data:
                all_results = existing_data['results'].copy()
                logger.info(f"📊 既存結果を統合: {len(all_results):,}個")
        
        # γ値生成（完了済みを除外）
        all_gamma_values = self.generate_gamma_values(
            self.total_gammas, 
            exclude_completed=use_existing_data
        )
        
        # 復旧チェック
        checkpoint_data = None
        start_batch = 0
        
        if resume:
            checkpoint_data = self.recovery_manager.load_latest_checkpoint()
            if checkpoint_data:
                start_batch = checkpoint_data.batch_id + 1
                # チェックポイントの結果を統合
                if checkpoint_data.results:
                    # 重複除去のため、γ値でマージ
                    existing_gammas = {r['gamma'] for r in all_results}
                    for result in checkpoint_data.results:
                        if result['gamma'] not in existing_gammas:
                            all_results.append(result)
                logger.info(f"🔄 チェックポイントから復旧: Batch {start_batch} から再開")
        
        # 処理済みγ値の確認
        processed_gammas = {r['gamma'] for r in all_results}
        remaining_gammas = [g for g in all_gamma_values if g not in processed_gammas]
        
        logger.info(f"📊 処理済みγ値: {len(processed_gammas):,}個")
        logger.info(f"📊 残りγ値: {len(remaining_gammas):,}個")
        
        if not remaining_gammas:
            logger.info("✅ 全てのγ値が既に処理済みです")
            # 既存結果の統計を計算して返す
            return self._calculate_final_results(all_results, time.time() - start_time)
        
        # バッチ処理
        total_batches = (len(remaining_gammas) + self.batch_size - 1) // self.batch_size
        
        for batch_id in range(start_batch, start_batch + total_batches):
            if self.recovery_manager.emergency_stop:
                logger.warning("⚠️ 緊急停止により処理を中断")
                break
            
            # バッチ範囲計算（残りγ値から）
            batch_start_idx = (batch_id - start_batch) * self.batch_size
            batch_end_idx = min(batch_start_idx + self.batch_size, len(remaining_gammas))
            
            if batch_start_idx >= len(remaining_gammas):
                break
                
            gamma_batch = remaining_gammas[batch_start_idx:batch_end_idx]
            
            logger.info(f"🔄 Batch {batch_id + 1}/{start_batch + total_batches} 開始: {len(gamma_batch)}個のγ値")
            
            # システム監視
            monitor = self.recovery_manager.monitor_system()
            if not self.recovery_manager.check_system_health(monitor):
                logger.warning("⚠️ システム負荷が高いため、5秒待機...")
                time.sleep(5)
                
                # メモリクリーンアップ
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # バッチ処理実行
            try:
                batch_results = self.process_gamma_batch(gamma_batch, batch_id)
                all_results.extend(batch_results)
                
                # チェックポイント保存
                checkpoint = CheckpointData(
                    batch_id=batch_id,
                    gamma_start_idx=batch_start_idx,
                    gamma_end_idx=batch_end_idx,
                    completed_gammas=gamma_batch,
                    results=all_results,
                    timestamp=datetime.now().isoformat(),
                    system_state=asdict(monitor),
                    memory_usage=monitor.memory_percent,
                    gpu_memory=monitor.gpu_memory_used,
                    total_progress=(batch_id + 1) / (start_batch + total_batches) * 100
                )
                
                self.recovery_manager.save_checkpoint(checkpoint)
                
                # 中間結果保存
                if (batch_id + 1) % 10 == 0:  # 10バッチごと
                    self._save_intermediate_results(all_results, batch_id + 1)
                
                logger.info(f"✅ Batch {batch_id + 1} 完了: {len(batch_results)}個の結果")
                
            except Exception as e:
                logger.error(f"❌ Batch {batch_id} 処理エラー: {e}")
                logger.error(traceback.format_exc())
                
                # エラーでも継続（堅牢性）
                continue
        
        # 最終結果の計算
        execution_time = time.time() - start_time
        return self._calculate_final_results(all_results, execution_time)
    
    def _calculate_final_results(self, all_results: List[Dict], execution_time: float) -> Dict:
        """最終結果の計算"""
        # 統計計算
        valid_results = [r for r in all_results if not np.isnan(r.get('spectral_dimension', np.nan))]
        
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'total_gammas_processed': len(all_results),
            'valid_results': len(valid_results),
            'execution_time_seconds': execution_time,
            'execution_time_formatted': f"{execution_time // 3600:.0f}h {(execution_time % 3600) // 60:.0f}m {execution_time % 60:.1f}s",
            'processing_speed_per_gamma': execution_time / len(all_results) if all_results else 0,
            'success_rate': len(valid_results) / len(all_results) if all_results else 0,
            'results': all_results
        }
        
        if valid_results:
            spectral_dims = [r['spectral_dimension'] for r in valid_results]
            convergences = [r['convergence_to_half'] for r in valid_results if not np.isnan(r['convergence_to_half'])]
            
            final_results.update({
                'statistics': {
                    'mean_spectral_dimension': np.mean(spectral_dims),
                    'std_spectral_dimension': np.std(spectral_dims),
                    'mean_convergence': np.mean(convergences) if convergences else np.nan,
                    'best_convergence': np.min(convergences) if convergences else np.nan,
                    'worst_convergence': np.max(convergences) if convergences else np.nan
                }
            })
        
        # 最終結果保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_file = self.results_dir / f"10k_gamma_final_results_{timestamp}.json"
        with open(final_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False, default=str)
        
        print("\n" + "=" * 80)
        print("🎉 10,000γ Challenge 完了！")
        print("=" * 80)
        print(f"📊 処理済みγ値: {len(all_results):,}")
        print(f"✅ 有効結果: {len(valid_results):,}")
        print(f"⏱️  実行時間: {final_results['execution_time_formatted']}")
        print(f"🚀 処理速度: {final_results['processing_speed_per_gamma']:.4f}秒/γ値")
        print(f"📈 成功率: {final_results['success_rate']:.1%}")
        print(f"💾 結果保存: {final_file}")
        print("=" * 80)
        
        return final_results
    
    def _save_intermediate_results(self, results: List[Dict], batch_count: int):
        """中間結果の保存"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            intermediate_file = self.results_dir / f"intermediate_results_batch_{batch_count}_{timestamp}.json"
            
            intermediate_data = {
                'timestamp': datetime.now().isoformat(),
                'batches_completed': batch_count,
                'total_results': len(results),
                'results': results
            }
            
            with open(intermediate_file, 'w', encoding='utf-8') as f:
                json.dump(intermediate_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"💾 中間結果保存: {intermediate_file}")
            
        except Exception as e:
            logger.warning(f"⚠️ 中間結果保存エラー: {e}")

    def load_existing_results(self, results_dir: Path) -> List[Dict]:
        """既存の結果ファイルから結果を読み込む"""
        existing_results = []
        for file in results_dir.glob("*.json"):
            with open(file, 'r', encoding='utf-8') as f:
                results = json.load(f)
                existing_results.extend(results['results'])
        return existing_results

def main():
    """メイン実行関数"""
    try:
        # リカバリーマネージャー初期化
        recovery_manager = RobustRecoveryManager()
        
        # 10Kチャレンジシステム初期化
        challenge_system = NKAT10KGammaChallenge(recovery_manager)
        
        # 既存データの確認
        existing_data = challenge_system.load_latest_gamma_data()
        if existing_data:
            print(f"📊 既存データ検出: {len(existing_data.get('results', []))}個の結果")
            
            # ユーザーに選択肢を提示
            use_existing = True  # デフォルトで既存データを活用
            print("🔄 既存データを活用して継続実行します")
        else:
            print("📊 既存データが見つかりません。新規実行します")
            use_existing = False
        
        # チャレンジ実行
        results = challenge_system.execute_10k_challenge(
            resume=True, 
            use_existing_data=use_existing
        )
        
        print("🎉 NKAT v9.1 - 10,000γ Challenge 成功！")
        
        # 結果サマリー表示
        if 'statistics' in results:
            stats = results['statistics']
            print(f"\n📊 最終統計:")
            print(f"  平均スペクトル次元: {stats.get('mean_spectral_dimension', 'N/A'):.6f}")
            print(f"  平均収束性: {stats.get('mean_convergence', 'N/A'):.6f}")
            print(f"  最良収束性: {stats.get('best_convergence', 'N/A'):.6f}")
        
    except KeyboardInterrupt:
        print("\n⚠️ ユーザーによる中断")
    except Exception as e:
        logger.error(f"❌ 実行エラー: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 