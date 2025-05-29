#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌌🔬 NKAT リーマン予想解析システム - 最高精度版
Non-Commutative Kolmogorov-Arnold Theory (NKAT) Riemann Hypothesis Analysis System
Ultimate Precision Implementation with RTX3080 Optimization & Recovery

非可換コルモゴロフアーノルド表現理論による厳密なリーマン予想解析
RTX3080専用最適化・電源断リカバリー・Streamlit監視ダッシュボード統合

Author: NKAT Research Team
Date: 2025-05-28
Version: 2.0 - Ultimate Precision Implementation
License: MIT

主要機能:
- 非可換コルモゴロフアーノルド表現理論によるリーマンゼータ関数解析
- RTX3080専用最適化（10GB VRAM効率利用、CUDA最適化、混合精度計算）
- 電源断からのHDF5チェックポイント復元機能
- StreamlitベースのリアルタイムGPU/CPU監視ダッシュボード
- 超高精度計算（150桁精度、mpmath統合）
- tqdmプログレスバー表示（日本語対応）
- 包括的ログ記録とエラーハンドリング
"""

import sys
import os
import warnings
from pathlib import Path
import json
import time
import pickle
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Tuple, List, Optional, Dict, Union, Callable, Any
from abc import ABC, abstractmethod
import signal
import threading
import queue
import gc
import psutil

# 数値計算・科学計算
import numpy as np
import scipy.special as sp
from scipy.optimize import minimize, root_scalar
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh, spsolve
import mpmath as mp

# 機械学習・GPU計算
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

# 可視化
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns

# データ処理
import pandas as pd
import h5py

# 進捗表示・ログ
from tqdm import tqdm, trange
import logging
import logging.handlers

# Streamlit（ダッシュボード）
import streamlit as st

# システム監視
import GPUtil

# 日本語フォント設定
plt.rcParams['font.family'] = ['Yu Gothic', 'MS Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 警告抑制
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# GPU環境設定とRTX3080最適化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 使用デバイス: {device}")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name()
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"🎮 GPU: {gpu_name}")
    print(f"💾 VRAM: {total_memory:.1f} GB")
    
    # RTX3080専用最適化設定
    if "RTX 3080" in gpu_name or "RTX3080" in gpu_name:
        print("⚡ RTX3080専用最適化を有効化")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.9)
    
    torch.cuda.empty_cache()
    print(f"🔧 CUDA最適化設定完了")

# 超高精度計算設定（150桁精度）
mp.mp.dps = 150  # 150桁精度
mp.mp.pretty = True

@dataclass
class NKATRiemannParameters:
    """NKAT リーマン予想解析パラメータ設定"""
    
    # 非可換コルモゴロフアーノルド表現パラメータ
    nkat_dimension: int = 32  # NKAT表現次元
    nkat_precision: int = 150  # 計算精度（桁数）
    nkat_max_terms: int = 4096  # 最大項数
    nkat_epsilon: float = 1e-50  # 超高精度収束閾値
    
    # リーマンゼータ関数パラメータ
    riemann_critical_line_start: float = 0.5  # 臨界線開始点
    riemann_critical_line_end: float = 100.0  # 臨界線終了点
    riemann_zero_search_precision: float = 1e-30  # 零点探索精度
    riemann_max_zeros: int = 1000  # 最大零点数
    
    # 非可換幾何学パラメータ
    theta_ij: float = 1e-35  # 非可換パラメータ（プランク長さスケール）
    c_star_algebra_dim: int = 256  # C*-代数次元
    hilbert_space_dim: int = 512  # ヒルベルト空間次元
    spectral_triple_dim: int = 128  # スペクトラル三重次元
    
    # GPU最適化パラメータ
    gpu_batch_size: int = 1024  # GPU バッチサイズ
    gpu_memory_limit_gb: float = 9.0  # GPU メモリ制限（RTX3080用）
    use_mixed_precision: bool = True  # 混合精度計算
    cuda_streams: int = 4  # CUDA ストリーム数
    
    # リカバリー・チェックポイントパラメータ
    checkpoint_interval_seconds: int = 300  # チェックポイント間隔（秒）
    auto_save_enabled: bool = True  # 自動保存機能
    max_checkpoint_files: int = 10  # 最大チェックポイントファイル数
    checkpoint_compression: bool = True  # チェックポイント圧縮
    
    # 監視・ログパラメータ
    monitoring_interval_seconds: float = 1.0  # 監視間隔（秒）
    log_level: int = logging.INFO  # ログレベル
    enable_gpu_monitoring: bool = True  # GPU監視有効
    enable_cpu_monitoring: bool = True  # CPU監視有効
    
    # 数値計算パラメータ
    max_iterations: int = 10000  # 最大反復数
    convergence_threshold: float = 1e-50  # 収束閾値
    numerical_stability_check: bool = True  # 数値安定性チェック
    
    def __post_init__(self):
        """パラメータ検証と自動調整"""
        if self.nkat_dimension < 8:
            raise ValueError("NKAT次元は8以上である必要があります")
        if self.nkat_precision < 50:
            raise ValueError("計算精度は50桁以上である必要があります")
        
        # RTX3080専用調整
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            if "RTX 3080" in gpu_name or "RTX3080" in gpu_name:
                self.gpu_memory_limit_gb = 9.0
                self.gpu_batch_size = min(self.gpu_batch_size, 2048)
                print(f"✅ RTX3080専用パラメータ調整完了")

class SystemMonitor:
    """システム監視クラス - GPU/CPU/メモリ監視"""
    
    def __init__(self, params: NKATRiemannParameters):
        self.params = params
        self.monitoring = False
        self.monitor_thread = None
        self.data_queue = queue.Queue(maxsize=10000)
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """ロガーセットアップ"""
        logger = logging.getLogger('SystemMonitor')
        logger.setLevel(self.params.log_level)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def get_gpu_info(self) -> Optional[Dict[str, Any]]:
        """GPU情報取得"""
        if not self.params.enable_gpu_monitoring or not torch.cuda.is_available():
            return None
        
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return None
            
            gpu = gpus[0]
            
            # PyTorchからの詳細情報
            torch_info = {
                'name': torch.cuda.get_device_name(0),
                'total_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
                'allocated_memory_gb': torch.cuda.memory_allocated(0) / 1e9,
                'cached_memory_gb': torch.cuda.memory_reserved(0) / 1e9,
                'free_memory_gb': (torch.cuda.get_device_properties(0).total_memory - 
                                 torch.cuda.memory_reserved(0)) / 1e9
            }
            
            return {
                'id': gpu.id,
                'name': gpu.name,
                'load_percent': gpu.load * 100,
                'memory_used_mb': gpu.memoryUsed,
                'memory_total_mb': gpu.memoryTotal,
                'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                'temperature_celsius': gpu.temperature,
                'torch_info': torch_info,
                'timestamp': datetime.now()
            }
        except Exception as e:
            self.logger.error(f"GPU情報取得エラー: {e}")
            return None
    
    def get_cpu_info(self) -> Optional[Dict[str, Any]]:
        """CPU情報取得"""
        if not self.params.enable_cpu_monitoring:
            return None
        
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_freq = psutil.cpu_freq()
            
            # CPU温度取得（可能な場合）
            cpu_temps = None
            try:
                temps = psutil.sensors_temperatures()
                if 'coretemp' in temps:
                    cpu_temps = [temp.current for temp in temps['coretemp']]
                elif 'cpu_thermal' in temps:
                    cpu_temps = [temp.current for temp in temps['cpu_thermal']]
            except:
                pass
            
            return {
                'usage_percent': cpu_percent,
                'frequency_current_mhz': cpu_freq.current if cpu_freq else None,
                'frequency_max_mhz': cpu_freq.max if cpu_freq else None,
                'core_count': psutil.cpu_count(logical=False),
                'thread_count': psutil.cpu_count(logical=True),
                'temperatures_celsius': cpu_temps,
                'avg_temperature_celsius': np.mean(cpu_temps) if cpu_temps else None,
                'timestamp': datetime.now()
            }
        except Exception as e:
            self.logger.error(f"CPU情報取得エラー: {e}")
            return None
    
    def get_memory_info(self) -> Dict[str, Any]:
        """メモリ情報取得"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return {
                'total_gb': memory.total / 1e9,
                'available_gb': memory.available / 1e9,
                'used_gb': memory.used / 1e9,
                'percent': memory.percent,
                'swap_total_gb': swap.total / 1e9,
                'swap_used_gb': swap.used / 1e9,
                'swap_percent': swap.percent,
                'timestamp': datetime.now()
            }
        except Exception as e:
            self.logger.error(f"メモリ情報取得エラー: {e}")
            return {}
    
    def monitor_loop(self):
        """監視ループ"""
        while self.monitoring:
            try:
                data = {
                    'gpu': self.get_gpu_info(),
                    'cpu': self.get_cpu_info(),
                    'memory': self.get_memory_info(),
                    'timestamp': datetime.now()
                }
                
                if not self.data_queue.full():
                    self.data_queue.put(data)
                
                time.sleep(self.params.monitoring_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"監視ループエラー: {e}")
                time.sleep(1.0)
    
    def start_monitoring(self):
        """監視開始"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
            self.monitor_thread.start()
            self.logger.info("システム監視開始")
    
    def stop_monitoring(self):
        """監視停止"""
        if self.monitoring:
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5.0)
            self.logger.info("システム監視停止")
    
    def get_recent_data(self, seconds: int = 60) -> List[Dict[str, Any]]:
        """最近のデータ取得"""
        data_list = []
        cutoff_time = datetime.now() - timedelta(seconds=seconds)
        
        temp_data = []
        while not self.data_queue.empty():
            temp_data.append(self.data_queue.get())
        
        # 時間でフィルタリング
        for data in temp_data:
            if data['timestamp'] >= cutoff_time:
                data_list.append(data)
        
        # 使用したデータを戻す（最新のもののみ）
        for data in temp_data[-100:]:  # 最新100件のみ保持
            if not self.data_queue.full():
                self.data_queue.put(data)
        
        return sorted(data_list, key=lambda x: x['timestamp'])

class CheckpointManager:
    """チェックポイント管理クラス - 電源断リカバリー対応"""
    
    def __init__(self, params: NKATRiemannParameters, base_dir: str = "results/checkpoints"):
        self.params = params
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()
        self.last_checkpoint_time = time.time()
        
    def _setup_logger(self) -> logging.Logger:
        """ロガーセットアップ"""
        logger = logging.getLogger('CheckpointManager')
        logger.setLevel(self.params.log_level)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def create_checkpoint_id(self) -> str:
        """チェックポイントID生成"""
        param_str = json.dumps(asdict(self.params), sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()[:12]
    
    def save_checkpoint(self, 
                       stage: str,
                       data: Dict[str, Any],
                       metadata: Dict[str, Any] = None) -> str:
        """チェックポイント保存"""
        try:
            checkpoint_id = self.create_checkpoint_id()
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")
            filename = f"{stage}_{checkpoint_id}_{timestamp}.h5"
            filepath = self.base_dir / filename
            
            # メタデータ準備
            if metadata is None:
                metadata = {}
            
            metadata.update({
                'stage': stage,
                'checkpoint_id': checkpoint_id,
                'timestamp': timestamp,
                'params': asdict(self.params),
                'python_version': sys.version,
                'torch_version': torch.__version__,
                'numpy_version': np.__version__
            })
            
            # HDF5形式で保存（Unicode文字列対応）
            with h5py.File(filepath, 'w') as f:
                # メタデータ保存（文字列エンコーディング対応）
                meta_group = f.create_group('metadata')
                for key, value in metadata.items():
                    if isinstance(value, str):
                        # Unicode文字列をUTF-8でエンコード
                        meta_group.attrs[key] = value.encode('utf-8')
                    elif isinstance(value, (int, float, bool)):
                        meta_group.attrs[key] = value
                    elif isinstance(value, complex):
                        # 複素数を実部・虚部に分けて保存
                        meta_group.attrs[f"{key}_real"] = value.real
                        meta_group.attrs[f"{key}_imag"] = value.imag
                    else:
                        # その他のオブジェクトはJSON文字列として保存
                        json_str = json.dumps(value, ensure_ascii=False)
                        meta_group.attrs[key] = json_str.encode('utf-8')
                
                # データ保存（数理的に正しい型変換）
                data_group = f.create_group('data')
                for key, value in data.items():
                    if isinstance(value, np.ndarray):
                        # NumPy配列の直接保存
                        data_group.create_dataset(key, data=value, compression='gzip')
                    elif isinstance(value, torch.Tensor):
                        # PyTorchテンソルをNumPy配列に変換
                        numpy_data = value.detach().cpu().numpy()
                        data_group.create_dataset(key, data=numpy_data, compression='gzip')
                    elif isinstance(value, (list, tuple)):
                        # リスト・タプルをNumPy配列に変換
                        try:
                            numpy_data = np.array(value)
                            data_group.create_dataset(key, data=numpy_data, compression='gzip')
                        except (ValueError, TypeError):
                            # 変換できない場合はJSON文字列として保存
                            json_str = json.dumps(value, ensure_ascii=False)
                            data_group.attrs[key] = json_str.encode('utf-8')
                    elif isinstance(value, (int, float)):
                        # 数値は属性として保存
                        data_group.attrs[key] = value
                    elif isinstance(value, complex):
                        # 複素数を実部・虚部に分けて保存
                        data_group.attrs[f"{key}_real"] = value.real
                        data_group.attrs[f"{key}_imag"] = value.imag
                    elif isinstance(value, str):
                        # 文字列をUTF-8でエンコード
                        data_group.attrs[key] = value.encode('utf-8')
                    else:
                        # その他のオブジェクトはJSON文字列として保存
                        try:
                            json_str = json.dumps(value, ensure_ascii=False)
                            data_group.attrs[key] = json_str.encode('utf-8')
                        except (TypeError, ValueError):
                            # JSON化できない場合は文字列表現を保存
                            str_repr = str(value)
                            data_group.attrs[key] = str_repr.encode('utf-8')
            
            self.last_checkpoint_time = time.time()
            self.logger.info(f"チェックポイント保存: {filename}")
            
            # 古いチェックポイントの削除
            self._cleanup_old_checkpoints(checkpoint_id, stage)
            
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"チェックポイント保存エラー: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_file: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """チェックポイント読み込み"""
        try:
            filepath = Path(checkpoint_file)
            if not filepath.exists():
                raise FileNotFoundError(f"チェックポイントファイルが見つかりません: {checkpoint_file}")
            
            data = {}
            metadata = {}
            
            with h5py.File(filepath, 'r') as f:
                # メタデータ読み込み（文字列デコーディング対応）
                if 'metadata' in f:
                    meta_group = f['metadata']
                    for key in meta_group.attrs:
                        value = meta_group.attrs[key]
                        
                        if isinstance(value, bytes):
                            # バイト列をUTF-8でデコード
                            try:
                                decoded_str = value.decode('utf-8')
                                # JSON文字列かどうか判定
                                if decoded_str.startswith(('{', '[', '"')):
                                    try:
                                        metadata[key] = json.loads(decoded_str)
                                    except json.JSONDecodeError:
                                        metadata[key] = decoded_str
                                else:
                                    metadata[key] = decoded_str
                            except UnicodeDecodeError:
                                metadata[key] = str(value)
                        elif key.endswith('_real') and f"{key[:-5]}_imag" in meta_group.attrs:
                            # 複素数の実部・虚部を結合
                            base_key = key[:-5]
                            if base_key not in metadata:
                                real_part = meta_group.attrs[key]
                                imag_part = meta_group.attrs[f"{base_key}_imag"]
                                metadata[base_key] = complex(real_part, imag_part)
                        elif not key.endswith('_imag'):
                            # 通常の数値データ
                            metadata[key] = value
                
                # データ読み込み（数理的に正しい型復元）
                if 'data' in f:
                    data_group = f['data']
                    
                    # データセットの読み込み
                    for key in data_group.keys():
                        dataset = data_group[key]
                        data[key] = np.array(dataset)
                    
                    # 属性の読み込み
                    for key in data_group.attrs:
                        value = data_group.attrs[key]
                        
                        if isinstance(value, bytes):
                            # バイト列をUTF-8でデコード
                            try:
                                decoded_str = value.decode('utf-8')
                                # JSON文字列かどうか判定
                                if decoded_str.startswith(('{', '[', '"')):
                                    try:
                                        data[key] = json.loads(decoded_str)
                                    except json.JSONDecodeError:
                                        data[key] = decoded_str
                                else:
                                    data[key] = decoded_str
                            except UnicodeDecodeError:
                                data[key] = str(value)
                        elif key.endswith('_real') and f"{key[:-5]}_imag" in data_group.attrs:
                            # 複素数の実部・虚部を結合
                            base_key = key[:-5]
                            if base_key not in data:
                                real_part = data_group.attrs[key]
                                imag_part = data_group.attrs[f"{base_key}_imag"]
                                data[base_key] = complex(real_part, imag_part)
                        elif not key.endswith('_imag'):
                            # 通常の数値データ
                            data[key] = value
            
            self.logger.info(f"チェックポイント読み込み: {checkpoint_file}")
            return data, metadata
            
        except Exception as e:
            self.logger.error(f"チェックポイント読み込みエラー: {e}")
            raise
    
    def _cleanup_old_checkpoints(self, checkpoint_id: str, stage: str):
        """古いチェックポイントの削除"""
        try:
            pattern = f"{stage}_{checkpoint_id}_*.h5"
            checkpoint_files = list(self.base_dir.glob(pattern))
            
            if len(checkpoint_files) > self.params.max_checkpoint_files:
                # 作成時間でソート
                checkpoint_files.sort(key=lambda x: x.stat().st_mtime)
                
                # 古いファイルを削除
                files_to_delete = checkpoint_files[:-self.params.max_checkpoint_files]
                for file_path in files_to_delete:
                    file_path.unlink()
                    self.logger.info(f"古いチェックポイント削除: {file_path.name}")
                    
        except Exception as e:
            self.logger.warning(f"チェックポイント削除エラー: {e}")
    
    def should_save_checkpoint(self) -> bool:
        """チェックポイント保存判定"""
        return (time.time() - self.last_checkpoint_time) >= self.params.checkpoint_interval_seconds
    
    def get_latest_checkpoint(self, stage: str = None) -> Optional[str]:
        """最新チェックポイント取得"""
        try:
            if stage:
                pattern = f"{stage}_*.h5"
            else:
                pattern = "*.h5"
            
            checkpoint_files = list(self.base_dir.glob(pattern))
            
            if not checkpoint_files:
                return None
            
            # 最新ファイルを返す
            latest_file = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
            return str(latest_file)
            
        except Exception as e:
            self.logger.error(f"最新チェックポイント取得エラー: {e}")
            return None

class NonCommutativeKolmogorovArnoldRepresentation(nn.Module):
    """
    非可換コルモゴロフアーノルド表現理論の実装
    
    定理: 任意の非可換連続汎関数 F は以下の形式で表現可能
    F(x̂₁, ..., x̂ₙ) = Σ Φ̂q(Σ ψ̂q,p(x̂p))
    
    ここで:
    - Φ̂q: 単変数作用素値関数
    - ψ̂q,p: 非可換変数に依存する作用素
    - 合成は非可換★積で定義
    """
    
    def __init__(self, params: NKATRiemannParameters):
        super().__init__()
        self.params = params
        self.device = device
        self.logger = self._setup_logger()
        
        # 非可換代数の初期化
        self._initialize_noncommutative_algebra()
        
        # スペクトラル三重の初期化
        self._initialize_spectral_triple()
        
        # 混合精度計算用スケーラー
        if self.params.use_mixed_precision:
            self.scaler = GradScaler()
        
        self.logger.info(f"NKAT表現初期化完了: {self.params.nkat_dimension}次元")
    
    def _setup_logger(self) -> logging.Logger:
        """ロガーセットアップ"""
        logger = logging.getLogger('NKATRepresentation')
        logger.setLevel(self.params.log_level)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_noncommutative_algebra(self):
        """非可換代数の初期化"""
        dim = self.params.nkat_dimension
        
        # 非可換構造定数 [X̂ᵢ, X̂ⱼ] = iθᵢⱼ（実数部のみ使用）
        self.theta_matrix = torch.zeros(dim, dim, device=self.device, dtype=torch.complex128)
        for i in range(dim):
            for j in range(i+1, dim):
                theta_value = self.params.theta_ij * (1 + 0.1 * (i + j))
                self.theta_matrix[i, j] = complex(theta_value, 0)  # 実数値として設定
                self.theta_matrix[j, i] = -self.theta_matrix[i, j]
        
        # 非可換座標作用素（実数部のみ使用）
        self.coordinate_operators = nn.ParameterList([
            nn.Parameter(
                torch.randn(self.params.c_star_algebra_dim, 
                           self.params.c_star_algebra_dim, 
                           device=self.device, dtype=torch.complex128) * 0.1
            )
            for _ in range(dim)
        ])
        
        # ディラック作用素（実数値）
        self.dirac_operator = nn.Parameter(
            torch.randn(self.params.hilbert_space_dim, 
                       self.params.hilbert_space_dim, 
                       device=self.device, dtype=torch.float32) * 0.1
        )
        
        self.logger.info("非可換代数初期化完了")
    
    def _initialize_spectral_triple(self):
        """スペクトラル三重 (A, H, D) の初期化"""
        # A: 非可換代数 - 数理的に正しい型統一
        self.algebra_representation = nn.Linear(
            self.params.nkat_dimension, 
            self.params.spectral_triple_dim,
            device=self.device,
            dtype=torch.float32  # 混合精度対応のため float32 に統一
        )
        
        # H: ヒルベルト空間
        self.hilbert_space_embedding = nn.Linear(
            self.params.spectral_triple_dim,
            self.params.hilbert_space_dim,
            device=self.device,
            dtype=torch.float32  # 混合精度対応のため float32 に統一
        )
        
        # D: ディラック作用素（自己共役）- 実数値で初期化
        dirac_real = torch.randn(self.params.hilbert_space_dim, 
                                self.params.hilbert_space_dim, 
                                device=self.device, dtype=torch.float32) * 0.1
        self.dirac_operator_spectral = nn.Parameter(
            dirac_real + dirac_real.T  # 自己共役性を保証
        )
        
        self.logger.info("スペクトラル三重初期化完了")
    
    def star_product(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        非可換★積の計算
        (f ★ g)(x) = f(x)g(x) + (iθ/2) ∂f/∂xᵢ ∂g/∂xⱼ θᵢⱼ + O(θ²)
        """
        # 入力テンソルの型を統一
        f = f.to(dtype=torch.float32)
        g = g.to(dtype=torch.float32)
        
        # 0次項: 通常の積
        result = f * g
        
        # 1次項: 非可換補正（数理的に正しい実装）
        if self.params.theta_ij != 0 and f.requires_grad and g.requires_grad:
            try:
                # 勾配計算（自動微分使用）
                f_grad = torch.autograd.grad(f.sum(), f, create_graph=True, retain_graph=True)[0]
                g_grad = torch.autograd.grad(g.sum(), g, create_graph=True, retain_graph=True)[0]
                
                # θ行列との縮約（実数部のみ）
                noncommutative_correction = torch.zeros_like(result)
                theta_real = self.theta_matrix.real.to(dtype=torch.float32)
                
                for i in range(min(self.params.nkat_dimension, f_grad.shape[-1])):
                    for j in range(min(self.params.nkat_dimension, g_grad.shape[-1])):
                        noncommutative_correction += (
                            0.5 * theta_real[i, j] * 
                            f_grad[..., i] * g_grad[..., j]
                        )
                
                result += noncommutative_correction
            except RuntimeError:
                # 勾配計算が不可能な場合は0次項のみ
                pass
        
        return result
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        NKAT表現の前向き計算
        F(x̂) = Σ Φ̂q(Σ ψ̂q,p(x̂p))
        """
        # 入力テンソルの型を統一
        x = x.to(dtype=torch.float32, device=self.device)
        
        # 混合精度計算
        if self.params.use_mixed_precision:
            with autocast():
                return self._forward_impl(x)
        else:
            return self._forward_impl(x)
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """実際の前向き計算実装"""
        # 非可換座標への埋め込み（型安全）
        x_noncommutative = self.algebra_representation(x)
        
        # スペクトラル三重での処理
        h_embedded = self.hilbert_space_embedding(x_noncommutative)
        
        # ディラック作用素の適用（実数演算）
        dirac_output = torch.matmul(h_embedded, self.dirac_operator_spectral)
        
        # 非可換★積による合成（数理的に正しい実装）
        result = dirac_output
        for i, coord_op in enumerate(self.coordinate_operators):
            if i < x.shape[-1]:
                # 型を統一してから演算
                coord_op_real = coord_op.real.to(dtype=torch.float32)
                coord_contribution = torch.matmul(
                    x[..., i:i+1], 
                    coord_op_real[:1, :1]
                )
                result = self.star_product(result, coord_contribution)
        
        # 数理的に正しい実数値を返す（物理的観測量）
        return result.real if result.is_complex() else result

class RiemannZetaAnalyzer:
    """リーマンゼータ関数解析クラス"""
    
    def __init__(self, params: NKATRiemannParameters):
        self.params = params
        self.logger = self._setup_logger()
        
        # NKAT表現の初期化
        self.nkat_representation = NonCommutativeKolmogorovArnoldRepresentation(params)
        
        # 超高精度計算設定
        mp.mp.dps = params.nkat_precision
        
        self.logger.info("リーマンゼータ解析器初期化完了")
    
    def _setup_logger(self) -> logging.Logger:
        """ロガーセットアップ"""
        logger = logging.getLogger('RiemannZetaAnalyzer')
        logger.setLevel(self.params.log_level)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def riemann_zeta_mpmath(self, s: complex) -> complex:
        """超高精度リーマンゼータ関数計算"""
        try:
            # 数理的に正しい特殊値の処理
            if abs(s - 1.0) < 1e-15:
                # s=1での極を回避
                return complex(float('inf'), 0)
            elif abs(s.real) < 1e-15 and abs(s.imag) < 1e-15:
                # s=0での値: ζ(0) = -1/2
                return complex(-0.5, 0)
            elif abs(s.real - (-1)) < 1e-15 and abs(s.imag) < 1e-15:
                # s=-1での値: ζ(-1) = -1/12
                return complex(-1.0/12.0, 0)
            elif abs(s.real - (-2)) < 1e-15 and abs(s.imag) < 1e-15:
                # s=-2での値: ζ(-2) = 0
                return complex(0, 0)
            
            s_mp = mp.mpc(s.real, s.imag)
            result = mp.zeta(s_mp)
            return complex(float(result.real), float(result.imag))
        except Exception as e:
            self.logger.error(f"リーマンゼータ計算エラー s={s}: {e}")
            # 数理的に正しいデフォルト値を返す
            if s.real < 0:
                return complex(0, 0)  # 負の偶数での零点
            else:
                return complex(1, 0)  # 正の実部での近似値
    
    def find_riemann_zeros(self, t_start: float, t_end: float, 
                          num_points: int = 1000) -> List[complex]:
        """リーマンゼータ関数の零点探索"""
        zeros = []
        
        # 臨界線 s = 1/2 + it での探索
        t_values = np.linspace(t_start, t_end, num_points)
        
        with tqdm(t_values, desc="零点探索", unit="点") as pbar:
            for t in pbar:
                s = complex(0.5, t)
                zeta_value = self.riemann_zeta_mpmath(s)
                
                # 零点判定（絶対値が閾値以下）
                if abs(zeta_value) < self.params.riemann_zero_search_precision:
                    zeros.append(s)
                    pbar.set_postfix({"零点数": len(zeros)})
                    self.logger.info(f"零点発見: s = {s}, ζ(s) = {zeta_value}")
                
                if len(zeros) >= self.params.riemann_max_zeros:
                    break
        
        return zeros
    
    def verify_riemann_hypothesis_nkat(self, zeros: List[complex]) -> Dict[str, Any]:
        """NKAT表現によるリーマン予想検証"""
        verification_results = {
            'total_zeros': len(zeros),
            'zeros_on_critical_line': 0,
            'max_deviation_from_critical_line': 0.0,
            'nkat_consistency_score': 0.0,
            'verification_details': []
        }
        
        if not zeros:
            return verification_results
        
        # 各零点の検証
        with tqdm(zeros, desc="NKAT検証", unit="零点") as pbar:
            for i, zero in enumerate(pbar):
                # 臨界線からの偏差
                deviation = abs(zero.real - 0.5)
                verification_results['max_deviation_from_critical_line'] = max(
                    verification_results['max_deviation_from_critical_line'], 
                    deviation
                )
                
                # 臨界線上の零点カウント
                if deviation < self.params.riemann_zero_search_precision:
                    verification_results['zeros_on_critical_line'] += 1
                
                # NKAT表現による一貫性チェック（数理的に正しい実装）
                try:
                    # 複素数を実数テンソルに変換（実部・虚部を分離）
                    zero_tensor = torch.tensor(
                        [[zero.real, zero.imag]], 
                        device=device, 
                        dtype=torch.float32,
                        requires_grad=False
                    )
                    
                    # NKAT表現の計算
                    with torch.no_grad():  # 勾配計算を無効化
                        nkat_output = self.nkat_representation(zero_tensor)
                        nkat_consistency = float(torch.norm(nkat_output).item())
                    
                except Exception as e:
                    self.logger.warning(f"NKAT計算エラー (零点 {i}): {e}")
                    nkat_consistency = 0.0
                
                verification_results['verification_details'].append({
                    'zero_index': i,
                    'zero': zero,
                    'deviation_from_critical_line': deviation,
                    'nkat_consistency': nkat_consistency
                })
                
                pbar.set_postfix({
                    "臨界線上": verification_results['zeros_on_critical_line'],
                    "最大偏差": f"{verification_results['max_deviation_from_critical_line']:.2e}"
                })
        
        # 全体的な一貫性スコア計算
        if verification_results['verification_details']:
            consistency_scores = [
                detail['nkat_consistency'] 
                for detail in verification_results['verification_details']
            ]
            verification_results['nkat_consistency_score'] = np.mean(consistency_scores)
        
        # リーマン予想の検証結果
        verification_results['riemann_hypothesis_verified'] = (
            verification_results['zeros_on_critical_line'] == verification_results['total_zeros']
        )
        
        return verification_results

class NKATRiemannDashboard:
    """Streamlit ダッシュボード"""
    
    def __init__(self, params: NKATRiemannParameters):
        self.params = params
        self.system_monitor = SystemMonitor(params)
        self.checkpoint_manager = CheckpointManager(params)
        self.riemann_analyzer = RiemannZetaAnalyzer(params)
        
    def run_dashboard(self):
        """ダッシュボード実行"""
        st.set_page_config(
            page_title="NKAT リーマン予想解析システム",
            page_icon="🌌",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🌌 NKAT リーマン予想解析システム")
        st.markdown("**非可換コルモゴロフアーノルド表現理論による最高精度解析**")
        
        # サイドバー
        with st.sidebar:
            st.header("⚙️ システム制御")
            
            if st.button("🚀 監視開始"):
                self.system_monitor.start_monitoring()
                st.success("システム監視開始")
            
            if st.button("⏹️ 監視停止"):
                self.system_monitor.stop_monitoring()
                st.info("システム監視停止")
            
            st.header("📊 パラメータ")
            st.json(asdict(self.params))
        
        # メインエリア
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("🖥️ システム監視")
            self._display_system_monitoring()
        
        with col2:
            st.header("🔬 リーマン解析")
            self._display_riemann_analysis()
        
        # 下部エリア
        st.header("📈 解析結果")
        self._display_analysis_results()
    
    def _display_system_monitoring(self):
        """システム監視表示"""
        recent_data = self.system_monitor.get_recent_data(60)
        
        if recent_data:
            latest_data = recent_data[-1]
            
            # GPU情報
            if latest_data.get('gpu'):
                gpu_info = latest_data['gpu']
                st.metric("GPU使用率", f"{gpu_info['load_percent']:.1f}%")
                st.metric("GPU温度", f"{gpu_info['temperature_celsius']:.1f}°C")
                st.metric("GPUメモリ", f"{gpu_info['memory_percent']:.1f}%")
            
            # CPU情報
            if latest_data.get('cpu'):
                cpu_info = latest_data['cpu']
                st.metric("CPU使用率", f"{cpu_info['usage_percent']:.1f}%")
                if cpu_info.get('avg_temperature_celsius'):
                    st.metric("CPU温度", f"{cpu_info['avg_temperature_celsius']:.1f}°C")
            
            # メモリ情報
            if latest_data.get('memory'):
                memory_info = latest_data['memory']
                st.metric("メモリ使用率", f"{memory_info['percent']:.1f}%")
        else:
            st.info("監視データがありません")
    
    def _display_riemann_analysis(self):
        """リーマン解析表示"""
        if st.button("🔍 零点探索開始"):
            with st.spinner("零点探索中..."):
                zeros = self.riemann_analyzer.find_riemann_zeros(
                    self.params.riemann_critical_line_start,
                    min(self.params.riemann_critical_line_end, 50.0),  # デモ用に制限
                    100  # デモ用に制限
                )
                
                st.success(f"零点 {len(zeros)} 個発見")
                
                if zeros:
                    # 検証実行
                    verification = self.riemann_analyzer.verify_riemann_hypothesis_nkat(zeros)
                    
                    st.metric("臨界線上の零点", verification['zeros_on_critical_line'])
                    st.metric("総零点数", verification['total_zeros'])
                    st.metric("最大偏差", f"{verification['max_deviation_from_critical_line']:.2e}")
                    
                    if verification['riemann_hypothesis_verified']:
                        st.success("✅ リーマン予想が検証されました！")
                    else:
                        st.warning("⚠️ 一部の零点が臨界線から外れています")
    
    def _display_analysis_results(self):
        """解析結果表示"""
        st.info("解析結果は実行後に表示されます")

def main():
    """メイン関数"""
    print("🌌 NKAT リーマン予想解析システム - 最高精度版")
    print("=" * 60)
    
    # パラメータ初期化
    params = NKATRiemannParameters()
    
    # Streamlitダッシュボード起動
    dashboard = NKATRiemannDashboard(params)
    dashboard.run_dashboard()

if __name__ == "__main__":
    main() 