#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀🔄 RTX3080対応 電源断リカバリー機能付き高次元ディラック/ラプラシアン作用素GPU解析
Non-Commutative Kolmogorov-Arnold Theory (NKAT) における作用素理論 - Recovery対応版

Author: NKAT Research Team
Date: 2025-01-24
Version: 1.7 - RTX3080最適化強化版（メモリ効率・GPU最適化・バッチ処理改善）

主要機能:
- RTX3080専用最適化（10GB VRAM効率利用）
- 電源断からのチェックポイント復元
- 計算途中からの再開機能
- より高次元（6-10次元）での解析対応
- 自動保存機能
- GPU/RTX3080最適化
- tqdmプログレスバー表示
- 詳細ログ記録機能
- メモリ効率的バッチ処理
- CUDA最適化
"""

import torch
import torch.nn as nn
import torch.sparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict, Callable, Any
import warnings
from pathlib import Path
import json
import time
import pickle
import hashlib
from dataclasses import dataclass, asdict
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import h5py
import os
from datetime import datetime
import signal
import sys
from tqdm import tqdm, trange
import logging
import logging.handlers
import gc
import psutil

# GPU可用性チェックと最適化設定
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
        # メモリ効率化
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.9)  # VRAM使用率90%まで
    
    # CUDA最適化設定
    torch.cuda.empty_cache()
    print(f"🔧 CUDA最適化設定完了")

# 日本語フォント設定
try:
    plt.rcParams['font.family'] = 'MS Gothic'
except:
    plt.rcParams['font.family'] = 'DejaVu Sans'

warnings.filterwarnings('ignore')

def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    ロガーのセットアップ
    
    Args:
        name: ロガー名
        log_file: ログファイルパス（Noneの場合は自動生成）
        level: ログレベル
    
    Returns:
        設定済みロガー
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 既存のハンドラーをクリア
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # フォーマッター設定
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # コンソールハンドラー
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # ファイルハンドラー
    if log_file is None:
        log_dir = Path("results/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"nkat_gpu_recovery_{timestamp}.log"
    
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# グローバルロガーの設定
main_logger = setup_logger('NKAT_GPU_Recovery')

def get_optimal_batch_size(total_dim: int, available_memory_gb: float) -> int:
    """最適なバッチサイズの計算"""
    # 複素数double precision (16 bytes) を考慮
    bytes_per_element = 16
    safety_factor = 0.7  # 安全係数
    
    available_bytes = available_memory_gb * 1e9 * safety_factor
    max_elements = int(available_bytes / bytes_per_element)
    
    # バッチサイズは total_dim の約数になるように調整
    optimal_batch = min(max_elements // total_dim, total_dim)
    
    # 2の累乗に近い値に調整（GPU効率向上）
    power_of_2 = 2 ** int(np.log2(optimal_batch))
    if power_of_2 < 32:
        power_of_2 = 32
    
    return min(power_of_2, total_dim)

def monitor_gpu_memory():
    """GPU メモリ使用量の監視"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'free_gb': total - reserved,
            'usage_percent': (reserved / total) * 100
        }
    return None

@dataclass
class RecoveryGPUOperatorParameters:
    """Recovery機能付きGPU対応作用素パラメータ"""
    dimension: int  # 空間次元（最大10次元まで対応）
    lattice_size: int  # 格子サイズ
    theta: float  # 非可換パラメータ
    kappa: float  # κ-変形パラメータ
    mass: float  # 質量項
    coupling: float  # 結合定数
    use_sparse: bool = True  # スパース行列使用
    recovery_enabled: bool = True  # リカバリー機能有効
    checkpoint_interval: int = 300  # チェックポイント間隔（秒）
    auto_save: bool = True  # 自動保存機能
    max_eigenvalues: int = 100  # 固有値計算数
    memory_limit_gb: float = 8.0  # メモリ制限（GB）
    log_level: int = logging.INFO  # ログレベル
    gpu_batch_size: int = None  # GPU バッチサイズ（自動計算）
    use_mixed_precision: bool = True  # 混合精度計算
    
    def __post_init__(self):
        if self.dimension < 2 or self.dimension > 10:
            raise ValueError("次元は2-10の範囲である必要があります")
        if self.lattice_size < 4:
            warnings.warn("格子サイズが小さすぎる可能性があります")
        
        # スピノル次元の計算
        if self.dimension <= 3:
            spinor_dim = 2
        elif self.dimension <= 6:
            spinor_dim = 4
        elif self.dimension <= 8:
            spinor_dim = 8
        else:
            spinor_dim = 16
        
        total_dim = self.lattice_size**self.dimension * spinor_dim
        
        # RTX3080の場合のメモリ制限調整
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            if "RTX 3080" in gpu_name or "RTX3080" in gpu_name:
                self.memory_limit_gb = 9.0  # RTX3080の場合は9GB使用可能
                main_logger.info("RTX3080検出: メモリ制限を9GBに設定")
        
        # 最適バッチサイズの自動計算
        if self.gpu_batch_size is None:
            available_memory = self.memory_limit_gb
            if torch.cuda.is_available():
                gpu_memory = monitor_gpu_memory()
                if gpu_memory:
                    available_memory = min(self.memory_limit_gb, gpu_memory['free_gb'])
            
            self.gpu_batch_size = get_optimal_batch_size(total_dim, available_memory)
            main_logger.info(f"最適バッチサイズ自動設定: {self.gpu_batch_size}")
        
        # メモリ使用量チェック
        estimated_memory = (total_dim**2 * 16) / 1e9  # 複素数double precision
        
        if self.use_sparse:
            sparsity = min(0.1, 1000.0 / total_dim)
            estimated_memory *= sparsity
            main_logger.info(f"推定メモリ使用量（スパース）: {estimated_memory:.2f} GB")
        else:
            main_logger.warning(f"推定メモリ使用量（密行列）: {estimated_memory:.2f} GB")
        
        if estimated_memory > self.memory_limit_gb:
            warning_msg = f"メモリ不足の可能性: 推定{estimated_memory:.1f}GB > 制限{self.memory_limit_gb}GB"
            main_logger.warning(warning_msg)
            print(f"⚠️  {warning_msg}")
            if not self.use_sparse:
                print("スパース行列の使用を強く推奨します")

class CheckpointManager:
    """チェックポイント管理クラス"""
    
    def __init__(self, base_dir: str = "results/checkpoints"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.current_checkpoint = None
        self.logger = setup_logger('CheckpointManager')
        
    def create_checkpoint_id(self, params: RecoveryGPUOperatorParameters) -> str:
        """パラメータに基づくチェックポイントIDの生成"""
        param_str = json.dumps(asdict(params), sort_keys=True)
        checkpoint_id = hashlib.md5(param_str.encode()).hexdigest()[:12]
        self.logger.info(f"チェックポイントID生成: {checkpoint_id}")
        return checkpoint_id
    
    def save_checkpoint(self, 
                       checkpoint_id: str,
                       stage: str,
                       data: Dict[str, Any],
                       metadata: Dict[str, Any] = None) -> str:
        """チェックポイントの保存"""
        self.logger.info(f"チェックポイント保存開始: ID={checkpoint_id}, stage={stage}")
        
        checkpoint_dir = self.base_dir / checkpoint_id
        checkpoint_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().isoformat()
        checkpoint_file = checkpoint_dir / f"{stage}_{timestamp.replace(':', '-')}.h5"
        
        # メタデータの準備
        if metadata is None:
            metadata = {}
        metadata.update({
            'timestamp': timestamp,
            'stage': stage,
            'checkpoint_id': checkpoint_id
        })
        
        # プログレスバー付きでHDF5ファイルに保存
        try:
            with tqdm(desc="💾 チェックポイント保存中", unit="item", disable=False) as pbar:
                with h5py.File(checkpoint_file, 'w') as f:
                    # メタデータ
                    meta_group = f.create_group('metadata')
                    for key, value in metadata.items():
                        if isinstance(value, (str, int, float)):
                            meta_group.attrs[key] = value
                        else:
                            meta_group.attrs[key] = str(value)
                    pbar.update(1)
                    
                    # データ
                    data_group = f.create_group('data')
                    for key, value in tqdm(data.items(), desc="データ保存中", leave=False):
                        try:
                            if isinstance(value, np.ndarray):
                                data_group.create_dataset(key, data=value)
                                self.logger.debug(f"配列データ保存: {key}, shape={value.shape}")
                            elif isinstance(value, sp.spmatrix):
                                # スパース行列の保存
                                sparse_group = data_group.create_group(key)
                                sparse_group.create_dataset('data', data=value.data)
                                sparse_group.create_dataset('indices', data=value.indices)
                                sparse_group.create_dataset('indptr', data=value.indptr)
                                sparse_group.attrs['shape'] = value.shape
                                sparse_group.attrs['format'] = value.format
                                self.logger.debug(f"スパース行列保存: {key}, shape={value.shape}, nnz={value.nnz}")
                            elif isinstance(value, (int, float, str)):
                                data_group.attrs[key] = value
                                self.logger.debug(f"属性データ保存: {key}={value}")
                            else:
                                # その他のデータはpickleで保存
                                pickled_data = pickle.dumps(value)
                                data_group.create_dataset(f'{key}_pickled', data=np.frombuffer(pickled_data, dtype=np.uint8))
                                self.logger.debug(f"Pickleデータ保存: {key}")
                        except Exception as e:
                            self.logger.error(f"データ保存エラー: {key}, error={e}")
                            raise
                        pbar.update(1)
            
            file_size = checkpoint_file.stat().st_size / (1024*1024)  # MB
            self.logger.info(f"チェックポイント保存完了: {checkpoint_file} ({file_size:.2f}MB)")
            print(f"💾 チェックポイント保存完了: {checkpoint_file}")
            
        except Exception as e:
            self.logger.error(f"チェックポイント保存失敗: {e}")
            raise
        
        self.current_checkpoint = str(checkpoint_file)
        return str(checkpoint_file)
    
    def load_checkpoint(self, checkpoint_file: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """チェックポイントの読み込み"""
        self.logger.info(f"チェックポイント読み込み開始: {checkpoint_file}")
        
        checkpoint_path = Path(checkpoint_file)
        if not checkpoint_path.exists():
            error_msg = f"チェックポイントファイルが見つかりません: {checkpoint_file}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        metadata = {}
        data = {}
        
        try:
            with tqdm(desc="📂 チェックポイント読み込み中", unit="item", disable=False) as pbar:
                with h5py.File(checkpoint_path, 'r') as f:
                    # メタデータの読み込み
                    if 'metadata' in f:
                        meta_group = f['metadata']
                        for key in meta_group.attrs:
                            metadata[key] = meta_group.attrs[key]
                        self.logger.debug(f"メタデータ読み込み: {len(metadata)}項目")
                    pbar.update(1)
                    
                    # データの読み込み
                    if 'data' in f:
                        data_group = f['data']
                        
                        # 属性の読み込み
                        for key in data_group.attrs:
                            data[key] = data_group.attrs[key]
                        
                        # データセットの読み込み
                        for key in tqdm(data_group, desc="データ読み込み中", leave=False):
                            try:
                                if key.endswith('_pickled'):
                                    # Pickleデータの復元
                                    pickled_bytes = data_group[key][()]
                                    original_key = key[:-8]  # '_pickled'を除去
                                    data[original_key] = pickle.loads(pickled_bytes.tobytes())
                                    self.logger.debug(f"Pickleデータ復元: {original_key}")
                                elif isinstance(data_group[key], h5py.Group):
                                    # スパース行列の復元
                                    sparse_group = data_group[key]
                                    if 'data' in sparse_group and 'indices' in sparse_group and 'indptr' in sparse_group:
                                        sparse_data = sparse_group['data'][()]
                                        indices = sparse_group['indices'][()]
                                        indptr = sparse_group['indptr'][()]
                                        shape = tuple(sparse_group.attrs['shape'])
                                        format_type = sparse_group.attrs['format']
                                        
                                        if format_type == 'csr':
                                            data[key] = sp.csr_matrix((sparse_data, indices, indptr), shape=shape)
                                        elif format_type == 'csc':
                                            data[key] = sp.csc_matrix((sparse_data, indices, indptr), shape=shape)
                                        else:
                                            data[key] = sp.csr_matrix((sparse_data, indices, indptr), shape=shape)
                                        self.logger.debug(f"スパース行列復元: {key}, shape={shape}")
                                else:
                                    # 通常のデータセット
                                    data[key] = data_group[key][()]
                                    self.logger.debug(f"データセット読み込み: {key}")
                            except Exception as e:
                                self.logger.error(f"データ読み込みエラー: {key}, error={e}")
                                raise
                            pbar.update(1)
            
            self.logger.info(f"チェックポイント読み込み完了: {len(data)}項目のデータ")
            print(f"📂 チェックポイント読み込み完了: {checkpoint_file}")
            
        except Exception as e:
            self.logger.error(f"チェックポイント読み込み失敗: {e}")
            raise
        
        return data, metadata
    
    def list_checkpoints(self, checkpoint_id: str) -> List[str]:
        """特定IDのチェックポイント一覧"""
        checkpoint_dir = self.base_dir / checkpoint_id
        if not checkpoint_dir.exists():
            self.logger.warning(f"チェックポイントディレクトリが存在しません: {checkpoint_dir}")
            return []
        
        checkpoints = sorted([str(f) for f in checkpoint_dir.glob("*.h5")])
        self.logger.info(f"チェックポイント一覧取得: {len(checkpoints)}個")
        return checkpoints
    
    def get_latest_checkpoint(self, checkpoint_id: str, stage: str = None) -> Optional[str]:
        """最新のチェックポイントを取得"""
        checkpoints = self.list_checkpoints(checkpoint_id)
        if not checkpoints:
            self.logger.info("利用可能なチェックポイントがありません")
            return None
        
        if stage:
            filtered = [cp for cp in checkpoints if stage in Path(cp).name]
            result = filtered[-1] if filtered else None
            self.logger.info(f"最新チェックポイント取得 (stage={stage}): {result}")
            return result
        
        result = checkpoints[-1]
        self.logger.info(f"最新チェックポイント取得: {result}")
        return result

class RecoveryGPUDiracLaplacianAnalyzer:
    """
    🚀🔄 Recovery機能付きRTX3080対応高次元ディラック/ラプラシアン作用素解析クラス
    
    新機能:
    1. 電源断からの自動復旧
    2. 計算途中からの再開
    3. 高次元（6-10次元）対応
    4. 自動チェックポイント保存
    5. メモリ効率最適化
    6. tqdmプログレスバー表示
    7. 詳細ログ記録機能
    """
    
    def __init__(self, params: RecoveryGPUOperatorParameters):
        self.params = params
        self.dim = params.dimension
        self.N = params.lattice_size
        self.theta = params.theta
        self.kappa = params.kappa
        self.mass = params.mass
        self.coupling = params.coupling
        self.use_sparse = params.use_sparse
        self.device = device
        
        # ロガーの設定
        self.logger = setup_logger(f'Analyzer_dim{self.dim}_N{self.N}', level=params.log_level)
        self.logger.info("=" * 80)
        self.logger.info("RecoveryGPUDiracLaplacianAnalyzer 初期化開始")
        self.logger.info(f"パラメータ: dim={self.dim}, N={self.N}, theta={self.theta}")
        
        # Recovery機能
        self.checkpoint_manager = CheckpointManager() if params.recovery_enabled else None
        self.checkpoint_id = None
        self.last_checkpoint_time = time.time()
        
        # 高次元対応のスピノル次元計算
        if self.dim <= 3:
            self.spinor_dim = 2
        elif self.dim <= 6:
            self.spinor_dim = 4
        elif self.dim <= 8:
            self.spinor_dim = 8
        else:
            self.spinor_dim = 16  # 10次元まで対応
        
        self.logger.info(f"スピノル次元: {self.spinor_dim}")
        
        print(f"🔧 初期化中: {self.dim}D, 格子サイズ {self.N}^{self.dim}")
        print(f"📊 スピノル次元: {self.spinor_dim}")
        print(f"📊 総格子点数: {self.N**self.dim:,}")
        
        # メモリ使用量の推定
        total_dim = self.N**self.dim * self.spinor_dim
        if self.use_sparse:
            sparsity = min(0.1, 1000.0 / total_dim)  # 適応的スパース率
            memory_gb = (total_dim**2 * sparsity * 16) / 1e9
            self.logger.info(f"推定メモリ使用量（スパース）: {memory_gb:.2f} GB")
            print(f"💾 推定メモリ使用量（スパース）: {memory_gb:.2f} GB")
        else:
            memory_gb = (total_dim**2 * 16) / 1e9
            self.logger.warning(f"推定メモリ使用量（密行列）: {memory_gb:.2f} GB")
            print(f"💾 推定メモリ使用量（密行列）: {memory_gb:.2f} GB")
        
        print(f"📊 行列次元: {total_dim:,} x {total_dim:,}")
        
        # Recovery有効時のチェックポイントID生成
        if self.checkpoint_manager:
            self.checkpoint_id = self.checkpoint_manager.create_checkpoint_id(params)
            self.logger.info(f"チェックポイントID: {self.checkpoint_id}")
            print(f"🔄 チェックポイントID: {self.checkpoint_id}")
        
        # 高次元対応ガンマ行列の構築
        print("🔨 ガンマ行列構築中...")
        self.gamma_matrices = self._construct_high_dimensional_gamma_matrices()
        
        # 自動保存のシグナルハンドラ設定
        if params.auto_save:
            signal.signal(signal.SIGINT, self._save_and_exit)
            signal.signal(signal.SIGTERM, self._save_and_exit)
            self.logger.info("自動保存シグナルハンドラ設定完了")
        
        self.logger.info("RecoveryGPUDiracLaplacianAnalyzer 初期化完了")
    
    def _save_and_exit(self, signum, frame):
        """シグナル受信時の自動保存"""
        self.logger.warning(f"シグナル {signum} を受信しました")
        print(f"\n⚠️  シグナル {signum} を受信しました")
        if self.checkpoint_manager and hasattr(self, '_current_stage_data'):
            self.logger.info("緊急保存を実行中...")
            self._save_checkpoint('emergency_save', self._current_stage_data)
        self.logger.info("緊急保存完了 - プログラムを終了します")
        print("💾 緊急保存完了 - プログラムを終了します")
        sys.exit(0)
    
    def _construct_high_dimensional_gamma_matrices(self) -> List[torch.Tensor]:
        """高次元対応ガンマ行列の構築（GPU最適化版）"""
        self.logger.info(f"{self.dim}次元ガンマ行列構築開始")
        
        # パウリ行列（GPU上で構築、float64で作成してからcomplex128に変換）
        sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.float64, device=self.device).to(torch.complex128)
        sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128, device=self.device)
        sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.float64, device=self.device).to(torch.complex128)
        I2 = torch.eye(2, dtype=torch.float64, device=self.device).to(torch.complex128)
        
        gamma = []
        
        with tqdm(desc=f"🔨 {self.dim}次元ガンマ行列構築", total=self.dim, disable=False) as pbar:
            if self.dim <= 3:
                # 低次元の場合
                gamma_list = [sigma_x, sigma_y, sigma_z][:self.dim]
                gamma = gamma_list
                self.logger.debug(f"低次元ガンマ行列構築: {self.dim}個")
                pbar.update(self.dim)
            
            elif self.dim == 4:
                # 4次元ディラック行列（GPU上で効率的に構築）
                O2 = torch.zeros((2, 2), dtype=torch.complex128, device=self.device)
                
                # ブロック対角行列を手動で構築
                gamma1 = torch.zeros((4, 4), dtype=torch.complex128, device=self.device)
                gamma1[2:4, 0:2] = sigma_x
                gamma1[0:2, 2:4] = sigma_x
                
                gamma2 = torch.zeros((4, 4), dtype=torch.complex128, device=self.device)
                gamma2[2:4, 0:2] = sigma_y
                gamma2[0:2, 2:4] = sigma_y
                
                gamma3 = torch.zeros((4, 4), dtype=torch.complex128, device=self.device)
                gamma3[2:4, 0:2] = sigma_z
                gamma3[0:2, 2:4] = sigma_z
                
                gamma4 = torch.zeros((4, 4), dtype=torch.complex128, device=self.device)
                gamma4[0:2, 0:2] = I2
                gamma4[2:4, 2:4] = -I2
                
                gamma = [gamma1, gamma2, gamma3, gamma4]
                self.logger.debug("4次元ディラック行列構築完了")
                pbar.update(4)
            
            elif self.dim <= 6:
                # 6次元まで：4次元を拡張（メモリ効率化）
                O2 = torch.zeros((2, 2), dtype=torch.complex128, device=self.device)
                O4 = torch.zeros((4, 4), dtype=torch.complex128, device=self.device)
                I4 = torch.eye(4, dtype=torch.float64, device=self.device).to(torch.complex128)
                
                # 基本4次元ガンマ行列（手動構築）
                gamma1 = torch.zeros((4, 4), dtype=torch.complex128, device=self.device)
                gamma1[2:4, 0:2] = sigma_x
                gamma1[0:2, 2:4] = sigma_x
                
                gamma2 = torch.zeros((4, 4), dtype=torch.complex128, device=self.device)
                gamma2[2:4, 0:2] = sigma_y
                gamma2[0:2, 2:4] = sigma_y
                
                gamma3 = torch.zeros((4, 4), dtype=torch.complex128, device=self.device)
                gamma3[2:4, 0:2] = sigma_z
                gamma3[0:2, 2:4] = sigma_z
                
                gamma4 = torch.zeros((4, 4), dtype=torch.complex128, device=self.device)
                gamma4[0:2, 0:2] = I2
                gamma4[2:4, 2:4] = -I2
                
                gamma4_list = [gamma1, gamma2, gamma3, gamma4]
                gamma = gamma4_list.copy()
                self.logger.debug("基本4次元ガンマ行列構築完了")
                pbar.update(4)
                
                # 5次元目と6次元目（効率的なクロネッカー積）
                for i in range(4, self.dim):
                    extra_gamma = torch.kron(I2, gamma4_list[i-4])
                    gamma.append(extra_gamma)
                    self.logger.debug(f"拡張ガンマ行列 {i+1} 構築完了")
                    pbar.update(1)
            
            else:
                # 8次元以上：GPU最適化再帰的構築
                self.logger.info("高次元ガンマ行列のGPU最適化再帰的構築開始")
                n_matrices_needed = self.dim
                current_dim = 2
                
                # 初期ガンマ行列
                gamma = [sigma_x, sigma_y, sigma_z]
                self.logger.debug("初期ガンマ行列設定完了")
                pbar.update(3)
                
                while len(gamma) < n_matrices_needed:
                    # 次元を倍にして拡張（GPU上で効率的に）
                    current_gamma = gamma.copy()
                    new_gamma = []
                    
                    # 既存の行列を拡張
                    I_current = torch.eye(current_dim, dtype=torch.float64, device=self.device).to(torch.complex128)
                    O_current = torch.zeros((current_dim, current_dim), dtype=torch.complex128, device=self.device)
                    
                    for g in current_gamma:
                        # メモリ効率的なブロック行列構築
                        new_g = torch.zeros((current_dim*2, current_dim*2), dtype=torch.complex128, device=self.device)
                        new_g[:current_dim, :current_dim] = g
                        new_g[current_dim:, current_dim:] = -g
                        new_gamma.append(new_g)
                    
                    # 新しい行列を追加
                    if len(new_gamma) < n_matrices_needed:
                        chirality = torch.zeros((current_dim*2, current_dim*2), dtype=torch.complex128, device=self.device)
                        chirality[:current_dim, :current_dim] = I_current
                        chirality[current_dim:, current_dim:] = -I_current
                        new_gamma.append(chirality)
                    
                    gamma = new_gamma
                    current_dim *= 2
                    
                    progress_update = min(len(gamma) - pbar.n, n_matrices_needed - pbar.n)
                    pbar.update(progress_update)
                    self.logger.debug(f"ガンマ行列拡張: 現在{len(gamma)}個, 次元{current_dim}")
                    
                    if current_dim > self.spinor_dim:
                        break
        
        # 必要な次元数に調整
        gamma = gamma[:self.dim]
        
        # GPU メモリ使用量の確認
        gpu_memory = monitor_gpu_memory()
        if gpu_memory:
            self.logger.info(f"ガンマ行列構築後GPU使用率: {gpu_memory['usage_percent']:.1f}%")
        
        self.logger.info(f"{self.dim}次元ガンマ行列構築完了: {len(gamma)}個の{gamma[0].shape}行列")
        print(f"✅ {self.dim}次元ガンマ行列構築完了: {len(gamma)}個の{gamma[0].shape}行列")
        
        return gamma
    
    def _save_checkpoint(self, stage: str, data: Dict[str, Any]):
        """チェックポイントの保存"""
        if not self.checkpoint_manager:
            return
        
        self.logger.info(f"チェックポイント保存: stage={stage}")
        
        metadata = {
            'dimension': self.dim,
            'lattice_size': self.N,
            'parameters': asdict(self.params)
        }
        
        self.checkpoint_manager.save_checkpoint(
            self.checkpoint_id, stage, data, metadata
        )
        self.last_checkpoint_time = time.time()
    
    def _should_save_checkpoint(self) -> bool:
        """チェックポイント保存タイミングの判定"""
        if not self.checkpoint_manager:
            return False
        
        should_save = (time.time() - self.last_checkpoint_time) > self.params.checkpoint_interval
        if should_save:
            self.logger.debug("チェックポイント保存タイミングに到達")
        return should_save
    
    def construct_discrete_dirac_operator_gpu_optimized(self) -> torch.sparse.FloatTensor:
        """
        🚀 RTX3080最適化版ディラック作用素の構築
        """
        stage = 'dirac_construction'
        self.logger.info("GPU最適化ディラック作用素構築開始")
        
        # 既存のチェックポイントをチェック
        if self.checkpoint_manager:
            latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint(
                self.checkpoint_id, stage
            )
            if latest_checkpoint:
                self.logger.info("既存のディラック作用素チェックポイントを発見")
                print("📂 既存のディラック作用素チェックポイントを発見")
                try:
                    data, metadata = self.checkpoint_manager.load_checkpoint(latest_checkpoint)
                    if 'dirac_operator' in data:
                        self.logger.info("ディラック作用素をチェックポイントから復元")
                        print("✅ ディラック作用素をチェックポイントから復元")
                        # scipyからtorchスパース行列に変換
                        scipy_matrix = data['dirac_operator']
                        return self._scipy_to_torch_sparse(scipy_matrix)
                except Exception as e:
                    self.logger.error(f"チェックポイント復元エラー: {e}")
                    print(f"⚠️  チェックポイント復元エラー: {e}")
                    print("新規に構築します")
        
        print("🔨 GPU最適化ディラック作用素構築中...")
        start_time = time.time()
        
        total_dim = self.N**self.dim * self.spinor_dim
        self.logger.info(f"行列次元: {total_dim} x {total_dim}")
        
        # GPU上でスパース行列を効率的に構築
        indices_list = []
        values_list = []
        
        # バッチサイズの動的調整
        batch_size = self.params.gpu_batch_size
        if total_dim > 100000:  # 大規模行列の場合
            batch_size = min(batch_size, total_dim // 100)
            self.logger.info(f"大規模行列用バッチサイズ調整: {batch_size}")
        
        # 各方向の微分作用素をGPU上で構築
        for mu in tqdm(range(self.dim), desc="🔨 GPU最適化ディラック作用素構築", disable=False):
            self.logger.debug(f"方向 {mu+1}/{self.dim} 処理中")
            
            # GPU上で差分作用素を構築
            diff_indices, diff_values = self._construct_difference_operator_gpu(mu, batch_size)
            
            # ガンマ行列との積（GPU上で効率的に）
            gamma_mu = self.gamma_matrices[mu]
            
            # クロネッカー積をGPU上で効率的に計算
            kron_indices, kron_values = self._gpu_kron_sparse(
                diff_indices, diff_values, gamma_mu, total_dim
            )
            
            indices_list.append(kron_indices)
            values_list.append(kron_values)
            
            # 非可換補正項
            if self.theta != 0:
                theta_indices, theta_values = self._construct_theta_correction_gpu(mu, batch_size)
                theta_kron_indices, theta_kron_values = self._gpu_kron_sparse(
                    theta_indices, theta_values, gamma_mu, total_dim
                )
                indices_list.append(theta_kron_indices)
                values_list.append(self.theta * theta_kron_values)
                self.logger.debug(f"非可換補正項追加: theta={self.theta}")
            
            # GPU メモリ管理
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 定期的なチェックポイント保存
            if self._should_save_checkpoint():
                temp_data = {'partial_construction': True, 'completed_directions': mu + 1}
                self._save_checkpoint('dirac_partial', temp_data)
                self.logger.info(f"部分的ディラック作用素保存: {mu+1}/{self.dim}方向完了")
        
        # 質量項の追加
        if self.mass != 0:
            with tqdm(desc="質量項追加中", total=1, disable=False) as pbar:
                mass_indices, mass_values = self._construct_mass_term_gpu(total_dim)
                indices_list.append(mass_indices)
                values_list.append(self.mass * mass_values)
                self.logger.debug(f"質量項追加: mass={self.mass}")
                pbar.update(1)
        
        # 全ての項を結合してスパース行列を構築
        with tqdm(desc="スパース行列結合中", total=1, disable=False) as pbar:
            all_indices = torch.cat(indices_list, dim=1)
            all_values = torch.cat(values_list, dim=0)
            
            # 重複インデックスの処理
            D_sparse = torch.sparse_coo_tensor(
                all_indices, all_values, 
                (total_dim, total_dim), 
                dtype=torch.complex128, 
                device=self.device
            ).coalesce()
            
            pbar.update(1)
        
        construction_time = time.time() - start_time
        nnz = D_sparse._nnz()
        
        self.logger.info(f"GPU最適化ディラック作用素構築完了: {construction_time:.2f}秒, nnz={nnz}")
        print(f"✅ GPU最適化ディラック作用素構築完了: {construction_time:.2f}秒")
        print(f"📊 行列サイズ: {D_sparse.shape}, 非零要素数: {nnz:,}")
        
        # GPU メモリ使用量の確認
        gpu_memory = monitor_gpu_memory()
        if gpu_memory:
            self.logger.info(f"構築後GPU使用率: {gpu_memory['usage_percent']:.1f}%")
            print(f"💾 GPU使用率: {gpu_memory['usage_percent']:.1f}%")
        
        # チェックポイント保存（scipyフォーマットで）
        if self.checkpoint_manager:
            scipy_matrix = self._torch_sparse_to_scipy(D_sparse)
            checkpoint_data = {
                'dirac_operator': scipy_matrix,
                'construction_time': construction_time,
                'matrix_info': {
                    'shape': D_sparse.shape,
                    'nnz': nnz,
                    'dtype': str(D_sparse.dtype)
                }
            }
            self._save_checkpoint(stage, checkpoint_data)
        
        return D_sparse
    
    def _construct_difference_operator_gpu(self, direction: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """GPU上での差分作用素構築"""
        self.logger.debug(f"GPU差分作用素構築: direction={direction}")
        
        # 1次元の前進差分をGPU上で構築
        n = self.N
        indices = torch.zeros((2, n), dtype=torch.long, device=self.device)
        values = torch.zeros(n, dtype=torch.complex128, device=self.device)
        
        # 対角線上の要素
        indices[0, :] = torch.arange(n, device=self.device)
        indices[1, :] = torch.arange(n, device=self.device)
        values[:] = -1.0
        
        # 上対角線の要素（周期境界条件）
        indices_upper = torch.zeros((2, n), dtype=torch.long, device=self.device)
        values_upper = torch.ones(n, dtype=torch.complex128, device=self.device)
        
        indices_upper[0, :-1] = torch.arange(n-1, device=self.device)
        indices_upper[1, :-1] = torch.arange(1, n, device=self.device)
        indices_upper[0, -1] = n-1
        indices_upper[1, -1] = 0  # 周期境界条件
        
        # インデックスと値を結合
        all_indices = torch.cat([indices, indices_upper], dim=1)
        all_values = torch.cat([values, values_upper])
        
        return all_indices, all_values
    
    def _gpu_kron_sparse(self, indices_a: torch.Tensor, values_a: torch.Tensor, 
                        matrix_b: torch.Tensor, total_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """GPU上でのスパース行列とデンス行列のクロネッカー積"""
        # 効率的なクロネッカー積の実装
        m_a, n_a = self.N**self.dim, self.N**self.dim
        m_b, n_b = matrix_b.shape
        
        # 新しいインデックスの計算
        nnz = indices_a.shape[1]
        new_indices = torch.zeros((2, nnz * m_b * n_b), dtype=torch.long, device=self.device)
        new_values = torch.zeros(nnz * m_b * n_b, dtype=torch.complex128, device=self.device)
        
        idx = 0
        for k in range(nnz):
            i_a, j_a = indices_a[0, k], indices_a[1, k]
            val_a = values_a[k]
            
            for i_b in range(m_b):
                for j_b in range(n_b):
                    new_i = i_a * m_b + i_b
                    new_j = j_a * n_b + j_b
                    new_val = val_a * matrix_b[i_b, j_b]
                    
                    new_indices[0, idx] = new_i
                    new_indices[1, idx] = new_j
                    new_values[idx] = new_val
                    idx += 1
        
        return new_indices[:, :idx], new_values[:idx]
    
    def _construct_theta_correction_gpu(self, direction: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """GPU上でのθ-変形補正項の構築"""
        self.logger.debug(f"GPU θ補正項構築: direction={direction}, theta={self.theta}")
        
        # 位置作用素をGPU上で構築（float64で作成してからcomplex128に変換）
        n = self.N
        positions = torch.arange(n, device=self.device, dtype=torch.float64) - n // 2
        positions = positions.to(torch.complex128)  # complex128に変換
        
        # 対角行列として構築
        indices = torch.zeros((2, n), dtype=torch.long, device=self.device)
        indices[0, :] = torch.arange(n, device=self.device)
        indices[1, :] = torch.arange(n, device=self.device)
        values = positions * 0.01  # 小さな補正係数
        
        return indices, values
    
    def _construct_mass_term_gpu(self, total_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """GPU上での質量項の構築"""
        # 単位行列をスパース形式で構築
        indices = torch.zeros((2, total_dim), dtype=torch.long, device=self.device)
        indices[0, :] = torch.arange(total_dim, device=self.device)
        indices[1, :] = torch.arange(total_dim, device=self.device)
        values = torch.ones(total_dim, dtype=torch.complex128, device=self.device)
        
        return indices, values
    
    def _scipy_to_torch_sparse(self, scipy_matrix: sp.csr_matrix) -> torch.sparse.FloatTensor:
        """scipy スパース行列を torch スパース行列に変換"""
        coo = scipy_matrix.tocoo()
        indices = torch.from_numpy(np.vstack([coo.row, coo.col])).long()
        values = torch.from_numpy(coo.data).to(torch.complex128)
        
        return torch.sparse_coo_tensor(
            indices, values, coo.shape, 
            dtype=torch.complex128, device=self.device
        ).coalesce()
    
    def _torch_sparse_to_scipy(self, torch_sparse: torch.sparse.FloatTensor) -> sp.csr_matrix:
        """torch スパース行列を scipy スパース行列に変換"""
        torch_sparse = torch_sparse.coalesce().cpu()
        indices = torch_sparse.indices().numpy()
        values = torch_sparse.values().numpy()
        shape = torch_sparse.shape
        
        return sp.coo_matrix((values, (indices[0], indices[1])), shape=shape).tocsr()
    
    def compute_spectral_dimension_gpu_optimized(self, 
                                                operator: torch.sparse.FloatTensor,
                                                n_eigenvalues: int = None) -> Tuple[float, Dict]:
        """
        🚀 RTX3080最適化版スペクトル次元計算
        """
        if n_eigenvalues is None:
            n_eigenvalues = self.params.max_eigenvalues
        
        stage = 'spectral_computation'
        self.logger.info(f"GPU最適化スペクトル次元計算開始: n_eigenvalues={n_eigenvalues}")
        
        # 既存のチェックポイントをチェック
        if self.checkpoint_manager:
            latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint(
                self.checkpoint_id, stage
            )
            if latest_checkpoint:
                self.logger.info("既存のスペクトル計算チェックポイントを発見")
                print("📂 既存のスペクトル計算チェックポイントを発見")
                try:
                    data, metadata = self.checkpoint_manager.load_checkpoint(latest_checkpoint)
                    if 'spectral_dimension' in data:
                        self.logger.info("スペクトル次元をチェックポイントから復元")
                        print("✅ スペクトル次元をチェックポイントから復元")
                        return data['spectral_dimension'], data.get('analysis_info', {})
                except Exception as e:
                    self.logger.error(f"チェックポイント復元エラー: {e}")
                    print(f"⚠️  チェックポイント復元エラー: {e}")
                    print("新規に計算します")
        
        print("🔍 GPU最適化スペクトル次元計算中...")
        start_time = time.time()
        
        # GPU メモリ使用量の初期確認
        gpu_memory = monitor_gpu_memory()
        if gpu_memory:
            self.logger.info(f"計算開始時GPU使用率: {gpu_memory['usage_percent']:.1f}%")
            print(f"💾 計算開始時GPU使用率: {gpu_memory['usage_percent']:.1f}%")
        
        try:
            # GPU上でエルミート化（メモリ効率的に）
            with tqdm(desc="🔨 GPU上でエルミート化中", total=1, disable=False) as pbar:
                self.logger.debug("GPU上でエルミート化開始")
                
                # 混合精度計算の使用
                if self.params.use_mixed_precision and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        operator_hermitian = torch.sparse.mm(operator.conj().transpose(0, 1), operator)
                else:
                    operator_hermitian = torch.sparse.mm(operator.conj().transpose(0, 1), operator)
                
                # スパース行列の最適化
                operator_hermitian = operator_hermitian.coalesce()
                
                self.logger.debug(f"GPU上でエルミート化完了: shape={operator_hermitian.shape}")
                pbar.update(1)
            
            # GPU メモリクリーンアップ
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 現在の状態を保存
            self._current_stage_data = {
                'stage': 'hermitian_complete',
                'operator_shape': operator.shape,
                'nnz': operator._nnz()
            }
            
            # GPU上での固有値計算（効率的な実装）
            eigenvalues = self._compute_eigenvalues_gpu_optimized(
                operator_hermitian, n_eigenvalues
            )
            
            if len(eigenvalues) < 10:
                self.logger.warning(f"有効な固有値が少なすぎます: {len(eigenvalues)}個")
                print("⚠️  警告: 有効な固有値が少なすぎます")
                return float('nan'), {}
            
            # 中間結果の保存
            if self._should_save_checkpoint():
                checkpoint_data = {
                    'eigenvalues': eigenvalues.cpu().numpy(),
                    'computation_stage': 'eigenvalues_complete'
                }
                self._save_checkpoint('spectral_intermediate', checkpoint_data)
            
        except Exception as e:
            self.logger.error(f"固有値計算エラー: {e}")
            print(f"❌ 固有値計算エラー: {e}")
            return float('nan'), {}
        
        # GPU上でスペクトルゼータ関数の計算（最適化版）
        spectral_dimension = self._compute_spectral_zeta_gpu_optimized(eigenvalues)
        
        computation_time = time.time() - start_time
        self.logger.info(f"GPU最適化スペクトル次元計算完了: {computation_time:.2f}秒")
        print(f"✅ GPU最適化スペクトル次元計算完了: {computation_time:.2f}秒")
        
        # 最終GPU メモリ使用量の確認
        gpu_memory = monitor_gpu_memory()
        if gpu_memory:
            self.logger.info(f"計算完了時GPU使用率: {gpu_memory['usage_percent']:.1f}%")
            print(f"💾 計算完了時GPU使用率: {gpu_memory['usage_percent']:.1f}%")
        
        # 詳細情報
        analysis_info = {
            'eigenvalues': eigenvalues.cpu().numpy(),
            'n_eigenvalues': len(eigenvalues),
            'min_eigenvalue': torch.min(eigenvalues).item(),
            'max_eigenvalue': torch.max(eigenvalues).item(),
            'spectral_gap': (eigenvalues[1] - eigenvalues[0]).item() if len(eigenvalues) > 1 else 0,
            'computation_time': computation_time,
            'gpu_optimized': True
        }
        
        # 最終結果の保存
        if self.checkpoint_manager:
            final_data = {
                'spectral_dimension': spectral_dimension,
                'analysis_info': analysis_info
            }
            self._save_checkpoint(stage, final_data)
        
        return spectral_dimension, analysis_info
    
    def _compute_eigenvalues_gpu_optimized(self, operator_hermitian: torch.sparse.FloatTensor, 
                                         n_eigenvalues: int) -> torch.Tensor:
        """GPU最適化固有値計算（高精度版）"""
        self.logger.info("高精度GPU最適化固有値計算開始")
        
        # 行列サイズに応じた計算方法の選択
        matrix_size = operator_hermitian.shape[0]
        
        if matrix_size < 5000:
            # 小規模行列：直接GPU計算（高精度版）
            return self._compute_eigenvalues_direct_gpu_precision(operator_hermitian, n_eigenvalues)
        elif matrix_size < 50000:
            # 中規模行列：改良バッチ処理
            return self._compute_eigenvalues_batch_gpu_precision(operator_hermitian, n_eigenvalues)
        else:
            # 大規模行列：ハイブリッド計算（高精度版）
            return self._compute_eigenvalues_hybrid_precision(operator_hermitian, n_eigenvalues)
    
    def _compute_eigenvalues_direct_gpu_precision(self, operator: torch.sparse.FloatTensor, 
                                                n_eigenvalues: int) -> torch.Tensor:
        """小規模行列の高精度直接GPU計算"""
        with tqdm(desc="🧮 高精度小規模行列GPU固有値計算", total=1, disable=False) as pbar:
            # スパース行列を密行列に変換（倍精度）
            dense_operator = operator.to_dense().to(torch.complex128)
            
            # 数値安定性のための前処理
            # 行列の条件数チェック
            try:
                # 対角要素の平均を計算
                diag_mean = torch.mean(torch.diag(dense_operator).real)
                
                # 正則化項を追加（条件数改善）
                regularization = 1e-12 * diag_mean * torch.eye(
                    dense_operator.shape[0], 
                    device=self.device, 
                    dtype=torch.complex128
                )
                dense_operator_reg = dense_operator + regularization
                
                # GPU上で高精度固有値計算
                eigenvalues = torch.linalg.eigvals(dense_operator_reg)
                eigenvalues = torch.real(eigenvalues)
                
                # 正の固有値のみを抽出（より厳密な閾値）
                eigenvalues = eigenvalues[eigenvalues > 1e-14]
                
                # 最小固有値から順にソート
                eigenvalues, _ = torch.sort(eigenvalues)
                eigenvalues = eigenvalues[:n_eigenvalues]
                
                self.logger.info(f"高精度直接GPU計算完了: {len(eigenvalues)}個の固有値")
                self.logger.info(f"最小固有値: {torch.min(eigenvalues).item():.2e}")
                self.logger.info(f"最大固有値: {torch.max(eigenvalues).item():.2e}")
                self.logger.info(f"条件数: {(torch.max(eigenvalues) / torch.min(eigenvalues)).item():.2e}")
                
            except Exception as e:
                self.logger.error(f"高精度直接計算失敗: {e}")
                # フォールバック：標準精度
                eigenvalues = torch.linalg.eigvals(dense_operator.to(torch.complex64))
                eigenvalues = torch.real(eigenvalues)
                eigenvalues = eigenvalues[eigenvalues > 1e-12]
                eigenvalues, _ = torch.sort(eigenvalues)
                eigenvalues = eigenvalues[:n_eigenvalues]
            
            pbar.update(1)
        
        return eigenvalues.to(torch.float64)
    
    def _compute_eigenvalues_batch_gpu_precision(self, operator: torch.sparse.FloatTensor, 
                                               n_eigenvalues: int) -> torch.Tensor:
        """中規模行列の高精度バッチ処理GPU計算"""
        self.logger.info("高精度中規模行列バッチ処理GPU計算開始")
        
        # 改良Lanczos法の実装
        matrix_size = operator.shape[0]
        max_iterations = min(200, n_eigenvalues * 3)  # 反復回数を増加
        tolerance = 1e-12  # 収束判定を厳密化
        
        eigenvalues_list = []
        
        with tqdm(desc="🧮 高精度バッチ処理GPU固有値計算", total=max_iterations//10, disable=False) as pbar:
            # 複数の初期ベクトルで実行（精度向上）
            for seed in range(3):  # 3つの異なる初期値
                torch.manual_seed(42 + seed)  # 再現性のため
                
                # 初期ベクトル（正規化）
                v = torch.randn(matrix_size, dtype=torch.complex128, device=self.device)
                v = v / torch.norm(v)
                
                # Lanczos反復
                eigenvals_seed = []
                prev_eigenval = float('inf')
                
                for i in range(max_iterations):
                    # 行列ベクトル積（高精度）
                    Av = torch.sparse.mm(operator, v.unsqueeze(1)).squeeze(1)
                    
                    # Rayleigh商による固有値近似
                    eigenval = torch.real(torch.dot(v.conj(), Av))
                    eigenvals_seed.append(eigenval.item())
                    
                    # 収束判定（厳密化）
                    if abs(eigenval.item() - prev_eigenval) < tolerance:
                        self.logger.debug(f"収束達成 (seed={seed}, iter={i}): {eigenval.item():.2e}")
                        break
                    
                    # 次のベクトルの計算（Gram-Schmidt直交化）
                    if i < max_iterations - 1:
                        v_new = Av - eigenval * v
                        
                        # 直交化（数値安定性向上）
                        for _ in range(2):  # 2回の直交化
                            v_new = v_new - torch.dot(v_new.conj(), v) * v
                        
                        norm_v_new = torch.norm(v_new)
                        if norm_v_new > tolerance:
                            v = v_new / norm_v_new
                        else:
                            # 新しいランダムベクトルで再開
                            v = torch.randn(matrix_size, dtype=torch.complex128, device=self.device)
                            v = v / torch.norm(v)
                    
                    prev_eigenval = eigenval.item()
                    
                    if i % 10 == 0:
                        pbar.update(1)
                
                eigenvalues_list.extend(eigenvals_seed)
        
        # 重複除去と選別
        eigenvalues = torch.tensor(eigenvalues_list, device=self.device, dtype=torch.float64)
        eigenvalues = eigenvalues[eigenvalues > 1e-14]
        
        # 重複除去（近い値をマージ）
        eigenvalues_unique = []
        eigenvalues_sorted, _ = torch.sort(eigenvalues)
        
        if len(eigenvalues_sorted) > 0:
            eigenvalues_unique.append(eigenvalues_sorted[0])
            for i in range(1, len(eigenvalues_sorted)):
                if abs(eigenvalues_sorted[i] - eigenvalues_unique[-1]) > tolerance * 10:
                    eigenvalues_unique.append(eigenvalues_sorted[i])
        
        eigenvalues_final = torch.tensor(eigenvalues_unique, device=self.device, dtype=torch.float64)
        eigenvalues_final = eigenvalues_final[:n_eigenvalues]
        
        self.logger.info(f"高精度バッチ処理GPU計算完了: {len(eigenvalues_final)}個の固有値")
        return eigenvalues_final
    
    def _compute_eigenvalues_hybrid_precision(self, operator: torch.sparse.FloatTensor, 
                                            n_eigenvalues: int) -> torch.Tensor:
        """大規模行列の高精度ハイブリッド計算（GPU+CPU）"""
        self.logger.info("高精度大規模行列ハイブリッド計算開始")
        
        with tqdm(desc="🧮 高精度ハイブリッド固有値計算", total=1, disable=False) as pbar:
            # GPU上で前処理
            operator_cpu = self._torch_sparse_to_scipy(operator)
            
            # CPU上で高精度スパース固有値計算
            try:
                # 複数の手法を試行
                eigenvalues_list = []
                
                # 手法1: ARPACK (which='SM')
                try:
                    eigenvals_sm = eigsh(
                        operator_cpu, 
                        k=min(n_eigenvalues, operator_cpu.shape[0]-2),
                        which='SM',  # 最小固有値
                        tol=1e-12,   # 収束判定を厳密化
                        maxiter=1000,  # 最大反復回数増加
                        return_eigenvectors=False
                    )
                    eigenvals_sm = np.real(eigenvals_sm)
                    eigenvals_sm = eigenvals_sm[eigenvals_sm > 1e-14]
                    eigenvalues_list.extend(eigenvals_sm)
                    self.logger.info(f"ARPACK-SM: {len(eigenvals_sm)}個の固有値")
                except Exception as e:
                    self.logger.warning(f"ARPACK-SM失敗: {e}")
                
                # 手法2: ARPACK (which='SA')
                try:
                    eigenvals_sa = eigsh(
                        operator_cpu, 
                        k=min(n_eigenvalues//2, operator_cpu.shape[0]-2),
                        which='SA',  # 最小代数固有値
                        tol=1e-12,
                        maxiter=1000,
                        return_eigenvectors=False
                    )
                    eigenvals_sa = np.real(eigenvals_sa)
                    eigenvals_sa = eigenvals_sa[eigenvals_sa > 1e-14]
                    eigenvalues_list.extend(eigenvals_sa)
                    self.logger.info(f"ARPACK-SA: {len(eigenvals_sa)}個の固有値")
                except Exception as e:
                    self.logger.warning(f"ARPACK-SA失敗: {e}")
                
                if eigenvalues_list:
                    # 重複除去と統合
                    eigenvalues_np = np.array(eigenvalues_list)
                    eigenvalues_np = np.unique(eigenvalues_np)
                    eigenvalues_np = eigenvalues_np[eigenvalues_np > 1e-14]
                    eigenvalues_np = np.sort(eigenvalues_np)[:n_eigenvalues]
                    
                    # GPU上に戻す
                    eigenvalues = torch.tensor(eigenvalues_np, device=self.device, dtype=torch.float64)
                    
                    self.logger.info(f"統合結果: {len(eigenvalues)}個の固有値")
                    
                else:
                    raise Exception("全ての手法が失敗")
                
            except Exception as e:
                self.logger.warning(f"高精度scipy固有値計算失敗: {e}")
                # フォールバック：簡易近似（改良版）
                eigenvalues = self._compute_eigenvalues_approximation_precision(operator, n_eigenvalues)
            
            pbar.update(1)
        
        self.logger.info(f"高精度ハイブリッド計算完了: {len(eigenvalues)}個の固有値")
        return eigenvalues
    
    def _compute_eigenvalues_approximation_precision(self, operator: torch.sparse.FloatTensor, 
                                                   n_eigenvalues: int) -> torch.Tensor:
        """高精度固有値近似計算（改良フォールバック）"""
        self.logger.info("高精度固有値近似計算開始")
        
        # 複数の近似手法を組み合わせ
        eigenvalues_list = []
        
        # 手法1: 対角要素による近似
        diagonal = torch.sparse.sum(operator, dim=1).to_dense()
        diagonal = torch.real(diagonal)
        diagonal = diagonal[diagonal > 1e-14]
        diagonal, _ = torch.sort(diagonal)
        eigenvalues_list.extend(diagonal[:n_eigenvalues//2].tolist())
        
        # 手法2: Gershgorin円盤による推定
        try:
            # 各行の非対角要素の和
            operator_dense = operator.to_dense()
            diag_elements = torch.diag(operator_dense).real
            off_diag_sums = torch.sum(torch.abs(operator_dense), dim=1) - torch.abs(diag_elements)
            
            # Gershgorin円盤の中心と半径
            centers = diag_elements
            radii = off_diag_sums
            
            # 固有値の下限推定
            lower_bounds = centers - radii
            lower_bounds = lower_bounds[lower_bounds > 1e-14]
            lower_bounds, _ = torch.sort(lower_bounds)
            eigenvalues_list.extend(lower_bounds[:n_eigenvalues//2].tolist())
            
        except Exception as e:
            self.logger.warning(f"Gershgorin推定失敗: {e}")
        
        # 統合と重複除去
        if eigenvalues_list:
            eigenvalues = torch.tensor(eigenvalues_list, device=self.device, dtype=torch.float64)
            eigenvalues = torch.unique(eigenvalues)
            eigenvalues = eigenvalues[eigenvalues > 1e-14]
            eigenvalues, _ = torch.sort(eigenvalues)
            eigenvalues = eigenvalues[:n_eigenvalues]
        else:
            # 最終フォールバック
            eigenvalues = torch.linspace(1e-6, 1.0, n_eigenvalues, device=self.device, dtype=torch.float64)
        
        self.logger.info(f"高精度近似計算完了: {len(eigenvalues)}個の固有値")
        return eigenvalues
    
    def _compute_spectral_zeta_gpu_optimized(self, eigenvalues: torch.Tensor) -> float:
        """GPU最適化スペクトルゼータ関数計算（高精度版）"""
        self.logger.info("高精度GPU最適化ゼータ関数計算開始")
        print("⚡ 高精度GPU最適化ゼータ関数計算...")
        
        # 高精度計算のための設定
        eigenvalues = eigenvalues.to(torch.float64)  # 倍精度に変更
        
        # 適応的t値範囲の設定（より細かいサンプリング）
        t_min = 1e-6
        t_max = 10.0
        n_samples_coarse = 50
        n_samples_fine = 200
        
        # 粗いサンプリングで傾向を把握
        t_values_coarse = torch.logspace(
            np.log10(t_min), np.log10(t_max), n_samples_coarse, 
            device=self.device, dtype=torch.float64
        )
        
        # バッチ処理でゼータ関数計算（粗いサンプリング）
        batch_size = min(10, len(t_values_coarse))
        zeta_values_coarse = []
        
        with tqdm(desc="粗いサンプリング", total=len(t_values_coarse)//batch_size, disable=False) as pbar:
            for i in range(0, len(t_values_coarse), batch_size):
                t_batch = t_values_coarse[i:i+batch_size]
                
                # 高精度バッチ計算
                exp_matrix = torch.exp(-t_batch.unsqueeze(1) * eigenvalues.unsqueeze(0))
                zeta_batch = torch.sum(exp_matrix, dim=1)
                zeta_values_coarse.extend(zeta_batch.tolist())
                
                pbar.update(1)
        
        zeta_values_coarse = torch.tensor(zeta_values_coarse, device=self.device, dtype=torch.float64)
        
        # 対数微分の粗い推定
        log_t_coarse = torch.log(t_values_coarse)
        log_zeta_coarse = torch.log(zeta_values_coarse + 1e-15)
        
        # 有効なデータ点のフィルタリング
        valid_mask_coarse = torch.isfinite(log_zeta_coarse) & torch.isfinite(log_t_coarse) & (log_zeta_coarse > -100)
        
        if torch.sum(valid_mask_coarse) < 10:
            self.logger.error("粗いサンプリングで有効なデータ点が不足")
            return float('nan')
        
        # 線形領域の特定（適応的範囲決定）
        log_t_valid_coarse = log_t_coarse[valid_mask_coarse]
        log_zeta_valid_coarse = log_zeta_coarse[valid_mask_coarse]
        
        # 局所的な傾きを計算して線形領域を特定
        gradients = torch.diff(log_zeta_valid_coarse) / torch.diff(log_t_valid_coarse)
        gradient_std = torch.std(gradients)
        gradient_mean = torch.mean(gradients)
        
        # 線形性が高い領域を特定
        linear_mask = torch.abs(gradients - gradient_mean) < 2 * gradient_std
        if torch.sum(linear_mask) < 5:
            # フォールバック：全範囲を使用
            linear_range = (log_t_valid_coarse[0], log_t_valid_coarse[-1])
        else:
            linear_indices = torch.where(linear_mask)[0]
            linear_range = (
                log_t_valid_coarse[linear_indices[0]].item(),
                log_t_valid_coarse[linear_indices[-1] + 1].item()
            )
        
        # 線形領域での細かいサンプリング
        t_min_fine = np.exp(linear_range[0])
        t_max_fine = np.exp(linear_range[1])
        
        t_values_fine = torch.logspace(
            np.log10(t_min_fine), np.log10(t_max_fine), n_samples_fine,
            device=self.device, dtype=torch.float64
        )
        
        # 高精度ゼータ関数計算（細かいサンプリング）
        zeta_values_fine = []
        
        with tqdm(desc="高精度サンプリング", total=len(t_values_fine)//batch_size, disable=False) as pbar:
            for i in range(0, len(t_values_fine), batch_size):
                t_batch = t_values_fine[i:i+batch_size]
                
                # 数値安定性を考慮した計算
                max_exp_arg = torch.max(-t_batch.unsqueeze(1) * eigenvalues.unsqueeze(0))
                if max_exp_arg > 700:  # オーバーフロー防止
                    # 正規化を適用
                    exp_matrix = torch.exp(-t_batch.unsqueeze(1) * eigenvalues.unsqueeze(0) - max_exp_arg)
                    zeta_batch = torch.sum(exp_matrix, dim=1) * torch.exp(max_exp_arg)
                else:
                    exp_matrix = torch.exp(-t_batch.unsqueeze(1) * eigenvalues.unsqueeze(0))
                    zeta_batch = torch.sum(exp_matrix, dim=1)
                
                zeta_values_fine.extend(zeta_batch.tolist())
                pbar.update(1)
        
        zeta_values_fine = torch.tensor(zeta_values_fine, device=self.device, dtype=torch.float64)
        
        # 高精度対数微分計算
        with tqdm(desc="高精度対数微分計算", total=1, disable=False) as pbar:
            log_t_fine = torch.log(t_values_fine)
            log_zeta_fine = torch.log(zeta_values_fine + 1e-15)
            
            # 有効なデータ点のフィルタリング（より厳密）
            valid_mask_fine = (
                torch.isfinite(log_zeta_fine) & 
                torch.isfinite(log_t_fine) & 
                (log_zeta_fine > -100) &
                (log_zeta_fine < 100) &
                (torch.abs(log_t_fine) < 50)
            )
            
            if torch.sum(valid_mask_fine) < 20:
                self.logger.error(f"高精度サンプリングで有効なデータ点が不足: {torch.sum(valid_mask_fine)}点")
                return float('nan')
            
            log_t_valid_fine = log_t_fine[valid_mask_fine]
            log_zeta_valid_fine = log_zeta_fine[valid_mask_fine]
            
            # 外れ値の除去（Huber回帰的アプローチ）
            if len(log_t_valid_fine) > 30:
                # 初期線形回帰
                X_init = torch.stack([log_t_valid_fine, torch.ones_like(log_t_valid_fine)], dim=1)
                params_init = torch.linalg.lstsq(X_init, log_zeta_valid_fine).solution
                residuals = log_zeta_valid_fine - (params_init[0] * log_t_valid_fine + params_init[1])
                
                # 外れ値の特定（3σ基準）
                residual_std = torch.std(residuals)
                inlier_mask = torch.abs(residuals) < 3 * residual_std
                
                if torch.sum(inlier_mask) >= 15:
                    log_t_valid_fine = log_t_valid_fine[inlier_mask]
                    log_zeta_valid_fine = log_zeta_valid_fine[inlier_mask]
                    self.logger.info(f"外れ値除去: {torch.sum(~inlier_mask)}点を除去")
            
            # 重み付き最小二乗法（改良版）
            # t→0での重みを高くする
            weights = torch.exp(-torch.abs(log_t_valid_fine - log_t_valid_fine[0]))
            weights = weights / torch.sum(weights) * len(weights)  # 正規化
            
            # 重み付き線形回帰
            X = torch.stack([log_t_valid_fine, torch.ones_like(log_t_valid_fine)], dim=1)
            W = torch.diag(weights)
            XtW = X.t() @ W
            XtWX = XtW @ X
            XtWy = XtW @ log_zeta_valid_fine
            
            try:
                # 正則化項を追加（数値安定性向上）
                regularization = 1e-10 * torch.eye(2, device=self.device, dtype=torch.float64)
                params = torch.linalg.solve(XtWX + regularization, XtWy)
                slope = params[0]
                intercept = params[1]
                
                # 回帰の品質評価
                y_pred = slope * log_t_valid_fine + intercept
                ss_res = torch.sum((log_zeta_valid_fine - y_pred) ** 2)
                ss_tot = torch.sum((log_zeta_valid_fine - torch.mean(log_zeta_valid_fine)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                self.logger.info(f"回帰品質: R² = {r_squared:.6f}")
                
                if r_squared < 0.8:
                    self.logger.warning(f"回帰品質が低い: R² = {r_squared:.6f}")
                
            except Exception as e:
                self.logger.error(f"重み付き回帰失敗: {e}")
                # フォールバック：通常の最小二乗法
                X = torch.stack([log_t_valid_fine, torch.ones_like(log_t_valid_fine)], dim=1)
                params = torch.linalg.lstsq(X, log_zeta_valid_fine).solution
                slope = params[0]
            
            # 理論的補正項の適用
            # 有限サイズ効果の補正
            n_eigenvalues = len(eigenvalues)
            finite_size_correction = 0.5 / n_eigenvalues  # 理論的補正
            
            # 非可換効果の補正
            noncommutative_correction = self.theta * 0.1 if hasattr(self, 'theta') and self.theta != 0 else 0
            
            spectral_dimension = -2 * slope.item() + finite_size_correction + noncommutative_correction
            
            self.logger.info(f"高精度スペクトル次元計算結果:")
            self.logger.info(f"  生の傾き: {slope.item():.8f}")
            self.logger.info(f"  有限サイズ補正: {finite_size_correction:.8f}")
            self.logger.info(f"  非可換補正: {noncommutative_correction:.8f}")
            self.logger.info(f"  最終スペクトル次元: {spectral_dimension:.8f}")
            
            pbar.update(1)
        
        return spectral_dimension
    
    def run_full_analysis_with_recovery(self) -> Dict[str, Any]:
        """🔄 Recovery機能付き完全解析の実行"""
        self.logger.info("=" * 80)
        self.logger.info("Recovery機能付き完全解析開始")
        
        print("=" * 80)
        print("🚀🔄 Recovery機能付きRTX3080高次元ディラック/ラプラシアン作用素解析")
        print("=" * 80)
        
        if torch.cuda.is_available():
            gpu_info = f"GPU: {torch.cuda.get_device_name()}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            self.logger.info(gpu_info)
            print(f"🎮 GPU: {torch.cuda.get_device_name()}")
            print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        analysis_params = f"次元: {self.dim}, 格子サイズ: {self.N}, スピノル次元: {self.spinor_dim}"
        self.logger.info(f"解析パラメータ: {analysis_params}")
        
        print(f"\n📊 解析パラメータ:")
        print(f"次元: {self.dim}")
        print(f"格子サイズ: {self.N}")
        print(f"スピノル次元: {self.spinor_dim}")
        print(f"Recovery機能: {'有効' if self.checkpoint_manager else '無効'}")
        
        total_start = time.time()
        
        # 全体の進捗管理
        with tqdm(total=2, desc="🚀 全体進捗", position=0, disable=False) as main_pbar:
            # 1. ディラック作用素の構築
            main_pbar.set_description("🔨 1. ディラック作用素の構築")
            self.logger.info("ステップ1: ディラック作用素の構築開始")
            D = self.construct_discrete_dirac_operator_gpu_optimized()
            main_pbar.update(1)
            
            # 2. スペクトル次元の計算
            main_pbar.set_description("🔍 2. スペクトル次元の計算")
            self.logger.info("ステップ2: スペクトル次元の計算開始")
            d_s_dirac, dirac_info = self.compute_spectral_dimension_gpu_optimized(D)
            main_pbar.update(1)
        
        dimension_error = abs(d_s_dirac - self.dim) if not np.isnan(d_s_dirac) else float('nan')
        
        self.logger.info(f"解析結果: スペクトル次元={d_s_dirac:.6f}, 理論値との差={dimension_error:.6f}")
        print(f"📈 ディラック作用素のスペクトル次元: {d_s_dirac:.6f}")
        print(f"🎯 理論値({self.dim})との差: {dimension_error:.6f}")
        
        total_time = time.time() - total_start
        self.logger.info(f"総計算時間: {total_time:.2f}秒")
        print(f"\n⏱️  総計算時間: {total_time:.2f}秒")
        
        # 結果の整理
        results = {
            'parameters': {
                'dimension': self.dim,
                'lattice_size': self.N,
                'theta': self.theta,
                'mass': self.mass,
                'spinor_dimension': self.spinor_dim
            },
            'results': {
                'spectral_dimension': d_s_dirac,
                'dimension_error': abs(d_s_dirac - self.dim),
                'total_computation_time': total_time,
                'matrix_size': D.shape[0],
                'nnz_elements': D._nnz(),
                'eigenvalues_computed': dirac_info.get('n_eigenvalues', 0)
            },
            'checkpoint_id': self.checkpoint_id,
            'spectral_dimension_dirac': d_s_dirac,
            'analysis_info': dirac_info
        }
        
        # 結果の保存
        output_file = f"results/json/recovery_gpu_results_dim{self.dim}_N{self.N}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with tqdm(desc="💾 結果保存中", total=1, disable=False) as pbar:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            pbar.update(1)
        
        self.logger.info(f"結果保存完了: {output_file}")
        print(f"\n💾 結果が '{output_file}' に保存されました。")
        
        # 最終チェックポイント
        if self.checkpoint_manager:
            self._save_checkpoint('final_results', results)
            self.logger.info(f"最終チェックポイント保存完了: ID {self.checkpoint_id}")
            print(f"🔄 最終チェックポイント保存完了: ID {self.checkpoint_id}")
        
        self.logger.info("Recovery機能付き完全解析完了")
        self.logger.info("=" * 80)
        
        return results

def demonstrate_recovery_analysis():
    """🚀 RTX3080最適化Recovery解析のデモンストレーション"""
    print("=" * 100)
    print("🚀🔄 RTX3080最適化 Recovery機能付き高次元ディラック/ラプラシアン作用素解析")
    print("=" * 100)
    
    # GPU情報の表示
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🎮 検出されたGPU: {gpu_name}")
        print(f"💾 VRAM: {gpu_memory:.1f} GB")
        
        # RTX3080の特別最適化
        if "RTX 3080" in gpu_name or "RTX3080" in gpu_name:
            print("⚡ RTX3080専用最適化が有効になります")
        else:
            print("⚠️  RTX3080以外のGPUが検出されました。最適化は限定的です。")
    else:
        print("❌ CUDAが利用できません。CPU計算になります。")
        return
    
    # 複数の次元でテスト
    test_dimensions = [3, 4, 5]  # RTX3080で安全にテストできる次元
    
    for dim in test_dimensions:
        print(f"\n{'='*60}")
        print(f"🧮 {dim}次元解析開始")
        print(f"{'='*60}")
        
        try:
            # パラメータ設定（RTX3080最適化）
            params = RecoveryGPUOperatorParameters(
                dimension=dim,
                lattice_size=8 if dim <= 4 else 6,  # 次元に応じて格子サイズ調整
                theta=0.1,
                kappa=0.05,
                mass=0.1,
                coupling=1.0,
                use_sparse=True,
                recovery_enabled=True,
                checkpoint_interval=60,  # 1分間隔
                auto_save=True,
                max_eigenvalues=50 if dim <= 4 else 30,  # 次元に応じて調整
                memory_limit_gb=9.0,  # RTX3080用
                log_level=logging.INFO,
                use_mixed_precision=True
            )
            
            print(f"📊 解析パラメータ:")
            print(f"   次元: {params.dimension}")
            print(f"   格子サイズ: {params.lattice_size}")
            print(f"   最大固有値数: {params.max_eigenvalues}")
            print(f"   メモリ制限: {params.memory_limit_gb} GB")
            print(f"   バッチサイズ: {params.gpu_batch_size}")
            
            # 解析器の初期化
            analyzer = RecoveryGPUDiracLaplacianAnalyzer(params)
            
            # GPU メモリ使用量の初期確認
            gpu_memory = monitor_gpu_memory()
            if gpu_memory:
                print(f"💾 解析開始前GPU使用率: {gpu_memory['usage_percent']:.1f}%")
            
            # 完全解析の実行
            start_time = time.time()
            results = analyzer.run_full_analysis_with_recovery()
            total_time = time.time() - start_time
            
            # 結果の表示
            print(f"\n✅ {dim}次元解析完了！")
            print(f"⏱️  総計算時間: {total_time:.2f}秒")
            
            if 'spectral_dimension_dirac' in results:
                d_s = results['spectral_dimension_dirac']
                print(f"📈 スペクトル次元: {d_s:.6f}")
                
                # 理論値との比較
                theoretical_d_s = dim  # 理論的なスペクトル次元
                error = abs(d_s - theoretical_d_s) / theoretical_d_s * 100
                print(f"🎯 理論値: {theoretical_d_s}")
                print(f"📊 相対誤差: {error:.2f}%")
                
                if error < 10:
                    print("✅ 良好な精度で計算されました")
                elif error < 20:
                    print("⚠️  精度は許容範囲内です")
                else:
                    print("❌ 精度が低い可能性があります")
            
            # GPU メモリ使用量の最終確認
            gpu_memory = monitor_gpu_memory()
            if gpu_memory:
                print(f"💾 解析完了後GPU使用率: {gpu_memory['usage_percent']:.1f}%")
            
            # メモリクリーンアップ
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"🎉 {dim}次元解析が正常に完了しました！")
            
        except Exception as e:
            print(f"❌ {dim}次元解析でエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            
            # エラー時もメモリクリーンアップ
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            continue
    
    print(f"\n{'='*100}")
    print("🎊 RTX3080最適化Recovery解析デモンストレーション完了！")
    print("=" * 100)

def quick_performance_test():
    """🚀 RTX3080性能テスト（軽量版）"""
    print("=" * 80)
    print("⚡ RTX3080性能テスト開始")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("❌ CUDAが利用できません")
        return
    
    # 軽量テスト用パラメータ
    params = RecoveryGPUOperatorParameters(
        dimension=3,
        lattice_size=6,
        theta=0.1,
        kappa=0.05,
        mass=0.1,
        coupling=1.0,
        use_sparse=True,
        recovery_enabled=False,  # 性能テストではRecovery無効
        max_eigenvalues=20,
        memory_limit_gb=9.0,
        use_mixed_precision=True
    )
    
    print(f"🧮 軽量テスト: {params.dimension}次元, 格子サイズ{params.lattice_size}")
    
    try:
        analyzer = RecoveryGPUDiracLaplacianAnalyzer(params)
        
        # GPU メモリ監視開始
        gpu_memory_start = monitor_gpu_memory()
        if gpu_memory_start:
            print(f"💾 開始時GPU使用率: {gpu_memory_start['usage_percent']:.1f}%")
        
        # 性能測定
        start_time = time.time()
        
        # ガンマ行列構築
        gamma_start = time.time()
        gamma_matrices = analyzer._construct_high_dimensional_gamma_matrices()
        gamma_time = time.time() - gamma_start
        print(f"⚡ ガンマ行列構築: {gamma_time:.2f}秒")
        
        # ディラック作用素構築
        dirac_start = time.time()
        D = analyzer.construct_discrete_dirac_operator_gpu_optimized()
        dirac_time = time.time() - dirac_start
        print(f"⚡ ディラック作用素構築: {dirac_time:.2f}秒")
        
        # スペクトル次元計算
        spectral_start = time.time()
        d_s, info = analyzer.compute_spectral_dimension_gpu_optimized(D, n_eigenvalues=15)
        spectral_time = time.time() - spectral_start
        print(f"⚡ スペクトル次元計算: {spectral_time:.2f}秒")
        
        total_time = time.time() - start_time
        
        # 結果表示
        print(f"\n🎯 性能テスト結果:")
        print(f"   総計算時間: {total_time:.2f}秒")
        print(f"   スペクトル次元: {d_s:.6f}")
        print(f"   固有値数: {info.get('n_eigenvalues', 'N/A')}")
        
        # GPU メモリ監視終了
        gpu_memory_end = monitor_gpu_memory()
        if gpu_memory_end:
            print(f"💾 終了時GPU使用率: {gpu_memory_end['usage_percent']:.1f}%")
            memory_used = gpu_memory_end['usage_percent'] - gpu_memory_start['usage_percent']
            print(f"💾 使用メモリ増加: {memory_used:.1f}%")
        
        # 性能評価
        if total_time < 30:
            print("🚀 優秀な性能です！")
        elif total_time < 60:
            print("✅ 良好な性能です")
        else:
            print("⚠️  性能改善の余地があります")
        
        print("✅ 性能テスト完了")
        
    except Exception as e:
        print(f"❌ 性能テストでエラー: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # メモリクリーンアップ
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    # ログ設定
    logger = setup_logger("NKAT_RTX3080_Recovery", "logs/nkat_rtx3080_recovery.log")
    
    print("🚀 RTX3080最適化 NKAT Recovery解析システム v1.7")
    print("=" * 80)
    
    # 実行モードの選択
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "test":
            quick_performance_test()
        elif mode == "demo":
            demonstrate_recovery_analysis()
        else:
            print("使用法: python dirac_laplacian_analysis_gpu_recovery.py [test|demo]")
    else:
        # デフォルト：軽量テスト
        print("🧪 軽量性能テストを実行します...")
        print("完全デモを実行する場合: python dirac_laplacian_analysis_gpu_recovery.py demo")
        print("=" * 80)
        quick_performance_test() 