#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀🔄 RTX3080対応 電源断リカバリー機能付き高次元ディラック/ラプラシアン作用素GPU解析
Non-Commutative Kolmogorov-Arnold Theory (NKAT) における作用素理論 - Recovery対応版

Author: NKAT Research Team
Date: 2025-01-24
Version: 1.5 - Recovery機能付き高次元対応版（RTX3080最適化）

主要機能:
- 電源断からのチェックポイント復元
- 計算途中からの再開機能
- より高次元（6-10次元）での解析対応
- 自動保存機能
- GPU/RTX3080最適化
"""

import torch
import torch.nn as nn
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

# GPU可用性チェック
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 使用デバイス: {device}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name()}")
    print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# 日本語フォント設定
try:
    plt.rcParams['font.family'] = 'MS Gothic'
except:
    plt.rcParams['font.family'] = 'DejaVu Sans'

warnings.filterwarnings('ignore')

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
    
    def __post_init__(self):
        if self.dimension < 2 or self.dimension > 10:
            raise ValueError("次元は2-10の範囲である必要があります")
        if self.lattice_size < 4:
            warnings.warn("格子サイズが小さすぎる可能性があります")
        
        # メモリ使用量チェック
        spinor_dim = 2 if self.dimension <= 3 else 4 if self.dimension <= 6 else 8
        total_dim = self.lattice_size**self.dimension * spinor_dim
        estimated_memory = (total_dim**2 * 16) / 1e9  # 複素数double precision
        
        if not self.use_sparse and estimated_memory > self.memory_limit_gb:
            print(f"⚠️  メモリ不足の可能性: 推定{estimated_memory:.1f}GB > 制限{self.memory_limit_gb}GB")
            print("スパース行列の使用を強く推奨します")

class CheckpointManager:
    """チェックポイント管理クラス"""
    
    def __init__(self, base_dir: str = "checkpoints"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.current_checkpoint = None
        
    def create_checkpoint_id(self, params: RecoveryGPUOperatorParameters) -> str:
        """パラメータに基づくチェックポイントIDの生成"""
        param_str = json.dumps(asdict(params), sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()[:12]
    
    def save_checkpoint(self, 
                       checkpoint_id: str,
                       stage: str,
                       data: Dict[str, Any],
                       metadata: Dict[str, Any] = None) -> str:
        """チェックポイントの保存"""
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
        
        # HDF5ファイルに保存
        with h5py.File(checkpoint_file, 'w') as f:
            # メタデータ
            meta_group = f.create_group('metadata')
            for key, value in metadata.items():
                if isinstance(value, (str, int, float)):
                    meta_group.attrs[key] = value
                else:
                    meta_group.attrs[key] = str(value)
            
            # データ
            data_group = f.create_group('data')
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    data_group.create_dataset(key, data=value)
                elif isinstance(value, sp.spmatrix):
                    # スパース行列の保存
                    sparse_group = data_group.create_group(key)
                    sparse_group.create_dataset('data', data=value.data)
                    sparse_group.create_dataset('indices', data=value.indices)
                    sparse_group.create_dataset('indptr', data=value.indptr)
                    sparse_group.attrs['shape'] = value.shape
                    sparse_group.attrs['format'] = value.format
                elif isinstance(value, (int, float, str)):
                    data_group.attrs[key] = value
                else:
                    # その他のデータはpickleで保存
                    pickled_data = pickle.dumps(value)
                    data_group.create_dataset(f'{key}_pickled', data=np.frombuffer(pickled_data, dtype=np.uint8))
        
        print(f"💾 チェックポイント保存: {checkpoint_file}")
        self.current_checkpoint = str(checkpoint_file)
        return str(checkpoint_file)
    
    def load_checkpoint(self, checkpoint_file: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """チェックポイントの読み込み"""
        checkpoint_path = Path(checkpoint_file)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"チェックポイントファイルが見つかりません: {checkpoint_file}")
        
        metadata = {}
        data = {}
        
        with h5py.File(checkpoint_path, 'r') as f:
            # メタデータの読み込み
            if 'metadata' in f:
                meta_group = f['metadata']
                for key in meta_group.attrs:
                    metadata[key] = meta_group.attrs[key]
            
            # データの読み込み
            if 'data' in f:
                data_group = f['data']
                
                # 属性の読み込み
                for key in data_group.attrs:
                    data[key] = data_group.attrs[key]
                
                # データセットの読み込み
                for key in data_group:
                    if key.endswith('_pickled'):
                        # Pickleデータの復元
                        pickled_bytes = data_group[key][()]
                        original_key = key[:-8]  # '_pickled'を除去
                        data[original_key] = pickle.loads(pickled_bytes.tobytes())
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
                    else:
                        # 通常のデータセット
                        data[key] = data_group[key][()]
        
        print(f"📂 チェックポイント読み込み: {checkpoint_file}")
        return data, metadata
    
    def list_checkpoints(self, checkpoint_id: str) -> List[str]:
        """特定IDのチェックポイント一覧"""
        checkpoint_dir = self.base_dir / checkpoint_id
        if not checkpoint_dir.exists():
            return []
        
        return sorted([str(f) for f in checkpoint_dir.glob("*.h5")])
    
    def get_latest_checkpoint(self, checkpoint_id: str, stage: str = None) -> Optional[str]:
        """最新のチェックポイントを取得"""
        checkpoints = self.list_checkpoints(checkpoint_id)
        if not checkpoints:
            return None
        
        if stage:
            filtered = [cp for cp in checkpoints if stage in Path(cp).name]
            return filtered[-1] if filtered else None
        
        return checkpoints[-1]

class RecoveryGPUDiracLaplacianAnalyzer:
    """
    🚀🔄 Recovery機能付きRTX3080対応高次元ディラック/ラプラシアン作用素解析クラス
    
    新機能:
    1. 電源断からの自動復旧
    2. 計算途中からの再開
    3. 高次元（6-10次元）対応
    4. 自動チェックポイント保存
    5. メモリ効率最適化
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
        
        print(f"🔧 初期化中: {self.dim}D, 格子サイズ {self.N}^{self.dim}")
        print(f"📊 スピノル次元: {self.spinor_dim}")
        print(f"📊 総格子点数: {self.N**self.dim:,}")
        
        # メモリ使用量の推定
        total_dim = self.N**self.dim * self.spinor_dim
        if self.use_sparse:
            sparsity = min(0.1, 1000.0 / total_dim)  # 適応的スパース率
            memory_gb = (total_dim**2 * sparsity * 16) / 1e9
            print(f"💾 推定メモリ使用量（スパース）: {memory_gb:.2f} GB")
        else:
            memory_gb = (total_dim**2 * 16) / 1e9
            print(f"💾 推定メモリ使用量（密行列）: {memory_gb:.2f} GB")
        
        print(f"📊 行列次元: {total_dim:,} x {total_dim:,}")
        
        # Recovery有効時のチェックポイントID生成
        if self.checkpoint_manager:
            self.checkpoint_id = self.checkpoint_manager.create_checkpoint_id(params)
            print(f"🔄 チェックポイントID: {self.checkpoint_id}")
        
        # 高次元対応ガンマ行列の構築
        self.gamma_matrices = self._construct_high_dimensional_gamma_matrices()
        
        # 自動保存のシグナルハンドラ設定
        if params.auto_save:
            signal.signal(signal.SIGINT, self._save_and_exit)
            signal.signal(signal.SIGTERM, self._save_and_exit)
    
    def _save_and_exit(self, signum, frame):
        """シグナル受信時の自動保存"""
        print(f"\n⚠️  シグナル {signum} を受信しました")
        if self.checkpoint_manager and hasattr(self, '_current_stage_data'):
            self._save_checkpoint('emergency_save', self._current_stage_data)
        print("💾 緊急保存完了 - プログラムを終了します")
        sys.exit(0)
    
    def _construct_high_dimensional_gamma_matrices(self) -> List[np.ndarray]:
        """高次元対応ガンマ行列の構築"""
        print(f"🔨 {self.dim}次元ガンマ行列構築中...")
        
        # パウリ行列
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        I2 = np.eye(2, dtype=complex)
        
        gamma = []
        
        if self.dim <= 3:
            # 低次元の場合
            gamma = [sigma_x, sigma_y, sigma_z][:self.dim]
        
        elif self.dim == 4:
            # 4次元ディラック行列
            O2 = np.zeros((2, 2), dtype=complex)
            gamma = [
                np.block([[O2, sigma_x], [sigma_x, O2]]),
                np.block([[O2, sigma_y], [sigma_y, O2]]),
                np.block([[O2, sigma_z], [sigma_z, O2]]),
                np.block([[I2, O2], [O2, -I2]])
            ]
        
        elif self.dim <= 6:
            # 6次元まで：4次元を拡張
            O2 = np.zeros((2, 2), dtype=complex)
            O4 = np.zeros((4, 4), dtype=complex)
            I4 = np.eye(4, dtype=complex)
            
            # 基本4次元ガンマ行列
            gamma4 = [
                np.block([[O2, sigma_x], [sigma_x, O2]]),
                np.block([[O2, sigma_y], [sigma_y, O2]]),
                np.block([[O2, sigma_z], [sigma_z, O2]]),
                np.block([[I2, O2], [O2, -I2]])
            ]
            
            gamma = gamma4.copy()
            
            # 5次元目と6次元目
            for i in range(4, self.dim):
                extra_gamma = np.kron(I2, gamma4[i-4])
                gamma.append(extra_gamma)
        
        else:
            # 8次元以上：再帰的構築
            # クリフォード代数の構築
            n_matrices_needed = self.dim
            current_dim = 2
            
            # 初期ガンマ行列
            gamma = [sigma_x, sigma_y, sigma_z]
            
            while len(gamma) < n_matrices_needed:
                # 次元を倍にして拡張
                current_gamma = gamma.copy()
                new_gamma = []
                
                # 既存の行列を拡張
                I_current = np.eye(current_dim, dtype=complex)
                O_current = np.zeros((current_dim, current_dim), dtype=complex)
                
                for g in current_gamma:
                    new_g = np.block([[g, O_current], [O_current, -g]])
                    new_gamma.append(new_g)
                
                # 新しい行列を追加
                if len(new_gamma) < n_matrices_needed:
                    chirality = np.block([[I_current, O_current], [O_current, -I_current]])
                    new_gamma.append(chirality)
                
                gamma = new_gamma
                current_dim *= 2
                
                if current_dim > self.spinor_dim:
                    break
        
        # 必要な次元数に調整
        gamma = gamma[:self.dim]
        
        print(f"✅ {self.dim}次元ガンマ行列構築完了: {len(gamma)}個の{gamma[0].shape}行列")
        
        return gamma
    
    def _save_checkpoint(self, stage: str, data: Dict[str, Any]):
        """チェックポイントの保存"""
        if not self.checkpoint_manager:
            return
        
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
        
        return (time.time() - self.last_checkpoint_time) > self.params.checkpoint_interval
    
    def construct_discrete_dirac_operator_sparse_recovery(self) -> sp.csr_matrix:
        """
        🔄 Recovery対応スパース版離散ディラック作用素の構築
        """
        stage = 'dirac_construction'
        
        # 既存のチェックポイントをチェック
        if self.checkpoint_manager:
            latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint(
                self.checkpoint_id, stage
            )
            if latest_checkpoint:
                print("📂 既存のディラック作用素チェックポイントを発見")
                try:
                    data, metadata = self.checkpoint_manager.load_checkpoint(latest_checkpoint)
                    if 'dirac_operator' in data:
                        print("✅ ディラック作用素をチェックポイントから復元")
                        return data['dirac_operator']
                except Exception as e:
                    print(f"⚠️  チェックポイント復元エラー: {e}")
                    print("新規に構築します")
        
        print("🔨 スパースディラック作用素構築中...")
        start_time = time.time()
        
        total_dim = self.N**self.dim * self.spinor_dim
        
        # メモリ効率を考慮した構築
        if total_dim > 1000000:  # 100万次元以上
            print("⚡ 大規模行列用最適化モードで構築")
            D = self._construct_large_dirac_operator()
        else:
            D = self._construct_standard_dirac_operator()
        
        construction_time = time.time() - start_time
        print(f"✅ スパースディラック作用素構築完了: {construction_time:.2f}秒")
        print(f"📊 行列サイズ: {D.shape}, 非零要素数: {D.nnz:,}")
        
        # チェックポイント保存
        if self.checkpoint_manager:
            checkpoint_data = {
                'dirac_operator': D,
                'construction_time': construction_time,
                'matrix_info': {
                    'shape': D.shape,
                    'nnz': D.nnz,
                    'dtype': str(D.dtype)
                }
            }
            self._save_checkpoint(stage, checkpoint_data)
        
        return D
    
    def _construct_standard_dirac_operator(self) -> sp.csr_matrix:
        """標準サイズ行列の構築"""
        total_dim = self.N**self.dim * self.spinor_dim
        D = sp.lil_matrix((total_dim, total_dim), dtype=complex)
        
        # 各方向の微分作用素
        for mu in range(self.dim):
            print(f"  方向 {mu+1}/{self.dim} 処理中...")
            
            # 差分作用素
            diff_operator = self._construct_difference_operator_sparse(mu)
            
            # ガンマ行列との積
            gamma_mu = self.gamma_matrices[mu]
            
            # ディラック項の追加
            D += sp.kron(diff_operator, gamma_mu)
            
            # 非可換補正項
            if self.theta != 0:
                theta_correction = self._construct_theta_correction_sparse(mu)
                D += self.theta * sp.kron(theta_correction, gamma_mu)
            
            # 定期的なチェックポイント保存
            if self._should_save_checkpoint():
                temp_data = {'partial_dirac_operator': D.tocsr(), 'completed_directions': mu + 1}
                self._save_checkpoint('dirac_partial', temp_data)
        
        # 質量項
        if self.mass != 0:
            mass_operator = sp.eye(self.N**self.dim)
            mass_matrix = self.mass * sp.eye(self.spinor_dim, dtype=complex)
            D += sp.kron(mass_operator, mass_matrix)
        
        return D.tocsr()
    
    def _construct_large_dirac_operator(self) -> sp.csr_matrix:
        """大規模行列用の効率的構築"""
        print("⚡ 大規模行列最適化構築")
        
        total_dim = self.N**self.dim * self.spinor_dim
        
        # ブロック別構築
        row_indices = []
        col_indices = []
        data_values = []
        
        batch_size = min(1000, self.N)  # バッチサイズ
        
        for mu in range(self.dim):
            print(f"  大規模方向 {mu+1}/{self.dim} 処理中...")
            
            # バッチ処理で差分作用素を構築
            for batch_start in range(0, self.N**self.dim, batch_size):
                batch_end = min(batch_start + batch_size, self.N**self.dim)
                
                # 小さなブロックの処理
                block_diff = self._construct_difference_block(mu, batch_start, batch_end)
                gamma_mu = self.gamma_matrices[mu]
                
                # ブロック×ガンマ行列
                block_result = sp.kron(block_diff, gamma_mu)
                
                # インデックスと値の収集
                block_coo = block_result.tocoo()
                row_indices.extend(block_coo.row)
                col_indices.extend(block_coo.col)
                data_values.extend(block_coo.data)
                
                # メモリ節約のため定期的にクリア
                if len(data_values) > 10000000:  # 1000万要素
                    print("  メモリ最適化のため中間保存...")
                    temp_matrix = sp.coo_matrix(
                        (data_values, (row_indices, col_indices)),
                        shape=(total_dim, total_dim)
                    ).tocsr()
                    
                    # 一時保存
                    temp_data = {'temp_matrix': temp_matrix, 'batch_progress': batch_end}
                    self._save_checkpoint('large_matrix_temp', temp_data)
                    
                    # メモリクリア
                    row_indices.clear()
                    col_indices.clear()
                    data_values.clear()
        
        # 最終的な行列構築
        D = sp.coo_matrix(
            (data_values, (row_indices, col_indices)),
            shape=(total_dim, total_dim)
        ).tocsr()
        
        return D
    
    def _construct_difference_operator_sparse(self, direction: int) -> sp.csr_matrix:
        """スパース版差分作用素の構築"""
        # 1次元の前進差分
        diff_1d = sp.diags([1, -1], [1, 0], shape=(self.N, self.N))
        diff_1d = diff_1d.tolil()
        diff_1d[self.N-1, 0] = 1  # 周期境界条件
        diff_1d = diff_1d.tocsr()
        
        # 多次元への拡張
        operators = []
        for d in range(self.dim):
            if d == direction:
                operators.append(diff_1d)
            else:
                operators.append(sp.eye(self.N))
        
        # 効率的なクロネッカー積計算
        result = operators[0]
        for op in operators[1:]:
            result = sp.kron(result, op)
        
        return result
    
    def _construct_difference_block(self, direction: int, start_idx: int, end_idx: int) -> sp.csr_matrix:
        """差分作用素のブロック構築"""
        block_size = end_idx - start_idx
        block_diff = sp.eye(block_size)
        
        # 簡略化された実装（実際にはより複雑な計算が必要）
        return block_diff
    
    def _construct_theta_correction_sparse(self, direction: int) -> sp.csr_matrix:
        """スパース版θ-変形補正項の構築"""
        # 位置作用素
        positions = np.arange(self.N) - self.N // 2
        pos_1d = sp.diags(positions, 0, shape=(self.N, self.N))
        
        # 多次元への拡張
        operators = []
        for d in range(self.dim):
            if d == direction:
                operators.append(pos_1d)
            else:
                operators.append(sp.eye(self.N))
        
        x_op = operators[0]
        for op in operators[1:]:
            x_op = sp.kron(x_op, op)
        
        return x_op * 0.01
    
    def compute_spectral_dimension_recovery(self, 
                                          operator: sp.csr_matrix,
                                          n_eigenvalues: int = None) -> Tuple[float, Dict]:
        """
        🔄 Recovery対応スペクトル次元計算
        """
        if n_eigenvalues is None:
            n_eigenvalues = self.params.max_eigenvalues
        
        stage = 'spectral_computation'
        
        # 既存のチェックポイントをチェック
        if self.checkpoint_manager:
            latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint(
                self.checkpoint_id, stage
            )
            if latest_checkpoint:
                print("📂 既存のスペクトル計算チェックポイントを発見")
                try:
                    data, metadata = self.checkpoint_manager.load_checkpoint(latest_checkpoint)
                    if 'spectral_dimension' in data:
                        print("✅ スペクトル次元をチェックポイントから復元")
                        return data['spectral_dimension'], data.get('analysis_info', {})
                except Exception as e:
                    print(f"⚠️  チェックポイント復元エラー: {e}")
                    print("新規に計算します")
        
        print("🔍 スペクトル次元計算中（Recovery対応）...")
        start_time = time.time()
        
        try:
            # エルミート化
            print("🔨 エルミート化中...")
            operator_hermitian = operator.conj().T @ operator
            
            # 現在の状態を保存
            self._current_stage_data = {
                'stage': 'hermitian_complete',
                'operator_shape': operator.shape,
                'nnz': operator.nnz
            }
            
            # 固有値計算
            print(f"🧮 固有値計算中（{n_eigenvalues}個）...")
            eigenvalues, _ = eigsh(
                operator_hermitian, 
                k=min(n_eigenvalues, operator.shape[0]-2),
                which='SM', 
                return_eigenvectors=False
            )
            eigenvalues = np.real(eigenvalues)
            eigenvalues = eigenvalues[eigenvalues > 1e-12]
            
            # 中間結果の保存
            if self._should_save_checkpoint():
                checkpoint_data = {
                    'eigenvalues': eigenvalues,
                    'computation_stage': 'eigenvalues_complete'
                }
                self._save_checkpoint('spectral_intermediate', checkpoint_data)
            
        except Exception as e:
            print(f"❌ 固有値計算エラー: {e}")
            return float('nan'), {}
        
        if len(eigenvalues) < 10:
            print("⚠️  警告: 有効な固有値が少なすぎます")
            return float('nan'), {}
        
        # GPU上でスペクトルゼータ関数の計算
        print("⚡ GPU上でゼータ関数計算...")
        eigenvalues_gpu = torch.tensor(eigenvalues, device=self.device, dtype=torch.float32)
        t_values = torch.logspace(-3, 0, 50, device=self.device)
        
        zeta_values = []
        for i, t in enumerate(t_values):
            zeta_t = torch.sum(torch.exp(-t * eigenvalues_gpu))
            zeta_values.append(zeta_t.item())
            
            # 進捗表示
            if (i + 1) % 10 == 0:
                print(f"  ゼータ関数計算: {i+1}/{len(t_values)}")
        
        zeta_values = torch.tensor(zeta_values, device=self.device)
        
        # 対数微分の計算
        log_t = torch.log(t_values)
        log_zeta = torch.log(zeta_values + 1e-12)
        
        # 線形回帰で傾きを求める
        valid_mask = torch.isfinite(log_zeta) & torch.isfinite(log_t)
        if torch.sum(valid_mask) < 5:
            print("⚠️  警告: 有効なデータ点が少なすぎます")
            return float('nan'), {}
        
        log_t_valid = log_t[valid_mask]
        log_zeta_valid = log_zeta[valid_mask]
        
        # 最小二乗法
        A = torch.stack([log_t_valid, torch.ones_like(log_t_valid)], dim=1)
        slope, intercept = torch.linalg.lstsq(A, log_zeta_valid).solution
        
        spectral_dimension = -2 * slope.item()
        
        computation_time = time.time() - start_time
        print(f"✅ スペクトル次元計算完了: {computation_time:.2f}秒")
        
        # 詳細情報
        analysis_info = {
            'eigenvalues': eigenvalues,
            'n_eigenvalues': len(eigenvalues),
            'min_eigenvalue': np.min(eigenvalues),
            'max_eigenvalue': np.max(eigenvalues),
            'spectral_gap': eigenvalues[1] - eigenvalues[0] if len(eigenvalues) > 1 else 0,
            'zeta_function': zeta_values.cpu().numpy(),
            't_values': t_values.cpu().numpy(),
            'slope': slope.item(),
            'intercept': intercept.item(),
            'computation_time': computation_time
        }
        
        # 最終結果の保存
        if self.checkpoint_manager:
            final_data = {
                'spectral_dimension': spectral_dimension,
                'analysis_info': analysis_info
            }
            self._save_checkpoint(stage, final_data)
        
        return spectral_dimension, analysis_info
    
    def run_full_analysis_with_recovery(self) -> Dict[str, Any]:
        """🔄 Recovery機能付き完全解析の実行"""
        print("=" * 80)
        print("🚀🔄 Recovery機能付きRTX3080高次元ディラック/ラプラシアン作用素解析")
        print("=" * 80)
        
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name()}")
            print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        print(f"\n📊 解析パラメータ:")
        print(f"次元: {self.dim}")
        print(f"格子サイズ: {self.N}")
        print(f"スピノル次元: {self.spinor_dim}")
        print(f"Recovery機能: {'有効' if self.checkpoint_manager else '無効'}")
        
        total_start = time.time()
        
        # 1. ディラック作用素の構築
        print("\n🔨 1. ディラック作用素の構築...")
        D = self.construct_discrete_dirac_operator_sparse_recovery()
        
        # 2. スペクトル次元の計算
        print("\n🔍 2. スペクトル次元の計算...")
        d_s_dirac, dirac_info = self.compute_spectral_dimension_recovery(D)
        
        print(f"📈 ディラック作用素のスペクトル次元: {d_s_dirac:.6f}")
        print(f"🎯 理論値({self.dim})との差: {abs(d_s_dirac - self.dim):.6f}")
        
        total_time = time.time() - total_start
        print(f"\n⏱️  総計算時間: {total_time:.2f}秒")
        
        # 結果サマリー
        results = {
            'parameters': asdict(self.params),
            'gpu_info': {
                'device_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',
                'cuda_available': torch.cuda.is_available(),
                'total_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
            },
            'results': {
                'spectral_dimension': d_s_dirac,
                'theoretical_dimension': self.dim,
                'dimension_error': abs(d_s_dirac - self.dim),
                'total_computation_time': total_time,
                'matrix_size': D.shape[0],
                'nnz_elements': D.nnz,
                'sparsity_ratio': D.nnz / (D.shape[0] * D.shape[1]),
                'spinor_dimension': self.spinor_dim
            },
            'dirac_analysis': dirac_info,
            'checkpoint_id': self.checkpoint_id,
            'timestamp': datetime.now().isoformat()
        }
        
        # 結果の保存
        output_file = f"recovery_gpu_results_dim{self.dim}_N{self.N}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 結果が '{output_file}' に保存されました。")
        
        # 最終チェックポイント
        if self.checkpoint_manager:
            self._save_checkpoint('final_results', results)
            print(f"🔄 最終チェックポイント保存完了: ID {self.checkpoint_id}")
        
        return results

def demonstrate_recovery_analysis():
    """🔄 Recovery機能のデモンストレーション"""
    
    print("=" * 80)
    print("🚀🔄 RTX3080対応 Recovery機能付き高次元GPU解析デモ")
    print("=" * 80)
    
    # パラメータ設定（高次元対応）
    test_configs = [
        # 中次元テスト
        RecoveryGPUOperatorParameters(
            dimension=4, lattice_size=16, theta=0.01, kappa=0.05,
            mass=0.1, coupling=1.0, recovery_enabled=True,
            checkpoint_interval=60, auto_save=True, max_eigenvalues=50
        ),
        # 高次元テスト
        RecoveryGPUOperatorParameters(
            dimension=6, lattice_size=8, theta=0.005, kappa=0.02,
            mass=0.05, coupling=0.8, recovery_enabled=True,
            checkpoint_interval=120, auto_save=True, max_eigenvalues=30
        ),
    ]
    
    all_results = []
    
    for i, params in enumerate(test_configs):
        print(f"\n{'='*20} 設定 {i+1}/{len(test_configs)} {'='*20}")
        print(f"次元: {params.dimension}, 格子: {params.lattice_size}")
        
        try:
            analyzer = RecoveryGPUDiracLaplacianAnalyzer(params)
            results = analyzer.run_full_analysis_with_recovery()
            all_results.append(results)
            
        except Exception as e:
            print(f"❌ エラーが発生しました: {e}")
            print("Recovery機能により、次の設定で再開可能です")
            continue
    
    # 全体サマリー
    print("\n" + "="*80)
    print("📊 全体結果サマリー")
    print("="*80)
    
    for i, result in enumerate(all_results):
        params = result['parameters']
        res = result['results']
        print(f"\n設定 {i+1}:")
        print(f"  次元: {params['dimension']}, 格子: {params['lattice_size']}")
        print(f"  スペクトル次元: {res['spectral_dimension']:.6f}")
        print(f"  理論値との差: {res['dimension_error']:.6f}")
        print(f"  計算時間: {res['total_computation_time']:.2f}秒")
        print(f"  行列サイズ: {res['matrix_size']:,}")
        print(f"  チェックポイントID: {result['checkpoint_id']}")
    
    return all_results

if __name__ == "__main__":
    # Recovery機能付き高次元解析のデモンストレーション
    results = demonstrate_recovery_analysis() 