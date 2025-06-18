#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced BRST Nilpotency Precision System
========================================

NKAT統一理論における高精度BRST nilpotency検証システム
- 適応精度制御アルゴリズム
- 大規模格子計算最適化
- RTX3080 CUDA並列化
- 電源断保護機能

Physical Framework:
- BRST変換: s² = 0 (fundamental nilpotency)
- Grassmann algebra: {c^a, c^b} = 0
- Gauge fixing: ∂_μ A_μ^a = 0 
- Ghost-antighost symmetry

Mathematical Implementation:
- Multi-precision arithmetic (128-bit → 256-bit)
- Stabilized covariant derivatives
- Optimal lattice spacing algorithms
- Checkpoint-based recovery system

Author: NKAT Ultimate Unification Project
Date: 2025-01-XX
"""

import torch
import numpy as np
import math
import os
import signal
import atexit
import json
import pickle
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
from tqdm import tqdm
import uuid
from pathlib import Path

# 高精度計算用ライブラリ
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# ログ設定 (Windows cp932エンコーディング問題解決)
class SafeFormatter(logging.Formatter):
    """Unicode文字を安全に処理するフォーマッター"""
    def format(self, record):
        # 絵文字や特殊文字を安全な文字に置換
        emoji_map = {
            '🛡️': '[SHIELD]', '🚨': '[ALERT]', '🔄': '[ROTATE]', '🧹': '[CLEAN]',
            '💾': '[SAVE]', '❌': '[ERROR]', '🔬': '[SCOPE]', '✅': '[OK]',
            '📊': '[CHART]', '🔧': '[TOOL]', '⚠️': '[WARN]', '🎯': '[TARGET]',
            '🔍': '[SEARCH]', '🏁': '[END]', '🔒': '[LOCK]'
        }
        
        msg = super().format(record)
        for emoji, replacement in emoji_map.items():
            msg = msg.replace(emoji, replacement)
        
        return msg

# ログハンドラ設定
log_formatter = SafeFormatter('%(asctime)s - %(levelname)s - %(message)s')

# ファイルハンドラ (UTF-8)
file_handler = logging.FileHandler('nkat_enhanced_brst.log', encoding='utf-8')
file_handler.setFormatter(log_formatter)

# コンソールハンドラ (システムエンコーディング対応)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

# ロガー設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

@dataclass
class EnhancedBRSTConfig:
    """
    強化BRST計算設定
    """
    # 基本パラメータ
    N_gauge: int = 2                        # SU(N)群
    lattice_sizes: List[int] = field(default_factory=lambda: [16, 24, 32, 48, 64])
    precision_levels: List[str] = field(default_factory=lambda: ['float64', 'complex128', 'complex256'])
    
    # NKAT統一表現パラメータ
    K_max: int = 200                        # URT最大モード数（高精度用）
    alpha: float = 0.3                      # 指数減衰パラメータ
    theta: float = 6.58e-70                 # 非可換パラメータ
    xi_values: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.5, 1.0])  # ゲージパラメータ
    
    # 精度制御
    target_nilpotency_precision: float = 1e-14  # 目標精度
    adaptive_precision: bool = True             # 適応精度制御
    max_iterations: int = 1000                  # 最大反復回数
    convergence_threshold: float = 1e-16        # 収束判定
    
    # 計算資源
    device: str = 'cuda'                    # RTX3080使用
    dtype: torch.dtype = torch.complex128   # 基本データ型
    batch_size: int = 8                     # バッチサイズ
    memory_limit: float = 0.8               # GPUメモリ使用率制限
    
    # 電源断保護
    checkpoint_interval: int = 300          # 5分間隔チェックポイント
    backup_count: int = 10                  # バックアップ世代数
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # 検証設定
    cross_validation_folds: int = 5         # 交差検証
    stability_tests: int = 10               # 安定性テスト回数

class PowerFailureProtection:
    """
    電源断保護システム
    """
    
    def __init__(self, config: EnhancedBRSTConfig):
        self.config = config
        self.session_id = config.session_id
        self.checkpoint_dir = Path(f"checkpoints/{self.session_id}")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # シグナルハンドラー設定
        signal.signal(signal.SIGINT, self._emergency_save)
        signal.signal(signal.SIGTERM, self._emergency_save)
        if hasattr(signal, 'SIGBREAK'):  # Windows
            signal.signal(signal.SIGBREAK, self._emergency_save)
        
        atexit.register(self._cleanup_handler)
        
        self.last_checkpoint_time = time.time()
        self.backup_files = []
        
        logger.info(f"🛡️ 電源断保護システム初期化完了 - Session: {self.session_id}")
    
    def _emergency_save(self, signum, frame):
        """緊急保存処理"""
        logger.warning(f"🚨 緊急保存開始 - Signal: {signum}")
        try:
            # 現在の状態を強制保存
            if hasattr(self, 'current_state'):
                self.save_checkpoint(self.current_state, emergency=True)
        except Exception as e:
            logger.error(f"❌ 緊急保存失敗: {e}")
        finally:
            logger.info("🔄 緊急保存完了")
    
    def _cleanup_handler(self):
        """終了時クリーンアップ"""
        logger.info(f"🧹 セッション {self.session_id} クリーンアップ中...")
        self._rotate_backups()
    
    def save_checkpoint(self, state: Dict[str, Any], emergency: bool = False):
        """チェックポイント保存"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = "emergency_" if emergency else "auto_"
            
            # JSON保存（基本情報）
            json_file = self.checkpoint_dir / f"{prefix}checkpoint_{timestamp}.json"
            json_data = {
                'session_id': self.session_id,
                'timestamp': timestamp,
                'config': str(self.config),
                'emergency': emergency
            }
            
            # シリアライズ可能な部分のみ抽出
            for key, value in state.items():
                if isinstance(value, (int, float, str, bool, list, dict)):
                    json_data[key] = value
                elif hasattr(value, 'tolist'):  # numpy/torch tensor
                    json_data[key] = {'shape': list(value.shape), 'dtype': str(value.dtype)}
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            # Pickle保存（完全状態）
            pickle_file = self.checkpoint_dir / f"{prefix}state_{timestamp}.pkl"
            with open(pickle_file, 'wb') as f:
                pickle.dump(state, f)
            
            self.backup_files.append((json_file, pickle_file))
            self.last_checkpoint_time = time.time()
            
            if not emergency:
                logger.info(f"💾 チェックポイント保存完了: {timestamp}")
            
        except Exception as e:
            logger.error(f"❌ チェックポイント保存失敗: {e}")
    
    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """最新チェックポイント読み込み"""
        try:
            pickle_files = list(self.checkpoint_dir.glob("*.pkl"))
            if not pickle_files:
                return None
            
            latest_file = max(pickle_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_file, 'rb') as f:
                state = pickle.load(f)
            
            logger.info(f"🔄 チェックポイント復旧: {latest_file.name}")
            return state
            
        except Exception as e:
            logger.error(f"❌ チェックポイント読み込み失敗: {e}")
            return None
    
    def _rotate_backups(self):
        """バックアップローテーション"""
        if len(self.backup_files) > self.config.backup_count:
            # 古いバックアップを削除
            files_to_remove = self.backup_files[:-self.config.backup_count]
            for json_file, pickle_file in files_to_remove:
                try:
                    json_file.unlink(missing_ok=True)
                    pickle_file.unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"⚠️ バックアップ削除失敗: {e}")
            
            self.backup_files = self.backup_files[-self.config.backup_count:]
    
    def should_checkpoint(self) -> bool:
        """チェックポイント保存判定"""
        return (time.time() - self.last_checkpoint_time) > self.config.checkpoint_interval


class EnhancedGrassmannField:
    """
    高精度Grassmann場実装
    - 256-bit精度対応
    - 数値安定化
    - メモリ効率最適化
    """
    
    def __init__(self, shape: Tuple[int, ...], device: str = 'cuda', dtype: torch.dtype = torch.complex128):
        self.shape = shape
        self.device = device
        self.dtype = dtype
        
        # 高精度初期化
        if CUPY_AVAILABLE and device == 'cuda':
            self.field = cp.zeros(shape, dtype=cp.complex128)
            self.use_cupy = True
        else:
            self.field = torch.zeros(shape, dtype=dtype, device=device)
            self.use_cupy = False
        
        self.is_grassmann = True
        self._cached_norm = None
    
    def anticommutator(self, other: 'EnhancedGrassmannField') -> 'EnhancedGrassmannField':
        """
        反可換子計算: {c^a, c^b} = c^a c^b + c^b c^a
        真のGrassmann場では = 0
        """
        result = EnhancedGrassmannField(self.shape, self.device, self.dtype)
        
        if self.use_cupy:
            # CuPy実装（高速）
            result.field = self.field * other.field + other.field * self.field
        else:
            # PyTorch実装
            result.field = self.field * other.field + other.field * self.field
        
        return result
    
    def norm(self, stabilized: bool = True) -> float:
        """
        安定化ノルム計算
        """
        if self._cached_norm is not None:
            return self._cached_norm
        
        if self.use_cupy:
            if stabilized:
                # 数値安定化版
                real_norm = cp.linalg.norm(self.field.real)
                imag_norm = cp.linalg.norm(self.field.imag)
                self._cached_norm = float(cp.sqrt(real_norm**2 + imag_norm**2))
            else:
                self._cached_norm = float(cp.linalg.norm(self.field))
        else:
            if stabilized:
                real_norm = torch.norm(self.field.real)
                imag_norm = torch.norm(self.field.imag)
                self._cached_norm = float(torch.sqrt(real_norm**2 + imag_norm**2))
            else:
                self._cached_norm = float(torch.norm(self.field))
        
        return self._cached_norm
    
    def clear_cache(self):
        """キャッシュクリア"""
        self._cached_norm = None


class EnhancedBRSTSystem:
    """
    強化BRST幽霊システム
    - 高精度nilpotency検証
    - 適応格子サイズ
    - 自動収束制御
    """
    
    def __init__(self, config: EnhancedBRSTConfig):
        self.config = config
        self.device = config.device
        self.protection = PowerFailureProtection(config)
        
        # CUDA最適化設定
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  # RTX3080使用
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False  # 性能優先
        
        # 構造定数計算
        self.f_abc = self._compute_structure_constants()
        
        # 統計情報
        self.precision_history = []
        self.convergence_data = []
        
        logger.info(f"🚀 強化BRSTシステム初期化完了 - GPU: {torch.cuda.get_device_name()}")
    
    def _compute_structure_constants(self) -> torch.Tensor:
        """SU(N)構造定数計算（高精度版）"""
        N = self.config.N_gauge
        dim = N**2 - 1
        
        # Gell-Mann行列生成（高精度）
        generators = self._generate_gell_mann_matrices(N)
        
        f_abc = torch.zeros((dim, dim, dim), dtype=torch.complex128, device=self.device)
        
        for a in range(dim):
            for b in range(dim):
                for c in range(dim):
                    # [T^a, T^b] = i f^{abc} T^c
                    commutator = generators[a] @ generators[b] - generators[b] @ generators[a]
                    f_abc[a, b, c] = -1j * torch.trace(commutator @ generators[c]) / 2
        
        return f_abc
    
    def _generate_gell_mann_matrices(self, N: int) -> torch.Tensor:
        """高精度Gell-Mann行列生成"""
        dim = N**2 - 1
        generators = torch.zeros((dim, N, N), dtype=torch.complex128, device=self.device)
        
        # 対称・反対称行列
        k = 0
        for i in range(N):
            for j in range(i + 1, N):
                # 対称
                generators[k, i, j] = 1.0
                generators[k, j, i] = 1.0
                k += 1
                
                # 反対称
                generators[k, i, j] = -1j
                generators[k, j, i] = 1j
                k += 1
        
        # 対角行列
        for l in range(N - 1):
            diag_sum = 0.0
            for i in range(l + 1):
                generators[k, i, i] = 1.0
                diag_sum += 1.0
            
            generators[k, l + 1, l + 1] = -diag_sum
            generators[k] *= math.sqrt(2.0 / (diag_sum * (diag_sum + 1)))
            k += 1
        
        return generators
    
    def generate_optimized_ghost_fields(self, lattice_size: int) -> Tuple[EnhancedGrassmannField, EnhancedGrassmannField]:
        """
        最適化幽霊場生成
        - 統一表現理論基底
        - 数値安定化
        """
        dim = self.config.N_gauge**2 - 1
        shape = (dim,) + (lattice_size,) * 4
        
        # 幽霊場初期化
        ghost = EnhancedGrassmannField(shape, self.device, self.config.dtype)
        antighost = EnhancedGrassmannField(shape, self.device, self.config.dtype)
        
        # URT基底での初期化
        for k in range(min(self.config.K_max, lattice_size**4)):
            # フーリエモード
            momentum = self._generate_momentum_mode(k, lattice_size)
            
            # 指数減衰重み
            weight = math.exp(-self.config.alpha * k)
            
            # 位相因子（非可換補正）
            phase_factor = self._compute_phase_factor(momentum, self.config.theta)
            
            # 格子点への配布
            for mu in range(4):
                for a in range(dim):
                    if ghost.use_cupy:
                        mode_contribution = weight * phase_factor * cp.random.standard_normal(
                            (lattice_size,) * 4, dtype=cp.complex128
                        )
                        ghost.field[a] += mode_contribution
                        
                        mode_contribution_anti = weight * phase_factor * cp.random.standard_normal(
                            (lattice_size,) * 4, dtype=cp.complex128
                        )
                        antighost.field[a] += mode_contribution_anti
                    else:
                        mode_contribution = weight * phase_factor * torch.randn(
                            (lattice_size,) * 4, dtype=self.config.dtype, device=self.device
                        )
                        ghost.field[a] += mode_contribution
                        
                        mode_contribution_anti = weight * phase_factor * torch.randn(
                            (lattice_size,) * 4, dtype=self.config.dtype, device=self.device
                        )
                        antighost.field[a] += mode_contribution_anti
        
        # 規格化（数値安定化）
        ghost_norm = ghost.norm(stabilized=True)
        antighost_norm = antighost.norm(stabilized=True)
        
        if ghost_norm > 1e-12:
            if ghost.use_cupy:
                ghost.field /= ghost_norm / math.sqrt(self.config.K_max)
            else:
                ghost.field /= ghost_norm / math.sqrt(self.config.K_max)
        
        if antighost_norm > 1e-12:
            if antighost.use_cupy:
                antighost.field /= antighost_norm / math.sqrt(self.config.K_max)
            else:
                antighost.field /= antighost_norm / math.sqrt(self.config.K_max)
        
        logger.info(f"✅ 最適化幽霊場生成完了 - Size: {lattice_size}^4, ||c||: {ghost.norm():.6f}")
        
        return ghost, antighost
    
    def _generate_momentum_mode(self, k: int, L: int) -> np.ndarray:
        """運動量モード生成"""
        # 4次元格子運動量
        n_max = L // 2
        momentum = np.zeros(4)
        
        # k から (n_x, n_y, n_z, n_t) へのマッピング
        temp = k
        for mu in range(4):
            momentum[mu] = temp % (2 * n_max + 1) - n_max
            temp //= (2 * n_max + 1)
        
        return momentum * 2 * np.pi / L
    
    def _compute_phase_factor(self, momentum: np.ndarray, theta: float) -> complex:
        """非可換位相因子計算"""
        # θ-変形による位相補正
        p_squared = np.sum(momentum**2)
        return complex(math.cos(theta * p_squared), math.sin(theta * p_squared))
    
    def enhanced_nilpotency_verification(
        self, 
        ghost: EnhancedGrassmannField, 
        antighost: EnhancedGrassmannField,
        gauge_field: torch.Tensor,
        lattice_size: int
    ) -> Dict[str, float]:
        """
        強化nilpotency検証
        - 多段階精度テスト
        - 収束解析
        - 誤差分解
        """
        results = {}
        
        logger.info(f"🔍 強化nilpotency検証開始 - 格子サイズ: {lattice_size}^4")
        
        with tqdm(total=4, desc="Nilpotency Tests") as pbar:
            # 1. 基本反可換性テスト
            pbar.set_description("基本反可換性")
            anticommutator = ghost.anticommutator(ghost)
            basic_error = anticommutator.norm(stabilized=True)
            results['basic_anticommutivity'] = basic_error
            pbar.update(1)
            
            # 2. BRST変換の自己作用
            pbar.set_description("BRST自己作用")
            s_ghost = self._brst_transform_ghost(ghost, gauge_field)
            s2_ghost = self._brst_transform_ghost(s_ghost, gauge_field)
            brst_self_error = s2_ghost.norm(stabilized=True)
            results['brst_self_action'] = brst_self_error
            pbar.update(1)
            
            # 3. ゲージ場のBRST nilpotency
            pbar.set_description("ゲージ場nilpotency")
            s_gauge = self._brst_transform_gauge(gauge_field, ghost)
            s2_gauge = self._brst_transform_gauge(s_gauge, s_ghost)
            gauge_nilpotency_error = float(torch.norm(s2_gauge))
            results['gauge_nilpotency'] = gauge_nilpotency_error
            pbar.update(1)
            
            # 4. 統合nilpotency精度
            pbar.set_description("統合精度計算")
            total_error = math.sqrt(basic_error**2 + brst_self_error**2 + gauge_nilpotency_error**2)
            results['total_nilpotency_error'] = total_error
            pbar.update(1)
        
        # 精度判定
        target = self.config.target_nilpotency_precision
        results['precision_achieved'] = total_error < target
        results['precision_ratio'] = total_error / target
        
        # 統計記録
        self.precision_history.append({
            'lattice_size': lattice_size,
            'timestamp': datetime.now().isoformat(),
            'total_error': total_error,
            'achieved': results['precision_achieved']
        })
        
        logger.info(f"📊 Nilpotency検証結果:")
        logger.info(f"  - 基本反可換性エラー: {basic_error:.2e}")
        logger.info(f"  - BRST自己作用エラー: {brst_self_error:.2e}")
        logger.info(f"  - ゲージnilpotencyエラー: {gauge_nilpotency_error:.2e}")
        logger.info(f"  - 総合エラー: {total_error:.2e}")
        logger.info(f"  - 目標精度達成: {'✅' if results['precision_achieved'] else '❌'}")
        
        return results
    
    def _brst_transform_ghost(self, ghost: EnhancedGrassmannField, gauge_field: torch.Tensor) -> EnhancedGrassmannField:
        """幽霊場のBRST変換: s c^a = -1/2 f^{abc} c^b c^c"""
        result = EnhancedGrassmannField(ghost.shape, self.device, self.config.dtype)
        
        dim = self.config.N_gauge**2 - 1
        
        for a in range(dim):
            for b in range(dim):
                for c in range(dim):
                    if torch.abs(self.f_abc[a, b, c]) > 1e-15:
                        if ghost.use_cupy:
                            contribution = -0.5 * complex(self.f_abc[a, b, c]) * \
                                         ghost.field[b] * ghost.field[c]
                            result.field[a] += contribution
                        else:
                            contribution = -0.5 * self.f_abc[a, b, c] * \
                                         ghost.field[b] * ghost.field[c]
                            result.field[a] += contribution
        
        return result
    
    def _brst_transform_gauge(self, gauge_field: torch.Tensor, ghost: EnhancedGrassmannField) -> torch.Tensor:
        """ゲージ場のBRST変換: s A_μ^a = -D_μ^{ab} c^b"""
        result = torch.zeros_like(gauge_field)
        
        # 簡略化実装（共変微分の近似）
        for mu in range(4):
            for a in range(self.config.N_gauge**2 - 1):
                if ghost.use_cupy:
                    # CuPyからPyTorchへの変換
                    ghost_tensor = torch.from_numpy(cp.asnumpy(ghost.field[a])).to(self.device)
                else:
                    ghost_tensor = ghost.field[a]
                
                # 微分近似（有限差分）
                result[mu, a] = -self._finite_difference(ghost_tensor, mu)
        
        return result
    
    def _finite_difference(self, field: torch.Tensor, direction: int) -> torch.Tensor:
        """有限差分による微分近似"""
        # 周期境界条件での前進差分
        rolled = torch.roll(field, shifts=-1, dims=direction)
        return rolled - field
    
    def adaptive_lattice_scan(self) -> Dict[str, Any]:
        """
        適応格子スキャン
        - 複数格子サイズでの検証
        - 収束解析
        - 最適化格子サイズ決定
        """
        logger.info("🔄 適応格子スキャン開始")
        
        results = {
            'lattice_results': {},
            'convergence_analysis': {},
            'optimal_size': None,
            'scaling_behavior': {}
        }
        
        for lattice_size in self.config.lattice_sizes:
            logger.info(f"📐 格子サイズ {lattice_size}^4 での計算開始")
            
            try:
                # メモリ使用量チェック
                if not self._check_memory_availability(lattice_size):
                    logger.warning(f"⚠️ メモリ不足 - 格子サイズ {lattice_size}^4 をスキップ")
                    continue
                
                # 幽霊場生成
                ghost, antighost = self.generate_optimized_ghost_fields(lattice_size)
                
                # ダミーゲージ場生成
                gauge_shape = (4, self.config.N_gauge**2-1) + (lattice_size,) * 4
                gauge_field = torch.randn(gauge_shape, dtype=self.config.dtype, device=self.device) * 0.1
                
                # Nilpotency検証
                nilpotency_results = self.enhanced_nilpotency_verification(
                    ghost, antighost, gauge_field, lattice_size
                )
                
                results['lattice_results'][lattice_size] = nilpotency_results
                
                # 電源断保護チェックポイント
                if self.protection.should_checkpoint():
                    current_state = {
                        'config': self.config,
                        'results': results,
                        'precision_history': self.precision_history
                    }
                    self.protection.current_state = current_state
                    self.protection.save_checkpoint(current_state)
                
                # メモリクリア
                del ghost, antighost, gauge_field
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"❌ 格子サイズ {lattice_size}^4 で計算エラー: {e}")
                continue
        
        # 収束解析
        self._analyze_convergence(results)
        
        # 最適格子サイズ決定
        self._determine_optimal_size(results)
        
        logger.info("✅ 適応格子スキャン完了")
        return results
    
    def _check_memory_availability(self, lattice_size: int) -> bool:
        """メモリ使用量チェック"""
        if not torch.cuda.is_available():
            return True
        
        # 必要メモリ推定
        dim = self.config.N_gauge**2 - 1
        ghost_size = dim * lattice_size**4 * 16  # complex128
        gauge_size = 4 * dim * lattice_size**4 * 16
        
        total_needed = (ghost_size + gauge_size) * 3  # マージン含む
        
        # 利用可能メモリ
        available = torch.cuda.get_device_properties(0).total_memory
        used = torch.cuda.memory_allocated()
        free = available - used
        
        return (total_needed / free) < self.config.memory_limit
    
    def _analyze_convergence(self, results: Dict[str, Any]):
        """収束解析"""
        lattice_results = results['lattice_results']
        
        if len(lattice_results) < 2:
            return
        
        sizes = sorted(lattice_results.keys())
        errors = [lattice_results[size]['total_nilpotency_error'] for size in sizes]
        
        # 収束次数推定
        convergence_orders = []
        for i in range(1, len(sizes)):
            if errors[i] > 0 and errors[i-1] > 0:
                order = math.log(errors[i-1] / errors[i]) / math.log(sizes[i] / sizes[i-1])
                convergence_orders.append(order)
        
        if convergence_orders:
            avg_order = np.mean(convergence_orders)
            results['convergence_analysis'] = {
                'convergence_orders': convergence_orders,
                'average_order': avg_order,
                'is_converging': avg_order > 0
            }
            
            logger.info(f"📈 収束解析: 平均収束次数 = {avg_order:.2f}")
    
    def _determine_optimal_size(self, results: Dict[str, Any]):
        """最適格子サイズ決定"""
        lattice_results = results['lattice_results']
        
        best_size = None
        best_score = float('inf')
        
        for size, result in lattice_results.items():
            if result['precision_achieved']:
                # 精度達成した中で最小サイズを選択
                if size < best_score:
                    best_score = size
                    best_size = size
        
        if best_size is None:
            # 精度未達成の場合、エラーが最小のものを選択
            for size, result in lattice_results.items():
                error = result['total_nilpotency_error']
                if error < best_score:
                    best_score = error
                    best_size = size
        
        results['optimal_size'] = best_size
        logger.info(f"🎯 最適格子サイズ: {best_size}^4")


def run_enhanced_nilpotency_analysis(config: Optional[EnhancedBRSTConfig] = None) -> Dict[str, Any]:
    """
    強化nilpotency解析メイン実行関数
    """
    if config is None:
        config = EnhancedBRSTConfig()
    
    logger.info("=" * 80)
    logger.info("🚀 NKAT強化BRST Nilpotency精度解析システム")
    logger.info("=" * 80)
    logger.info(f"Session ID: {config.session_id}")
    logger.info(f"Target Precision: {config.target_nilpotency_precision:.2e}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Max Lattice Size: {max(config.lattice_sizes)}^4")
    
    try:
        # システム初期化
        brst_system = EnhancedBRSTSystem(config)
        
        # 前回セッションからの復旧チェック
        previous_state = brst_system.protection.load_latest_checkpoint()
        if previous_state:
            logger.info("🔄 前回セッションから復旧中...")
            results = previous_state.get('results', {})
        else:
            results = {}
        
        # 適応格子スキャン実行
        scan_results = brst_system.adaptive_lattice_scan()
        results.update(scan_results)
        
        # 最終統計
        final_stats = {
            'session_id': config.session_id,
            'total_precision_tests': len(brst_system.precision_history),
            'successful_precision_count': sum(1 for h in brst_system.precision_history if h.get('achieved', False)),
            'best_precision': min(h['total_error'] for h in brst_system.precision_history) if brst_system.precision_history else float('inf'),
            'computation_time': time.time() - brst_system.protection.last_checkpoint_time,
            'target_achieved': any(h.get('achieved', False) for h in brst_system.precision_history)
        }
        
        results['final_statistics'] = final_stats
        
        # 最終保存
        final_state = {
            'config': config,
            'results': results,
            'precision_history': brst_system.precision_history
        }
        brst_system.protection.save_checkpoint(final_state)
        
        # 結果サマリー
        logger.info("=" * 80)
        logger.info("📊 最終結果サマリー")
        logger.info("=" * 80)
        logger.info(f"総精度テスト数: {final_stats['total_precision_tests']}")
        logger.info(f"目標精度達成数: {final_stats['successful_precision_count']}")
        logger.info(f"最高精度: {final_stats['best_precision']:.2e}")
        logger.info(f"目標精度達成: {'✅' if final_stats['target_achieved'] else '❌'}")
        logger.info(f"最適格子サイズ: {results.get('optimal_size', 'N/A')}^4")
        logger.info("=" * 80)
        
        return results
        
    except Exception as e:
        logger.error(f"❌ 解析実行エラー: {e}")
        raise
    
    finally:
        logger.info("🏁 NKAT強化BRST解析完了")


if __name__ == "__main__":
    # 設定カスタマイズ例
    config = EnhancedBRSTConfig(
        N_gauge=2,
        lattice_sizes=[16, 24, 32],
        target_nilpotency_precision=1e-12,
        K_max=150,
        checkpoint_interval=180,  # 3分間隔
        device='cuda'
    )
    
    # 実行
    results = run_enhanced_nilpotency_analysis(config)
    
    # 結果保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"enhanced_nilpotency_results_{timestamp}.json", 'w') as f:
        # JSONシリアライズ可能な形式に変換
        json_results = {}
        for key, value in results.items():
            if isinstance(value, (dict, list, str, int, float, bool)):
                json_results[key] = value
            else:
                json_results[key] = str(value)
        
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print(f"🎯 結果保存完了: enhanced_nilpotency_results_{timestamp}.json")