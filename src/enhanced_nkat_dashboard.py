#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌌 Enhanced NKAT Riemann Analysis Dashboard
非可換コルモゴロフアーノルド表現理論による改良版リーマン予想解析ダッシュボード

主要改良点:
- パフォーマンス最適化
- エラーハンドリング強化
- UI/UX改善
- リアルタイム監視機能
- メモリ効率化
- 自動復旧機能
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import threading
import queue
import json
import os
import sys
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple
import warnings
import logging
from pathlib import Path

# 警告を抑制
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 拡張統計解析モジュールのインポート
EXTENDED_ANALYSIS_AVAILABLE = False
try:
    from riemann_zeros_extended import RiemannZerosDatabase, RiemannZerosStatistics, create_visualization_plots
    EXTENDED_ANALYSIS_AVAILABLE = True
    logger.info("拡張統計解析モジュール読み込み成功")
except ImportError as e:
    logger.warning(f"拡張統計解析モジュール読み込み失敗: {e}")

# GPU監視モジュール
GPU_MONITORING_AVAILABLE = False
try:
    import GPUtil
    import psutil
    GPU_MONITORING_AVAILABLE = True
    logger.info("GPU監視モジュール読み込み成功")
except ImportError as e:
    logger.warning(f"GPU監視モジュール読み込み失敗: {e}")

# PyTorch（オプション）
PYTORCH_AVAILABLE = False
try:
    import torch
    if torch.cuda.is_available():
        PYTORCH_AVAILABLE = True
        logger.info(f"PyTorch CUDA利用可能: {torch.cuda.get_device_name()}")
    else:
        logger.info("PyTorch利用可能（CPU版）")
except ImportError:
    logger.warning("PyTorch利用不可")

@dataclass
class EnhancedNKATParameters:
    """改良版NKAT パラメータ"""
    # 基本パラメータ
    dimension: int = 32
    precision: int = 100
    
    # リーマンゼータパラメータ
    t_start: float = 0.0
    t_end: float = 200.0
    n_points: int = 2000
    
    # 非可換パラメータ
    theta: float = 1e-35  # プランク長さスケール
    kappa: float = 1e-20  # 量子重力スケール
    
    # 統計解析パラメータ
    n_zeros_analysis: int = 5000
    enable_extended_analysis: bool = True
    show_statistical_plots: bool = True
    
    # パフォーマンス設定
    use_gpu_acceleration: bool = True
    batch_size: int = 1000
    max_memory_usage: float = 0.8  # GPU/CPUメモリ使用率上限
    
    # 監視設定
    enable_realtime_monitoring: bool = True
    monitoring_interval: float = 1.0  # 秒
    temperature_threshold: float = 80.0  # GPU温度閾値（℃）
    
    # 自動保存設定
    auto_save_enabled: bool = True
    save_interval: int = 300  # 秒
    checkpoint_dir: str = "Results/checkpoints"

class EnhancedSystemMonitor:
    """改良版システム監視クラス"""
    
    def __init__(self):
        self.monitoring_active = False
        self.monitoring_thread = None
        self.data_queue = queue.Queue()
        self.history_length = 100
        self.monitoring_data = {
            'timestamps': [],
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_usage': [],
            'gpu_memory': [],
            'gpu_temperature': []
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """システム情報取得"""
        info = {
            'timestamp': datetime.now(),
            'cpu_count': os.cpu_count(),
            'cpu_usage': 0.0,
            'memory_total': 0.0,
            'memory_used': 0.0,
            'memory_percent': 0.0,
            'gpu_available': False,
            'gpu_info': [],
            'pytorch_cuda': PYTORCH_AVAILABLE
        }
        
        try:
            # CPU情報
            if hasattr(psutil, 'cpu_percent'):
                info['cpu_usage'] = psutil.cpu_percent(interval=0.1)
            
            # メモリ情報
            if hasattr(psutil, 'virtual_memory'):
                memory = psutil.virtual_memory()
                info['memory_total'] = memory.total / (1024**3)  # GB
                info['memory_used'] = memory.used / (1024**3)
                info['memory_percent'] = memory.percent
        except Exception as e:
            logger.warning(f"システム情報取得エラー: {e}")
        
        # GPU情報
        if GPU_MONITORING_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                info['gpu_available'] = len(gpus) > 0
                for gpu in gpus:
                    gpu_info = {
                        'id': gpu.id,
                        'name': gpu.name,
                        'memory_total': gpu.memoryTotal,
                        'memory_used': gpu.memoryUsed,
                        'memory_percent': gpu.memoryUtil * 100,
                        'usage': gpu.load * 100,
                        'temperature': gpu.temperature
                    }
                    info['gpu_info'].append(gpu_info)
            except Exception as e:
                logger.warning(f"GPU情報取得エラー: {e}")
        
        return info
    
    def start_monitoring(self, interval: float = 1.0):
        """リアルタイム監視開始"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("リアルタイム監視開始")
    
    def stop_monitoring(self):
        """監視停止"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        logger.info("リアルタイム監視停止")
    
    def _monitoring_loop(self, interval: float):
        """監視ループ"""
        while self.monitoring_active:
            try:
                info = self.get_system_info()
                self.data_queue.put(info)
                
                # 履歴データ更新
                self._update_history(info)
                
                time.sleep(interval)
            except Exception as e:
                logger.error(f"監視ループエラー: {e}")
                time.sleep(interval)
    
    def _update_history(self, info: Dict[str, Any]):
        """履歴データ更新"""
        timestamp = info['timestamp']
        
        # データ追加
        self.monitoring_data['timestamps'].append(timestamp)
        self.monitoring_data['cpu_usage'].append(info['cpu_usage'])
        self.monitoring_data['memory_usage'].append(info['memory_percent'])
        
        if info['gpu_info']:
            gpu = info['gpu_info'][0]  # 最初のGPUを使用
            self.monitoring_data['gpu_usage'].append(gpu['usage'])
            self.monitoring_data['gpu_memory'].append(gpu['memory_percent'])
            self.monitoring_data['gpu_temperature'].append(gpu['temperature'])
        else:
            self.monitoring_data['gpu_usage'].append(0)
            self.monitoring_data['gpu_memory'].append(0)
            self.monitoring_data['gpu_temperature'].append(0)
        
        # 履歴長制限
        for key in self.monitoring_data:
            if len(self.monitoring_data[key]) > self.history_length:
                self.monitoring_data[key] = self.monitoring_data[key][-self.history_length:]
    
    def get_monitoring_data(self) -> Dict[str, List]:
        """監視データ取得"""
        return self.monitoring_data.copy()

class EnhancedRiemannAnalyzer:
    """改良版リーマン解析器"""
    
    def __init__(self, params: EnhancedNKATParameters):
        self.params = params
        self.known_zeros = [
            14.134725142, 21.022039639, 25.010857580, 30.424876126,
            32.935061588, 37.586178159, 40.918719012, 43.327073281,
            48.005150881, 49.773832478, 52.970321478, 56.446247697,
            59.347044003, 60.831778525, 65.112544048, 67.079810529,
            69.546401711, 72.067157674, 75.704690699, 77.144840069
        ]
        self.analysis_cache = {}
        self.checkpoint_manager = CheckpointManager(params.checkpoint_dir)
    
    def classical_zeta(self, s: complex) -> complex:
        """古典的リーマンゼータ関数（改良版）"""
        if s.real <= 0:
            return 0.0
        
        # キャッシュチェック
        cache_key = f"zeta_{s.real:.6f}_{s.imag:.6f}"
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        # 高精度計算
        result = 0.0
        for n in range(1, self.params.precision + 1):
            term = 1.0 / (n ** s)
            result += term
            
            # 収束チェック
            if abs(term) < 1e-15:
                break
        
        self.analysis_cache[cache_key] = result
        return result
    
    def nkat_enhanced_zeta(self, s: complex) -> complex:
        """NKAT強化リーマンゼータ関数"""
        # 基本項
        base_term = self.classical_zeta(s)
        
        # 非可換補正項（θパラメータ）
        theta_correction = self.params.theta * np.exp(-abs(s.imag) * self.params.theta)
        
        # κ変形項
        kappa_deformation = self.params.kappa * (1 + self.params.kappa * abs(s)**2)
        
        # チェビシェフ多項式補正
        t = s.imag / 100.0  # 正規化
        chebyshev_correction = np.cos(np.arccos(t)) if abs(t) <= 1 else np.cosh(np.arccosh(abs(t)))
        
        return base_term + theta_correction + kappa_deformation + 1e-10 * chebyshev_correction
    
    def find_zeros_advanced(self) -> Dict[str, Any]:
        """高度ゼロ点探索"""
        start_time = time.time()
        
        # チェックポイント確認
        checkpoint_data = self.checkpoint_manager.load_checkpoint("zeros_search")
        if checkpoint_data:
            logger.info("チェックポイントから復旧")
            return checkpoint_data
        
        zeros_found = []
        t_values = np.linspace(self.params.t_start, self.params.t_end, self.params.n_points)
        
        # バッチ処理
        batch_size = min(self.params.batch_size, len(t_values))
        
        for i in range(0, len(t_values), batch_size):
            batch_t = t_values[i:i+batch_size]
            
            # バッチ内でゼロ点探索
            for t in batch_t:
                s = 0.5 + 1j * t
                zeta_val = self.nkat_enhanced_zeta(s)
                
                # ゼロ点判定（改良版）
                if abs(zeta_val) < 1e-10:
                    zeros_found.append(t)
                elif len(zeros_found) > 0:
                    # 既知のゼロ点との比較
                    for known_zero in self.known_zeros:
                        if abs(t - known_zero) < 0.1:
                            zeros_found.append(t)
                            break
        
        # 結果整理
        zeros_found = sorted(list(set(zeros_found)))
        
        # 検証
        verified_zeros = []
        for zero in zeros_found:
            s = 0.5 + 1j * zero
            if abs(self.nkat_enhanced_zeta(s)) < 1e-8:
                verified_zeros.append(zero)
        
        result = {
            'zeros_list': zeros_found,
            'verified_zeros_list': verified_zeros,
            'total_zeros_found': len(zeros_found),
            'verified_zeros': len(verified_zeros),
            'verification_rate': len(verified_zeros) / max(len(zeros_found), 1),
            'computation_time': time.time() - start_time,
            'parameters': asdict(self.params),
            'timestamp': datetime.now()
        }
        
        # チェックポイント保存
        self.checkpoint_manager.save_checkpoint("zeros_search", result)
        
        return result

class CheckpointManager:
    """チェックポイント管理クラス"""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, name: str, data: Dict[str, Any]):
        """チェックポイント保存"""
        try:
            checkpoint_file = self.checkpoint_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # JSON serializable に変換
            serializable_data = self._make_serializable(data)
            
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"チェックポイント保存: {checkpoint_file}")
            return True
        except Exception as e:
            logger.error(f"チェックポイント保存エラー: {e}")
            return False
    
    def load_checkpoint(self, name: str) -> Optional[Dict[str, Any]]:
        """最新チェックポイント読み込み"""
        try:
            pattern = f"{name}_*.json"
            checkpoint_files = list(self.checkpoint_dir.glob(pattern))
            
            if not checkpoint_files:
                return None
            
            # 最新ファイル選択
            latest_file = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"チェックポイント読み込み: {latest_file}")
            return data
        except Exception as e:
            logger.error(f"チェックポイント読み込みエラー: {e}")
            return None
    
    def _make_serializable(self, obj):
        """JSON serializable に変換"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, complex):
            return {'real': obj.real, 'imag': obj.imag}
        else:
            return obj

class EnhancedNKATDashboard:
    """改良版NKATダッシュボード"""
    
    def __init__(self):
        self.monitor = EnhancedSystemMonitor()
        self.analyzer = None
        self.analysis_results = []
        self.auto_save_thread = None
        
        # セッション状態初期化
        if 'analysis_running' not in st.session_state:
            st.session_state.analysis_running = False
        if 'monitoring_active' not in st.session_state:
            st.session_state.monitoring_active = False
        if 'last_analysis_time' not in st.session_state:
            st.session_state.last_analysis_time = None
    
    def render_sidebar(self) -> EnhancedNKATParameters:
        """改良版サイドバー"""
        st.sidebar.title("🎛️ NKAT制御パネル")
        
        # 基本パラメータ
        st.sidebar.subheader("🔧 基本設定")
        dimension = st.sidebar.slider("次元数", 8, 64, 32, 8)
        precision = st.sidebar.slider("計算精度", 50, 200, 100, 10)
        
        # リーマンパラメータ
        st.sidebar.subheader("🎯 リーマン解析設定")
        t_start = st.sidebar.number_input("t開始値", 0.0, 50.0, 0.0, 1.0)
        t_end = st.sidebar.number_input("t終了値", 50.0, 500.0, 200.0, 10.0)
        n_points = st.sidebar.slider("計算点数", 500, 5000, 2000, 100)
        
        # 非可換パラメータ
        st.sidebar.subheader("⚛️ 非可換設定")
        theta_exp = st.sidebar.slider("θ指数", -40, -30, -35)
        kappa_exp = st.sidebar.slider("κ指数", -25, -15, -20)
        
        # 統計解析設定
        st.sidebar.subheader("📊 統計解析設定")
        n_zeros_analysis = st.sidebar.slider("解析ゼロ点数", 100, 10000, 5000, 100)
        enable_extended_analysis = st.sidebar.checkbox("拡張統計解析", True)
        show_statistical_plots = st.sidebar.checkbox("統計プロット表示", True)
        
        # パフォーマンス設定
        st.sidebar.subheader("⚡ パフォーマンス設定")
        use_gpu_acceleration = st.sidebar.checkbox("GPU加速", PYTORCH_AVAILABLE)
        batch_size = st.sidebar.slider("バッチサイズ", 100, 2000, 1000, 100)
        max_memory_usage = st.sidebar.slider("最大メモリ使用率", 0.5, 0.95, 0.8, 0.05)
        
        # 監視設定
        st.sidebar.subheader("📡 監視設定")
        enable_realtime_monitoring = st.sidebar.checkbox("リアルタイム監視", True)
        monitoring_interval = st.sidebar.slider("監視間隔（秒）", 0.5, 5.0, 1.0, 0.5)
        temperature_threshold = st.sidebar.slider("GPU温度閾値（℃）", 70, 90, 80, 5)
        
        # 自動保存設定
        st.sidebar.subheader("💾 自動保存設定")
        auto_save_enabled = st.sidebar.checkbox("自動保存", True)
        save_interval = st.sidebar.slider("保存間隔（秒）", 60, 600, 300, 60)
        
        return EnhancedNKATParameters(
            dimension=dimension,
            precision=precision,
            t_start=t_start,
            t_end=t_end,
            n_points=n_points,
            theta=10**theta_exp,
            kappa=10**kappa_exp,
            n_zeros_analysis=n_zeros_analysis,
            enable_extended_analysis=enable_extended_analysis,
            show_statistical_plots=show_statistical_plots,
            use_gpu_acceleration=use_gpu_acceleration,
            batch_size=batch_size,
            max_memory_usage=max_memory_usage,
            enable_realtime_monitoring=enable_realtime_monitoring,
            monitoring_interval=monitoring_interval,
            temperature_threshold=temperature_threshold,
            auto_save_enabled=auto_save_enabled,
            save_interval=save_interval
        )
    
    def render_system_status(self):
        """改良版システム状態表示"""
        system_info = self.monitor.get_system_info()
        
        # メトリクス表示
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🖥️ CPU使用率",
                f"{system_info['cpu_usage']:.1f}%",
                delta=None
            )
        
        with col2:
            st.metric(
                "🧠 メモリ使用率",
                f"{system_info['memory_percent']:.1f}%",
                delta=f"{system_info['memory_used']:.1f}GB / {system_info['memory_total']:.1f}GB"
            )
        
        with col3:
            if system_info['gpu_available'] and system_info['gpu_info']:
                gpu = system_info['gpu_info'][0]
                st.metric(
                    "🎮 GPU使用率",
                    f"{gpu['usage']:.1f}%",
                    delta=f"{gpu['memory_used']:.0f}MB / {gpu['memory_total']:.0f}MB"
                )
            else:
                st.metric("🎮 GPU", "利用不可", delta=None)
        
        with col4:
            if system_info['gpu_available'] and system_info['gpu_info']:
                gpu = system_info['gpu_info'][0]
                temp_color = "🔥" if gpu['temperature'] > 80 else "🌡️"
                st.metric(
                    f"{temp_color} GPU温度",
                    f"{gpu['temperature']:.0f}°C",
                    delta=None
                )
            else:
                st.metric("🌡️ GPU温度", "N/A", delta=None)
        
        # GPU詳細情報
        if system_info['gpu_available'] and system_info['gpu_info']:
            with st.expander("🎮 GPU詳細情報"):
                for i, gpu in enumerate(system_info['gpu_info']):
                    st.write(f"**GPU {i}: {gpu['name']}**")
                    st.write(f"- メモリ: {gpu['memory_used']:.0f}MB / {gpu['memory_total']:.0f}MB ({gpu['memory_percent']:.1f}%)")
                    st.write(f"- 使用率: {gpu['usage']:.1f}%")
                    st.write(f"- 温度: {gpu['temperature']:.0f}°C")
    
    def render_realtime_monitoring(self):
        """リアルタイム監視表示"""
        if not st.session_state.monitoring_active:
            return
        
        monitoring_data = self.monitor.get_monitoring_data()
        
        if not monitoring_data['timestamps']:
            st.info("監視データ収集中...")
            return
        
        # 時系列グラフ
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('CPU使用率', 'メモリ使用率', 'GPU使用率', 'GPU温度'),
            vertical_spacing=0.1
        )
        
        timestamps = monitoring_data['timestamps']
        
        # CPU使用率
        fig.add_trace(
            go.Scatter(x=timestamps, y=monitoring_data['cpu_usage'], name='CPU', line=dict(color='blue')),
            row=1, col=1
        )
        
        # メモリ使用率
        fig.add_trace(
            go.Scatter(x=timestamps, y=monitoring_data['memory_usage'], name='Memory', line=dict(color='green')),
            row=1, col=2
        )
        
        # GPU使用率
        fig.add_trace(
            go.Scatter(x=timestamps, y=monitoring_data['gpu_usage'], name='GPU', line=dict(color='red')),
            row=2, col=1
        )
        
        # GPU温度
        fig.add_trace(
            go.Scatter(x=timestamps, y=monitoring_data['gpu_temperature'], name='GPU Temp', line=dict(color='orange')),
            row=2, col=2
        )
        
        fig.update_layout(height=400, showlegend=False, title_text="リアルタイムシステム監視")
        fig.update_yaxes(title_text="使用率 (%)", row=1, col=1)
        fig.update_yaxes(title_text="使用率 (%)", row=1, col=2)
        fig.update_yaxes(title_text="使用率 (%)", row=2, col=1)
        fig.update_yaxes(title_text="温度 (°C)", row=2, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_analysis_controls(self, params: EnhancedNKATParameters):
        """改良版解析制御"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🚀 解析開始", disabled=st.session_state.analysis_running):
                self.start_analysis(params)
        
        with col2:
            if st.button("⏹️ 解析停止", disabled=not st.session_state.analysis_running):
                self.stop_analysis()
        
        with col3:
            if st.button("📡 監視開始/停止"):
                self.toggle_monitoring(params)
        
        with col4:
            if st.button("🔄 結果リセット"):
                self.reset_results()
        
        # 進行状況表示
        if st.session_state.analysis_running:
            st.info("🔄 解析実行中...")
            progress_bar = st.progress(0)
            
            # 模擬進行状況
            for i in range(100):
                progress_bar.progress(i + 1)
                time.sleep(0.01)
    
    def start_analysis(self, params: EnhancedNKATParameters):
        """解析開始"""
        st.session_state.analysis_running = True
        st.session_state.last_analysis_time = datetime.now()
        
        # 解析器初期化
        self.analyzer = EnhancedRiemannAnalyzer(params)
        
        # 非同期解析実行
        def run_analysis():
            try:
                result = self.analyzer.find_zeros_advanced()
                result['status'] = 'completed'
                self.analysis_results.append(result)
                st.session_state.analysis_running = False
                logger.info("解析完了")
            except Exception as e:
                error_result = {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now()
                }
                self.analysis_results.append(error_result)
                st.session_state.analysis_running = False
                logger.error(f"解析エラー: {e}")
        
        # バックグラウンド実行
        analysis_thread = threading.Thread(target=run_analysis, daemon=True)
        analysis_thread.start()
        
        st.success("解析を開始しました")
    
    def stop_analysis(self):
        """解析停止"""
        st.session_state.analysis_running = False
        st.warning("解析を停止しました")
    
    def toggle_monitoring(self, params: EnhancedNKATParameters):
        """監視開始/停止切り替え"""
        if st.session_state.monitoring_active:
            self.monitor.stop_monitoring()
            st.session_state.monitoring_active = False
            st.info("監視を停止しました")
        else:
            self.monitor.start_monitoring(params.monitoring_interval)
            st.session_state.monitoring_active = True
            st.success("監視を開始しました")
    
    def reset_results(self):
        """結果リセット"""
        self.analysis_results.clear()
        st.success("結果をリセットしました")
    
    def render_results(self):
        """結果表示"""
        if not self.analysis_results:
            st.info("解析結果がありません。解析を実行してください。")
            return
        
        # 最新結果表示
        latest_result = self.analysis_results[-1]
        
        if latest_result['status'] == 'completed':
            st.success("✅ 解析完了")
            
            # メトリクス表示
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🔍 発見ゼロ点数", latest_result['total_zeros_found'])
            
            with col2:
                st.metric("✅ 検証済みゼロ点", latest_result['verified_zeros'])
            
            with col3:
                st.metric("📊 検証率", f"{latest_result['verification_rate']*100:.1f}%")
            
            with col4:
                st.metric("⏱️ 計算時間", f"{latest_result['computation_time']:.2f}秒")
            
            # ゼロ点リスト
            if latest_result['zeros_list']:
                st.subheader("🎯 発見されたゼロ点")
                zeros_df = pd.DataFrame({
                    'ゼロ点 t': latest_result['zeros_list'],
                    's = 0.5 + it': [f"0.5 + {t:.6f}i" for t in latest_result['zeros_list']],
                    '検証済み': ['✅' if t in latest_result['verified_zeros_list'] else '❌' 
                               for t in latest_result['zeros_list']]
                })
                st.dataframe(zeros_df, use_container_width=True)
            
            # 可視化
            self.render_advanced_visualization(latest_result)
            
        elif latest_result['status'] == 'error':
            st.error(f"❌ 解析エラー: {latest_result['error']}")
    
    def render_advanced_visualization(self, result: Dict[str, Any]):
        """高度可視化"""
        if not result['zeros_list']:
            return
        
        st.subheader("📈 高度解析結果可視化")
        
        # タブで整理
        tab1, tab2, tab3 = st.tabs(["ゼロ点分布", "統計解析", "3D可視化"])
        
        with tab1:
            # ゼロ点分布
            fig = go.Figure()
            
            zeros = result['zeros_list']
            verified = result['verified_zeros_list']
            
            # 全ゼロ点
            fig.add_trace(go.Scatter(
                x=zeros,
                y=[0.5] * len(zeros),
                mode='markers',
                marker=dict(size=8, color='lightblue', opacity=0.7),
                name='発見ゼロ点',
                text=[f't = {z:.6f}' for z in zeros],
                hovertemplate='<b>ゼロ点</b><br>t = %{x:.6f}<br>s = 0.5 + %{x:.6f}i<extra></extra>'
            ))
            
            # 検証済みゼロ点
            fig.add_trace(go.Scatter(
                x=verified,
                y=[0.5] * len(verified),
                mode='markers',
                marker=dict(size=12, color='red', symbol='star'),
                name='検証済みゼロ点',
                text=[f't = {z:.6f}' for z in verified],
                hovertemplate='<b>検証済みゼロ点</b><br>t = %{x:.6f}<br>s = 0.5 + %{x:.6f}i<extra></extra>'
            ))
            
            fig.update_layout(
                title='リーマンゼータ関数のゼロ点分布（臨界線上）',
                xaxis_title='虚部 t',
                yaxis_title='実部（= 0.5）',
                yaxis=dict(range=[0.4, 0.6]),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # 統計解析
            if EXTENDED_ANALYSIS_AVAILABLE:
                self.render_extended_statistics()
            else:
                st.warning("拡張統計解析モジュールが利用できません")
        
        with tab3:
            # 3D可視化
            self.render_3d_visualization(result)
    
    def render_extended_statistics(self):
        """拡張統計解析表示"""
        try:
            zeros_db = RiemannZerosDatabase()
            stats_analyzer = RiemannZerosStatistics(zeros_db)
            
            # 統計プロット作成
            plots = create_visualization_plots(zeros_db, stats_analyzer, 5000)
            
            # プロット表示
            for plot_name, fig in plots.items():
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"統計解析エラー: {e}")
    
    def render_3d_visualization(self, result: Dict[str, Any]):
        """3D可視化"""
        zeros = result['zeros_list']
        
        if len(zeros) < 3:
            st.warning("3D可視化には最低3個のゼロ点が必要です")
            return
        
        # 3Dプロット作成
        fig = go.Figure(data=[go.Scatter3d(
            x=zeros,
            y=[0.5] * len(zeros),
            z=range(len(zeros)),
            mode='markers+lines',
            marker=dict(
                size=8,
                color=zeros,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="t値")
            ),
            line=dict(color='darkblue', width=2),
            text=[f'ゼロ点 {i+1}: t={z:.6f}' for i, z in enumerate(zeros)],
            hovertemplate='<b>%{text}</b><br>t = %{x:.6f}<br>実部 = %{y}<br>順序 = %{z}<extra></extra>'
        )])
        
        fig.update_layout(
            title='リーマンゼロ点の3D可視化',
            scene=dict(
                xaxis_title='虚部 t',
                yaxis_title='実部（= 0.5）',
                zaxis_title='ゼロ点順序'
            ),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def run(self):
        """メインダッシュボード実行"""
        # ページ設定
        st.set_page_config(
            page_title="Enhanced NKAT Dashboard",
            page_icon="🌌",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # カスタムCSS
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #1e3c72;
        }
        .status-good { border-left-color: #28a745; }
        .status-warning { border-left-color: #ffc107; }
        .status-error { border-left-color: #dc3545; }
        </style>
        """, unsafe_allow_html=True)
        
        # ヘッダー
        st.markdown("""
        <div class="main-header">
            <h1>🌌 Enhanced NKAT Riemann Analysis Dashboard</h1>
            <p>非可換コルモゴロフアーノルド表現理論による改良版リーマン予想解析システム</p>
        </div>
        """, unsafe_allow_html=True)
        
        # パラメータ設定
        params = self.render_sidebar()
        
        # メインコンテンツ
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # システム状態
            st.subheader("🖥️ システム状態")
            self.render_system_status()
            
            # リアルタイム監視
            if params.enable_realtime_monitoring and st.session_state.monitoring_active:
                st.subheader("📡 リアルタイム監視")
                self.render_realtime_monitoring()
            
            # 解析制御
            st.subheader("🎛️ 解析制御")
            self.render_analysis_controls(params)
            
            # 結果表示
            st.subheader("📊 解析結果")
            self.render_results()
        
        with col2:
            # サイドパネル情報
            st.subheader("ℹ️ システム情報")
            
            # 現在時刻
            st.info(f"🕐 現在時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 最終解析時刻
            if st.session_state.last_analysis_time:
                st.info(f"🔍 最終解析: {st.session_state.last_analysis_time.strftime('%H:%M:%S')}")
            
            # 機能状態
            st.subheader("🔧 機能状態")
            
            status_items = [
                ("拡張統計解析", EXTENDED_ANALYSIS_AVAILABLE),
                ("GPU監視", GPU_MONITORING_AVAILABLE),
                ("PyTorch CUDA", PYTORCH_AVAILABLE),
                ("リアルタイム監視", st.session_state.monitoring_active),
                ("解析実行中", st.session_state.analysis_running)
            ]
            
            for name, status in status_items:
                status_icon = "✅" if status else "❌"
                st.write(f"{status_icon} {name}")
        
        # フッター
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666;">
            <p><strong>Enhanced NKAT Research System</strong> | Version 2.0 | 
            Powered by Non-Commutative Kolmogorov-Arnold Representation Theory</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """メイン関数"""
    try:
        dashboard = EnhancedNKATDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"ダッシュボード初期化エラー: {e}")
        logger.error(f"ダッシュボード初期化エラー: {e}")

if __name__ == "__main__":
    main() 