#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🖥️📊 NKAT GPU/CPU リアルタイム監視ダッシュボード
Streamlitベースのシステム監視とNKAT解析統合インターフェース

Author: NKAT Research Team
Date: 2025-01-24
Version: 1.0 - Streamlit GPU監視統合版

主要機能:
- GPU使用率・温度・メモリ使用量監視
- CPU使用率・メモリ使用量監視
- NKAT解析のリアルタイム実行
- プログレスバー表示
- ログ表示
- チェックポイント管理
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import time
import threading
import queue
import psutil
import GPUtil
import torch
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import logging

# NKAT解析モジュールのインポート
sys.path.append(str(Path(__file__).parent.parent))
from gpu.dirac_laplacian_analysis_gpu_recovery import (
    RecoveryGPUOperatorParameters,
    RecoveryGPUDiracLaplacianAnalyzer,
    setup_logger
)

# Streamlit設定
st.set_page_config(
    page_title="NKAT GPU/CPU 監視ダッシュボード",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff6b6b;
    }
    .gpu-card {
        border-left-color: #4ecdc4;
    }
    .cpu-card {
        border-left-color: #45b7d1;
    }
    .memory-card {
        border-left-color: #96ceb4;
    }
    .temp-card {
        border-left-color: #feca57;
    }
    .stProgress .st-bo {
        background-color: #e1e5e9;
    }
</style>
""", unsafe_allow_html=True)

class SystemMonitor:
    """システム監視クラス"""
    
    def __init__(self):
        self.logger = setup_logger('SystemMonitor')
        self.data_queue = queue.Queue(maxsize=1000)
        self.monitoring = False
        self.monitor_thread = None
        
    def get_gpu_info(self):
        """GPU情報の取得"""
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return None
            
            gpu = gpus[0]  # 最初のGPUを使用
            
            # PyTorchからの追加情報
            torch_info = {}
            if torch.cuda.is_available():
                torch_info = {
                    'name': torch.cuda.get_device_name(0),
                    'total_memory': torch.cuda.get_device_properties(0).total_memory / 1e9,
                    'allocated_memory': torch.cuda.memory_allocated(0) / 1e9,
                    'cached_memory': torch.cuda.memory_reserved(0) / 1e9,
                }
            
            return {
                'id': gpu.id,
                'name': gpu.name,
                'load': gpu.load * 100,  # 使用率（%）
                'memory_used': gpu.memoryUsed,  # MB
                'memory_total': gpu.memoryTotal,  # MB
                'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                'temperature': gpu.temperature,  # 摂氏
                'torch_info': torch_info,
                'timestamp': datetime.now()
            }
        except Exception as e:
            self.logger.error(f"GPU情報取得エラー: {e}")
            return None
    
    def get_cpu_info(self):
        """CPU情報の取得"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_freq = psutil.cpu_freq()
            cpu_temps = None
            
            # CPU温度の取得（可能な場合）
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
                'frequency_current': cpu_freq.current if cpu_freq else None,
                'frequency_max': cpu_freq.max if cpu_freq else None,
                'core_count': psutil.cpu_count(logical=False),
                'thread_count': psutil.cpu_count(logical=True),
                'temperatures': cpu_temps,
                'avg_temperature': np.mean(cpu_temps) if cpu_temps else None,
                'timestamp': datetime.now()
            }
        except Exception as e:
            self.logger.error(f"CPU情報取得エラー: {e}")
            return None
    
    def get_memory_info(self):
        """メモリ情報の取得"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return {
                'total': memory.total / 1e9,  # GB
                'available': memory.available / 1e9,  # GB
                'used': memory.used / 1e9,  # GB
                'percent': memory.percent,
                'swap_total': swap.total / 1e9,  # GB
                'swap_used': swap.used / 1e9,  # GB
                'swap_percent': swap.percent,
                'timestamp': datetime.now()
            }
        except Exception as e:
            self.logger.error(f"メモリ情報取得エラー: {e}")
            return None
    
    def get_disk_info(self):
        """ディスク情報の取得"""
        try:
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            return {
                'total': disk.total / 1e9,  # GB
                'used': disk.used / 1e9,  # GB
                'free': disk.free / 1e9,  # GB
                'percent': (disk.used / disk.total) * 100,
                'read_bytes': disk_io.read_bytes if disk_io else 0,
                'write_bytes': disk_io.write_bytes if disk_io else 0,
                'timestamp': datetime.now()
            }
        except Exception as e:
            self.logger.error(f"ディスク情報取得エラー: {e}")
            return None
    
    def monitor_loop(self):
        """監視ループ"""
        while self.monitoring:
            try:
                data = {
                    'gpu': self.get_gpu_info(),
                    'cpu': self.get_cpu_info(),
                    'memory': self.get_memory_info(),
                    'disk': self.get_disk_info(),
                    'timestamp': datetime.now()
                }
                
                if not self.data_queue.full():
                    self.data_queue.put(data)
                else:
                    # キューが満杯の場合、古いデータを削除
                    try:
                        self.data_queue.get_nowait()
                        self.data_queue.put(data)
                    except queue.Empty:
                        pass
                
                time.sleep(1)  # 1秒間隔で監視
                
            except Exception as e:
                self.logger.error(f"監視ループエラー: {e}")
                time.sleep(5)
    
    def start_monitoring(self):
        """監視開始"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
            self.monitor_thread.start()
            self.logger.info("システム監視開始")
    
    def stop_monitoring(self):
        """監視停止"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        self.logger.info("システム監視停止")
    
    def get_recent_data(self, seconds=60):
        """最近のデータを取得"""
        data_list = []
        temp_queue = queue.Queue()
        
        # キューからデータを取得
        while not self.data_queue.empty():
            try:
                data = self.data_queue.get_nowait()
                temp_queue.put(data)
                data_list.append(data)
            except queue.Empty:
                break
        
        # データを戻す
        while not temp_queue.empty():
            self.data_queue.put(temp_queue.get())
        
        # 指定秒数以内のデータのみ返す
        cutoff_time = datetime.now() - timedelta(seconds=seconds)
        recent_data = [d for d in data_list if d['timestamp'] > cutoff_time]
        
        return recent_data

class NKATAnalysisRunner:
    """NKAT解析実行クラス"""
    
    def __init__(self):
        self.logger = setup_logger('NKATAnalysisRunner')
        self.current_analysis = None
        self.analysis_thread = None
        self.progress_queue = queue.Queue()
        self.log_queue = queue.Queue()
        
    def run_analysis(self, params):
        """解析実行"""
        try:
            self.logger.info("NKAT解析開始")
            
            # プログレス更新
            self.progress_queue.put({"stage": "初期化", "progress": 0})
            
            analyzer = RecoveryGPUDiracLaplacianAnalyzer(params)
            
            self.progress_queue.put({"stage": "ディラック作用素構築", "progress": 25})
            
            results = analyzer.run_full_analysis_with_recovery()
            
            self.progress_queue.put({"stage": "完了", "progress": 100})
            
            self.logger.info("NKAT解析完了")
            return results
            
        except Exception as e:
            self.logger.error(f"NKAT解析エラー: {e}")
            self.progress_queue.put({"stage": "エラー", "progress": 0, "error": str(e)})
            return None
    
    def start_analysis_async(self, params):
        """非同期解析開始"""
        if self.analysis_thread and self.analysis_thread.is_alive():
            return False
        
        self.analysis_thread = threading.Thread(
            target=lambda: self.run_analysis(params), 
            daemon=True
        )
        self.analysis_thread.start()
        return True
    
    def get_progress(self):
        """進捗取得"""
        progress_data = []
        while not self.progress_queue.empty():
            try:
                progress_data.append(self.progress_queue.get_nowait())
            except queue.Empty:
                break
        return progress_data

# グローバル変数
if 'monitor' not in st.session_state:
    st.session_state.monitor = SystemMonitor()
    st.session_state.monitor.start_monitoring()

if 'analysis_runner' not in st.session_state:
    st.session_state.analysis_runner = NKATAnalysisRunner()

def create_gpu_chart(data_list):
    """GPU使用率チャート作成"""
    if not data_list:
        return go.Figure()
    
    timestamps = [d['timestamp'] for d in data_list if d['gpu']]
    gpu_loads = [d['gpu']['load'] for d in data_list if d['gpu']]
    gpu_temps = [d['gpu']['temperature'] for d in data_list if d['gpu']]
    memory_percents = [d['gpu']['memory_percent'] for d in data_list if d['gpu']]
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('GPU使用率 (%)', 'GPU温度 (°C)', 'GPUメモリ使用率 (%)'),
        vertical_spacing=0.1
    )
    
    # GPU使用率
    fig.add_trace(
        go.Scatter(x=timestamps, y=gpu_loads, name='GPU使用率', line=dict(color='#4ecdc4')),
        row=1, col=1
    )
    
    # GPU温度
    fig.add_trace(
        go.Scatter(x=timestamps, y=gpu_temps, name='GPU温度', line=dict(color='#feca57')),
        row=2, col=1
    )
    
    # GPUメモリ使用率
    fig.add_trace(
        go.Scatter(x=timestamps, y=memory_percents, name='GPUメモリ', line=dict(color='#ff6b6b')),
        row=3, col=1
    )
    
    fig.update_layout(height=600, showlegend=False)
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    
    return fig

def create_cpu_chart(data_list):
    """CPU使用率チャート作成"""
    if not data_list:
        return go.Figure()
    
    timestamps = [d['timestamp'] for d in data_list if d['cpu']]
    cpu_usage = [d['cpu']['usage_percent'] for d in data_list if d['cpu']]
    cpu_temps = [d['cpu']['avg_temperature'] for d in data_list if d['cpu'] and d['cpu']['avg_temperature']]
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('CPU使用率 (%)', 'CPU温度 (°C)'),
        vertical_spacing=0.15
    )
    
    # CPU使用率
    fig.add_trace(
        go.Scatter(x=timestamps, y=cpu_usage, name='CPU使用率', line=dict(color='#45b7d1')),
        row=1, col=1
    )
    
    # CPU温度
    if cpu_temps:
        temp_timestamps = [d['timestamp'] for d in data_list if d['cpu'] and d['cpu']['avg_temperature']]
        fig.add_trace(
            go.Scatter(x=temp_timestamps, y=cpu_temps, name='CPU温度', line=dict(color='#feca57')),
            row=2, col=1
        )
    
    fig.update_layout(height=400, showlegend=False)
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    
    return fig

def create_memory_chart(data_list):
    """メモリ使用量チャート作成"""
    if not data_list:
        return go.Figure()
    
    timestamps = [d['timestamp'] for d in data_list if d['memory']]
    memory_used = [d['memory']['used'] for d in data_list if d['memory']]
    memory_total = [d['memory']['total'] for d in data_list if d['memory']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps, y=memory_used, 
        name='使用中', fill='tonexty', 
        line=dict(color='#96ceb4')
    ))
    
    fig.add_trace(go.Scatter(
        x=timestamps, y=memory_total, 
        name='総容量', 
        line=dict(color='#ddd', dash='dash')
    ))
    
    fig.update_layout(
        title='メモリ使用量 (GB)',
        height=300,
        yaxis_title='メモリ (GB)'
    )
    
    return fig

def main():
    """メイン関数"""
    st.title("🖥️📊 NKAT GPU/CPU 監視ダッシュボード")
    st.markdown("---")
    
    # サイドバー
    st.sidebar.title("⚙️ 設定")
    
    # 監視設定
    st.sidebar.subheader("📊 監視設定")
    monitor_interval = st.sidebar.slider("更新間隔 (秒)", 1, 10, 2)
    data_history = st.sidebar.slider("データ履歴 (分)", 1, 60, 5)
    
    # NKAT解析設定
    st.sidebar.subheader("🚀 NKAT解析設定")
    dimension = st.sidebar.selectbox("次元", [3, 4, 5, 6], index=0)
    lattice_size = st.sidebar.selectbox("格子サイズ", [8, 16, 32], index=0)
    max_eigenvalues = st.sidebar.slider("最大固有値数", 10, 100, 20)
    
    # メインエリア
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 リアルタイム監視")
        
        # データ取得
        recent_data = st.session_state.monitor.get_recent_data(data_history * 60)
        
        if recent_data:
            latest_data = recent_data[-1]
            
            # メトリクス表示
            metric_cols = st.columns(4)
            
            with metric_cols[0]:
                if latest_data['gpu']:
                    st.markdown('<div class="metric-card gpu-card">', unsafe_allow_html=True)
                    st.metric(
                        "🎮 GPU使用率", 
                        f"{latest_data['gpu']['load']:.1f}%",
                        delta=None
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
            
            with metric_cols[1]:
                if latest_data['cpu']:
                    st.markdown('<div class="metric-card cpu-card">', unsafe_allow_html=True)
                    st.metric(
                        "💻 CPU使用率", 
                        f"{latest_data['cpu']['usage_percent']:.1f}%",
                        delta=None
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
            
            with metric_cols[2]:
                if latest_data['memory']:
                    st.markdown('<div class="metric-card memory-card">', unsafe_allow_html=True)
                    st.metric(
                        "💾 メモリ使用率", 
                        f"{latest_data['memory']['percent']:.1f}%",
                        delta=None
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
            
            with metric_cols[3]:
                if latest_data['gpu'] and latest_data['gpu']['temperature']:
                    st.markdown('<div class="metric-card temp-card">', unsafe_allow_html=True)
                    st.metric(
                        "🌡️ GPU温度", 
                        f"{latest_data['gpu']['temperature']:.1f}°C",
                        delta=None
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # チャート表示
            tab1, tab2, tab3 = st.tabs(["🎮 GPU", "💻 CPU", "💾 メモリ"])
            
            with tab1:
                gpu_chart = create_gpu_chart(recent_data)
                st.plotly_chart(gpu_chart, use_container_width=True)
            
            with tab2:
                cpu_chart = create_cpu_chart(recent_data)
                st.plotly_chart(cpu_chart, use_container_width=True)
            
            with tab3:
                memory_chart = create_memory_chart(recent_data)
                st.plotly_chart(memory_chart, use_container_width=True)
        
        else:
            st.warning("監視データを取得中...")
    
    with col2:
        st.subheader("🚀 NKAT解析実行")
        
        # 解析パラメータ表示
        st.write("**解析パラメータ:**")
        st.write(f"- 次元: {dimension}")
        st.write(f"- 格子サイズ: {lattice_size}")
        st.write(f"- 最大固有値数: {max_eigenvalues}")
        
        # 解析実行ボタン
        if st.button("🚀 解析開始", type="primary"):
            params = RecoveryGPUOperatorParameters(
                dimension=dimension,
                lattice_size=lattice_size,
                theta=0.01,
                kappa=0.05,
                mass=0.1,
                coupling=1.0,
                recovery_enabled=True,
                checkpoint_interval=60,
                auto_save=True,
                max_eigenvalues=max_eigenvalues,
                log_level=logging.INFO
            )
            
            if st.session_state.analysis_runner.start_analysis_async(params):
                st.success("解析を開始しました！")
            else:
                st.warning("解析が既に実行中です")
        
        # 進捗表示
        progress_data = st.session_state.analysis_runner.get_progress()
        if progress_data:
            latest_progress = progress_data[-1]
            st.write("**解析進捗:**")
            st.write(f"ステージ: {latest_progress['stage']}")
            st.progress(latest_progress['progress'] / 100)
            
            if 'error' in latest_progress:
                st.error(f"エラー: {latest_progress['error']}")
        
        # チェックポイント一覧
        st.subheader("💾 チェックポイント")
        checkpoint_dir = Path("results/checkpoints")
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("*/"))
            if checkpoints:
                st.write(f"**保存済み: {len(checkpoints)}個**")
                for cp in checkpoints[-3:]:  # 最新3個を表示
                    st.write(f"- {cp.name}")
            else:
                st.write("チェックポイントなし")
        
        # ログファイル一覧
        st.subheader("📝 ログファイル")
        log_dir = Path("results/logs")
        if log_dir.exists():
            log_files = list(log_dir.glob("*.log"))
            if log_files:
                st.write(f"**ログファイル: {len(log_files)}個**")
                latest_log = sorted(log_files)[-1]
                if st.button("📖 最新ログ表示"):
                    try:
                        with open(latest_log, 'r', encoding='utf-8') as f:
                            log_content = f.read()
                        st.text_area("ログ内容", log_content[-2000:], height=200)
                    except Exception as e:
                        st.error(f"ログ読み込みエラー: {e}")
            else:
                st.write("ログファイルなし")
    
    # 自動更新
    time.sleep(monitor_interval)
    st.rerun()

if __name__ == "__main__":
    main() 