#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT リーマン予想解析 Streamlit ダッシュボード
Real-time GPU Monitoring and Riemann Analysis Dashboard

RTX3080 GPU監視とリーマン予想解析の進行状況をリアルタイムで表示
- GPU使用率・温度・メモリ監視
- リーマン予想解析の進行状況
- 電源断リカバリー状況
- 解析結果の可視化
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
import threading
import queue
import logging

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.riemann_analysis.nkat_riemann_analyzer import (
        NKATRiemannConfig, RiemannZetaAnalyzer, GPUMonitor, RecoveryManager
    )
    import torch
    import psutil
    import GPUtil
except ImportError as e:
    st.error(f"必要なモジュールのインポートに失敗: {e}")
    st.stop()

# ページ設定
st.set_page_config(
    page_title="NKAT リーマン予想解析ダッシュボード",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .status-good {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-danger {
        color: #dc3545;
        font-weight: bold;
    }
    .analysis-progress {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        color: white;
        padding: 0.5rem;
        border-radius: 0.25rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class DashboardState:
    """ダッシュボードの状態管理"""
    
    def __init__(self):
        self.gpu_monitor = GPUMonitor()
        self.recovery_manager = RecoveryManager()
        self.analyzer = None
        self.analysis_running = False
        self.analysis_thread = None
        self.data_queue = queue.Queue()
        
        # データ履歴
        self.gpu_history = []
        self.analysis_history = []
        self.max_history = 100
        
        # 設定
        self.config = NKATRiemannConfig()
        
    def initialize_analyzer(self):
        """解析器の初期化"""
        if self.analyzer is None:
            self.analyzer = RiemannZetaAnalyzer(self.config)
    
    def get_gpu_status(self):
        """GPU状態の取得"""
        status = self.gpu_monitor.get_gpu_status()
        status['timestamp'] = datetime.now()
        
        # 履歴に追加
        self.gpu_history.append(status)
        if len(self.gpu_history) > self.max_history:
            self.gpu_history.pop(0)
        
        return status
    
    def get_system_status(self):
        """システム状態の取得"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / 1024**3,
            'disk_percent': disk.percent,
            'disk_free_gb': disk.free / 1024**3,
            'timestamp': datetime.now()
        }
    
    def start_analysis(self, max_dimension: int):
        """解析の開始"""
        if self.analysis_running:
            return False
        
        self.initialize_analyzer()
        self.analysis_running = True
        
        def analysis_worker():
            try:
                results = self.analyzer.analyze_riemann_hypothesis(max_dimension)
                self.data_queue.put(('analysis_complete', results))
            except Exception as e:
                self.data_queue.put(('analysis_error', str(e)))
            finally:
                self.analysis_running = False
        
        self.analysis_thread = threading.Thread(target=analysis_worker)
        self.analysis_thread.start()
        
        return True
    
    def stop_analysis(self):
        """解析の停止"""
        self.analysis_running = False
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=5)

# セッション状態の初期化
if 'dashboard_state' not in st.session_state:
    st.session_state.dashboard_state = DashboardState()

dashboard_state = st.session_state.dashboard_state

def main():
    """メインダッシュボード"""
    
    # ヘッダー
    st.markdown('<h1 class="main-header">🌌 NKAT リーマン予想解析ダッシュボード</h1>', unsafe_allow_html=True)
    st.markdown("**非可換コルモゴロフアーノルド表現理論によるリーマン予想解析**")
    st.markdown("---")
    
    # サイドバー設定
    with st.sidebar:
        st.header("⚙️ 設定")
        
        # 解析設定
        st.subheader("解析パラメータ")
        max_dimension = st.slider("最大次元", 15, 100, 50, 5)
        critical_dimension = st.slider("臨界次元", 10, 30, 15)
        
        # GPU設定
        st.subheader("GPU設定")
        gpu_memory_fraction = st.slider("GPU メモリ使用率", 0.5, 1.0, 0.95, 0.05)
        enable_mixed_precision = st.checkbox("混合精度", True)
        
        # 更新設定
        st.subheader("表示設定")
        auto_refresh = st.checkbox("自動更新", True)
        refresh_interval = st.slider("更新間隔 (秒)", 1, 10, 3)
        
        # 解析制御
        st.subheader("解析制御")
        if st.button("🚀 解析開始", disabled=dashboard_state.analysis_running):
            dashboard_state.config.max_dimension = max_dimension
            dashboard_state.config.critical_dimension = critical_dimension
            dashboard_state.config.gpu_memory_fraction = gpu_memory_fraction
            dashboard_state.config.enable_mixed_precision = enable_mixed_precision
            
            if dashboard_state.start_analysis(max_dimension):
                st.success("解析を開始しました")
            else:
                st.error("解析の開始に失敗しました")
        
        if st.button("⏹️ 解析停止", disabled=not dashboard_state.analysis_running):
            dashboard_state.stop_analysis()
            st.info("解析を停止しました")
    
    # メインコンテンツ
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # GPU監視セクション
        st.header("🎮 RTX3080 GPU 監視")
        
        gpu_status = dashboard_state.get_gpu_status()
        
        if gpu_status['available']:
            # GPU メトリクス
            gpu_cols = st.columns(4)
            
            with gpu_cols[0]:
                utilization = gpu_status.get('gpu_utilization', 0)
                if utilization is None:
                    utilization = 0
                color = "normal" if utilization < 80 else "inverse"
                st.metric(
                    "GPU 使用率",
                    f"{utilization:.1f}%",
                    delta=None
                )
            
            with gpu_cols[1]:
                temp = gpu_status.get('temperature', 0)
                if temp is None:
                    temp = 0
                temp_status = "🟢" if temp < 70 else "🟡" if temp < 80 else "🔴"
                st.metric(
                    "温度",
                    f"{temp}°C {temp_status}",
                    delta=None
                )
            
            with gpu_cols[2]:
                memory_util = gpu_status.get('memory_utilization', 0)
                if memory_util is None:
                    memory_util = 0
                st.metric(
                    "VRAM 使用率",
                    f"{memory_util:.1f}%",
                    delta=None
                )
            
            with gpu_cols[3]:
                power_draw = gpu_status.get('power_draw', 0)
                power_limit = gpu_status.get('power_limit', 300)
                
                # None値の処理
                if power_draw is None:
                    power_draw = 0
                if power_limit is None:
                    power_limit = 300
                
                power_percent = (power_draw / power_limit * 100) if power_limit > 0 else 0
                st.metric(
                    "電力使用",
                    f"{power_draw:.0f}W ({power_percent:.0f}%)",
                    delta=None
                )
            
            # GPU履歴グラフ
            if len(dashboard_state.gpu_history) > 1:
                fig_gpu = create_gpu_history_chart(dashboard_state.gpu_history)
                st.plotly_chart(fig_gpu, use_container_width=True)
        
        else:
            st.error("❌ GPU が利用できません")
    
    with col2:
        # システム監視
        st.header("💻 システム状態")
        
        system_status = dashboard_state.get_system_status()
        
        # None値の処理
        cpu_percent = system_status.get('cpu_percent', 0)
        memory_percent = system_status.get('memory_percent', 0)
        memory_available_gb = system_status.get('memory_available_gb', 0)
        disk_percent = system_status.get('disk_percent', 0)
        
        if cpu_percent is None:
            cpu_percent = 0
        if memory_percent is None:
            memory_percent = 0
        if memory_available_gb is None:
            memory_available_gb = 0
        if disk_percent is None:
            disk_percent = 0
        
        st.metric("CPU 使用率", f"{cpu_percent:.1f}%")
        st.metric("メモリ使用率", f"{memory_percent:.1f}%")
        st.metric("利用可能メモリ", f"{memory_available_gb:.1f} GB")
        st.metric("ディスク使用率", f"{disk_percent:.1f}%")
        
        # 解析状況
        st.header("🔬 解析状況")
        
        if dashboard_state.analysis_running:
            st.markdown('<div class="analysis-progress">🔄 解析実行中...</div>', unsafe_allow_html=True)
            st.progress(0.5)  # 進行状況は簡略化
        else:
            st.info("解析待機中")
        
        # リカバリー状況
        st.header("💾 リカバリー状況")
        
        checkpoint_state, is_valid = dashboard_state.recovery_manager.load_checkpoint()
        
        if checkpoint_state:
            st.success("✅ チェックポイント利用可能")
            if 'analysis_timestamp' in checkpoint_state:
                timestamp = checkpoint_state['analysis_timestamp']
                st.text(f"最終保存: {timestamp}")
        else:
            st.info("チェックポイントなし")
    
    # 解析結果セクション
    st.header("📊 リーマン予想解析結果")
    
    # キューからデータを取得
    try:
        while not dashboard_state.data_queue.empty():
            event_type, data = dashboard_state.data_queue.get_nowait()
            
            if event_type == 'analysis_complete':
                st.success("🎉 解析完了!")
                display_analysis_results(data)
            elif event_type == 'analysis_error':
                st.error(f"❌ 解析エラー: {data}")
    except queue.Empty:
        pass
    
    # 既存の結果を表示
    if checkpoint_state and 'final_assessment' in checkpoint_state:
        display_analysis_results(checkpoint_state)
    
    # 自動更新
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

def create_gpu_history_chart(gpu_history):
    """GPU履歴チャートの作成"""
    if not gpu_history:
        return go.Figure()
    
    df = pd.DataFrame(gpu_history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('GPU使用率', '温度', 'VRAM使用率', '電力使用'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # GPU使用率
    if 'gpu_utilization' in df.columns:
        gpu_util_clean = df['gpu_utilization'].fillna(0)
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=gpu_util_clean,
                      name='GPU使用率', line=dict(color='blue')),
            row=1, col=1
        )
    
    # 温度
    if 'temperature' in df.columns:
        temp_clean = df['temperature'].fillna(0)
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=temp_clean,
                      name='温度', line=dict(color='red')),
            row=1, col=2
        )
    
    # VRAM使用率
    if 'memory_utilization' in df.columns:
        memory_util_clean = df['memory_utilization'].fillna(0)
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=memory_util_clean,
                      name='VRAM使用率', line=dict(color='green')),
            row=2, col=1
        )
    
    # 電力使用
    if 'power_draw' in df.columns:
        power_draw_clean = df['power_draw'].fillna(0)
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=power_draw_clean,
                      name='電力使用', line=dict(color='orange')),
            row=2, col=2
        )
    
    fig.update_layout(
        title="RTX3080 GPU 履歴",
        height=400,
        showlegend=False
    )
    
    return fig

def display_analysis_results(results):
    """解析結果の表示"""
    if 'final_assessment' not in results:
        st.warning("解析結果が不完全です")
        return
    
    assessment = results['final_assessment']
    
    # 総合評価
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("総合評価", assessment['assessment'])
    
    with col2:
        st.metric("信頼度", assessment['confidence'])
    
    with col3:
        st.metric("総合スコア", f"{assessment['overall_score']:.4f}")
    
    # 詳細スコア
    if 'component_scores' in assessment:
        st.subheader("詳細スコア")
        
        scores = assessment['component_scores']
        score_df = pd.DataFrame([
            {'指標': 'NKAT-ゼータ対応', 'スコア': scores.get('nkat_zeta_correspondence', 0)},
            {'指標': 'ゼロ点検証', 'スコア': scores.get('zero_verification', 0)},
            {'指標': '臨界線選好', 'スコア': scores.get('critical_line_preference', 0)},
            {'指標': '収束性', 'スコア': scores.get('convergence', 0)},
            {'指標': '超収束一致', 'スコア': scores.get('superconvergence_agreement', 0)}
        ])
        
        fig_scores = px.bar(score_df, x='指標', y='スコア', 
                           title="各指標のスコア",
                           color='スコア',
                           color_continuous_scale='viridis')
        fig_scores.update_layout(height=400)
        st.plotly_chart(fig_scores, use_container_width=True)
    
    # 次元別解析結果
    if 'dimensions_analyzed' in results and 'convergence_data' in results:
        st.subheader("次元別収束性")
        
        dims = results['dimensions_analyzed']
        convergence = results['convergence_data']
        
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(
            x=dims, y=convergence,
            mode='lines+markers',
            name='収束スコア',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))
        
        fig_conv.update_layout(
            title="次元別収束性",
            xaxis_title="次元",
            yaxis_title="収束スコア",
            height=400
        )
        
        st.plotly_chart(fig_conv, use_container_width=True)
    
    # 実行情報
    if 'execution_time' in results:
        st.info(f"⏱️ 実行時間: {results['execution_time']:.2f}秒")
    
    if 'analysis_timestamp' in results:
        st.info(f"📅 解析日時: {results['analysis_timestamp']}")

def create_real_time_monitor():
    """リアルタイム監視ページ"""
    st.header("📡 リアルタイム監視")
    
    # プレースホルダー
    gpu_placeholder = st.empty()
    chart_placeholder = st.empty()
    
    # 監視ループ
    for i in range(60):  # 1分間監視
        gpu_status = dashboard_state.get_gpu_status()
        
        with gpu_placeholder.container():
            if gpu_status['available']:
                cols = st.columns(4)
                
                with cols[0]:
                    gpu_util = gpu_status.get('gpu_utilization', 0)
                    if gpu_util is None:
                        gpu_util = 0
                    st.metric("GPU使用率", f"{gpu_util:.1f}%")
                
                with cols[1]:
                    temp = gpu_status.get('temperature', 0)
                    if temp is None:
                        temp = 0
                    st.metric("温度", f"{temp}°C")
                
                with cols[2]:
                    memory_util = gpu_status.get('memory_utilization', 0)
                    if memory_util is None:
                        memory_util = 0
                    st.metric("VRAM", f"{memory_util:.1f}%")
                
                with cols[3]:
                    power_draw = gpu_status.get('power_draw', 0)
                    if power_draw is None:
                        power_draw = 0
                    st.metric("電力", f"{power_draw:.0f}W")
        
        # チャート更新
        if len(dashboard_state.gpu_history) > 1:
            fig = create_gpu_history_chart(dashboard_state.gpu_history[-20:])  # 最新20件
            chart_placeholder.plotly_chart(fig, use_container_width=True)
        
        time.sleep(1)

if __name__ == "__main__":
    main() 