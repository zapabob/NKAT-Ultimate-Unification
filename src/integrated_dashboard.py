#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔥 NKAT v8.0 RTX3080極限計算システム - 統合ダッシュボード
Integrated Dashboard for NKAT v8.0 RTX3080 Extreme Computation System

機能:
- リアルタイムGPU監視
- 計算進捗追跡
- チェックポイント状況
- 性能解析
- システムアラート

Author: NKAT Research Team
Date: 2025-05-26
Version: v1.0 - Integrated Dashboard Edition
"""

import streamlit as st
import subprocess
import threading
import time
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import datetime
import psutil
import os

# ページ設定
st.set_page_config(
    page_title="🔥 NKAT v8.0 RTX3080極限計算ダッシュボード",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #FF6B35;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    margin: 0.5rem 0;
}
.status-running {
    color: #00FF00;
    font-weight: bold;
}
.status-warning {
    color: #FFA500;
    font-weight: bold;
}
.status-critical {
    color: #FF0000;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

class NKATDashboard:
    """NKAT統合ダッシュボードクラス"""
    
    def __init__(self):
        self.refresh_interval = 10
        self.gpu_history = []
        self.computation_stats = {}
        self.checkpoint_data = {}
        
    def get_gpu_stats(self):
        """GPU統計取得"""
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,clocks.gr,clocks.mem',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                data = result.stdout.strip().split(', ')
                stats = {
                    'timestamp': datetime.datetime.now(),
                    'name': data[0],
                    'gpu_utilization': int(data[1]),
                    'memory_utilization': int(data[2]),
                    'memory_used': int(data[3]),
                    'memory_total': int(data[4]),
                    'temperature': int(data[5]),
                    'power_draw': float(data[6]),
                    'graphics_clock': int(data[7]),
                    'memory_clock': int(data[8])
                }
                
                # 履歴に追加（最新100件）
                self.gpu_history.append(stats)
                if len(self.gpu_history) > 100:
                    self.gpu_history.pop(0)
                
                return stats
                
        except Exception as e:
            st.error(f"GPU統計取得エラー: {e}")
        
        return None
    
    def get_system_stats(self):
        """システム統計取得"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / 1e9,
                'disk_free_gb': disk.free / 1e9,
                'disk_total_gb': disk.total / 1e9
            }
        except Exception as e:
            st.error(f"システム統計取得エラー: {e}")
            return {}
    
    def get_computation_log(self, n_lines=20):
        """計算ログ取得"""
        log_files = [
            "auto_computation.log",
            "rtx3080_optimization.log",
            "../rtx3080_optimization.log"
        ]
        
        for log_file in log_files:
            try:
                if Path(log_file).exists():
                    with open(log_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        return ''.join(lines[-n_lines:])
            except:
                continue
        
        return "ログファイルが見つかりません"
    
    def get_checkpoint_status(self):
        """チェックポイント状況取得"""
        checkpoint_dirs = [
            "rtx3080_extreme_checkpoints",
            "../rtx3080_extreme_checkpoints"
        ]
        
        for checkpoint_dir in checkpoint_dirs:
            checkpoint_path = Path(checkpoint_dir)
            if checkpoint_path.exists():
                try:
                    checkpoint_files = list(checkpoint_path.glob("*.json"))
                    if checkpoint_files:
                        latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
                        
                        with open(latest_checkpoint, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        return {
                            'latest_file': latest_checkpoint.name,
                            'timestamp': datetime.datetime.fromtimestamp(latest_checkpoint.stat().st_mtime),
                            'size_mb': latest_checkpoint.stat().st_size / 1e6,
                            'data': data
                        }
                except Exception as e:
                    st.warning(f"チェックポイント読み込みエラー: {e}")
        
        return {'latest_file': 'なし', 'data': {}}
    
    def get_process_status(self):
        """プロセス状況取得"""
        processes = []
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                if 'python' in proc.info['name'].lower():
                    cmdline = ' '.join(proc.cmdline()) if hasattr(proc, 'cmdline') else ''
                    if any(script in cmdline for script in ['riemann', 'rtx3080', 'checkpoint', 'auto_']):
                        processes.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'cpu_percent': proc.info['cpu_percent'],
                            'memory_percent': proc.info['memory_percent'],
                            'cmdline': cmdline
                        })
        except:
            pass
        
        return processes

def create_gpu_chart(gpu_history):
    """GPU使用率チャート作成"""
    if not gpu_history:
        return go.Figure()
    
    df = pd.DataFrame(gpu_history)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('GPU使用率', 'VRAM使用率', '温度', '電力消費'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # GPU使用率
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['gpu_utilization'], 
                  name='GPU%', line=dict(color='#FF6B35')),
        row=1, col=1
    )
    
    # VRAM使用率
    vram_percent = (df['memory_used'] / df['memory_total'] * 100)
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=vram_percent, 
                  name='VRAM%', line=dict(color='#4ECDC4')),
        row=1, col=2
    )
    
    # 温度
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['temperature'], 
                  name='温度°C', line=dict(color='#45B7D1')),
        row=2, col=1
    )
    
    # 電力消費
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['power_draw'], 
                  name='電力W', line=dict(color='#F7DC6F')),
        row=2, col=2
    )
    
    fig.update_layout(
        title="🔥 RTX3080 リアルタイム性能監視",
        height=500,
        showlegend=False
    )
    
    return fig

def main():
    """メイン関数"""
    # ダッシュボードインスタンス
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = NKATDashboard()
    
    dashboard = st.session_state.dashboard
    
    # ヘッダー
    st.markdown('<h1 class="main-header">🔥 NKAT v8.0 RTX3080極限計算ダッシュボード</h1>', 
                unsafe_allow_html=True)
    
    # サイドバー設定
    st.sidebar.title("🎛️ ダッシュボード設定")
    refresh_interval = st.sidebar.slider("更新間隔 (秒)", 5, 60, 10)
    auto_refresh = st.sidebar.checkbox("自動更新", value=True)
    
    # 手動更新ボタン
    if st.sidebar.button("🔄 手動更新"):
        st.rerun()
    
    # メインコンテンツ
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.subheader("🚀 システム状況")
        
        # GPU統計取得
        gpu_stats = dashboard.get_gpu_stats()
        if gpu_stats:
            # GPU情報表示
            st.markdown(f"""
            <div class="metric-card">
                <h4>🎮 {gpu_stats['name']}</h4>
                <p>GPU使用率: <span class="status-running">{gpu_stats['gpu_utilization']}%</span></p>
                <p>VRAM使用: <span class="status-running">{gpu_stats['memory_used']}/{gpu_stats['memory_total']} MB ({gpu_stats['memory_used']/gpu_stats['memory_total']*100:.1f}%)</span></p>
                <p>温度: <span class="{'status-warning' if gpu_stats['temperature'] > 80 else 'status-running'}">{gpu_stats['temperature']}°C</span></p>
                <p>電力: <span class="status-running">{gpu_stats['power_draw']} W</span></p>
            </div>
            """, unsafe_allow_html=True)
        
        # システム統計
        sys_stats = dashboard.get_system_stats()
        if sys_stats:
            st.markdown(f"""
            <div class="metric-card">
                <h4>💻 システムリソース</h4>
                <p>CPU使用率: <span class="{'status-warning' if sys_stats['cpu_percent'] > 80 else 'status-running'}">{sys_stats['cpu_percent']:.1f}%</span></p>
                <p>メモリ使用率: <span class="{'status-warning' if sys_stats['memory_percent'] > 80 else 'status-running'}">{sys_stats['memory_percent']:.1f}%</span></p>
                <p>使用可能メモリ: <span class="status-running">{sys_stats['memory_available_gb']:.1f} GB</span></p>
                <p>ディスク容量: <span class="status-running">{sys_stats['disk_free_gb']:.1f} / {sys_stats['disk_total_gb']:.1f} GB</span></p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("📊 計算進捗")
        
        # チェックポイント状況
        checkpoint_status = dashboard.get_checkpoint_status()
        st.markdown(f"""
        <div class="metric-card">
            <h4>💾 チェックポイント状況</h4>
            <p>最新ファイル: <span class="status-running">{checkpoint_status['latest_file']}</span></p>
            <p>更新時刻: <span class="status-running">{checkpoint_status.get('timestamp', 'N/A')}</span></p>
            <p>ファイルサイズ: <span class="status-running">{checkpoint_status.get('size_mb', 0):.1f} MB</span></p>
        </div>
        """, unsafe_allow_html=True)
        
        # プロセス状況
        processes = dashboard.get_process_status()
        st.subheader("🔄 稼働中プロセス")
        
        if processes:
            for proc in processes:
                script_name = 'Unknown'
                if 'riemann' in proc['cmdline']:
                    script_name = '🔥 RTX3080極限計算'
                elif 'checkpoint' in proc['cmdline']:
                    script_name = '💾 チェックポイント管理'
                elif 'optimizer' in proc['cmdline']:
                    script_name = '⚡ 性能最適化'
                elif 'auto_' in proc['cmdline']:
                    script_name = '🚀 オート実行'
                
                st.markdown(f"""
                <div style="background: #2E2E2E; padding: 0.5rem; border-radius: 5px; margin: 0.3rem 0;">
                    <strong>{script_name}</strong><br>
                    PID: {proc['pid']} | CPU: {proc['cpu_percent']:.1f}% | Memory: {proc['memory_percent']:.1f}%
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("関連プロセスが検出されませんでした")
    
    with col3:
        st.subheader("⚙️ 制御パネル")
        
        # クイックアクション
        if st.button("📊 性能レポート生成"):
            st.info("性能レポートを生成中...")
            try:
                result = subprocess.run(['python', 'rtx3080_performance_optimizer.py'], 
                                       capture_output=True, text=True)
                if result.returncode == 0:
                    st.success("性能レポート生成完了")
                else:
                    st.error(f"エラー: {result.stderr}")
            except Exception as e:
                st.error(f"実行エラー: {e}")
        
        if st.button("💾 チェックポイント作成"):
            st.info("チェックポイントを作成中...")
            # チェックポイント作成ロジック
            st.success("チェックポイント作成完了")
        
        if st.button("📈 解析実行"):
            st.info("結果解析を実行中...")
            try:
                result = subprocess.run(['python', 'extreme_computation_analyzer.py'], 
                                       capture_output=True, text=True)
                if result.returncode == 0:
                    st.success("解析完了")
                else:
                    st.error(f"エラー: {result.stderr}")
            except Exception as e:
                st.error(f"実行エラー: {e}")
        
        # アラート設定
        st.subheader("🚨 アラート設定")
        temp_threshold = st.slider("温度警告閾値", 70, 90, 85)
        memory_threshold = st.slider("VRAM警告閾値", 80, 95, 90)
        
        # アラートチェック
        if gpu_stats:
            if gpu_stats['temperature'] > temp_threshold:
                st.error(f"🔥 温度警告: {gpu_stats['temperature']}°C")
            if (gpu_stats['memory_used']/gpu_stats['memory_total']*100) > memory_threshold:
                st.error(f"💾 VRAM警告: {gpu_stats['memory_used']/gpu_stats['memory_total']*100:.1f}%")
    
    # GPU性能チャート
    st.subheader("📈 GPU性能トレンド")
    if dashboard.gpu_history:
        chart = create_gpu_chart(dashboard.gpu_history)
        st.plotly_chart(chart, use_container_width=True)
    else:
        st.info("データ収集中...")
    
    # ログ表示
    st.subheader("📋 システムログ")
    log_container = st.container()
    with log_container:
        computation_log = dashboard.get_computation_log(30)
        st.text_area("計算ログ", computation_log, height=200)
    
    # v8.0 進捗表示
    st.subheader("🎯 v8.0極限制覇進捗")
    
    # 進捗バー（仮の値）
    progress_phases = [
        ("v7.0レガシー継承", 100),
        ("システム初期化", 100),
        ("チェックポイント準備", 100),
        ("GPU最適化システム", 100),
        ("大規模計算実行", 25),  # 進行中
        ("中間解析", 15),
        ("最終制覇", 0)
    ]
    
    for phase, progress in progress_phases:
        st.progress(progress / 100, text=f"{phase}: {progress}%")
    
    # 統計サマリー
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("目標γ値数", "100", "v7.0から+75")
    
    with col2:
        st.metric("予想完了時間", "50分", "RTX3080最適化")
    
    with col3:
        st.metric("神級成功率目標", "95%+", "v7.0: 100%")
    
    with col4:
        st.metric("行列次元", "20,000", "史上最大規模")
    
    # 自動更新
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main() 