#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT v11 包括的リカバリーダッシュボード
電源断対応・自動復旧・リアルタイム監視システム

作成者: NKAT Research Team
作成日: 2025年5月26日
バージョン: v11.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
import time
import psutil
import subprocess
import threading
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# GPU監視用
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

class SystemState:
    """システム状態管理クラス"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.last_backup = None
        self.running_processes = {}
        self.checkpoints = []
        self.system_metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_usage': [],
            'timestamps': []
        }
        
    def update_metrics(self):
        """システムメトリクスを更新"""
        current_time = datetime.now()
        
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # メモリ使用率
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # GPU使用率
        gpu_percent = 0
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_percent = gpus[0].load * 100
            except:
                pass
        
        # メトリクス追加
        self.system_metrics['cpu_usage'].append(cpu_percent)
        self.system_metrics['memory_usage'].append(memory_percent)
        self.system_metrics['gpu_usage'].append(gpu_percent)
        self.system_metrics['timestamps'].append(current_time)
        
        # 最新100件のみ保持
        if len(self.system_metrics['timestamps']) > 100:
            for key in self.system_metrics:
                self.system_metrics[key] = self.system_metrics[key][-100:]
    
    def get_running_processes(self):
        """実行中のPythonプロセスを取得"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
            try:
                if 'python' in proc.info['name'].lower():
                    processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cmdline': ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else '',
                        'cpu_percent': proc.info['cpu_percent'],
                        'memory_percent': proc.info['memory_percent']
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return processes
    
    def create_checkpoint(self):
        """チェックポイントを作成"""
        checkpoint_dir = "recovery_data"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{timestamp}.json")
        
        checkpoint_data = {
            'timestamp': timestamp,
            'system_metrics': self.system_metrics,
            'running_processes': self.get_running_processes(),
            'working_directory': os.getcwd(),
            'environment_variables': dict(os.environ)
        }
        
        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.checkpoints.append(checkpoint_file)
            self.last_backup = datetime.now()
            return True, checkpoint_file
        except Exception as e:
            return False, str(e)

# グローバル状態
if 'system_state' not in st.session_state:
    st.session_state.system_state = SystemState()

def load_results_data():
    """結果データを読み込み"""
    results = {}
    
    # 高精度リーマン結果
    if os.path.exists('high_precision_riemann_results.json'):
        with open('high_precision_riemann_results.json', 'r') as f:
            results['riemann'] = json.load(f)
    
    # その他の結果ファイル
    result_files = [
        'ultimate_mastery_riemann_results.json',
        'extended_riemann_results.json',
        'improved_riemann_results.json'
    ]
    
    for file in result_files:
        if os.path.exists(file):
            try:
                with open(file, 'r') as f:
                    key = file.replace('.json', '').replace('_results', '')
                    results[key] = json.load(f)
            except:
                continue
    
    return results

def create_system_monitoring_dashboard():
    """システム監視ダッシュボードを作成"""
    st.header("🖥️ システム監視")
    
    # メトリクス更新
    st.session_state.system_state.update_metrics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cpu_usage = psutil.cpu_percent()
        st.metric("CPU使用率", f"{cpu_usage:.1f}%")
    
    with col2:
        memory = psutil.virtual_memory()
        st.metric("メモリ使用率", f"{memory.percent:.1f}%")
    
    with col3:
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_usage = gpus[0].load * 100
                    st.metric("GPU使用率", f"{gpu_usage:.1f}%")
                else:
                    st.metric("GPU使用率", "N/A")
            except:
                st.metric("GPU使用率", "エラー")
        else:
            st.metric("GPU使用率", "未対応")
    
    with col4:
        uptime = datetime.now() - st.session_state.system_state.start_time
        st.metric("稼働時間", f"{uptime.seconds//3600}h {(uptime.seconds//60)%60}m")
    
    # リアルタイムグラフ
    if st.session_state.system_state.system_metrics['timestamps']:
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('CPU使用率', 'メモリ使用率', 'GPU使用率'),
            vertical_spacing=0.1
        )
        
        timestamps = st.session_state.system_state.system_metrics['timestamps']
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=st.session_state.system_state.system_metrics['cpu_usage'],
                name='CPU',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=st.session_state.system_state.system_metrics['memory_usage'],
                name='Memory',
                line=dict(color='green')
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=st.session_state.system_state.system_metrics['gpu_usage'],
                name='GPU',
                line=dict(color='red')
            ),
            row=3, col=1
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

def create_process_monitoring():
    """プロセス監視セクション"""
    st.header("🔄 プロセス監視")
    
    processes = st.session_state.system_state.get_running_processes()
    
    if processes:
        df = pd.DataFrame(processes)
        st.dataframe(df, use_container_width=True)
        
        # プロセス制御
        st.subheader("プロセス制御")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 プロセス更新"):
                st.rerun()
        
        with col2:
            if st.button("⚠️ 全Pythonプロセス終了"):
                for proc in processes:
                    try:
                        psutil.Process(proc['pid']).terminate()
                    except:
                        continue
                st.success("プロセス終了要求を送信しました")
                time.sleep(2)
                st.rerun()
    else:
        st.info("実行中のPythonプロセスはありません")

def create_checkpoint_management():
    """チェックポイント管理セクション"""
    st.header("💾 チェックポイント管理")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📸 チェックポイント作成"):
            success, result = st.session_state.system_state.create_checkpoint()
            if success:
                st.success(f"チェックポイント作成: {result}")
            else:
                st.error(f"エラー: {result}")
    
    with col2:
        if st.session_state.system_state.last_backup:
            last_backup_str = st.session_state.system_state.last_backup.strftime("%H:%M:%S")
            st.info(f"最終バックアップ: {last_backup_str}")
        else:
            st.warning("バックアップなし")
    
    with col3:
        checkpoint_count = len(st.session_state.system_state.checkpoints)
        st.metric("チェックポイント数", checkpoint_count)
    
    # チェックポイント一覧
    if st.session_state.system_state.checkpoints:
        st.subheader("チェックポイント一覧")
        for checkpoint in st.session_state.system_state.checkpoints[-10:]:  # 最新10件
            st.text(f"📁 {os.path.basename(checkpoint)}")

def create_results_analysis():
    """結果分析セクション"""
    st.header("📊 結果分析")
    
    results = load_results_data()
    
    if not results:
        st.warning("分析可能な結果データがありません")
        return
    
    # タブで分類
    tabs = st.tabs(["収束分析", "スペクトル解析", "統計評価", "詳細データ"])
    
    with tabs[0]:  # 収束分析
        st.subheader("収束分析")
        
        if 'riemann' in results:
            riemann_data = results['riemann']
            
            # 収束値の表示
            if 'overall_statistics' in riemann_data:
                stats = riemann_data['overall_statistics']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("平均収束度", f"{stats.get('mean_convergence', 0):.6f}")
                with col2:
                    st.metric("標準偏差", f"{stats.get('std_convergence', 0):.6f}")
                with col3:
                    success_rate = stats.get('success_rate', 0) * 100
                    st.metric("成功率", f"{success_rate:.1f}%")
                
                # 収束度の評価
                mean_conv = stats.get('mean_convergence', 0)
                if mean_conv > 0.495:
                    st.success(f"🎉 優秀な収束結果: {mean_conv:.6f}")
                elif mean_conv > 0.49:
                    st.info(f"✅ 良好な収束結果: {mean_conv:.6f}")
                else:
                    st.warning(f"⚠️ 改善の余地あり: {mean_conv:.6f}")
    
    with tabs[1]:  # スペクトル解析
        st.subheader("スペクトル解析")
        
        if 'riemann' in results and 'spectral_dimensions_all' in results['riemann']:
            spectral_data = results['riemann']['spectral_dimensions_all']
            
            if spectral_data:
                # スペクトル次元の可視化
                fig = go.Figure()
                
                for i, spectrum in enumerate(spectral_data):
                    fig.add_trace(go.Scatter(
                        y=spectrum,
                        mode='lines+markers',
                        name=f'スペクトル {i+1}'
                    ))
                
                fig.update_layout(
                    title="スペクトル次元分析",
                    xaxis_title="γ値インデックス",
                    yaxis_title="スペクトル次元"
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:  # 統計評価
        st.subheader("統計評価")
        
        # 全結果の統計比較
        comparison_data = []
        
        for key, data in results.items():
            if isinstance(data, dict) and 'overall_statistics' in data:
                stats = data['overall_statistics']
                comparison_data.append({
                    'データセット': key,
                    '平均収束度': stats.get('mean_convergence', 0),
                    '標準偏差': stats.get('std_convergence', 0),
                    '最小値': stats.get('min_convergence', 0),
                    '最大値': stats.get('max_convergence', 0)
                })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True)
            
            # 比較グラフ
            fig = px.bar(df, x='データセット', y='平均収束度', 
                        title="データセット別収束度比較")
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:  # 詳細データ
        st.subheader("詳細データ")
        
        selected_dataset = st.selectbox("データセット選択", list(results.keys()))
        
        if selected_dataset:
            st.json(results[selected_dataset])

def main():
    """メイン関数"""
    st.set_page_config(
        page_title="NKAT v11 包括的リカバリーダッシュボード",
        page_icon="🔄",
        layout="wide"
    )
    
    st.title("🔄 NKAT v11 包括的リカバリーダッシュボード")
    st.markdown("**電源断対応・自動復旧・リアルタイム監視システム**")
    
    # サイドバー
    st.sidebar.title("🎛️ 制御パネル")
    
    # 自動更新設定
    auto_refresh = st.sidebar.checkbox("自動更新 (30秒)", value=True)
    if auto_refresh:
        time.sleep(30)
        # Streamlit バージョン互換性対応
        try:
            st.rerun()
        except AttributeError:
            st.experimental_rerun()
    
    # 緊急停止ボタン
    if st.sidebar.button("🚨 緊急停止", type="primary"):
        st.sidebar.error("緊急停止が実行されました")
        # チェックポイント作成
        st.session_state.system_state.create_checkpoint()
        st.stop()
    
    # メインコンテンツ
    tab1, tab2, tab3, tab4 = st.tabs([
        "🖥️ システム監視", 
        "🔄 プロセス管理", 
        "💾 チェックポイント", 
        "📊 結果分析"
    ])
    
    with tab1:
        create_system_monitoring_dashboard()
    
    with tab2:
        create_process_monitoring()
    
    with tab3:
        create_checkpoint_management()
    
    with tab4:
        create_results_analysis()
    
    # フッター
    st.markdown("---")
    st.markdown(
        "**NKAT v11 Research System** | "
        f"起動時刻: {st.session_state.system_state.start_time.strftime('%Y-%m-%d %H:%M:%S')} | "
        "🔄 自動リカバリー対応"
    )

if __name__ == "__main__":
    main() 