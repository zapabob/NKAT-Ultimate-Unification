#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 NKAT Streamlit ダッシュボード - バージョン対応監視システム
NKAT Research Progress Dashboard with Version Compatibility

Author: NKAT Research Consortium
Date: 2025-05-26
Version: 9.1 - Universal Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
import psutil
import torch
import glob
import os
import sys
from typing import Dict, List, Optional, Any
import logging

# ページ設定
st.set_page_config(
    page_title="NKAT Research Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 日本語フォント設定
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .status-good { color: #28a745; }
    .status-warning { color: #ffc107; }
    .status-error { color: #dc3545; }
</style>
""", unsafe_allow_html=True)

class NKATDashboard:
    """NKAT研究進捗ダッシュボード"""
    
    def __init__(self):
        self.base_path = Path(".")
        self.results_paths = [
            "10k_gamma_results",
            "analysis_results", 
            "results",
            "../10k_gamma_results",
            "../analysis_results",
            "../results"
        ]
        self.checkpoint_paths = [
            "10k_gamma_checkpoints_production",
            "10k_gamma_checkpoints",
            "checkpoints",
            "../10k_gamma_checkpoints_production",
            "../checkpoints"
        ]
        
    def get_system_info(self) -> Dict:
        """システム情報取得"""
        try:
            # CPU・メモリ情報
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # GPU情報
            gpu_info = {"available": False}
            if torch.cuda.is_available():
                gpu_info = {
                    "available": True,
                    "name": torch.cuda.get_device_name(),
                    "memory_total": torch.cuda.get_device_properties(0).total_memory / 1e9,
                    "memory_used": torch.cuda.memory_allocated() / 1e9,
                    "memory_cached": torch.cuda.memory_reserved() / 1e9
                }
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_total": memory.total / 1e9,
                "memory_used": memory.used / 1e9,
                "gpu": gpu_info,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            st.error(f"システム情報取得エラー: {e}")
            return {}
    
    def find_latest_results(self, pattern: str) -> Optional[Path]:
        """最新の結果ファイルを検索"""
        for results_path in self.results_paths:
            path = Path(results_path)
            if path.exists():
                files = list(path.glob(pattern))
                if files:
                    return max(files, key=lambda x: x.stat().st_mtime)
        return None
    
    def load_entanglement_results(self) -> Optional[Dict]:
        """量子もつれ解析結果の読み込み"""
        try:
            # 複数のパターンで検索
            patterns = [
                "nkat_v91_entanglement_results.json",
                "*entanglement*.json",
                "nkat_*_entanglement*.json"
            ]
            
            for pattern in patterns:
                file_path = self.find_latest_results(pattern)
                if file_path and file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
            
            return None
        except Exception as e:
            st.error(f"量子もつれ結果読み込みエラー: {e}")
            return None
    
    def load_10k_gamma_results(self) -> Optional[Dict]:
        """10,000γチャレンジ結果の読み込み"""
        try:
            patterns = [
                "10k_gamma_final_results_*.json",
                "*10k*gamma*.json",
                "intermediate_results_*.json"
            ]
            
            for pattern in patterns:
                file_path = self.find_latest_results(pattern)
                if file_path and file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
            
            return None
        except Exception as e:
            st.error(f"10Kγ結果読み込みエラー: {e}")
            return None
    
    def load_checkpoint_status(self) -> Optional[Dict]:
        """チェックポイント状況の読み込み"""
        try:
            for checkpoint_path in self.checkpoint_paths:
                path = Path(checkpoint_path)
                if path.exists():
                    # 最新のチェックポイントファイルを検索
                    checkpoint_files = list(path.glob("checkpoint_batch_*.json"))
                    if checkpoint_files:
                        latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
                        with open(latest_checkpoint, 'r', encoding='utf-8') as f:
                            return json.load(f)
            
            return None
        except Exception as e:
            st.error(f"チェックポイント読み込みエラー: {e}")
            return None
    
    def display_header(self):
        """ヘッダー表示"""
        st.markdown('<h1 class="main-header">🚀 NKAT Research Dashboard</h1>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"📅 **更新時刻**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        with col2:
            st.info(f"🔬 **NKAT Version**: v9.1 - Quantum Entanglement")
        with col3:
            if st.button("🔄 更新", key="refresh_button"):
                # Streamlit バージョン互換性対応
                try:
                    st.rerun()
                except AttributeError:
                    st.experimental_rerun()
    
    def display_system_status(self):
        """システム状況表示"""
        st.header("🖥️ システム状況")
        
        system_info = self.get_system_info()
        if not system_info:
            st.error("システム情報を取得できませんでした")
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            cpu_color = "🟢" if system_info["cpu_percent"] < 80 else "🟡" if system_info["cpu_percent"] < 95 else "🔴"
            st.metric(
                label=f"{cpu_color} CPU使用率",
                value=f"{system_info['cpu_percent']:.1f}%"
            )
        
        with col2:
            mem_color = "🟢" if system_info["memory_percent"] < 80 else "🟡" if system_info["memory_percent"] < 90 else "🔴"
            st.metric(
                label=f"{mem_color} メモリ使用率",
                value=f"{system_info['memory_percent']:.1f}%",
                delta=f"{system_info['memory_used']:.1f}/{system_info['memory_total']:.1f} GB"
            )
        
        with col3:
            if system_info["gpu"]["available"]:
                gpu_usage = (system_info["gpu"]["memory_used"] / system_info["gpu"]["memory_total"]) * 100
                gpu_color = "🟢" if gpu_usage < 80 else "🟡" if gpu_usage < 95 else "🔴"
                st.metric(
                    label=f"{gpu_color} GPU メモリ",
                    value=f"{gpu_usage:.1f}%",
                    delta=f"{system_info['gpu']['memory_used']:.1f}/{system_info['gpu']['memory_total']:.1f} GB"
                )
            else:
                st.metric(label="❌ GPU", value="利用不可")
        
        with col4:
            if system_info["gpu"]["available"]:
                st.metric(
                    label="🎮 GPU",
                    value="利用可能",
                    delta=system_info["gpu"]["name"]
                )
            else:
                st.metric(label="🎮 GPU", value="なし")
    
    def display_entanglement_analysis(self):
        """量子もつれ解析結果表示"""
        st.header("🔬 量子もつれ解析結果")
        
        entanglement_data = self.load_entanglement_results()
        if not entanglement_data:
            st.warning("量子もつれ解析結果が見つかりません")
            return
        
        # メトリクス表示
        if "entanglement_metrics" in entanglement_data:
            metrics = entanglement_data["entanglement_metrics"]
            
            # データフレーム作成
            df = pd.DataFrame(metrics)
            
            # サマリー統計
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_concurrence = df["concurrence"].mean()
                st.metric("平均Concurrence", f"{avg_concurrence:.6f}")
            with col2:
                avg_entropy = df["entanglement_entropy"].mean()
                st.metric("平均エントロピー", f"{avg_entropy:.6f}")
            with col3:
                avg_negativity = df["negativity"].mean()
                st.metric("平均Negativity", f"{avg_negativity:.6f}")
            with col4:
                detection_rate = (df["concurrence"] > 0.01).mean() * 100
                st.metric("もつれ検出率", f"{detection_rate:.1f}%")
            
            # グラフ表示
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Concurrence", "Entanglement Entropy", "Negativity", "Quantum Discord"),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Concurrence
            fig.add_trace(
                go.Scatter(x=df["gamma"], y=df["concurrence"], mode="lines+markers", name="Concurrence"),
                row=1, col=1
            )
            
            # Entanglement Entropy
            fig.add_trace(
                go.Scatter(x=df["gamma"], y=df["entanglement_entropy"], mode="lines+markers", name="Entropy"),
                row=1, col=2
            )
            
            # Negativity
            fig.add_trace(
                go.Scatter(x=df["gamma"], y=df["negativity"], mode="lines+markers", name="Negativity"),
                row=2, col=1
            )
            
            # Quantum Discord
            fig.add_trace(
                go.Scatter(x=df["gamma"], y=df["quantum_discord"], mode="lines+markers", name="Discord"),
                row=2, col=2
            )
            
            fig.update_layout(height=600, title_text="量子もつれメトリクス")
            st.plotly_chart(fig, use_container_width=True)
            
            # データテーブル
            st.subheader("📊 詳細データ")
            st.dataframe(df)
    
    def display_10k_gamma_progress(self):
        """10,000γチャレンジ進捗表示"""
        st.header("🎯 10,000γ Challenge 進捗")
        
        # チェックポイント状況
        checkpoint_data = self.load_checkpoint_status()
        results_data = self.load_10k_gamma_results()
        
        if checkpoint_data:
            st.subheader("📊 現在の進捗")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("完了バッチ", f"{checkpoint_data.get('batch_id', 0) + 1}")
            with col2:
                progress = checkpoint_data.get('total_progress', 0)
                st.metric("進捗率", f"{progress:.1f}%")
            with col3:
                completed_gammas = len(checkpoint_data.get('completed_gammas', []))
                st.metric("処理済みγ値", f"{completed_gammas:,}")
            with col4:
                timestamp = checkpoint_data.get('timestamp', '')
                if timestamp:
                    last_update = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    st.metric("最終更新", last_update.strftime('%H:%M:%S'))
            
            # プログレスバー
            progress_value = checkpoint_data.get('total_progress', 0) / 100
            st.progress(progress_value)
            
            # システム状態
            if 'system_state' in checkpoint_data:
                sys_state = checkpoint_data['system_state']
                st.subheader("💻 実行時システム状態")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("CPU使用率", f"{sys_state.get('cpu_percent', 0):.1f}%")
                with col2:
                    st.metric("メモリ使用率", f"{sys_state.get('memory_percent', 0):.1f}%")
                with col3:
                    st.metric("GPU メモリ", f"{checkpoint_data.get('gpu_memory', 0):.1f} GB")
        
        if results_data:
            st.subheader("📈 最終結果")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                total_processed = results_data.get('total_gammas_processed', 0)
                st.metric("総処理γ値", f"{total_processed:,}")
            with col2:
                valid_results = results_data.get('valid_results', 0)
                st.metric("有効結果", f"{valid_results:,}")
            with col3:
                success_rate = results_data.get('success_rate', 0) * 100
                st.metric("成功率", f"{success_rate:.1f}%")
            with col4:
                exec_time = results_data.get('execution_time_formatted', 'N/A')
                st.metric("実行時間", exec_time)
            
            # 統計情報
            if 'statistics' in results_data:
                stats = results_data['statistics']
                st.subheader("📊 統計サマリー")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    mean_dim = stats.get('mean_spectral_dimension', 0)
                    st.metric("平均スペクトル次元", f"{mean_dim:.6f}")
                with col2:
                    mean_conv = stats.get('mean_convergence', 0)
                    st.metric("平均収束値", f"{mean_conv:.6f}")
                with col3:
                    best_conv = stats.get('best_convergence', 0)
                    st.metric("最良収束値", f"{best_conv:.6f}")
            
            # 結果のグラフ化
            if 'results' in results_data:
                results = results_data['results']
                if results:
                    df_results = pd.DataFrame(results)
                    
                    # 有効な結果のみフィルタ
                    valid_df = df_results.dropna(subset=['spectral_dimension'])
                    
                    if not valid_df.empty:
                        fig = make_subplots(
                            rows=1, cols=2,
                            subplot_titles=("スペクトル次元分布", "収束性分布")
                        )
                        
                        # スペクトル次元ヒストグラム
                        fig.add_trace(
                            go.Histogram(x=valid_df['spectral_dimension'], name="スペクトル次元"),
                            row=1, col=1
                        )
                        
                        # 収束性ヒストグラム
                        if 'convergence_to_half' in valid_df.columns:
                            convergence_data = valid_df['convergence_to_half'].dropna()
                            if not convergence_data.empty:
                                fig.add_trace(
                                    go.Histogram(x=convergence_data, name="収束性"),
                                    row=1, col=2
                                )
                        
                        fig.update_layout(height=400, title_text="結果分布")
                        st.plotly_chart(fig, use_container_width=True)
        
        if not checkpoint_data and not results_data:
            st.warning("10,000γチャレンジのデータが見つかりません")
    
    def display_file_browser(self):
        """ファイルブラウザ"""
        st.header("📁 ファイルブラウザ")
        
        # 結果ファイル一覧
        st.subheader("📊 結果ファイル")
        for results_path in self.results_paths:
            path = Path(results_path)
            if path.exists():
                files = list(path.glob("*.json"))
                if files:
                    st.write(f"**{results_path}/**")
                    for file in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True):
                        file_size = file.stat().st_size / 1024  # KB
                        file_time = datetime.fromtimestamp(file.stat().st_mtime)
                        st.write(f"  📄 {file.name} ({file_size:.1f} KB, {file_time.strftime('%Y-%m-%d %H:%M')})")
        
        # チェックポイントファイル一覧
        st.subheader("💾 チェックポイント")
        for checkpoint_path in self.checkpoint_paths:
            path = Path(checkpoint_path)
            if path.exists():
                files = list(path.glob("checkpoint_*.json"))
                if files:
                    st.write(f"**{checkpoint_path}/**")
                    for file in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:  # 最新5個
                        file_size = file.stat().st_size / 1024  # KB
                        file_time = datetime.fromtimestamp(file.stat().st_mtime)
                        st.write(f"  💾 {file.name} ({file_size:.1f} KB, {file_time.strftime('%Y-%m-%d %H:%M')})")
    
    def display_version_info(self):
        """バージョン情報表示"""
        st.header("ℹ️ バージョン情報")
        
        version_info = {
            "NKAT Version": "v9.1 - Quantum Entanglement Revolution",
            "Python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "PyTorch": torch.__version__ if hasattr(torch, '__version__') else "不明",
            "CUDA Available": "✅" if torch.cuda.is_available() else "❌",
            "Streamlit": st.__version__ if hasattr(st, '__version__') else "不明",
            "Dashboard Version": "1.0.0",
            "Last Updated": "2025-05-26"
        }
        
        for key, value in version_info.items():
            st.write(f"**{key}**: {value}")

def main():
    """メイン関数"""
    dashboard = NKATDashboard()
    
    # ヘッダー表示
    dashboard.display_header()
    
    # サイドバー
    st.sidebar.title("🎛️ ダッシュボード設定")
    
    # 表示オプション
    show_system = st.sidebar.checkbox("🖥️ システム状況", value=True)
    show_entanglement = st.sidebar.checkbox("🔬 量子もつれ解析", value=True)
    show_10k_gamma = st.sidebar.checkbox("🎯 10,000γ Challenge", value=True)
    show_files = st.sidebar.checkbox("📁 ファイルブラウザ", value=False)
    show_version = st.sidebar.checkbox("ℹ️ バージョン情報", value=False)
    
    # 自動更新設定
    auto_refresh = st.sidebar.checkbox("🔄 自動更新", value=False)
    if auto_refresh:
        refresh_interval = st.sidebar.slider("更新間隔（秒）", 5, 60, 30)
        time.sleep(refresh_interval)
        # Streamlit バージョン互換性対応
        try:
            st.rerun()
        except AttributeError:
            st.experimental_rerun()
    
    # 表示内容
    if show_system:
        dashboard.display_system_status()
        st.divider()
    
    if show_entanglement:
        dashboard.display_entanglement_analysis()
        st.divider()
    
    if show_10k_gamma:
        dashboard.display_10k_gamma_progress()
        st.divider()
    
    if show_files:
        dashboard.display_file_browser()
        st.divider()
    
    if show_version:
        dashboard.display_version_info()

if __name__ == "__main__":
    main() 