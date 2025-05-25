#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 NKAT v11 包括的リカバリーダッシュボード - 電源断対応監視システム
NKAT v11 Comprehensive Recovery Dashboard with Power Failure Protection

Author: NKAT Research Consortium
Date: 2025-05-26
Version: 11.0 - Comprehensive Recovery System
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
import pickle
import threading
import subprocess
from dataclasses import dataclass, asdict
import warnings

# ページ設定
st.set_page_config(
    page_title="NKAT v11 Recovery Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 日本語フォント設定とスタイル
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .status-good { color: #28a745; font-weight: bold; }
    .status-warning { color: #ffc107; font-weight: bold; }
    .status-error { color: #dc3545; font-weight: bold; }
    .recovery-panel {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    .progress-bar {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        height: 20px;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class SystemState:
    """システム状態管理"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    gpu_available: bool
    gpu_memory_used: float
    active_processes: List[str]
    last_checkpoint: Optional[str]
    verification_progress: float
    critical_line_convergence: float
    
class NKATRecoveryDashboard:
    """NKAT v11 包括的リカバリーダッシュボード"""
    
    def __init__(self):
        self.base_path = Path(".")
        self.state_file = Path("nkat_v11_system_state.pkl")
        self.recovery_log = Path("nkat_v11_recovery.log")
        
        # 結果ファイルパス（優先順位付き）
        self.results_paths = [
            "rigorous_verification_results",
            "enhanced_verification_results", 
            "10k_gamma_results",
            "analysis_results",
            "../rigorous_verification_results",
            "../10k_gamma_results",
            "../analysis_results"
        ]
        
        # チェックポイントパス
        self.checkpoint_paths = [
            "10k_gamma_checkpoints_production",
            "test_checkpoints",
            "checkpoints",
            "../10k_gamma_checkpoints_production",
            "../checkpoints"
        ]
        
        # ログ設定
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.recovery_log, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # セッション状態初期化
        if 'system_state' not in st.session_state:
            st.session_state.system_state = self.load_system_state()
        if 'auto_recovery' not in st.session_state:
            st.session_state.auto_recovery = True
        if 'monitoring_active' not in st.session_state:
            st.session_state.monitoring_active = False
    
    def save_system_state(self, state: SystemState):
        """システム状態の保存"""
        try:
            with open(self.state_file, 'wb') as f:
                pickle.dump(asdict(state), f)
            self.logger.info(f"システム状態保存完了: {state.timestamp}")
        except Exception as e:
            self.logger.error(f"システム状態保存エラー: {e}")
    
    def load_system_state(self) -> Optional[SystemState]:
        """システム状態の読み込み"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'rb') as f:
                    state_dict = pickle.load(f)
                return SystemState(**state_dict)
        except Exception as e:
            self.logger.error(f"システム状態読み込みエラー: {e}")
        return None
    
    def get_current_system_info(self) -> SystemState:
        """現在のシステム情報取得"""
        try:
            # CPU・メモリ情報
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # GPU情報
            gpu_available = torch.cuda.is_available()
            gpu_memory_used = 0.0
            if gpu_available:
                gpu_memory_used = torch.cuda.memory_allocated() / 1e9
            
            # アクティブプロセス
            active_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] and 'python' in proc.info['name'].lower():
                        cmdline = proc.info['cmdline']
                        if cmdline and any('nkat' in str(cmd).lower() for cmd in cmdline):
                            active_processes.append(f"{proc.info['name']} (PID: {proc.info['pid']})")
                except:
                    continue
            
            # 最新チェックポイント
            last_checkpoint = self.find_latest_checkpoint()
            
            # 検証進捗
            verification_progress = self.get_verification_progress()
            
            # 臨界線収束度
            critical_line_convergence = self.get_critical_line_convergence()
            
            return SystemState(
                timestamp=datetime.now().isoformat(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                gpu_available=gpu_available,
                gpu_memory_used=gpu_memory_used,
                active_processes=active_processes,
                last_checkpoint=last_checkpoint,
                verification_progress=verification_progress,
                critical_line_convergence=critical_line_convergence
            )
        except Exception as e:
            self.logger.error(f"システム情報取得エラー: {e}")
            return None
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """最新チェックポイントの検索"""
        try:
            latest_time = 0
            latest_file = None
            
            for checkpoint_path in self.checkpoint_paths:
                path = Path(checkpoint_path)
                if path.exists():
                    for file_path in path.rglob("*.json"):
                        if file_path.stat().st_mtime > latest_time:
                            latest_time = file_path.stat().st_mtime
                            latest_file = str(file_path)
            
            return latest_file
        except Exception as e:
            self.logger.error(f"チェックポイント検索エラー: {e}")
            return None
    
    def get_verification_progress(self) -> float:
        """検証進捗の取得"""
        try:
            # 最新の検証結果ファイルを検索
            patterns = [
                "*rigorous_verification*.json",
                "*enhanced_verification*.json",
                "*10k_gamma*.json"
            ]
            
            for pattern in patterns:
                for results_path in self.results_paths:
                    path = Path(results_path)
                    if path.exists():
                        files = list(path.glob(pattern))
                        if files:
                            latest_file = max(files, key=lambda x: x.stat().st_mtime)
                            with open(latest_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                
                            # 進捗計算ロジック
                            if 'critical_line_verification' in data:
                                gamma_count = len(data['critical_line_verification'].get('gamma_values', []))
                                return min(gamma_count / 15.0, 1.0) * 100
                            elif 'verification_results' in data:
                                return data.get('progress_percentage', 0.0)
            
            return 0.0
        except Exception as e:
            self.logger.error(f"検証進捗取得エラー: {e}")
            return 0.0
    
    def get_critical_line_convergence(self) -> float:
        """臨界線収束度の取得"""
        try:
            # 最新の厳密検証結果を検索
            for results_path in self.results_paths:
                path = Path(results_path)
                if path.exists():
                    files = list(path.glob("*rigorous_verification*.json"))
                    if files:
                        latest_file = max(files, key=lambda x: x.stat().st_mtime)
                        with open(latest_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        if 'critical_line_verification' in data:
                            spectral_analysis = data['critical_line_verification'].get('spectral_analysis', [])
                            if spectral_analysis:
                                convergences = [item.get('convergence_to_half', 1.0) for item in spectral_analysis]
                                return 1.0 - np.mean(convergences)  # 1に近いほど良い収束
            
            return 0.0
        except Exception as e:
            self.logger.error(f"臨界線収束度取得エラー: {e}")
            return 0.0
    
    def display_header(self):
        """ヘッダー表示"""
        st.markdown('<h1 class="main-header">🚀 NKAT v11 包括的リカバリーダッシュボード</h1>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.info(f"📅 **更新時刻**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        with col2:
            st.info(f"🔬 **NKAT Version**: v11.0 - Recovery System")
        with col3:
            if st.button("🔄 手動更新", key="manual_refresh"):
                st.experimental_rerun()
        with col4:
            auto_refresh = st.checkbox("🔄 自動更新 (30秒)", value=True)
            if auto_refresh:
                time.sleep(30)
                st.experimental_rerun()
    
    def display_recovery_panel(self):
        """リカバリーパネル表示"""
        st.header("🛡️ 電源断リカバリーシステム")
        
        with st.container():
            st.markdown('<div class="recovery-panel">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("📊 システム状態")
                current_state = self.get_current_system_info()
                if current_state:
                    self.save_system_state(current_state)
                    st.session_state.system_state = current_state
                    
                    st.metric("CPU使用率", f"{current_state.cpu_percent:.1f}%")
                    st.metric("メモリ使用率", f"{current_state.memory_percent:.1f}%")
                    
                    if current_state.gpu_available:
                        st.metric("GPU VRAM", f"{current_state.gpu_memory_used:.1f} GB")
                        st.success("✅ GPU利用可能")
                    else:
                        st.error("❌ GPU利用不可")
            
            with col2:
                st.subheader("🔄 アクティブプロセス")
                if current_state and current_state.active_processes:
                    for process in current_state.active_processes:
                        st.text(f"🟢 {process}")
                else:
                    st.warning("⚠️ NKATプロセスが検出されません")
                
                if st.button("🚀 検証プロセス再開", key="restart_verification"):
                    self.restart_verification_process()
            
            with col3:
                st.subheader("💾 チェックポイント状況")
                if current_state and current_state.last_checkpoint:
                    checkpoint_time = Path(current_state.last_checkpoint).stat().st_mtime
                    checkpoint_dt = datetime.fromtimestamp(checkpoint_time)
                    st.success(f"✅ 最新: {checkpoint_dt.strftime('%H:%M:%S')}")
                    st.text(f"📁 {Path(current_state.last_checkpoint).name}")
                else:
                    st.warning("⚠️ チェックポイントなし")
                
                if st.button("💾 手動チェックポイント", key="manual_checkpoint"):
                    self.create_manual_checkpoint()
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def display_verification_progress(self):
        """検証進捗表示"""
        st.header("📊 高精度検証進捗")
        
        current_state = st.session_state.system_state
        if not current_state:
            st.error("システム状態が取得できません")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 検証進捗")
            progress = current_state.verification_progress
            st.progress(progress / 100.0)
            st.metric("完了率", f"{progress:.1f}%")
            
            # 進捗バー（カスタム）
            st.markdown(f"""
            <div style="background-color: #e0e0e0; border-radius: 10px; padding: 3px;">
                <div class="progress-bar" style="width: {progress}%;"></div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("🔍 臨界線収束性")
            convergence = current_state.critical_line_convergence
            st.metric("収束度", f"{convergence:.6f}")
            
            # 収束性の評価
            if convergence > 0.497:
                st.success("🎉 優秀な収束性！")
            elif convergence > 0.49:
                st.info("✅ 良好な収束性")
            else:
                st.warning("⚠️ 収束性要改善")
    
    def display_detailed_analysis(self):
        """詳細分析表示"""
        st.header("🔬 高精度結果詳細分析")
        
        # 最新の厳密検証結果を読み込み
        latest_results = self.load_latest_rigorous_results()
        if not latest_results:
            st.error("厳密検証結果が見つかりません")
            return
        
        # タブで分割表示
        tab1, tab2, tab3, tab4 = st.tabs(["📊 収束分析", "🎯 スペクトル解析", "📈 統計評価", "🔍 詳細データ"])
        
        with tab1:
            self.display_convergence_analysis(latest_results)
        
        with tab2:
            self.display_spectral_analysis(latest_results)
        
        with tab3:
            self.display_statistical_evaluation(latest_results)
        
        with tab4:
            self.display_detailed_data(latest_results)
    
    def load_latest_rigorous_results(self) -> Optional[Dict]:
        """最新の厳密検証結果読み込み"""
        try:
            for results_path in self.results_paths:
                path = Path(results_path)
                if path.exists():
                    files = list(path.glob("*rigorous_verification*.json"))
                    if files:
                        latest_file = max(files, key=lambda x: x.stat().st_mtime)
                        with open(latest_file, 'r', encoding='utf-8') as f:
                            return json.load(f)
            return None
        except Exception as e:
            self.logger.error(f"厳密検証結果読み込みエラー: {e}")
            return None
    
    def display_convergence_analysis(self, results: Dict):
        """収束分析表示"""
        if 'critical_line_verification' not in results:
            st.error("臨界線検証データがありません")
            return
        
        spectral_analysis = results['critical_line_verification'].get('spectral_analysis', [])
        if not spectral_analysis:
            st.error("スペクトル解析データがありません")
            return
        
        # データ準備
        gamma_values = [item['gamma'] for item in spectral_analysis]
        convergences = [item['convergence_to_half'] for item in spectral_analysis]
        real_parts = [item['real_part'] for item in spectral_analysis]
        
        # 収束性グラフ
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('臨界線収束性 (|Re - 1/2|)', '実部の値'),
            vertical_spacing=0.1
        )
        
        # 収束性プロット
        fig.add_trace(
            go.Scatter(
                x=gamma_values,
                y=convergences,
                mode='lines+markers',
                name='収束度',
                line=dict(color='red', width=2),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        # 理論値ライン (0.5)
        fig.add_hline(y=0.5, line_dash="dash", line_color="green", 
                     annotation_text="理論値 (1/2)", row=1, col=1)
        
        # 実部プロット
        fig.add_trace(
            go.Scatter(
                x=gamma_values,
                y=real_parts,
                mode='lines+markers',
                name='実部',
                line=dict(color='blue', width=2),
                marker=dict(size=8)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="🎯 臨界線収束性詳細分析",
            height=600,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="γ値", row=2, col=1)
        fig.update_yaxes(title_text="収束度", row=1, col=1)
        fig.update_yaxes(title_text="実部", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 統計サマリー
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("平均収束度", f"{np.mean(convergences):.6f}")
        with col2:
            st.metric("標準偏差", f"{np.std(convergences):.6f}")
        with col3:
            st.metric("最良収束", f"{np.min(convergences):.6f}")
    
    def display_spectral_analysis(self, results: Dict):
        """スペクトル解析表示"""
        spectral_analysis = results['critical_line_verification'].get('spectral_analysis', [])
        if not spectral_analysis:
            return
        
        # スペクトル次元分析
        gamma_values = [item['gamma'] for item in spectral_analysis]
        spectral_dims = [item['spectral_dimension'] for item in spectral_analysis]
        
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=gamma_values,
                y=spectral_dims,
                mode='lines+markers',
                name='スペクトル次元',
                line=dict(color='purple', width=3),
                marker=dict(size=10, color='purple')
            )
        )
        
        fig.update_layout(
            title="🔬 スペクトル次元分析",
            xaxis_title="γ値",
            yaxis_title="スペクトル次元",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # レベル間隔統計
        st.subheader("📊 レベル間隔統計")
        level_stats_data = []
        for item in spectral_analysis:
            if 'level_spacing_stats' in item:
                stats = item['level_spacing_stats']
                level_stats_data.append({
                    'γ値': item['gamma'],
                    '平均間隔': stats.get('mean_spacing', 0),
                    '正規化平均': stats.get('normalized_mean', 0),
                    '正規化分散': stats.get('normalized_variance', 0),
                    'GUE偏差': stats.get('gue_statistical_distance', 0)
                })
        
        if level_stats_data:
            df = pd.DataFrame(level_stats_data)
            st.dataframe(df, use_container_width=True)
    
    def display_statistical_evaluation(self, results: Dict):
        """統計評価表示"""
        st.subheader("📈 統計的評価")
        
        # 全体統計
        if 'overall_statistics' in results:
            stats = results['overall_statistics']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("数学的厳密性", f"{stats.get('mathematical_rigor', 0):.3f}")
            with col2:
                st.metric("証明完全性", f"{stats.get('proof_completeness', 0):.3f}")
            with col3:
                st.metric("統計的有意性", f"{stats.get('statistical_significance', 0):.3f}")
            with col4:
                success_rate = stats.get('success_rate', 0)
                st.metric("成功率", f"{success_rate:.1%}")
        
        # 検証結果サマリー
        if 'verification_summary' in results:
            summary = results['verification_summary']
            
            st.subheader("🎯 検証サマリー")
            for key, value in summary.items():
                if isinstance(value, bool):
                    status = "✅ 成功" if value else "❌ 失敗"
                    st.write(f"**{key}**: {status}")
                elif isinstance(value, (int, float)):
                    st.write(f"**{key}**: {value}")
                else:
                    st.write(f"**{key}**: {value}")
    
    def display_detailed_data(self, results: Dict):
        """詳細データ表示"""
        st.subheader("🔍 詳細データ")
        
        # JSON表示
        with st.expander("📄 完全なJSON結果"):
            st.json(results)
        
        # 主要メトリクス表
        if 'critical_line_verification' in results:
            spectral_analysis = results['critical_line_verification'].get('spectral_analysis', [])
            if spectral_analysis:
                df_data = []
                for item in spectral_analysis:
                    df_data.append({
                        'γ値': f"{item['gamma']:.6f}",
                        'スペクトル次元': f"{item['spectral_dimension']:.8f}",
                        '実部': f"{item['real_part']:.8f}",
                        '収束度': f"{item['convergence_to_half']:.8f}",
                        '固有値数': item.get('eigenvalue_count', 0)
                    })
                
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True)
    
    def restart_verification_process(self):
        """検証プロセス再開"""
        try:
            st.info("🚀 検証プロセスを再開しています...")
            
            # 最新の検証スクリプトを実行
            verification_scripts = [
                "nkat_v11_rigorous_mathematical_verification.py",
                "nkat_v11_enhanced_large_scale_verification.py",
                "riemann_high_precision.py"
            ]
            
            for script in verification_scripts:
                if Path(script).exists():
                    subprocess.Popen([sys.executable, script], 
                                   cwd=self.base_path,
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
                    st.success(f"✅ {script} を開始しました")
                    break
            else:
                st.error("❌ 検証スクリプトが見つかりません")
                
        except Exception as e:
            st.error(f"❌ プロセス再開エラー: {e}")
            self.logger.error(f"プロセス再開エラー: {e}")
    
    def create_manual_checkpoint(self):
        """手動チェックポイント作成"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_data = {
                "timestamp": timestamp,
                "system_state": asdict(st.session_state.system_state) if st.session_state.system_state else {},
                "manual_checkpoint": True
            }
            
            checkpoint_dir = Path("manual_checkpoints")
            checkpoint_dir.mkdir(exist_ok=True)
            
            checkpoint_file = checkpoint_dir / f"manual_checkpoint_{timestamp}.json"
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
            st.success(f"✅ 手動チェックポイント作成: {checkpoint_file.name}")
            self.logger.info(f"手動チェックポイント作成: {checkpoint_file}")
            
        except Exception as e:
            st.error(f"❌ チェックポイント作成エラー: {e}")
            self.logger.error(f"チェックポイント作成エラー: {e}")

def main():
    """メイン関数"""
    dashboard = NKATRecoveryDashboard()
    
    # ヘッダー表示
    dashboard.display_header()
    
    # サイドバー
    with st.sidebar:
        st.header("🛠️ 制御パネル")
        
        # 自動リカバリー設定
        auto_recovery = st.checkbox("🛡️ 自動リカバリー", value=st.session_state.auto_recovery)
        st.session_state.auto_recovery = auto_recovery
        
        # 監視設定
        monitoring_active = st.checkbox("👁️ 連続監視", value=st.session_state.monitoring_active)
        st.session_state.monitoring_active = monitoring_active
        
        # 緊急停止
        if st.button("🛑 緊急停止", type="secondary"):
            st.error("🛑 緊急停止が実行されました")
            st.stop()
        
        # システム情報
        st.subheader("💻 システム情報")
        if torch.cuda.is_available():
            st.success(f"🎮 GPU: {torch.cuda.get_device_name()}")
            st.info(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            st.warning("⚠️ GPU利用不可")
    
    # メインコンテンツ
    dashboard.display_recovery_panel()
    dashboard.display_verification_progress()
    dashboard.display_detailed_analysis()
    
    # フッター
    st.markdown("---")
    st.markdown("🚀 **NKAT v11 包括的リカバリーダッシュボード** - 電源断対応監視システム")
    st.markdown(f"📅 最終更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 