#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 NKAT Streamlit監視ダッシュボード起動スクリプト
GPU/CPU監視とNKAT解析統合ダッシュボードの起動

Author: NKAT Research Team
Date: 2025-01-24
Version: 1.0
"""

import subprocess
import sys
import os
from pathlib import Path
import webbrowser
import time

def check_dependencies():
    """依存関係のチェック"""
    required_packages = [
        'streamlit',
        'plotly',
        'psutil',
        'GPUtil',
        'torch'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - インストール済み")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - 未インストール")
    
    if missing_packages:
        print(f"\n⚠️  以下のパッケージが不足しています:")
        for package in missing_packages:
            print(f"  - {package}")
        print(f"\n以下のコマンドでインストールしてください:")
        print(f"py -3 -m pip install {' '.join(missing_packages)}")
        return False
    
    return True

def start_dashboard():
    """ダッシュボードの起動"""
    print("🖥️📊 NKAT GPU/CPU監視ダッシュボード起動中...")
    print("=" * 60)
    
    # 依存関係チェック
    if not check_dependencies():
        print("\n❌ 依存関係が不足しています。先にインストールしてください。")
        return False
    
    # Streamlitアプリのパス
    app_path = Path(__file__).parent.parent / "src" / "gpu" / "streamlit_gpu_monitor.py"
    
    if not app_path.exists():
        print(f"❌ Streamlitアプリが見つかりません: {app_path}")
        return False
    
    print(f"📁 アプリパス: {app_path}")
    print("🚀 Streamlitサーバー起動中...")
    
    # Streamlitコマンドの構築
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", "8501",
        "--server.address", "localhost",
        "--browser.gatherUsageStats", "false",
        "--server.headless", "false"
    ]
    
    try:
        # Streamlitプロセスの起動
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )
        
        print("⏳ サーバー起動を待機中...")
        time.sleep(3)
        
        # ブラウザを開く
        url = "http://localhost:8501"
        print(f"🌐 ブラウザでダッシュボードを開いています: {url}")
        webbrowser.open(url)
        
        print("\n" + "=" * 60)
        print("🎉 NKAT監視ダッシュボードが起動しました！")
        print("=" * 60)
        print("📊 機能:")
        print("  - GPU使用率・温度・メモリ監視")
        print("  - CPU使用率・温度監視")
        print("  - システムメモリ監視")
        print("  - NKAT解析のリアルタイム実行")
        print("  - プログレスバー表示")
        print("  - ログ表示")
        print("  - チェックポイント管理")
        print("\n🔗 アクセス URL: http://localhost:8501")
        print("⏹️  停止するには Ctrl+C を押してください")
        print("=" * 60)
        
        # プロセスの監視
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n⏹️  ダッシュボードを停止中...")
            process.terminate()
            process.wait()
            print("✅ ダッシュボードが停止しました")
        
        return True
        
    except Exception as e:
        print(f"❌ ダッシュボード起動エラー: {e}")
        return False

def main():
    """メイン関数"""
    print("🖥️📊 NKAT Streamlit監視ダッシュボード")
    print("=" * 60)
    print("GPU/CPU監視とNKAT解析統合ダッシュボード")
    print("=" * 60)
    
    # 現在のディレクトリ確認
    current_dir = Path.cwd()
    print(f"📁 現在のディレクトリ: {current_dir}")
    
    # プロジェクトルートに移動
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    print(f"📁 プロジェクトルート: {project_root}")
    
    # ダッシュボード起動
    success = start_dashboard()
    
    if not success:
        print("\n❌ ダッシュボードの起動に失敗しました")
        print("📋 トラブルシューティング:")
        print("  1. 依存関係がインストールされているか確認")
        print("  2. ポート8501が使用可能か確認")
        print("  3. ファイアウォール設定を確認")
        sys.exit(1)

if __name__ == "__main__":
    main() 