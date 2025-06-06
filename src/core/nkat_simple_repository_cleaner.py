#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT Simple Repository Cleaner
シンプル・効果的リポジトリクリーナー

Don't hold back. Give it your all!!
"""

import os
import shutil
import time
from pathlib import Path
from datetime import datetime

def main():
    """メインクリーニング処理"""
    print("🧹 NKAT Simple Repository Cleaner")
    print("Don't hold back. Give it your all!!")
    print("=" * 50)
    
    # 1. 危険な無限再帰ディレクトリを削除
    print("\n⚡ 無限再帰ディレクトリクリーンアップ...")
    try:
        if os.path.exists("recovery_data"):
            shutil.rmtree("recovery_data")
            print("✅ recovery_data 完全削除完了")
    except Exception as e:
        print(f"⚠️ recovery_data削除エラー: {e}")
    
    # 2. __pycache__ クリーンアップ
    print("\n🧽 __pycache__ クリーンアップ...")
    removed_count = 0
    for root, dirs, files in os.walk("."):
        if "__pycache__" in dirs:
            pycache_path = os.path.join(root, "__pycache__")
            try:
                shutil.rmtree(pycache_path)
                print(f"🗑️ 削除: {pycache_path}")
                removed_count += 1
            except Exception as e:
                print(f"❌ エラー: {pycache_path} - {e}")
    
    print(f"✅ __pycache__ 削除: {removed_count} 個")
    
    # 3. .pyc ファイルクリーンアップ
    print("\n🧽 .pyc ファイルクリーンアップ...")
    pyc_count = 0
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith('.pyc'):
                pyc_path = os.path.join(root, file)
                try:
                    os.remove(pyc_path)
                    pyc_count += 1
                except Exception as e:
                    print(f"❌ エラー: {pyc_path} - {e}")
    
    print(f"✅ .pyc ファイル削除: {pyc_count} 個")
    
    # 4. クリーンな recovery_data ディレクトリ作成
    print("\n📂 クリーンなrecovery_dataディレクトリ作成...")
    os.makedirs("recovery_data", exist_ok=True)
    print("✅ recovery_data ディレクトリ作成完了")
    
    # 5. 完了レポート
    print("\n🎉 NKAT Simple Repository Cleaner 完了！")
    print("=" * 50)
    print(f"⏰ 完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("✅ リポジトリが整理整頓されました")
    print("🚀 Git コミット・プッシュ準備完了")

if __name__ == "__main__":
    main() 