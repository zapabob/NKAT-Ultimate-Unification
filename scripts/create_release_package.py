#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📦 NKAT リーマン予想解析システム - リリースパッケージ作成スクリプト
Release Package Creator for NKAT Riemann Hypothesis Analysis System

本番リリース用のパッケージを作成し、配布可能な形式でアーカイブします。

Author: NKAT Research Team
Date: 2025-05-28
Version: 2.0.0 - Production Release
License: MIT
"""

import os
import sys
import shutil
import zipfile
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import hashlib

# プロジェクトルート
project_root = Path(__file__).parent.parent

def create_release_info() -> Dict[str, Any]:
    """リリース情報作成"""
    return {
        "name": "NKAT リーマン予想解析システム",
        "version": "2.0.0",
        "release_date": datetime.now().isoformat(),
        "description": "非可換コルモゴロフアーノルド表現理論による最高精度リーマン予想解析システム",
        "author": "NKAT Research Team",
        "license": "MIT",
        "requirements": {
            "python": ">=3.8",
            "gpu": "NVIDIA RTX 3080 (推奨)",
            "vram": "8GB以上",
            "ram": "16GB以上",
            "storage": "50GB以上"
        },
        "features": [
            "非可換コルモゴロフアーノルド表現理論",
            "200桁超高精度計算",
            "RTX3080最適化",
            "電源断リカバリー",
            "Streamlit監視ダッシュボード",
            "リアルタイムGPU監視"
        ],
        "files": {
            "main_system": "src/nkat_riemann_ultimate_precision_system.py",
            "launcher": "scripts/production_launcher.py",
            "config": "config/production_config.json",
            "startup": "launch_production.bat",
            "readme": "README_Production_Release.md"
        }
    }

def get_files_to_include() -> List[str]:
    """リリースに含めるファイルリスト"""
    return [
        # メインシステム
        "src/nkat_riemann_ultimate_precision_system.py",
        
        # 起動スクリプト
        "scripts/production_launcher.py",
        "scripts/test_nkat_riemann_ultimate_system.py",
        "launch_production.bat",
        
        # 設定ファイル
        "config/production_config.json",
        "requirements.txt",
        
        # ドキュメント
        "README_Production_Release.md",
        "README.md",
        "LICENSE",
        
        # 既存の重要ファイル
        "src/kolmogorov_arnold_quantum_unified_theory.py",
        "src/gpu/streamlit_gpu_monitor.py",
        "src/gpu/dirac_laplacian_analysis_gpu_recovery.py",
        
        # 設定・データファイル
        "nkat_config.json",
        "kaq_colab_config.json",
    ]

def get_directories_to_include() -> List[str]:
    """リリースに含めるディレクトリリスト"""
    return [
        "config",
        "docs",
        "figures",
        "scripts",
        "src/core",
        "src/gpu",
        "src/mathematical",
        "src/quantum",
    ]

def calculate_file_hash(file_path: Path) -> str:
    """ファイルのSHA256ハッシュ計算"""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception:
        return "error"

def create_file_manifest(release_dir: Path) -> Dict[str, Any]:
    """ファイルマニフェスト作成"""
    manifest = {
        "created": datetime.now().isoformat(),
        "files": {},
        "total_files": 0,
        "total_size": 0
    }
    
    for file_path in release_dir.rglob("*"):
        if file_path.is_file():
            relative_path = file_path.relative_to(release_dir)
            file_size = file_path.stat().st_size
            file_hash = calculate_file_hash(file_path)
            
            manifest["files"][str(relative_path)] = {
                "size": file_size,
                "hash": file_hash,
                "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            }
            
            manifest["total_files"] += 1
            manifest["total_size"] += file_size
    
    return manifest

def copy_files_to_release(release_dir: Path):
    """リリースディレクトリにファイルをコピー"""
    print("📁 ファイルコピー開始...")
    
    # 個別ファイルコピー
    files_to_include = get_files_to_include()
    for file_path in files_to_include:
        source = project_root / file_path
        if source.exists():
            dest = release_dir / file_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            print(f"   ✅ {file_path}")
        else:
            print(f"   ⚠️  {file_path} (見つかりません)")
    
    # ディレクトリコピー
    directories_to_include = get_directories_to_include()
    for dir_path in directories_to_include:
        source = project_root / dir_path
        if source.exists() and source.is_dir():
            dest = release_dir / dir_path
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(source, dest, ignore=shutil.ignore_patterns(
                '*.pyc', '__pycache__', '*.log', '*.tmp', '.git*'
            ))
            print(f"   ✅ {dir_path}/ (ディレクトリ)")
        else:
            print(f"   ⚠️  {dir_path}/ (見つかりません)")

def create_startup_scripts(release_dir: Path):
    """起動スクリプト作成"""
    print("🚀 起動スクリプト作成...")
    
    # Windows用クイックスタート
    quick_start_bat = release_dir / "QUICK_START.bat"
    with open(quick_start_bat, 'w', encoding='utf-8') as f:
        f.write("""@echo off
chcp 65001 > nul
echo.
echo 🌌 NKAT リーマン予想解析システム - クイックスタート
echo ================================================================
echo.
echo 📋 システムチェック実行中...
py -3 scripts\\production_launcher.py --check-only
if %errorLevel% neq 0 (
    echo ❌ システムチェック失敗
    pause
    exit /b 1
)
echo.
echo ✅ システムチェック完了
echo 🚀 本番システム起動中...
echo.
launch_production.bat
""")
    
    # Linux/Mac用スクリプト
    quick_start_sh = release_dir / "quick_start.sh"
    with open(quick_start_sh, 'w', encoding='utf-8') as f:
        f.write("""#!/bin/bash
echo "🌌 NKAT リーマン予想解析システム - クイックスタート"
echo "================================================================"
echo
echo "📋 システムチェック実行中..."
python3 scripts/production_launcher.py --check-only
if [ $? -ne 0 ]; then
    echo "❌ システムチェック失敗"
    exit 1
fi
echo
echo "✅ システムチェック完了"
echo "🚀 本番システム起動中..."
echo
python3 scripts/production_launcher.py
""")
    
    # 実行権限付与（Unix系）
    try:
        os.chmod(quick_start_sh, 0o755)
    except:
        pass
    
    print("   ✅ QUICK_START.bat")
    print("   ✅ quick_start.sh")

def create_installation_guide(release_dir: Path):
    """インストールガイド作成"""
    print("📖 インストールガイド作成...")
    
    install_guide = release_dir / "INSTALL.md"
    with open(install_guide, 'w', encoding='utf-8') as f:
        f.write("""# 🚀 NKAT システム インストールガイド

## クイックスタート

### Windows
1. `QUICK_START.bat` をダブルクリック
2. ブラウザで http://localhost:8501 にアクセス

### Linux/Mac
```bash
chmod +x quick_start.sh
./quick_start.sh
```

## 詳細インストール

### 1. Python環境
```bash
# Python 3.8以上が必要
python --version

# 仮想環境作成（推奨）
python -m venv nkat_env
source nkat_env/bin/activate  # Linux/Mac
nkat_env\\Scripts\\activate   # Windows
```

### 2. 依存関係インストール
```bash
pip install -r requirements.txt
```

### 3. GPU環境確認
```bash
# CUDA確認
nvidia-smi

# PyTorch CUDA確認
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 4. システムチェック
```bash
python scripts/production_launcher.py --check-only
```

### 5. 起動
```bash
# Windows
launch_production.bat

# Linux/Mac
python scripts/production_launcher.py
```

## トラブルシューティング

詳細は `README_Production_Release.md` を参照してください。
""")
    
    print("   ✅ INSTALL.md")

def create_release_archive(release_dir: Path, version: str) -> Path:
    """リリースアーカイブ作成"""
    print("📦 アーカイブ作成中...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"NKAT_Riemann_System_v{version}_{timestamp}.zip"
    archive_path = project_root / "releases" / archive_name
    
    # リリースディレクトリ作成
    archive_path.parent.mkdir(exist_ok=True)
    
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        for file_path in release_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(release_dir)
                zipf.write(file_path, arcname)
    
    return archive_path

def main():
    """メイン関数"""
    print("🌌 NKAT リーマン予想解析システム - リリースパッケージ作成")
    print("=" * 70)
    
    # リリース情報
    release_info = create_release_info()
    version = release_info["version"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # リリースディレクトリ作成
    release_dir = project_root / "releases" / f"NKAT_v{version}_{timestamp}"
    if release_dir.exists():
        shutil.rmtree(release_dir)
    release_dir.mkdir(parents=True)
    
    print(f"📁 リリースディレクトリ: {release_dir}")
    print()
    
    try:
        # ファイルコピー
        copy_files_to_release(release_dir)
        print()
        
        # 起動スクリプト作成
        create_startup_scripts(release_dir)
        print()
        
        # インストールガイド作成
        create_installation_guide(release_dir)
        print()
        
        # リリース情報保存
        release_info_file = release_dir / "release_info.json"
        with open(release_info_file, 'w', encoding='utf-8') as f:
            json.dump(release_info, f, indent=2, ensure_ascii=False)
        print("   ✅ release_info.json")
        print()
        
        # ファイルマニフェスト作成
        print("📋 ファイルマニフェスト作成...")
        manifest = create_file_manifest(release_dir)
        manifest_file = release_dir / "MANIFEST.json"
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"   ✅ ファイル数: {manifest['total_files']}")
        print(f"   ✅ 総サイズ: {manifest['total_size'] / 1024 / 1024:.1f} MB")
        print()
        
        # アーカイブ作成
        archive_path = create_release_archive(release_dir, version)
        archive_size = archive_path.stat().st_size / 1024 / 1024
        print(f"   ✅ アーカイブ: {archive_path.name}")
        print(f"   ✅ サイズ: {archive_size:.1f} MB")
        print()
        
        # 成功メッセージ
        print("🎉 リリースパッケージ作成完了!")
        print("=" * 70)
        print(f"📦 パッケージ: {archive_path}")
        print(f"📁 展開先: {release_dir}")
        print(f"🌐 ダッシュボード: http://localhost:8501")
        print()
        print("📋 配布方法:")
        print(f"   1. {archive_path.name} を配布")
        print("   2. 展開後、QUICK_START.bat を実行")
        print("   3. README_Production_Release.md を参照")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 