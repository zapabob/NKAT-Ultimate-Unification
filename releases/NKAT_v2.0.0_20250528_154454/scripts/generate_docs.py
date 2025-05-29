#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT理論 API ドキュメント自動生成スクリプト
pdocを使用してHTMLドキュメントを生成

Usage:
    python generate_docs.py
"""

import os
import subprocess
import sys
from pathlib import Path

def generate_documentation():
    """NKAT理論のAPIドキュメントを生成"""
    
    print("=" * 60)
    print("NKAT Theory API Documentation Generator")
    print("=" * 60)
    
    # ドキュメント出力ディレクトリ
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    
    # ドキュメント化対象ファイル
    target_files = [
        "nkat_core_theory.py",
        "kappa_deformed_bspline_theory.py", 
        "dirac_laplacian_analysis.py",
        "kappa_minkowski_theta_relationship.py",
        "nkat_gpu_optimized.py",
        "nkat_implementation.py",
        "quantum_gravity_implementation.py"
    ]
    
    # pdocの確認とインストール
    try:
        subprocess.run(["pdoc", "--version"], check=True, capture_output=True)
        print("✅ pdoc が利用可能です")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("📦 pdoc をインストール中...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pdoc3"], check=True)
    
    # 各ファイルのドキュメント生成
    for file_name in target_files:
        if Path(file_name).exists():
            print(f"📄 {file_name} のドキュメントを生成中...")
            
            try:
                # HTMLドキュメント生成
                cmd = [
                    "pdoc", 
                    "--html", 
                    "--output-dir", str(docs_dir),
                    "--force",
                    file_name
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"   ✅ {file_name} → docs/{file_name.replace('.py', '.html')}")
                else:
                    print(f"   ❌ エラー: {result.stderr}")
                    
            except Exception as e:
                print(f"   ❌ 例外: {e}")
        else:
            print(f"   ⚠️  {file_name} が見つかりません")
    
    # インデックスページの生成
    generate_index_page(docs_dir, target_files)
    
    # README for docs
    generate_docs_readme(docs_dir)
    
    print(f"\n🎉 ドキュメント生成完了！")
    print(f"📂 出力先: {docs_dir.absolute()}")
    print(f"🌐 ブラウザで docs/index.html を開いてください")

def generate_index_page(docs_dir: Path, target_files: list):
    """ドキュメントのインデックスページを生成"""
    
    index_html = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NKAT Theory API Documentation</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                 color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }
        .module-list { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .module-card { border: 1px solid #ddd; border-radius: 8px; padding: 20px; 
                      background: #f9f9f9; transition: transform 0.2s; }
        .module-card:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        .module-title { color: #333; font-size: 1.2em; font-weight: bold; margin-bottom: 10px; }
        .module-desc { color: #666; margin-bottom: 15px; }
        .module-link { background: #667eea; color: white; padding: 8px 16px; 
                      text-decoration: none; border-radius: 4px; display: inline-block; }
        .module-link:hover { background: #5a6fd8; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🌌 NKAT Theory API Documentation</h1>
        <p>Non-Commutative Kolmogorov-Arnold Theory - 非可換コルモゴロフ・アーノルド理論</p>
        <p>Generated on: """ + str(Path().absolute()) + """</p>
    </div>
    
    <div class="module-list">
"""
    
    # モジュール情報
    module_info = {
        "nkat_core_theory.py": {
            "title": "NKAT Core Theory",
            "desc": "κ-変形B-スプライン、スペクトル次元、θ-λ関係の統合実装"
        },
        "kappa_deformed_bspline_theory.py": {
            "title": "κ-Deformed B-Spline Theory", 
            "desc": "κ-変形B-スプライン関数の厳密な定義と数学的証明"
        },
        "dirac_laplacian_analysis.py": {
            "title": "Dirac/Laplacian Analysis",
            "desc": "ディラック・ラプラシアン作用素のスペクトル解析"
        },
        "kappa_minkowski_theta_relationship.py": {
            "title": "κ-Minkowski θ Relationship",
            "desc": "κ-ミンコフスキー空間とθパラメータの関係解析"
        },
        "nkat_gpu_optimized.py": {
            "title": "NKAT GPU Optimized",
            "desc": "GPU最適化されたNKAT実装（CUDA対応）"
        },
        "nkat_implementation.py": {
            "title": "NKAT Basic Implementation", 
            "desc": "NKAT理論の基本実装"
        },
        "quantum_gravity_implementation.py": {
            "title": "Quantum Gravity Implementation",
            "desc": "量子重力理論の実装"
        }
    }
    
    for file_name in target_files:
        if Path(file_name).exists():
            module_name = file_name.replace('.py', '')
            info = module_info.get(file_name, {"title": module_name, "desc": "モジュール説明"})
            
            index_html += f"""
        <div class="module-card">
            <div class="module-title">{info['title']}</div>
            <div class="module-desc">{info['desc']}</div>
            <a href="{module_name}.html" class="module-link">📖 ドキュメントを見る</a>
        </div>
"""
    
    index_html += """
    </div>
    
    <div style="margin-top: 40px; padding: 20px; background: #f0f0f0; border-radius: 8px;">
        <h3>🔗 関連リンク</h3>
        <ul>
            <li><a href="https://github.com/NKAT-Research/Ultimate-Unification">GitHub Repository</a></li>
            <li><a href="https://arxiv.org/abs/2025.XXXX">arXiv Preprint</a></li>
            <li><a href="../README.md">Project README</a></li>
        </ul>
    </div>
</body>
</html>
"""
    
    with open(docs_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(index_html)
    
    print("📄 インデックスページを生成しました: docs/index.html")

def generate_docs_readme(docs_dir: Path):
    """ドキュメント用READMEを生成"""
    
    readme_content = """# NKAT Theory API Documentation

このディレクトリには、NKAT (Non-Commutative Kolmogorov-Arnold Theory) の API ドキュメントが含まれています。

## 📚 ドキュメント構成

- `index.html` - メインインデックスページ
- `*.html` - 各モジュールのAPIドキュメント

## 🌐 閲覧方法

1. ブラウザで `index.html` を開く
2. 各モジュールのリンクをクリックしてAPIドキュメントを閲覧

## 🔄 ドキュメント更新

```bash
python generate_docs.py
```

## 📖 主要モジュール

### Core Theory
- **nkat_core_theory** - NKAT統合理論実装
- **kappa_deformed_bspline_theory** - κ-変形B-スプライン理論
- **dirac_laplacian_analysis** - ディラック/ラプラシアン解析

### Implementation
- **nkat_gpu_optimized** - GPU最適化実装
- **quantum_gravity_implementation** - 量子重力実装

### Analysis
- **kappa_minkowski_theta_relationship** - θ-λ関係解析

---

Generated by NKAT Documentation Generator
"""
    
    with open(docs_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print("📄 ドキュメント用READMEを生成しました: docs/README.md")

if __name__ == "__main__":
    generate_documentation() 