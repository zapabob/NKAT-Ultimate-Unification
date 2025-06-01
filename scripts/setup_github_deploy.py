#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT-Transformer GitHub Deploy Setup
GitHub公開用ファイル整理・デプロイ準備スクリプト

機能:
1. 独立版ファイル整理
2. デモ・テスト用スクリプト作成
3. ライセンス・設定ファイル準備
4. Git準備
"""

import os
import shutil
import json
from datetime import datetime

def create_directory_structure():
    """GitHub用ディレクトリ構造作成"""
    dirs = [
        'nkat-transformer-standalone',
        'nkat-transformer-standalone/examples',
        'nkat-transformer-standalone/docs',
        'nkat-transformer-standalone/tests',
        'nkat-transformer-standalone/models',
        'nkat-transformer-standalone/results'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ Created: {dir_path}")

def copy_core_files():
    """コアファイルをコピー"""
    files_to_copy = {
        'nkat_core_standalone.py': 'nkat-transformer-standalone/nkat_core_standalone.py',
        'README_NKAT_Standalone.md': 'nkat-transformer-standalone/README.md',
        'requirements_standalone.txt': 'nkat-transformer-standalone/requirements.txt'
    }
    
    for src, dst in files_to_copy.items():
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"✅ Copied: {src} → {dst}")
        else:
            print(f"⚠️ Missing: {src}")

def create_license():
    """MITライセンス作成"""
    license_text = f"""MIT License

Copyright (c) {datetime.now().year} NKAT Advanced Computing Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
    
    with open('nkat-transformer-standalone/LICENSE', 'w', encoding='utf-8') as f:
        f.write(license_text)
    print("✅ Created: LICENSE")

def create_examples():
    """使用例スクリプト作成"""
    
    # 1. クイックデモ
    quick_demo = '''#!/usr/bin/env python3
"""
Quick Demo - NKAT-Transformer
クイックデモ（5エポック軽量版）
"""

from nkat_core_standalone import NKATTrainer, NKATConfig

def main():
    print("🚀 NKAT-Transformer Quick Demo")
    print("5エポックの軽量デモを実行します...")
    
    # 軽量設定
    config = NKATConfig()
    config.num_epochs = 5
    config.batch_size = 32
    
    # 学習実行
    trainer = NKATTrainer(config)
    accuracy = trainer.train()
    
    print(f"\\n✅ デモ完了！")
    print(f"精度: {accuracy:.2f}%")
    print("本格的な99%+学習はnum_epochs=100で実行してください")

if __name__ == "__main__":
    main()
'''
    
    # 2. 本格訓練
    full_training = '''#!/usr/bin/env python3
"""
Full Training - NKAT-Transformer
本格的な99%+精度訓練
"""

from nkat_core_standalone import NKATTrainer, NKATConfig

def main():
    print("🎯 NKAT-Transformer Full Training")
    print("99%+精度を目指した本格訓練を開始します...")
    
    # 本格設定
    config = NKATConfig()
    config.num_epochs = 100
    config.batch_size = 64
    
    # GPUメモリが不足する場合の調整
    # config.batch_size = 32
    
    print(f"設定:")
    print(f"• エポック数: {config.num_epochs}")
    print(f"• バッチサイズ: {config.batch_size}")
    print(f"• 困難クラス: {config.difficult_classes}")
    
    # 学習実行
    trainer = NKATTrainer(config)
    accuracy = trainer.train()
    
    print(f"\\n🎉 訓練完了！")
    print(f"最終精度: {accuracy:.2f}%")
    
    if accuracy >= 99.0:
        print("🏆 99%+達成おめでとうございます！")
    else:
        print("📈 さらなる調整で99%+を目指しましょう")

if __name__ == "__main__":
    main()
'''
    
    # 3. カスタム設定例
    custom_example = '''#!/usr/bin/env python3
"""
Custom Configuration Example
カスタム設定例
"""

from nkat_core_standalone import NKATTrainer, NKATConfig

# カスタム設定クラス
class FastConfig(NKATConfig):
    """高速訓練用設定"""
    def __init__(self):
        super().__init__()
        self.num_epochs = 50
        self.batch_size = 128
        self.learning_rate = 2e-4

class PreciseConfig(NKATConfig):
    """高精度追求用設定"""
    def __init__(self):
        super().__init__()
        self.num_epochs = 200
        self.batch_size = 32
        self.learning_rate = 5e-5
        self.class_weight_boost = 2.0

def main():
    print("🔧 Custom Configuration Examples")
    
    choice = input("設定を選択してください (1: 高速, 2: 高精度): ")
    
    if choice == "1":
        config = FastConfig()
        print("⚡ 高速訓練設定を選択")
    elif choice == "2":
        config = PreciseConfig()
        print("🎯 高精度追求設定を選択")
    else:
        config = NKATConfig()
        print("📋 標準設定を使用")
    
    trainer = NKATTrainer(config)
    accuracy = trainer.train()
    
    print(f"結果: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
'''
    
    examples = {
        'nkat-transformer-standalone/examples/quick_demo.py': quick_demo,
        'nkat-transformer-standalone/examples/full_training.py': full_training,
        'nkat-transformer-standalone/examples/custom_config.py': custom_example
    }
    
    for filename, content in examples.items():
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Created: {filename}")

def create_test_script():
    """テストスクリプト作成"""
    test_script = '''#!/usr/bin/env python3
"""
Test Script - NKAT-Transformer
基本動作テスト
"""

import torch
import sys
import os

# パス追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nkat_core_standalone import NKATConfig, NKATVisionTransformer, load_pretrained

def test_model_creation():
    """モデル作成テスト"""
    print("🧪 Model Creation Test")
    
    config = NKATConfig()
    model = NKATVisionTransformer(config)
    
    # パラメータ数確認
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model created: {total_params:,} parameters")
    
    return model

def test_forward_pass():
    """順伝播テスト"""
    print("🔄 Forward Pass Test")
    
    config = NKATConfig()
    model = NKATVisionTransformer(config)
    
    # ダミー入力
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1, 28, 28)
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"✅ Forward pass: Input {dummy_input.shape} → Output {output.shape}")
    print(f"✅ Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    return output

def test_cuda_availability():
    """CUDA動作テスト"""
    print("🎮 CUDA Availability Test")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"✅ CUDA version: {torch.version.cuda}")
        print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️ CUDA not available - using CPU")

def main():
    print("🚀 NKAT-Transformer Test Suite")
    print("=" * 50)
    
    try:
        # CUDA テスト
        test_cuda_availability()
        print()
        
        # モデル作成テスト
        model = test_model_creation()
        print()
        
        # 順伝播テスト
        output = test_forward_pass()
        print()
        
        print("🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
'''
    
    with open('nkat-transformer-standalone/tests/test_basic.py', 'w', encoding='utf-8') as f:
        f.write(test_script)
    print("✅ Created: tests/test_basic.py")

def create_github_workflow():
    """GitHub Actions CI/CD作成"""
    workflow = '''name: NKAT-Transformer CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10"]

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run basic tests
      run: |
        python tests/test_basic.py
    
    - name: Run quick demo
      run: |
        python examples/quick_demo.py
'''
    
    os.makedirs('nkat-transformer-standalone/.github/workflows', exist_ok=True)
    with open('nkat-transformer-standalone/.github/workflows/ci.yml', 'w', encoding='utf-8') as f:
        f.write(workflow)
    print("✅ Created: .github/workflows/ci.yml")

def create_gitignore():
    """GitIgnore作成"""
    gitignore = '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyTorch
*.pth
*.pt

# Data
data/
*.csv
*.json

# Results
results/
figures/
logs/
checkpoints/
nkat_models/
nkat_reports/

# Jupyter
.ipynb_checkpoints

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Virtual environments
venv/
env/
ENV/

# Testing
.pytest_cache/
.coverage
htmlcov/

# Temporary files
*.tmp
*.log
'''
    
    with open('nkat-transformer-standalone/.gitignore', 'w', encoding='utf-8') as f:
        f.write(gitignore)
    print("✅ Created: .gitignore")

def create_setup_info():
    """セットアップ情報作成"""
    setup_info = {
        "name": "nkat-transformer",
        "version": "1.0.0",
        "description": "99%+ MNIST Vision Transformer - Lightweight & Educational",
        "author": "NKAT Advanced Computing Team",
        "license": "MIT",
        "python_requires": ">=3.8",
        "keywords": ["vision transformer", "pytorch", "mnist", "deep learning", "ai"],
        "classifiers": [
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Topic :: Scientific/Engineering :: Artificial Intelligence"
        ],
        "repository": "https://github.com/your-username/nkat-transformer",
        "documentation": "https://github.com/your-username/nkat-transformer/blob/main/README.md",
        "created": datetime.now().isoformat(),
        "features": [
            "99.20% MNIST accuracy",
            "Single file implementation",
            "Educational friendly",
            "Production ready",
            "CUDA optimized"
        ]
    }
    
    with open('nkat-transformer-standalone/setup_info.json', 'w', encoding='utf-8') as f:
        json.dump(setup_info, f, indent=2, ensure_ascii=False)
    print("✅ Created: setup_info.json")

def create_deployment_guide():
    """デプロイガイド作成"""
    guide = '''# NKAT-Transformer Deployment Guide

## GitHub Pages デプロイ

### 1. リポジトリ作成
```bash
cd nkat-transformer-standalone
git init
git add .
git commit -m "Initial commit: NKAT-Transformer v1.0.0"
git branch -M main
git remote add origin https://github.com/yourusername/nkat-transformer.git
git push -u origin main
```

### 2. リリース作成
1. GitHub → Releases → Create a new release
2. Tag: v1.0.0
3. Title: "NKAT-Transformer v1.0.0 - 99%+ MNIST Accuracy"
4. Description: READMEの主要部分を記載

### 3. GitHub Pages設定
1. Settings → Pages
2. Source: Deploy from a branch
3. Branch: main / (root)

## Note.com 記事投稿

### 記事構成
1. **導入**: 99%達成の成果
2. **技術解説**: Vision Transformer基礎
3. **実装詳細**: 独自改良点
4. **結果分析**: クラス別精度など
5. **コード公開**: GitHubリンク
6. **応用可能性**: 今後の展開

### 投稿スケジュール
- [ ] 技術解説記事
- [ ] 実装チュートリアル
- [ ] 結果分析記事
- [ ] 教育活用記事

## 宣伝・共有

### SNS
- Twitter: #AI #VisionTransformer #PyTorch #MNIST
- LinkedIn: 技術記事として投稿
- Qiita: 技術解説記事

### コミュニティ
- Reddit: r/MachineLearning, r/deeplearning
- Discord: AI関連サーバー
- Stack Overflow: 関連質問への回答

### 学術関連
- arXiv: 技術レポート投稿検討
- 学会: 教育セッションでの発表

## メンテナンス

### 定期更新
- [ ] PyTorchバージョン対応
- [ ] 新機能追加
- [ ] ドキュメント改善
- [ ] Issue対応

### バージョン管理
- v1.0.x: バグフィックス
- v1.1.x: 機能追加
- v2.0.x: 大幅改良

## 成功指標

### GitHub
- [ ] ⭐100+ Stars
- [ ] 🍴20+ Forks
- [ ] 📝10+ Issues/PRs

### Note
- [ ] 👀1000+ Views
- [ ] ❤️100+ Likes
- [ ] 💬50+ Comments

### 技術的インパクト
- [ ] 教育利用事例
- [ ] 研究引用
- [ ] 商用利用報告
'''
    
    with open('nkat-transformer-standalone/docs/deployment_guide.md', 'w', encoding='utf-8') as f:
        f.write(guide)
    print("✅ Created: docs/deployment_guide.md")

def main():
    """メイン実行"""
    print("🚀 NKAT-Transformer GitHub Deploy Setup")
    print("=" * 60)
    
    # 1. ディレクトリ構造作成
    print("\n📁 Creating directory structure...")
    create_directory_structure()
    
    # 2. コアファイルコピー
    print("\n📄 Copying core files...")
    copy_core_files()
    
    # 3. ライセンス作成
    print("\n📜 Creating license...")
    create_license()
    
    # 4. 使用例作成
    print("\n💻 Creating examples...")
    create_examples()
    
    # 5. テストスクリプト作成
    print("\n🧪 Creating test scripts...")
    create_test_script()
    
    # 6. GitHub Actions作成
    print("\n🔄 Creating GitHub workflows...")
    create_github_workflow()
    
    # 7. GitIgnore作成
    print("\n🚫 Creating .gitignore...")
    create_gitignore()
    
    # 8. セットアップ情報作成
    print("\n⚙️ Creating setup info...")
    create_setup_info()
    
    # 9. デプロイガイド作成
    print("\n📖 Creating deployment guide...")
    create_deployment_guide()
    
    print("\n" + "=" * 60)
    print("✅ GitHub Deploy Setup Complete!")
    print("=" * 60)
    print("\n📁 Output directory: nkat-transformer-standalone/")
    print("\n🚀 Next steps:")
    print("1. cd nkat-transformer-standalone")
    print("2. git init")
    print("3. git add .")
    print("4. git commit -m 'Initial commit'")
    print("5. Create GitHub repository")
    print("6. git remote add origin <your-repo-url>")
    print("7. git push -u origin main")
    print("\n📝 Note記事作成:")
    print("• Note発表用_記事テンプレート.md を参考に記事作成")
    print("• 画像・グラフを追加して公開")
    
    print("\n🎯 公開準備完了！")

if __name__ == "__main__":
    main() 