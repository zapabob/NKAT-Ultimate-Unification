# 🚀 NKAT システム インストールガイド

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
nkat_env\Scripts\activate   # Windows
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
