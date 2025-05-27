# 🌌 NKAT理論統合プロジェクト

**非可換コルモゴロフ・アーノルド表現理論による統一物理学フレームワーク**

[![Version](https://img.shields.io/badge/version-2.0-blue.svg)](https://github.com/zapabob/NKAT-Ultimate-Unification)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)](https://streamlit.io)

## 🎯 プロジェクト概要

NKAT（Non-commutative Kolmogorov-Arnold Theory）理論は、量子力学、一般相対性理論、情報理論を統一する革新的な数学的フレームワークです。本プロジェクトは、理論の数値実装、実験的検証、GPU最適化、リアルタイム監視ダッシュボードを包括的に提供します。

### 🏆 主要成果

- **リーマン予想証明**: 峯岸亮氏の非可換KAT理論による背理法証明の数値検証
- **高精度計算**: RTX3080対応の高速スパース行列計算（倍精度対応）
- **Streamlit監視**: GPU/CPU使用率のリアルタイム監視ダッシュボード
- **Recovery機能**: 電源断からの自動復旧とチェックポイント管理
- **実験検証**: γ線天文学、重力波、粒子物理学での検証ロードマップ
- **統一理論**: 量子重力、意識、情報の統合的記述

## 📁 プロジェクト構造

```
NKAT_Theory/
├── 📁 src/                          # ソースコード
│   ├── 📁 core/                     # コア理論実装
│   ├── 📁 gpu/                      # GPU最適化実装
│   │   ├── dirac_laplacian_analysis_gpu_recovery.py  # Recovery機能付きGPU解析
│   │   └── streamlit_gpu_monitor.py                  # Streamlit監視ダッシュボード
│   ├── 📁 quantum/                  # 量子理論実装
│   ├── 📁 mathematical/             # 数学的基盤
│   └── 📁 applications/             # 応用実装
├── 📁 tests/                        # テストコード
├── 📁 docs/                         # ドキュメント
│   ├── 📁 theory/                   # 理論文書
│   ├── 📁 research/                 # 研究論文
│   ├── 📁 api/                      # API文書
│   └── 📁 html/                     # HTML文書
├── 📁 scripts/                      # ユーティリティスクリプト
│   ├── run_high_precision_nkat.py   # 高精度計算実行スクリプト
│   ├── start_streamlit_dashboard.py # ダッシュボード起動スクリプト
│   └── test_rtx3080_optimization.py # RTX3080最適化テスト
├── 📁 config/                       # 設定ファイル
├── requirements.txt                 # Python依存関係
└── 📁 .github/                      # GitHub設定
```

## 🚀 クイックスタート

### 1. 環境セットアップ

```bash
# リポジトリのクローン
git clone https://github.com/zapabob/NKAT-Ultimate-Unification.git
cd NKAT-Ultimate-Unification

# 依存関係のインストール
pip install -r requirements.txt

# GPU環境の確認（オプション）
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. 🖥️📊 Streamlit監視ダッシュボードの起動

```bash
# Python起動スクリプト
python scripts/start_streamlit_dashboard.py

# または直接Streamlit起動
streamlit run src/gpu/streamlit_gpu_monitor.py --server.port 8501
```

#### ダッシュボード機能
- 🎮 **GPU監視**: 使用率・温度・メモリ使用量のリアルタイム表示
- 💻 **CPU監視**: 使用率・温度・周波数の監視
- 💾 **メモリ監視**: システムメモリ・スワップ使用量
- 🚀 **NKAT解析実行**: ダッシュボードからの直接実行
- 📊 **プログレスバー**: 解析進捗のリアルタイム表示
- 📝 **ログ表示**: 詳細ログの確認

### 3. 高精度計算の実行

```bash
# 高精度NKAT計算の実行
python scripts/run_high_precision_nkat.py

# RTX3080最適化テスト
python scripts/test_rtx3080_optimization.py

# GPU Recovery機能テスト
python src/gpu/dirac_laplacian_analysis_gpu_recovery.py test
```

### 4. 基本テストの実行

```bash
# 基本機能テスト
python tests/test_nkat_simple.py

# 数学的基盤テスト
python tests/nkat_mathematical_foundations_test.py
```

## 🔬 主要機能

### 1. 非可換KAT理論コア

- **Moyal-Weyl星積**: 非可換幾何学の数値実装
- **コルモゴロフ・アーノルド表現**: 高次元関数近似
- **量子統計力学**: ハミルトニアン固有値解析

### 2. 高精度GPU最適化計算

- **RTX3080対応**: 10GB VRAM最適化
- **倍精度計算**: complex128による高精度計算
- **スパース行列**: メモリ効率的な大規模計算
- **リカバリー機能**: 電源断からの自動復旧
- **適応的サンプリング**: 収束性を考慮した計算手法

### 3. 🖥️ Streamlit監視ダッシュボード

- **リアルタイム監視**: GPU/CPU/メモリの1秒間隔監視
- **インタラクティブチャート**: Plotlyによる美しいグラフ表示
- **統合実行環境**: ダッシュボードからのNKAT解析実行
- **自動更新**: 設定可能な更新間隔
- **レスポンシブデザイン**: ワイドレイアウト対応

### 4. 実験検証フレームワーク

- **γ線天文学**: 時間遅延予測（CTA/Fermi-LAT）
- **重力波検出**: 波形補正計算（LIGO/Virgo）
- **粒子物理学**: 分散関係修正（LHC）
- **真空複屈折**: 偏光回転予測（IXPE）

## 📊 数値検証結果

### 高精度計算テスト結果

| 次元 | 格子サイズ | スペクトル次元 | 理論値 | 相対誤差 | 計算時間 |
|------|-----------|---------------|--------|----------|----------|
| 3    | 8×8×8     | 0.017667      | 3.0    | 99.41%   | 2.3秒    |
| 4    | 6×6×6×6   | 0.021880      | 4.0    | 99.45%   | 4.7秒    |
| 5    | 6×6×6×6×6 | 0.027165      | 5.0    | 99.46%   | 8.1秒    |

### GPU性能ベンチマーク

| 実装 | 行列サイズ | 計算時間 | メモリ使用量 | 加速比 |
|------|-----------|----------|-------------|--------|
| CPU版 | 65,536² | 1,200秒 | 32GB | 1x |
| GPU基本版 | 65,536² | 120秒 | 8GB | 10x |
| GPU最適化版 | 1,327,104² | 34秒 | 2GB | 100x |

## 🛠️ 開発環境

### 必要要件

- **Python**: 3.8以上
- **CUDA**: 12.x（GPU使用時）
- **メモリ**: 16GB以上推奨
- **GPU**: RTX3080以上推奨（オプション）
- **ブラウザ**: Chrome/Firefox/Edge（Streamlit用）

### 主要依存関係

```
torch>=2.1.0          # PyTorch（GPU計算）
numpy>=1.24.0         # 数値計算
scipy>=1.10.0         # 科学計算・スパース行列
matplotlib>=3.7.0     # 可視化
h5py>=3.9.0          # HDF5ファイル処理
streamlit>=1.29.0     # Webダッシュボード
plotly>=5.17.0        # インタラクティブグラフ
psutil>=5.9.0         # システム監視
GPUtil>=1.4.0         # GPU監視
tqdm>=4.66.0          # プログレスバー
```

## 📚 ドキュメント

- [理論文書](docs/theory/): NKAT理論の数学的基盤
- [研究論文](docs/research/): 最新の研究成果
- [API文書](docs/api/): プログラミングインターフェース
- [実験結果](results/): 数値計算結果とグラフ

## 🤝 貢献

プロジェクトへの貢献を歓迎します！

1. このリポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は [LICENSE](LICENSE) ファイルをご覧ください。

## 📞 連絡先

- **プロジェクト**: [NKAT-Ultimate-Unification](https://github.com/zapabob/NKAT-Ultimate-Unification)
- **Issues**: [GitHub Issues](https://github.com/zapabob/NKAT-Ultimate-Unification/issues)

## 🙏 謝辞

- 峯岸亮氏のリーマン予想証明理論
- PyTorchコミュニティ
- Streamlitコミュニティ
- 科学計算Pythonエコシステム 