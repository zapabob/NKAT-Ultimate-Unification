# Non-Commutative Kolmogorov-Arnold Transformer (NKAT)

非可換コルモゴロフ・アーノルドの定理に基づくTransformerモデルの実装。

## 概要

このリポジトリは、非可換コルモゴロフ・アーノルドの定理を拡張したTransformerモデル（NKAT）の実装を含みます。このモデルは、非可換性を考慮した特徴抽出と、量子情報理論との関連性を持つ新しいアーキテクチャを提供します。

## 主な特徴

- 非可換性を考慮したマルチヘッドアテンション
- 量子情報理論との関連性
- 高次元データの効率的な処理
- 転移学習のサポート

## インストール

```bash
git clone https://github.com/zapabob/NKAT.git
cd NKAT
pip install -r requirements.txt
```

## 使用方法

### MNISTでの学習

```python
python nkat_implementation.py
```

### 転移学習（Fashion-MNIST）

```python
python nkat_implementation.py --transfer
```

## モデルアーキテクチャ

- 入力次元: 784 (28x28)
- モデル次元: 256
- ヘッド数: 8
- レイヤー数: 4
- フィードフォワード次元: 1024
- 非可換性パラメータ: 0.1

## 性能

- MNIST: 97.76% 精度
- Fashion-MNIST: 転移学習による高い性能

## ライセンス

MIT License 