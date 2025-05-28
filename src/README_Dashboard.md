# 🚀 NKAT Research Dashboard

## 📋 概要

NKAT v9.1用のStreamlitダッシュボードは、リーマン予想研究の進捗をリアルタイムで監視できるWebベースの統合監視システムです。バージョンが変わっても継続的に使用できるよう、柔軟な設計になっています。

## 🎯 主要機能

### 🖥️ システム監視
- **CPU使用率**: リアルタイム監視
- **メモリ使用率**: システム・GPU メモリ
- **GPU状況**: CUDA対応、VRAM使用量
- **温度監視**: システム温度（対応ハードウェア）

### 🔬 量子もつれ解析
- **Concurrence**: もつれ度の可視化
- **Entanglement Entropy**: エンタングルメント・エントロピー
- **Negativity**: 負性による分離可能性判定
- **Quantum Discord**: 量子不協和
- **Bell Violation**: ベル不等式違反度

### 🎯 10,000γ Challenge 監視
- **進捗状況**: リアルタイム進捗率
- **チェックポイント**: 自動保存状況
- **結果統計**: スペクトル次元、収束性
- **実行時間**: 推定完了時刻

### 📁 ファイル管理
- **結果ファイル**: 自動検索・表示
- **チェックポイント**: バックアップ状況
- **バージョン情報**: システム詳細

## 🚀 起動方法

### 方法1: バッチファイル（推奨）
```bash
# Windowsの場合
start_dashboard.bat
```

### 方法2: 直接実行
```bash
# 依存関係インストール
py -3 -m pip install -r requirements_dashboard.txt

# ダッシュボード起動
py -3 -m streamlit run nkat_streamlit_dashboard.py
```

### 方法3: カスタムポート
```bash
py -3 -m streamlit run nkat_streamlit_dashboard.py --server.port 8502
```

## 🌐 アクセス方法

ダッシュボード起動後、ブラウザで以下にアクセス：
- **ローカル**: http://localhost:8501
- **ネットワーク**: http://[IPアドレス]:8501

## 🎛️ 使用方法

### 基本操作
1. **サイドバー**: 表示項目の選択
2. **更新ボタン**: 手動データ更新
3. **自動更新**: 定期的な自動更新設定

### 表示オプション
- ✅ **システム状況**: CPU/GPU/メモリ監視
- ✅ **量子もつれ解析**: エンタングルメント結果
- ✅ **10,000γ Challenge**: 進捗・結果
- ⬜ **ファイルブラウザ**: ファイル一覧
- ⬜ **バージョン情報**: システム詳細

### 自動更新設定
- **更新間隔**: 5-60秒で設定可能
- **リアルタイム監視**: 長時間実行時に便利

## 📊 データソース

ダッシュボードは以下のファイルを自動検索：

### 結果ファイル
```
10k_gamma_results/
analysis_results/
results/
../10k_gamma_results/
../analysis_results/
../results/
```

### チェックポイント
```
10k_gamma_checkpoints_production/
10k_gamma_checkpoints/
checkpoints/
../10k_gamma_checkpoints_production/
../checkpoints/
```

### 検索パターン
- **量子もつれ**: `*entanglement*.json`
- **10Kγ結果**: `*10k*gamma*.json`
- **チェックポイント**: `checkpoint_batch_*.json`

## 🔧 カスタマイズ

### パス設定
```python
# nkat_streamlit_dashboard.py の NKATDashboard.__init__() で設定
self.results_paths = [
    "your_custom_results_path",
    # 追加パス
]
```

### 表示項目追加
```python
def display_custom_analysis(self):
    """カスタム解析結果表示"""
    st.header("🔬 カスタム解析")
    # カスタム実装
```

## 🛠️ トラブルシューティング

### よくある問題

#### 1. ダッシュボードが起動しない
```bash
# 依存関係の再インストール
py -3 -m pip install --upgrade streamlit plotly psutil
```

#### 2. データが表示されない
- 結果ファイルのパスを確認
- ファイルの権限を確認
- ファイル形式（JSON）を確認

#### 3. GPU情報が表示されない
```bash
# PyTorchのCUDA対応確認
py -3 -c "import torch; print(torch.cuda.is_available())"
```

#### 4. ポートが使用中
```bash
# 別のポートで起動
py -3 -m streamlit run nkat_streamlit_dashboard.py --server.port 8502
```

### ログ確認
```bash
# Streamlitログの確認
py -3 -m streamlit run nkat_streamlit_dashboard.py --logger.level debug
```

## 📈 パフォーマンス最適化

### 大量データ処理
- **データフィルタリング**: 表示データの制限
- **キャッシュ活用**: `@st.cache_data` デコレータ
- **非同期読み込み**: バックグラウンド処理

### メモリ使用量削減
```python
# データフレームの最適化
df = df.astype({'column': 'category'})  # カテゴリ型使用
df = df.sample(n=1000)  # サンプリング
```

## 🔄 バージョン対応

### 新バージョン対応
1. **ファイルパターン追加**: 新しい結果ファイル形式
2. **メトリクス追加**: 新しい解析指標
3. **表示項目拡張**: 新機能の可視化

### 後方互換性
- 古いファイル形式の自動検出
- エラー処理による堅牢性
- デフォルト値による安全性

## 📚 参考資料

### Streamlit公式
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Documentation](https://plotly.com/python/)

### NKAT関連
- `NKAT_v91_Comprehensive_Report.md`: 詳細仕様
- `nkat_v91_quantum_entanglement.py`: 量子もつれ実装
- `nkat_10000_gamma_challenge_robust.py`: 10Kγシステム

---

**🎉 NKAT Research Dashboard で研究進捗を効率的に監視しましょう！**

*NKAT Research Consortium*  
*2025年5月26日*  
*Dashboard Version 1.0.0* 