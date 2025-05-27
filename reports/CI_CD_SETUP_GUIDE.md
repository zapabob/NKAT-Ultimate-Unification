# 🤖 NKAT理論フレームワーク CI/CD セットアップガイド

## 概要

NKAT理論GPU加速フレームワークの **GitHub Actions CI/CD** システムの詳細セットアップガイドです。

## 🚀 Phase ② `ci-enable` 完了内容

### 1. GitHub Actions ワークフロー

**ファイル**: `.github/workflows/nkat_gpu_ci.yml`

#### 🔄 自動実行トリガー
- **Push**: `main`, `develop` ブランチ
- **Pull Request**: `main` ブランチ
- **定期実行**: 毎週月曜日 6:00 UTC
- **手動実行**: GitHub UI から `workflow_dispatch`

#### 📊 実行ジョブ構成

| ジョブ | 実行時間 | 説明 |
|--------|----------|------|
| `cpu-benchmark` | 30分 | CPU版基本動作確認 |
| `gpu-benchmark` | 45分 | GPU版CUDA環境実行 |
| `performance-analysis` | 15分 | CPU vs GPU 比較分析 |
| `create-summary` | 10分 | 実行結果サマリー生成 |

### 2. ベンチマーク CLI ツール

**ファイル**: `src/bench_gpu.py`

```bash
# 基本実行
python src/bench_gpu.py --maxN 10

# 詳細出力
python src/bench_gpu.py --maxN 10 --verbose

# カスタム設定
python src/bench_gpu.py --maxN 12 --precision complex128 --eig 256
```

### 3. 依存関係更新

**ファイル**: `requirements.txt`

CI/CD用追加パッケージ:
- `psutil>=5.9.0` - システム監視
- `pytest>=7.4.0` - テストフレームワーク
- `pytest-cov>=4.0.0` - カバレッジ測定
- `pyyaml>=6.0` - 設定ファイル

## 🛠️ セットアップ手順

### Step 1: リポジトリ準備

```bash
# 1. リポジトリクローン
git clone https://github.com/your-username/NKAT-Ultimate-Unification.git
cd NKAT-Ultimate-Unification

# 2. ブランチ作成
git checkout -b ci-enable

# 3. ファイル確認
ls -la .github/workflows/
ls -la src/bench_gpu.py
```

### Step 2: GitHub リポジトリ設定

#### 2.1 Secrets 設定（必要に応じて）

GitHub リポジトリの Settings > Secrets and variables > Actions で設定:

```
# GPU環境用（オプション）
CUDA_VERSION: "12.1"
TORCH_VERSION: "2.2.0"

# 通知用（オプション）  
SLACK_WEBHOOK_URL: "https://hooks.slack.com/..."
```

#### 2.2 Actions 有効化

1. GitHub リポジトリの **Actions** タブを開く
2. **I understand my workflows, go ahead and enable them** をクリック
3. ワークフローが表示されることを確認

### Step 3: 初回実行

```bash
# 1. 変更をコミット
git add .github/workflows/nkat_gpu_ci.yml
git add src/bench_gpu.py
git add requirements.txt
git add README.md
git commit -m "CI: add GPU benchmark workflow + CLI tools"

# 2. プッシュして自動実行開始
git push origin ci-enable

# 3. GitHub Actions 確認
# https://github.com/your-username/NKAT-Ultimate-Unification/actions
```

### Step 4: 結果確認

#### 4.1 実行ログ確認

1. GitHub Actions タブで実行状況を確認
2. 各ジョブのログを詳細確認
3. エラーがある場合は修正

#### 4.2 Artifacts ダウンロード

実行完了後、以下のファイルがダウンロード可能:

- `cpu-benchmark-results` - CPU版結果
- `gpu-benchmark-results` - GPU版結果
- `performance-analysis` - 比較分析図
- `ci-summary` - 実行サマリー

## 📊 CI/CD 実行例

### 成功例

```
🚀 NKAT GPU加速理論フレームワーク CI/CD

✅ cpu-benchmark (30分)
   - 基本ライブラリ動作確認: 完了
   - CPU版8³格子ベンチマーク: 完了
   - 精度検証: 25.4% > 10% ✅
   - 成功率検証: 40% > 20% ✅

✅ gpu-benchmark (45分)
   - CUDA環境確認: GPU検出 ✅
   - GPU版8³格子ベンチマーク: 完了
   - GPU版10³格子ベンチマーク: 完了
   - 精度検証: 60.38% > 30% ✅
   - 計算時間: 0.83秒 < 60秒 ✅

✅ performance-analysis (15分)
   - CPU vs GPU 比較分析: 完了
   - 改善率計算: 精度96.7%改善, 速度57×向上
   - 可視化生成: ci_performance_comparison.png

✅ create-summary (10分)
   - 実行サマリー生成: 完了
   - Artifacts 保存: 4個のファイル
```

### エラー対処例

#### GPU未検出エラー

```yaml
# .github/workflows/nkat_gpu_ci.yml の修正
gpu-benchmark:
  runs-on: ubuntu-latest
  container:
    image: nvidia/cuda:12.1.1-devel-ubuntu22.04
    options: --gpus all  # この行を追加
```

#### 依存関係エラー

```bash
# requirements.txt に追加
cupy-cuda12x>=12.0.0  # コメントアウトを解除
```

#### メモリ不足エラー

```python
# src/bench_gpu.py の修正
# 格子サイズを小さく調整
lattice_sizes = [6, 8]  # 10 → 8 に変更
```

## 🔧 カスタマイズ

### 1. 実行頻度の変更

```yaml
# .github/workflows/nkat_gpu_ci.yml
schedule:
  - cron: '0 6 * * 1'     # 毎週月曜日
  # - cron: '0 6 * * *'   # 毎日
  # - cron: '0 6 1 * *'   # 毎月1日
```

### 2. 格子サイズの調整

```python
# src/bench_gpu.py
def benchmark_performance(max_lattice_size=10):
    lattice_sizes = [8]
    if max_lattice_size >= 10:
        lattice_sizes.append(10)
    if max_lattice_size >= 12:
        lattice_sizes.append(12)  # 大きな格子を追加
```

### 3. 通知の追加

```yaml
# .github/workflows/nkat_gpu_ci.yml に追加
- name: 📧 Slack通知
  if: always()
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    webhook_url: ${{ secrets.SLACK_WEBHOOK_URL }}
```

## 📈 パフォーマンス監視

### 1. 実行時間の追跡

```bash
# 実行時間ログの確認
grep "実行時間" ci_summary.md
grep "計算時間" ci_gpu_lattice*.json
```

### 2. 精度の追跡

```python
# 精度推移の分析
import json
import matplotlib.pyplot as plt

# 複数回の実行結果を比較
results = []
for file in ['ci_gpu_lattice8_*.json']:
    with open(file) as f:
        data = json.load(f)
        precision = data['performance_metrics']['precision_achieved']
        results.append(float(precision.replace('%', '')))

plt.plot(results)
plt.ylabel('理論予測精度 (%)')
plt.title('CI/CD 精度推移')
plt.show()
```

### 3. リソース使用量の監視

```python
# src/bench_gpu.py に追加
import psutil

def monitor_resources():
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    print(f"CPU使用率: {cpu_percent}%")
    print(f"メモリ使用率: {memory_percent}%")
```

## 🚀 次のフェーズ候補

### Phase ③ `holo-viz`
AdS/CFT ホログラフィック可視化スクリプト生成

### Phase ④ `pack-release`
Zenodo & arXiv 用 ZIP＋DOI パッケージ作成

### Phase ⑤ `mail-send`
CTA / LIGO 連絡メール自動生成

### Phase ⑥ `multi-gpu`
NVLink 対応マルチ GPU スケルトン

## 📞 サポート

### トラブルシューティング

1. **GitHub Actions ログ確認**
   - Actions タブ → 該当ワークフロー → 詳細ログ

2. **ローカル再現**
   ```bash
   # 同じ環境でローカル実行
   python src/bench_gpu.py --maxN 8
   ```

3. **Issue 報告**
   - GitHub Issues で詳細な実行ログと共に報告

### 連絡先

- **GitHub Issues**: [NKAT-Ultimate-Unification/issues](https://github.com/your-username/NKAT-Ultimate-Unification/issues)
- **研究チーム**: NKAT Research Team

---

**🎯 Phase ② `ci-enable` 完了**

**GitHub Actions**: ✅ 完全対応  
**自動ベンチマーク**: ✅ 4ジョブ構成  
**CLI ツール**: ✅ bench_gpu.py  
**依存関係**: ✅ CI/CD対応  

**NKAT Research Team - 2025年5月24日** 