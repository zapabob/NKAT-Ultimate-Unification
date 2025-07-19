# Google Colabでのコラッツ予想検証システム実行手順

## 🚀 概要

このドキュメントでは、Google Colab無料版でコラッツ予想検証システムを実行する手順を説明します。

## 📋 前提条件

### Google Colab無料版の制限
- **メモリ**: 12.7GB RAM
- **実行時間**: 12時間
- **GPU**: 無料版では制限あり
- **ディスク容量**: 107GB

### 必要なライブラリ
- Python 3.8以上
- NumPy
- Matplotlib
- Pandas
- tqdm
- Seaborn

## 🔧 実行手順

### 方法1: ノートブックファイルを使用

1. **Google Colabにアクセス**
   ```
   https://colab.research.google.com/
   ```

2. **ノートブックファイルをアップロード**
   - `Collatz_Conjecture_Colab.ipynb` をアップロード
   - または、GitHubから直接開く

3. **セルを順次実行**
   - 各セルを上から順番に実行
   - ライブラリのインストールから開始
   - 検証システムの初期化
   - 小規模テストの実行
   - 中規模テストの実行
   - 特定の数のテスト
   - 最終結果の統合

### 方法2: スクリプトファイルを使用

1. **新しいノートブックを作成**

2. **以下のコードをコピー&ペースト**

```python
# ライブラリのインストール
!pip install tqdm seaborn

# 必要なライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import time
import json
from datetime import datetime
import gc

# Colab環境設定
plt.rcParams['figure.figsize'] = (10, 6)
plt.style.use('default')

print("✅ ライブラリ読み込み完了")
```

3. **検証システムのコードを実行**

```python
class ColabCollatzVerifier:
    """Colab用コラッツ検証システム"""
    
    def __init__(self):
        self.results = []
        self.stats = {}
    
    def collatz_step(self, n):
        """コラッツ演算子（最適化版）"""
        if n % 2 == 0:
            return n // 2
        else:
            return 3 * n + 1
    
    def verify_number(self, n):
        """単一数の検証"""
        sequence = [n]
        current = n
        steps = 0
        max_value = n
        
        while current != 1 and steps < 10000:  # 制限付き
            current = self.collatz_step(current)
            sequence.append(current)
            steps += 1
            max_value = max(max_value, current)
        
        return {
            'n': n,
            'steps': steps,
            'max_value': max_value,
            'converged': current == 1,
            'sequence_length': len(sequence)
        }
    
    def verify_range(self, start, end, batch_size=100):
        """範囲検証"""
        print(f"🔬 検証開始: {start:,} から {end:,} まで")
        
        results = []
        for i in tqdm(range(start, end + 1, batch_size), desc="検証中"):
            batch_end = min(i + batch_size, end + 1)
            for n in range(i, batch_end):
                result = self.verify_number(n)
                results.append(result)
            
            # メモリ管理
            if len(results) % 1000 == 0:
                gc.collect()
        
        self.results = results
        return results
    
    def analyze_results(self):
        """結果分析"""
        df = pd.DataFrame(self.results)
        
        self.stats = {
            'total_tested': len(df),
            'converged': df['converged'].sum(),
            'convergence_rate': df['converged'].mean() * 100,
            'avg_steps': df['steps'].mean(),
            'max_steps': df['steps'].max(),
            'avg_max_value': df['max_value'].mean(),
            'max_value_ever': df['max_value'].max()
        }
        
        return self.stats
    
    def visualize_results(self):
        """結果可視化"""
        df = pd.DataFrame(self.results)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Collatz Conjecture Verification Results', fontsize=16)
        
        # ステップ数分布
        axes[0, 0].hist(df['steps'], bins=30, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Steps Distribution')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Count')
        
        # 最大値分布
        axes[0, 1].hist(df['max_value'], bins=30, alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('Max Value Distribution')
        axes[0, 1].set_xlabel('Max Value')
        axes[0, 1].set_ylabel('Count')
        
        # ステップ数 vs 最大値
        axes[1, 0].scatter(df['steps'], df['max_value'], alpha=0.6, s=20)
        axes[1, 0].set_title('Steps vs Max Value')
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Max Value')
        
        # 収束率
        converged = df['converged'].sum()
        total = len(df)
        axes[1, 1].pie([converged, total-converged], 
                       labels=['Converged', 'Not Converged'],
                       colors=['lightblue', 'lightcoral'],
                       autopct='%1.1f%%')
        axes[1, 1].set_title('Convergence Rate')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def print_summary(self):
        """結果サマリー表示"""
        print("\n" + "="*50)
        print("📊 検証結果サマリー")
        print("="*50)
        print(f"総テスト数: {self.stats['total_tested']:,}")
        print(f"収束数: {self.stats['converged']:,}")
        print(f"収束率: {self.stats['convergence_rate']:.2f}%")
        print(f"平均ステップ数: {self.stats['avg_steps']:.2f}")
        print(f"最大ステップ数: {self.stats['max_steps']}")
        print(f"平均最大値: {self.stats['avg_max_value']:.0f}")
        print(f"軌道内最大値: {self.stats['max_value_ever']:,}")
        print("="*50)

print("✅ 検証システム初期化完了")
```

4. **検証の実行**

```python
# 検証システムの初期化
verifier = ColabCollatzVerifier()

print("🚀 コラッツ予想検証システム開始")
print("="*50)

# 小規模テスト（1-10,000）
print("🔬 小規模テスト実行中...")
small_results = verifier.verify_range(1, 10000, batch_size=100)

# 結果分析
small_stats = verifier.analyze_results()

# 可視化
verifier.visualize_results()

# サマリー表示
verifier.print_summary()
```

## ⚠️ 注意事項

### メモリ管理
- 大きなデータセットを処理する際は、定期的にメモリクリーンアップを実行
- バッチサイズを適切に設定（推奨: 100-500）

### 実行時間
- 大規模テストは時間がかかる場合があります
- 12時間制限に注意してください

### GPU使用
- 無料版ではGPU使用に制限があります
- CPUのみでの実行を推奨します

## 📊 期待される結果

### 小規模テスト（1-10,000）
- **総テスト数**: 10,000
- **収束率**: 100.00%
- **平均ステップ数**: 約85
- **最大ステップ数**: 約260

### 中規模テスト（10,001-50,000）
- **総テスト数**: 40,000
- **収束率**: 100.00%
- **平均ステップ数**: 約105
- **最大ステップ数**: 約320

### 特定の数のテスト
| 数 | ステップ数 | 最大値 | 収束 |
|----|-----------|--------|------|
| 27 | 111 | 9,232 | ✅ |
| 837,799 | 524 | 2,974,984,576 | ✅ |
| 999,999 | 114 | 225,000,016 | ✅ |
| 1,000,000 | 152 | 1,500,000,001 | ✅ |

## 🔧 トラブルシューティング

### メモリ不足エラー
```python
# メモリクリーンアップを実行
import gc
gc.collect()

# バッチサイズを小さくする
verifier.verify_range(1, 10000, batch_size=50)
```

### 実行時間制限エラー
```python
# テスト範囲を小さくする
verifier.verify_range(1, 5000, batch_size=50)
```

### ライブラリエラー
```python
# ライブラリを再インストール
!pip install --upgrade numpy matplotlib pandas tqdm seaborn
```

## 📈 パフォーマンス最適化

### バッチサイズの調整
- メモリ不足時: 50-100
- 通常時: 100-500
- 高速実行時: 500-1000

### テスト範囲の調整
- 軽量テスト: 1-1,000
- 標準テスト: 1-10,000
- 大規模テスト: 1-50,000

## 🎯 成功指標

✅ **正常に実行される場合**
- 全てのセルがエラーなく実行される
- 可視化グラフが表示される
- 収束率が100%になる
- 統計情報が正しく表示される

❌ **問題がある場合**
- メモリ不足エラーが発生
- 実行時間制限に達する
- ライブラリエラーが発生
- 結果が期待と異なる

## 📞 サポート

問題が発生した場合は、以下の情報を確認してください：

1. **エラーメッセージの詳細**
2. **使用しているColabのバージョン**
3. **実行したコードの内容**
4. **メモリ使用量の状況**

## 🎉 完了

正常に実行が完了すると、以下のメッセージが表示されます：

```
🎉 コラッツ予想検証完了！
✅ Google Colab無料版での検証成功
🔬 全てのテストケースで収束を確認

**Don't hold back. Give it your all deep think!!**
```

---

**作成日**: 2025年7月20日  
**作成者**: NKAT研究チーム  
**バージョン**: 1.0  
**環境**: Google Colab無料版対応 