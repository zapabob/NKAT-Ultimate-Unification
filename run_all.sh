#!/bin/bash
# NKAT理論 - 弦理論・ホログラフィック統合フレームワーク
# ワンクリック実行スクリプト (Linux/Mac)
# Version: 2025-05-24

echo "================================================================================"
echo "🎯 NKAT理論 - 弦理論・ホログラフィック統合フレームワーク"
echo "================================================================================"
echo "📅 実行日時: $(date)"
echo "🔬 統合理論: 弦理論 + ホログラフィック原理 + AdS/CFT + 量子重力"
echo "🌌 高次元理論: 超対称性 + M理論 + カラビ・ヤウ多様体"
echo "================================================================================"

# 現在のディレクトリを保存
ORIGINAL_DIR=$(pwd)

# スクリプトのディレクトリに移動
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Pythonの確認
echo "🔍 Python環境の確認..."
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "❌ Pythonが見つかりません。Python 3.8以上をインストールしてください。"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo "✅ Python確認完了: $($PYTHON_CMD --version)"

# 依存関係のインストール
echo "📦 依存関係のインストール..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "⚠️ 依存関係のインストールに失敗しました。手動でインストールしてください。"
    echo "pip install -r requirements.txt"
fi

# srcディレクトリに移動
cd src

echo ""
echo "🚀 NKAT理論フレームワーク実行開始..."
echo ""

# 1. 弦理論・ホログラフィック統合フレームワーク
echo "🌌 1. 弦理論・ホログラフィック統合フレームワーク実行中..."
$PYTHON_CMD riemann_string_holographic_framework.py
if [ $? -eq 0 ]; then
    echo "✅ 弦理論・ホログラフィック統合フレームワーク完了"
else
    echo "❌ 弦理論・ホログラフィック統合フレームワークの実行に失敗しました。"
fi

echo ""

# 2. AdS/CFT対応可視化
echo "🎨 2. AdS/CFT対応可視化実行中..."
$PYTHON_CMD plot_ads_cft_correspondence.py
if [ $? -eq 0 ]; then
    echo "✅ AdS/CFT対応可視化完了"
else
    echo "❌ AdS/CFT対応可視化の実行に失敗しました。"
fi

echo ""

# 3. 超高精度16⁴格子検証 (オプション - 時間がかかる場合)
echo "🔬 3. 超高精度16⁴格子検証を実行しますか？ (時間がかかります)"
read -p "実行する場合は 'y' を入力してください (y/n): " choice
if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
    echo "🧮 超高精度16⁴格子検証実行中..."
    $PYTHON_CMD riemann_ultra_precision_16_lattice.py
    if [ $? -eq 0 ]; then
        echo "✅ 超高精度16⁴格子検証完了"
    else
        echo "❌ 超高精度16⁴格子検証の実行に失敗しました。"
    fi
else
    echo "⏭️ 超高精度16⁴格子検証をスキップしました。"
fi

echo ""

# 4. 高精度リーマン予想検証 (軽量版)
echo "🎯 4. 高精度リーマン予想検証実行中..."
$PYTHON_CMD riemann_high_precision.py
if [ $? -eq 0 ]; then
    echo "✅ 高精度リーマン予想検証完了"
else
    echo "❌ 高精度リーマン予想検証の実行に失敗しました。"
fi

echo ""

# 結果の確認
echo "📊 生成された結果ファイル:"
echo ""
if [ -f "string_holographic_ultimate_results.json" ]; then
    echo "✅ string_holographic_ultimate_results.json"
fi
if [ -f "ads_cft_holographic_correspondence.png" ]; then
    echo "✅ ads_cft_holographic_correspondence.png"
fi
if [ -f "ads_cft_holographic_analysis.json" ]; then
    echo "✅ ads_cft_holographic_analysis.json"
fi
if ls ultra_precision_16_lattice_results_*.json 1> /dev/null 2>&1; then
    echo "✅ ultra_precision_16_lattice_results_*.json"
fi
if [ -f "high_precision_riemann_results.json" ]; then
    echo "✅ high_precision_riemann_results.json"
fi

echo ""
echo "🎉 NKAT理論フレームワーク実行完了！"
echo ""
echo "📈 結果の概要:"
echo "  - 弦理論・ホログラフィック統合による数値検証"
echo "  - AdS/CFT対応の可視化"
echo "  - リーマン予想の高精度数値検証"
echo "  - 非可換幾何学による新しいアプローチ"
echo ""
echo "📁 結果ファイルは src ディレクトリに保存されています。"
echo "🌟 論文・研究発表にご活用ください！"
echo ""

# 元のディレクトリに戻る
cd "$ORIGINAL_DIR"

echo "実行完了。" 