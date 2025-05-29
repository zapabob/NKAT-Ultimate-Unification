#!/bin/bash
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
