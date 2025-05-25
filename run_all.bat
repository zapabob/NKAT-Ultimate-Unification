@echo off
REM NKAT理論 - 弦理論・ホログラフィック統合フレームワーク
REM ワンクリック実行スクリプト (Windows)
REM Version: 2025-05-24

echo ================================================================================
echo 🎯 NKAT理論 - 弦理論・ホログラフィック統合フレームワーク
echo ================================================================================
echo 📅 実行日時: %date% %time%
echo 🔬 統合理論: 弦理論 + ホログラフィック原理 + AdS/CFT + 量子重力
echo 🌌 高次元理論: 超対称性 + M理論 + カラビ・ヤウ多様体
echo ================================================================================

REM 現在のディレクトリを保存
set ORIGINAL_DIR=%CD%

REM スクリプトのディレクトリに移動
cd /d "%~dp0"

REM Pythonの確認
echo 🔍 Python環境の確認...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Pythonが見つかりません。Python 3.8以上をインストールしてください。
    pause
    exit /b 1
)

REM 依存関係のインストール
echo 📦 依存関係のインストール...
pip install -r requirements.txt

if errorlevel 1 (
    echo ⚠️ 依存関係のインストールに失敗しました。手動でインストールしてください。
    echo pip install -r requirements.txt
    pause
)

REM srcディレクトリに移動
cd src

echo.
echo 🚀 NKAT理論フレームワーク実行開始...
echo.

REM 1. 弦理論・ホログラフィック統合フレームワーク
echo 🌌 1. 弦理論・ホログラフィック統合フレームワーク実行中...
python riemann_string_holographic_framework.py
if errorlevel 1 (
    echo ❌ 弦理論・ホログラフィック統合フレームワークの実行に失敗しました。
) else (
    echo ✅ 弦理論・ホログラフィック統合フレームワーク完了
)

echo.

REM 2. AdS/CFT対応可視化
echo 🎨 2. AdS/CFT対応可視化実行中...
python plot_ads_cft_correspondence.py
if errorlevel 1 (
    echo ❌ AdS/CFT対応可視化の実行に失敗しました。
) else (
    echo ✅ AdS/CFT対応可視化完了
)

echo.

REM 3. 超高精度16⁴格子検証 (オプション - 時間がかかる場合)
echo 🔬 3. 超高精度16⁴格子検証を実行しますか？ (時間がかかります)
set /p choice="実行する場合は 'y' を入力してください (y/n): "
if /i "%choice%"=="y" (
    echo 🧮 超高精度16⁴格子検証実行中...
    python riemann_ultra_precision_16_lattice.py
    if errorlevel 1 (
        echo ❌ 超高精度16⁴格子検証の実行に失敗しました。
    ) else (
        echo ✅ 超高精度16⁴格子検証完了
    )
) else (
    echo ⏭️ 超高精度16⁴格子検証をスキップしました。
)

echo.

REM 4. 高精度リーマン予想検証 (軽量版)
echo 🎯 4. 高精度リーマン予想検証実行中...
python riemann_high_precision.py
if errorlevel 1 (
    echo ❌ 高精度リーマン予想検証の実行に失敗しました。
) else (
    echo ✅ 高精度リーマン予想検証完了
)

echo.

REM 結果の確認
echo 📊 生成された結果ファイル:
echo.
if exist "string_holographic_ultimate_results.json" (
    echo ✅ string_holographic_ultimate_results.json
)
if exist "ads_cft_holographic_correspondence.png" (
    echo ✅ ads_cft_holographic_correspondence.png
)
if exist "ads_cft_holographic_analysis.json" (
    echo ✅ ads_cft_holographic_analysis.json
)
if exist "ultra_precision_16_lattice_results_*.json" (
    echo ✅ ultra_precision_16_lattice_results_*.json
)
if exist "high_precision_riemann_results.json" (
    echo ✅ high_precision_riemann_results.json
)

echo.
echo 🎉 NKAT理論フレームワーク実行完了！
echo.
echo 📈 結果の概要:
echo   - 弦理論・ホログラフィック統合による数値検証
echo   - AdS/CFT対応の可視化
echo   - リーマン予想の高精度数値検証
echo   - 非可換幾何学による新しいアプローチ
echo.
echo 📁 結果ファイルは src ディレクトリに保存されています。
echo 🌟 論文・研究発表にご活用ください！
echo.

REM 元のディレクトリに戻る
cd /d "%ORIGINAL_DIR%"

echo 実行完了。何かキーを押してください...
pause >nul 