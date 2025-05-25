@echo off
chcp 65001 > nul
title NKAT v11 統合システム起動

echo ========================================
echo 🚀 NKAT v11 統合システム起動
echo 電源断対応・自動復旧・リアルタイム監視
echo ========================================
echo.

:: Python環境チェック
echo 🔍 Python環境をチェック中...
py -3 --version > nul 2>&1
if errorlevel 1 (
    echo ❌ Python 3が見つかりません
    echo    Python 3をインストールしてください
    pause
    exit /b 1
)

py -3 --version
echo ✅ Python環境OK
echo.

:: 作業ディレクトリ確認
echo 📁 作業ディレクトリ: %CD%
echo.

:: 必要ファイル確認
echo 🔍 システムファイルをチェック中...
if not exist "nkat_v11_integrated_launcher.py" (
    echo ❌ nkat_v11_integrated_launcher.py が見つかりません
    pause
    exit /b 1
)
echo ✅ 統合ランチャーファイルOK
echo.

:: 統合ランチャー起動
echo 🚀 NKAT v11 統合ランチャーを起動中...
echo.
py -3 nkat_v11_integrated_launcher.py

echo.
echo 👋 NKAT v11 システムを終了しました
pause 