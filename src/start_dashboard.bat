@echo off
echo 🚀 NKAT Research Dashboard 起動中...
echo ================================================================================
echo 📅 起動時刻: %date% %time%
echo 🔬 NKAT Version: v9.1 - Quantum Entanglement Revolution
echo 🌐 ダッシュボード: http://localhost:8501
echo ================================================================================
echo.

REM 必要なパッケージのインストールチェック
echo 📦 依存関係チェック中...
py -3 -c "import streamlit, plotly, psutil" 2>nul
if errorlevel 1 (
    echo ⚠️ 必要なパッケージがインストールされていません
    echo 📦 インストール中...
    py -3 -m pip install streamlit plotly psutil
    if errorlevel 1 (
        echo ❌ パッケージのインストールに失敗しました
        pause
        exit /b 1
    )
)

echo ✅ 依存関係チェック完了
echo.

REM Streamlitダッシュボード起動
echo 🚀 ダッシュボード起動中...
echo 💡 ブラウザが自動で開かない場合は、手動で http://localhost:8501 にアクセスしてください
echo 🛑 終了するには Ctrl+C を押してください
echo.

py -3 -m streamlit run nkat_streamlit_dashboard.py --server.port 8501 --server.address localhost

echo.
echo 📊 ダッシュボードが終了しました
pause 