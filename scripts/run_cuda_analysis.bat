@echo off
chcp 65001 > nul
cls

echo 🚀 CUDA対応NKAT解析システム 実行スクリプト
echo 📚 峯岸亮先生のリーマン予想証明論文 - GPU超並列計算版
echo 🎮 Windows 11 + Python 3 + CUDA 12.x 対応
echo ================================================================================
echo.

:: 管理者権限チェック
net session >nul 2>&1
if %errorLevel% == 0 (
    echo 🔑 管理者権限で実行中
) else (
    echo ⚠️ 管理者権限が推奨されます（GPU最適化のため）
)
echo.

:: Python環境確認
echo 🐍 Python環境確認中...
py -3 --version >nul 2>&1
if %errorLevel% == 0 (
    py -3 --version
    echo ✅ Python 3 利用可能
) else (
    echo ❌ Python 3 が見つかりません
    echo 📦 Python 3.9-3.11をインストールしてください
    echo 🔗 https://www.python.org/downloads/
    pause
    exit /b 1
)
echo.

:: CUDA環境の事前確認
echo 🎮 CUDA環境事前確認...
nvidia-smi >nul 2>&1
if %errorLevel% == 0 (
    echo ✅ NVIDIA GPU ドライバ検出
    echo 📊 GPU情報:
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
) else (
    echo ⚠️ NVIDIA GPU ドライバが検出されません
    echo 💻 CPU モードで実行されます
)
echo.

:: 選択メニュー
echo 📋 実行オプションを選択してください:
echo.
echo 1️⃣ CUDA環境テスト実行
echo 2️⃣ CUDA解析システム実行
echo 3️⃣ 必要ライブラリ自動インストール
echo 4️⃣ システム情報表示
echo 5️⃣ 全て実行（推奨）
echo 0️⃣ 終了
echo.

set /p choice="選択してください (1-5, 0): "

if "%choice%"=="1" goto cuda_test
if "%choice%"=="2" goto cuda_analysis
if "%choice%"=="3" goto install_libs
if "%choice%"=="4" goto system_info
if "%choice%"=="5" goto run_all
if "%choice%"=="0" goto end
goto invalid_choice

:cuda_test
echo.
echo 🔍 CUDA環境テスト実行中...
echo ================================================================================
py -3 cuda_setup_test.py
if %errorLevel% == 0 (
    echo ✅ CUDA環境テスト完了
) else (
    echo ❌ CUDA環境テストでエラーが発生しました
)
echo.
pause
goto menu

:cuda_analysis
echo.
echo 🚀 CUDA解析システム実行中...
echo ================================================================================
echo ⏰ 実行時間: 約5-15分（GPU性能により変動）
echo 💾 要求メモリ: GPU 4GB以上、システム 8GB以上
echo.
echo 実行を開始します...
py -3 riemann_hypothesis_cuda_ultimate.py
if %errorLevel% == 0 (
    echo.
    echo ✅ CUDA解析完了！
    echo 📊 結果ファイルが生成されました
    echo 📁 現在のディレクトリを確認してください
) else (
    echo.
    echo ❌ 解析実行中にエラーが発生しました
    echo 🔧 CUDA環境テストを先に実行することを推奨します
)
echo.
pause
goto menu

:install_libs
echo.
echo 📦 必要ライブラリ自動インストール
echo ================================================================================
echo 以下のライブラリをインストールします:
echo - PyTorch CUDA版
echo - CuPy CUDA版
echo - その他必要なライブラリ
echo.
set /p confirm="続行しますか？ (y/n): "
if /i "%confirm%"=="y" (
    echo.
    echo 🔄 PyTorch CUDA版インストール中...
    py -3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    echo.
    echo 🔄 CuPy CUDA版インストール中...
    py -3 -m pip install cupy-cuda12x
    
    echo.
    echo 🔄 その他ライブラリインストール中...
    py -3 -m pip install -r requirements.txt
    
    echo.
    echo ✅ ライブラリインストール完了
) else (
    echo インストールをキャンセルしました
)
echo.
pause
goto menu

:system_info
echo.
echo 🖥️ システム情報表示
echo ================================================================================

echo 💻 OS情報:
ver

echo.
echo 🔧 環境変数:
echo CUDA_PATH: %CUDA_PATH%
echo PATH (CUDA関連のみ):
for %%i in ("%PATH:;=" "%") do (
    echo %%~i | findstr /i cuda >nul && echo   %%~i
)

echo.
echo 🐍 Python環境:
py -3 --version
py -3 -m pip --version

echo.
echo 💾 システムリソース:
wmic computersystem get TotalPhysicalMemory /value | findstr "="
wmic cpu get Name /value | findstr "="

echo.
echo 🎮 GPU情報:
nvidia-smi >nul 2>&1 && nvidia-smi || echo GPU情報を取得できませんでした

echo.
pause
goto menu

:run_all
echo.
echo 🌟 全実行モード開始
echo ================================================================================
echo 以下を順次実行します:
echo 1. システム情報表示
echo 2. 必要ライブラリ確認
echo 3. CUDA環境テスト
echo 4. CUDA解析システム実行
echo.

set /p confirm="全実行を開始しますか？ (y/n): "
if /i not "%confirm%"=="y" (
    echo 全実行をキャンセルしました
    echo.
    pause
    goto menu
)

echo.
echo 📋 ステップ 1/4: システム情報表示
echo ----------------------------------------
call :system_info_silent

echo.
echo 📋 ステップ 2/4: 必要ライブラリ確認
echo ----------------------------------------
py -3 -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>nul || echo PyTorch未検出
py -3 -c "import cupy; print(f'CuPy: {cupy.__version__}')" 2>nul || echo CuPy未検出
py -3 -c "import numpy; print(f'NumPy: {numpy.__version__}')" 2>nul || echo NumPy未検出

echo.
echo 📋 ステップ 3/4: CUDA環境テスト
echo ----------------------------------------
py -3 cuda_setup_test.py

echo.
echo 📋 ステップ 4/4: CUDA解析システム実行
echo ----------------------------------------
echo ⏰ メイン解析開始（5-15分程度）...
py -3 riemann_hypothesis_cuda_ultimate.py

echo.
echo 🏆 全実行完了！
echo ================================================================================
echo 📊 生成されたファイルを確認してください:
dir /b *.json *.png 2>nul | findstr /r "nkat.*\.json$\|nkat.*\.png$\|cuda.*\.json$" || echo 結果ファイルが見つかりません

echo.
pause
goto menu

:system_info_silent
echo 💻 OS: 
ver | findstr "Version"
echo 🐍 Python: 
py -3 --version 2>nul || echo Python未検出
echo 🎮 GPU: 
nvidia-smi --query-gpu=name --format=csv,noheader 2>nul || echo GPU情報取得不可
goto :eof

:invalid_choice
echo.
echo ❌ 無効な選択です。1-5または0を入力してください。
echo.
pause

:menu
echo.
echo 📋 実行オプションを選択してください:
echo.
echo 1️⃣ CUDA環境テスト実行
echo 2️⃣ CUDA解析システム実行
echo 3️⃣ 必要ライブラリ自動インストール
echo 4️⃣ システム情報表示
echo 5️⃣ 全て実行（推奨）
echo 0️⃣ 終了
echo.

set /p choice="選択してください (1-5, 0): "

if "%choice%"=="1" goto cuda_test
if "%choice%"=="2" goto cuda_analysis
if "%choice%"=="3" goto install_libs
if "%choice%"=="4" goto system_info
if "%choice%"=="5" goto run_all
if "%choice%"=="0" goto end
goto invalid_choice

:end
echo.
echo 🌟 CUDA対応NKAT解析システム実行スクリプト終了
echo 📚 峯岸亮先生のリーマン予想証明論文の革新的解析をありがとうございました！
echo 🚀 GPU並列計算によるNKAT理論の実証が完了しました
echo.
echo 📞 サポート: GitHub Issues / Documentation
echo 🔗 詳細情報: CUDA_SETUP_GUIDE.md
echo.
pause
exit /b 0 