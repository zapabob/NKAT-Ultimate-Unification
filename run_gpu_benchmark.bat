@echo off
echo ================================================================================
echo 🚀 GPU加速NKAT理論フレームワーク v2.0 - 自動ベンチマーク
echo ================================================================================
echo.

REM 現在時刻の取得
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YY=%dt:~2,2%" & set "YYYY=%dt:~0,4%" & set "MM=%dt:~4,2%" & set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%" & set "Min=%dt:~10,2%" & set "Sec=%dt:~12,2%"
set "timestamp=%YYYY%%MM%%DD%_%HH%%Min%%Sec%"

echo 📅 実行開始時刻: %YYYY%-%MM%-%DD% %HH%:%Min%:%Sec%
echo.

REM GPU情報の確認
echo 🎮 GPU情報確認中...
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits 2>nul
if %errorlevel% neq 0 (
    echo ⚠️ NVIDIA GPU が検出されませんでした。CPU版で実行します。
    set GPU_FLAG=--no-gpu
) else (
    echo ✅ NVIDIA GPU が検出されました。GPU加速モードで実行します。
    set GPU_FLAG=
)
echo.

REM Python環境の確認
echo 🐍 Python環境確認中...
py -3 --version
if %errorlevel% neq 0 (
    echo ❌ Python 3 が見つかりません。インストールしてください。
    pause
    exit /b 1
)
echo.

REM 必要なライブラリの確認
echo 📦 必要ライブラリの確認中...
py -3 -c "import numpy, scipy, matplotlib; print('✅ 基本ライブラリ OK')" 2>nul
if %errorlevel% neq 0 (
    echo ❌ 基本ライブラリが不足しています。requirements.txtからインストールしてください。
    echo pip install -r requirements.txt
    pause
    exit /b 1
)

REM CuPyの確認（GPU使用時のみ）
if not defined GPU_FLAG (
    py -3 -c "import cupy; print('✅ CuPy GPU加速 OK')" 2>nul
    if %errorlevel% neq 0 (
        echo ⚠️ CuPy が見つかりません。CPU版にフォールバックします。
        set GPU_FLAG=--no-gpu
    )
)
echo.

REM ディレクトリ移動
cd /d "%~dp0src"
if %errorlevel% neq 0 (
    echo ❌ srcディレクトリが見つかりません。
    pause
    exit /b 1
)

echo 🚀 GPU加速ベンチマーク実行中...
echo ================================================================================
echo.

REM ベンチマーク実行（複数設定）
echo 📊 設定1: 16³格子, complex128, 512固有値
py -3 riemann_gpu_accelerated.py --lattice 16 --precision complex128 --sparse csr --eig 512 --save gpu_benchmark_16c128_%timestamp%.json %GPU_FLAG%
set exit_code1=%errorlevel%
echo.

if %exit_code1% equ 0 (
    echo ✅ 設定1 完了
) else (
    echo ❌ 設定1 エラー
)
echo.

echo 📊 設定2: 12³格子, complex128, 1024固有値（高精度）
py -3 riemann_gpu_accelerated.py --lattice 12 --precision complex128 --sparse csr --eig 1024 --save gpu_benchmark_12c128_hq_%timestamp%.json %GPU_FLAG%
set exit_code2=%errorlevel%
echo.

if %exit_code2% equ 0 (
    echo ✅ 設定2 完了
) else (
    echo ❌ 設定2 エラー
)
echo.

echo 📊 設定3: 10³格子, complex64, 256固有値（高速）
py -3 riemann_gpu_accelerated.py --lattice 10 --precision complex64 --sparse csr --eig 256 --save gpu_benchmark_10c64_fast_%timestamp%.json %GPU_FLAG%
set exit_code3=%errorlevel%
echo.

if %exit_code3% equ 0 (
    echo ✅ 設定3 完了
) else (
    echo ❌ 設定3 エラー
)
echo.

REM 結果サマリー
echo ================================================================================
echo 📊 ベンチマーク結果サマリー
echo ================================================================================
echo.

if %exit_code1% equ 0 (
    echo ✅ 16³格子 complex128: 成功
) else (
    echo ❌ 16³格子 complex128: 失敗
)

if %exit_code2% equ 0 (
    echo ✅ 12³格子 complex128 高精度: 成功
) else (
    echo ❌ 12³格子 complex128 高精度: 失敗
)

if %exit_code3% equ 0 (
    echo ✅ 10³格子 complex64 高速: 成功
) else (
    echo ❌ 10³格子 complex64 高速: 失敗
)

echo.

REM 生成ファイルの確認
echo 📁 生成されたファイル:
dir /b gpu_benchmark_*_%timestamp%.* 2>nul
dir /b gpu_nkat_benchmark_*.json 2>nul
dir /b gpu_nkat_benchmark_analysis_*.png 2>nul
echo.

REM パフォーマンス統計の表示
echo 📈 パフォーマンス統計:
for %%f in (gpu_benchmark_*_%timestamp%.json) do (
    echo ファイル: %%f
    py -3 -c "
import json, sys
try:
    with open('%%f', 'r', encoding='utf-8') as f:
        data = json.load(f)
    metrics = data.get('performance_metrics', {})
    print(f'  総計算時間: {metrics.get(\"total_computation_time\", 0):.2f}秒')
    print(f'  平均精度: {metrics.get(\"precision_achieved\", \"N/A\")}')
    print(f'  改善率: {metrics.get(\"improvement_factor\", 1):.2f}×')
    print(f'  成功率: {metrics.get(\"success_rate\", 0)*100:.1f}%%')
except Exception as e:
    print(f'  エラー: {e}')
    " 2>nul
    echo.
)

REM 完了時刻
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YY=%dt:~2,2%" & set "YYYY=%dt:~0,4%" & set "MM=%dt:~4,2%" & set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%" & set "Min=%dt:~10,2%" & set "Sec=%dt:~12,2%"

echo ================================================================================
echo 🎉 GPU加速ベンチマーク完了！
echo 📅 完了時刻: %YYYY%-%MM%-%DD% %HH%:%Min%:%Sec%
echo ================================================================================
echo.

REM 次のステップの提案
echo 🚀 次のステップ:
echo   1. 結果ファイル（*.json）を確認
echo   2. 可視化画像（*.png）を確認  
echo   3. より大きな格子サイズでの実行
echo   4. GitHub Actions CI設定
echo   5. arXiv論文パッケージ作成
echo.

echo 続行するには何かキーを押してください...
pause >nul 