@echo off
chcp 65001 > nul
cd /d "%~dp0"

echo NKAT + Odlyzko-Schonhage背理法証明システム
echo ============================================
echo.

echo RTX3080 CUDA環境でリーマン予想背理法証明を実行中...
python nkat_simple_riemann_contradiction_analysis.py > nkat_proof_output.txt 2>&1

echo.
echo 実行完了。結果を確認します...
echo.

if exist nkat_simple_riemann_proof_*.json (
    echo ✅ JSON結果ファイル生成済み
    dir nkat_simple_riemann_proof_*.json
) else (
    echo ❌ JSON結果ファイルが見つかりません
)

if exist nkat_simple_proof_viz_*.png (
    echo ✅ 可視化ファイル生成済み
    dir nkat_simple_proof_viz_*.png
) else (
    echo ❌ 可視化ファイルが見つかりません
)

echo.
echo 実行ログ（最初の50行）:
echo ========================
type nkat_proof_output.txt | more

pause 