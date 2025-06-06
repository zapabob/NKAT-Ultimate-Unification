@echo off
chcp 65001 > nul
cd /d "%~dp0"

echo NKAT Simple CUDA Analysis
echo ========================
echo.

python simple_cuda_test.py
echo.

echo Running NKAT Analysis...
python riemann_hypothesis_cuda_ultimate_v5_nkat.py

echo.
echo Analysis completed. Check output files for results.
pause 