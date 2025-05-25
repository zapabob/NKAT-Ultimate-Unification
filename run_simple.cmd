@echo off
chcp 65001
echo NKAT v8.0 arxiv submission system starting...
echo.
python --version
echo.
echo Installing dependencies...
pip install matplotlib numpy tqdm
echo.
echo Running arxiv submission automation...
cd src
python arxiv_v8_submission_automation.py
cd ..
echo.
echo Process completed.
pause 