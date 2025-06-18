# RTX3080最適化NKAT実行スクリプト
# PowerShell版

Write-Host "======================================"
Write-Host "    NKAT RTX3080最適化実行システム    "
Write-Host "======================================"

# 環境変数設定
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

# プロジェクトディレクトリに移動
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host "[INFO] プロジェクトディレクトリ: $ProjectRoot"
Write-Host "[INFO] 環境変数設定完了"

# Python環境確認
try {
    $PythonVersion = py -3 --version
    Write-Host "[INFO] Python: $PythonVersion"
} catch {
    Write-Host "[ERROR] Python 3が見つかりません"
    exit 1
}

# CUDA確認
try {
    $CudaInfo = py -3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
    Write-Host "[INFO] $CudaInfo"
} catch {
    Write-Host "[WARNING] CUDA確認に失敗しました"
}

# 必要なディレクトリ作成
$Directories = @("checkpoints", "logs", "results")
foreach ($Dir in $Directories) {
    if (-not (Test-Path $Dir)) {
        New-Item -ItemType Directory -Path $Dir
        Write-Host "[INFO] ディレクトリ作成: $Dir"
    }
}

# 実行開始
Write-Host ""
Write-Host "[INFO] NKAT Clay Millennium Solver開始..."
Write-Host "[INFO] RTX3080最適化モードで実行中..."

try {
    # RTX3080最適化設定で実行
    py -3 rtx3080_run_config.py
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "[SUCCESS] 実行完了！"
        Write-Host "[INFO] 結果ファイルを確認してください"
    } else {
        Write-Host ""
        Write-Host "[ERROR] 実行中にエラーが発生しました (Exit Code: $LASTEXITCODE)"
    }
    
} catch {
    Write-Host "[ERROR] 実行失敗: $($_.Exception.Message)"
}

# ログファイル確認
$LogFiles = Get-ChildItem -Filter "*.log" | Sort-Object LastWriteTime -Descending
if ($LogFiles.Count -gt 0) {
    Write-Host ""
    Write-Host "[INFO] 最新ログファイル:"
    foreach ($LogFile in $LogFiles[0..2]) {  # 最新3件
        Write-Host "  - $($LogFile.Name) ($(Get-Date $LogFile.LastWriteTime -Format 'yyyy-MM-dd HH:mm:ss'))"
    }
}

# 結果ファイル確認
$ResultFiles = Get-ChildItem -Filter "*results*.json" | Sort-Object LastWriteTime -Descending
if ($ResultFiles.Count -gt 0) {
    Write-Host ""
    Write-Host "[INFO] 結果ファイル:"
    foreach ($ResultFile in $ResultFiles[0..2]) {  # 最新3件
        Write-Host "  - $($ResultFile.Name) ($(($ResultFile.Length / 1KB).ToString('F1')) KB)"
    }
}

Write-Host ""
Write-Host "[INFO] RTX3080実行システム終了"
Write-Host "======================================"

# 終了時にキー入力待ち（オプション）
# Read-Host "Enterキーを押して終了..." 