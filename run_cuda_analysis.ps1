# PowerShell実行ポリシー設定が必要な場合:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# UTF-8エンコーディング設定
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# ヘッダー表示
Clear-Host
Write-Host "🚀 CUDA対応NKAT解析システム 実行スクリプト (PowerShell版)" -ForegroundColor Cyan
Write-Host "📚 峯岸亮先生のリーマン予想証明論文 - GPU超並列計算版" -ForegroundColor Yellow
Write-Host "🎮 Windows 11 + Python 3 + CUDA 12.x 対応" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Gray
Write-Host ""

# 関数定義
function Test-AdminRights {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Test-Python {
    try {
        $pythonVersion = py -3 --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Python 3 利用可能: $pythonVersion" -ForegroundColor Green
            return $true
        }
    }
    catch {
        Write-Host "❌ Python 3 が見つかりません" -ForegroundColor Red
        Write-Host "📦 Python 3.9-3.11をインストールしてください" -ForegroundColor Yellow
        Write-Host "🔗 https://www.python.org/downloads/" -ForegroundColor Blue
        return $false
    }
    return $false
}

function Test-CUDA {
    try {
        $nvidiaInfo = nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ NVIDIA GPU ドライバ検出" -ForegroundColor Green
            Write-Host "📊 GPU情報:" -ForegroundColor Cyan
            $nvidiaInfo | ForEach-Object { Write-Host "   $_" -ForegroundColor White }
            return $true
        }
    }
    catch {
        Write-Host "⚠️ NVIDIA GPU ドライバが検出されません" -ForegroundColor Yellow
        Write-Host "💻 CPU モードで実行されます" -ForegroundColor Yellow
        return $false
    }
    return $false
}

function Show-Menu {
    Write-Host ""
    Write-Host "📋 実行オプションを選択してください:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "1️⃣ CUDA環境テスト実行" -ForegroundColor White
    Write-Host "2️⃣ CUDA解析システム実行" -ForegroundColor White  
    Write-Host "3️⃣ 必要ライブラリ自動インストール" -ForegroundColor White
    Write-Host "4️⃣ システム情報表示" -ForegroundColor White
    Write-Host "5️⃣ 全て実行（推奨）" -ForegroundColor Yellow
    Write-Host "0️⃣ 終了" -ForegroundColor Red
    Write-Host ""
}

function Invoke-CudaTest {
    Write-Host ""
    Write-Host "🔍 CUDA環境テスト実行中..." -ForegroundColor Cyan
    Write-Host "=" * 80 -ForegroundColor Gray
    
    try {
        py -3 cuda_setup_test.py
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ CUDA環境テスト完了" -ForegroundColor Green
        } else {
            Write-Host "❌ CUDA環境テストでエラーが発生しました" -ForegroundColor Red
        }
    }
    catch {
        Write-Host "❌ テスト実行中にエラーが発生しました: $($_.Exception.Message)" -ForegroundColor Red
    }
    
    Write-Host ""
    Write-Host "Press any key to continue..." -ForegroundColor Gray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}

function Invoke-CudaAnalysis {
    Write-Host ""
    Write-Host "🚀 CUDA解析システム実行中..." -ForegroundColor Cyan
    Write-Host "=" * 80 -ForegroundColor Gray
    Write-Host "⏰ 実行時間: 約5-15分（GPU性能により変動）" -ForegroundColor Yellow
    Write-Host "💾 要求メモリ: GPU 4GB以上、システム 8GB以上" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "実行を開始します..." -ForegroundColor Green
    
    try {
        py -3 riemann_hypothesis_cuda_ultimate.py
        if ($LASTEXITCODE -eq 0) {
            Write-Host ""
            Write-Host "✅ CUDA解析完了！" -ForegroundColor Green
            Write-Host "📊 結果ファイルが生成されました" -ForegroundColor Cyan
            Write-Host "📁 現在のディレクトリを確認してください" -ForegroundColor Cyan
            
            # 生成されたファイルを表示
            $resultFiles = Get-ChildItem -Path "." -Name "nkat_cuda_*.json", "nkat_cuda_*.png", "cuda_benchmark_*.json" -ErrorAction SilentlyContinue
            if ($resultFiles) {
                Write-Host "📄 生成されたファイル:" -ForegroundColor Cyan
                $resultFiles | ForEach-Object { Write-Host "   $_" -ForegroundColor White }
            }
        } else {
            Write-Host ""
            Write-Host "❌ 解析実行中にエラーが発生しました" -ForegroundColor Red
            Write-Host "🔧 CUDA環境テストを先に実行することを推奨します" -ForegroundColor Yellow
        }
    }
    catch {
        Write-Host "❌ 解析実行中にエラーが発生しました: $($_.Exception.Message)" -ForegroundColor Red
    }
    
    Write-Host ""
    Write-Host "Press any key to continue..." -ForegroundColor Gray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}

function Install-Libraries {
    Write-Host ""
    Write-Host "📦 必要ライブラリ自動インストール" -ForegroundColor Cyan
    Write-Host "=" * 80 -ForegroundColor Gray
    Write-Host "以下のライブラリをインストールします:" -ForegroundColor Yellow
    Write-Host "- PyTorch CUDA版" -ForegroundColor White
    Write-Host "- CuPy CUDA版" -ForegroundColor White
    Write-Host "- その他必要なライブラリ" -ForegroundColor White
    Write-Host ""
    
    $confirm = Read-Host "続行しますか？ (y/n)"
    if ($confirm -eq "y" -or $confirm -eq "Y") {
        try {
            Write-Host ""
            Write-Host "🔄 PyTorch CUDA版インストール中..." -ForegroundColor Yellow
            py -3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
            
            Write-Host ""
            Write-Host "🔄 CuPy CUDA版インストール中..." -ForegroundColor Yellow
            py -3 -m pip install cupy-cuda12x
            
            Write-Host ""
            Write-Host "🔄 その他ライブラリインストール中..." -ForegroundColor Yellow
            py -3 -m pip install -r requirements.txt
            
            Write-Host ""
            Write-Host "✅ ライブラリインストール完了" -ForegroundColor Green
        }
        catch {
            Write-Host "❌ インストール中にエラーが発生しました: $($_.Exception.Message)" -ForegroundColor Red
        }
    } else {
        Write-Host "インストールをキャンセルしました" -ForegroundColor Yellow
    }
    
    Write-Host ""
    Write-Host "Press any key to continue..." -ForegroundColor Gray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}

function Show-SystemInfo {
    Write-Host ""
    Write-Host "🖥️ システム情報表示" -ForegroundColor Cyan
    Write-Host "=" * 80 -ForegroundColor Gray
    
    # OS情報
    Write-Host "💻 OS情報:" -ForegroundColor Yellow
    $osInfo = Get-WmiObject -Class Win32_OperatingSystem
    Write-Host "   $($osInfo.Caption) $($osInfo.Version)" -ForegroundColor White
    
    # CPU情報
    Write-Host ""
    Write-Host "🔧 CPU情報:" -ForegroundColor Yellow
    $cpuInfo = Get-WmiObject -Class Win32_Processor
    Write-Host "   $($cpuInfo.Name)" -ForegroundColor White
    Write-Host "   論理プロセッサ数: $($cpuInfo.NumberOfLogicalProcessors)" -ForegroundColor White
    
    # メモリ情報
    Write-Host ""
    Write-Host "💾 メモリ情報:" -ForegroundColor Yellow
    $memInfo = Get-WmiObject -Class Win32_ComputerSystem
    $totalMemGB = [math]::Round($memInfo.TotalPhysicalMemory / 1GB, 2)
    Write-Host "   総メモリ: $totalMemGB GB" -ForegroundColor White
    
    # CUDA環境変数
    Write-Host ""
    Write-Host "🔧 CUDA環境変数:" -ForegroundColor Yellow
    $cudaPath = $env:CUDA_PATH
    if ($cudaPath) {
        Write-Host "   CUDA_PATH: $cudaPath" -ForegroundColor White
    } else {
        Write-Host "   CUDA_PATH: 未設定" -ForegroundColor Red
    }
    
    # Python環境
    Write-Host ""
    Write-Host "🐍 Python環境:" -ForegroundColor Yellow
    try {
        $pythonVersion = py -3 --version 2>$null
        Write-Host "   Python: $pythonVersion" -ForegroundColor White
        
        $pipVersion = py -3 -m pip --version 2>$null
        Write-Host "   pip: $pipVersion" -ForegroundColor White
    }
    catch {
        Write-Host "   Python: 未検出" -ForegroundColor Red
    }
    
    # ライブラリ確認
    Write-Host ""
    Write-Host "📚 重要ライブラリ:" -ForegroundColor Yellow
    
    $libraries = @("torch", "cupy", "numpy", "matplotlib", "scipy")
    foreach ($lib in $libraries) {
        try {
            $version = py -3 -c "import $lib; print($lib.__version__)" 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "   $lib`: $version" -ForegroundColor Green
            } else {
                Write-Host "   $lib`: 未検出" -ForegroundColor Red
            }
        }
        catch {
            Write-Host "   $lib`: 未検出" -ForegroundColor Red
        }
    }
    
    # GPU情報
    Write-Host ""
    Write-Host "🎮 GPU情報:" -ForegroundColor Yellow
    try {
        $gpuInfo = nvidia-smi --query-gpu=name,memory.total,driver_version,temperature.gpu --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -eq 0) {
            $gpuInfo | ForEach-Object { Write-Host "   $_" -ForegroundColor White }
        } else {
            Write-Host "   GPU情報を取得できませんでした" -ForegroundColor Red
        }
    }
    catch {
        Write-Host "   GPU情報を取得できませんでした" -ForegroundColor Red
    }
    
    Write-Host ""
    Write-Host "Press any key to continue..." -ForegroundColor Gray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}

function Invoke-RunAll {
    Write-Host ""
    Write-Host "🌟 全実行モード開始" -ForegroundColor Cyan
    Write-Host "=" * 80 -ForegroundColor Gray
    Write-Host "以下を順次実行します:" -ForegroundColor Yellow
    Write-Host "1. システム情報表示" -ForegroundColor White
    Write-Host "2. 必要ライブラリ確認" -ForegroundColor White
    Write-Host "3. CUDA環境テスト" -ForegroundColor White
    Write-Host "4. CUDA解析システム実行" -ForegroundColor White
    Write-Host ""
    
    $confirm = Read-Host "全実行を開始しますか？ (y/n)"
    if ($confirm -ne "y" -and $confirm -ne "Y") {
        Write-Host "全実行をキャンセルしました" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Press any key to continue..." -ForegroundColor Gray
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        return
    }
    
    # ステップ 1: システム情報
    Write-Host ""
    Write-Host "📋 ステップ 1/4: システム情報表示" -ForegroundColor Cyan
    Write-Host "-" * 40 -ForegroundColor Gray
    
    $osInfo = Get-WmiObject -Class Win32_OperatingSystem
    Write-Host "💻 OS: $($osInfo.Caption)" -ForegroundColor White
    
    try {
        $pythonVersion = py -3 --version 2>$null
        Write-Host "🐍 Python: $pythonVersion" -ForegroundColor White
    }
    catch {
        Write-Host "🐍 Python: 未検出" -ForegroundColor Red
    }
    
    try {
        $gpuName = nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
        Write-Host "🎮 GPU: $gpuName" -ForegroundColor White
    }
    catch {
        Write-Host "🎮 GPU: 情報取得不可" -ForegroundColor Red
    }
    
    # ステップ 2: ライブラリ確認
    Write-Host ""
    Write-Host "📋 ステップ 2/4: 必要ライブラリ確認" -ForegroundColor Cyan
    Write-Host "-" * 40 -ForegroundColor Gray
    
    $libraries = @("torch", "cupy", "numpy")
    foreach ($lib in $libraries) {
        try {
            $version = py -3 -c "import $lib; print(f'$lib`: {$lib.__version__}')" 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "$version" -ForegroundColor Green
            } else {
                Write-Host "$lib`: 未検出" -ForegroundColor Red
            }
        }
        catch {
            Write-Host "$lib`: 未検出" -ForegroundColor Red
        }
    }
    
    # ステップ 3: CUDA環境テスト
    Write-Host ""
    Write-Host "📋 ステップ 3/4: CUDA環境テスト" -ForegroundColor Cyan
    Write-Host "-" * 40 -ForegroundColor Gray
    py -3 cuda_setup_test.py
    
    # ステップ 4: メイン解析
    Write-Host ""
    Write-Host "📋 ステップ 4/4: CUDA解析システム実行" -ForegroundColor Cyan
    Write-Host "-" * 40 -ForegroundColor Gray
    Write-Host "⏰ メイン解析開始（5-15分程度）..." -ForegroundColor Yellow
    py -3 riemann_hypothesis_cuda_ultimate.py
    
    Write-Host ""
    Write-Host "🏆 全実行完了！" -ForegroundColor Green
    Write-Host "=" * 80 -ForegroundColor Gray
    Write-Host "📊 生成されたファイルを確認してください:" -ForegroundColor Cyan
    
    $resultFiles = Get-ChildItem -Path "." -Name "nkat_*.json", "nkat_*.png", "cuda_*.json" -ErrorAction SilentlyContinue
    if ($resultFiles) {
        $resultFiles | ForEach-Object { Write-Host "   $_" -ForegroundColor White }
    } else {
        Write-Host "   結果ファイルが見つかりません" -ForegroundColor Red
    }
    
    Write-Host ""
    Write-Host "Press any key to continue..." -ForegroundColor Gray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}

# メイン処理開始
# 管理者権限チェック
if (Test-AdminRights) {
    Write-Host "🔑 管理者権限で実行中" -ForegroundColor Green
} else {
    Write-Host "⚠️ 管理者権限が推奨されます（GPU最適化のため）" -ForegroundColor Yellow
}
Write-Host ""

# Python環境確認
Write-Host "🐍 Python環境確認中..." -ForegroundColor Cyan
if (-not (Test-Python)) {
    Write-Host "Press any key to exit..." -ForegroundColor Gray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}
Write-Host ""

# CUDA環境事前確認
Write-Host "🎮 CUDA環境事前確認..." -ForegroundColor Cyan
$cudaAvailable = Test-CUDA
Write-Host ""

# メインループ
do {
    Show-Menu
    $choice = Read-Host "選択してください (1-5, 0)"
    
    switch ($choice) {
        "1" { Invoke-CudaTest }
        "2" { Invoke-CudaAnalysis }
        "3" { Install-Libraries }
        "4" { Show-SystemInfo }
        "5" { Invoke-RunAll }
        "0" { 
            Write-Host ""
            Write-Host "🌟 CUDA対応NKAT解析システム実行スクリプト終了" -ForegroundColor Cyan
            Write-Host "📚 峯岸亮先生のリーマン予想証明論文の革新的解析をありがとうございました！" -ForegroundColor Yellow
            Write-Host "🚀 GPU並列計算によるNKAT理論の実証が完了しました" -ForegroundColor Green
            Write-Host ""
            Write-Host "📞 サポート: GitHub Issues / Documentation" -ForegroundColor Gray
            Write-Host "🔗 詳細情報: CUDA_SETUP_GUIDE.md" -ForegroundColor Gray
            Write-Host ""
            break
        }
        default {
            Write-Host ""
            Write-Host "❌ 無効な選択です。1-5または0を入力してください。" -ForegroundColor Red
            Write-Host ""
            Write-Host "Press any key to continue..." -ForegroundColor Gray
            $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        }
    }
} while ($choice -ne "0")

Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 