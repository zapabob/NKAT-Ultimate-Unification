# Lean 4 セットアップスクリプト (Windows PowerShell)
# ボブにゃんの「なんJ実況テンション」でLean環境を構築するで〜

Write-Host "🚀 Lean 4 セットアップ開始！" -ForegroundColor Green

# 1. elan インストール
Write-Host "📦 elan をインストール中..." -ForegroundColor Yellow
try {
    # elan のインストール
    Invoke-WebRequest -Uri "https://github.com/leanprover/elan/releases/latest/download/elan-x86_64-pc-windows-msvc.zip" -OutFile "elan.zip"
    Expand-Archive -Path "elan.zip" -DestinationPath "elan" -Force
    Remove-Item "elan.zip"
    
    # PATH に追加
    $env:PATH = ".\elan\bin;$env:PATH"
    [Environment]::SetEnvironmentVariable("PATH", ".\elan\bin;$env:PATH", "User")
    
    Write-Host "✅ elan インストール完了！" -ForegroundColor Green
} catch {
    Write-Host "❌ elan インストール失敗: $_" -ForegroundColor Red
    exit 1
}

# 2. Lean 4 ツールチェーンインストール
Write-Host "🔧 Lean 4 ツールチェーンをインストール中..." -ForegroundColor Yellow
try {
    .\elan\bin\elan.exe toolchain install leanprover/lean4:stable
    .\elan\bin\elan.exe default leanprover/lean4:stable
    
    Write-Host "✅ Lean 4 ツールチェーンインストール完了！" -ForegroundColor Green
} catch {
    Write-Host "❌ Lean 4 ツールチェーンインストール失敗: $_" -ForegroundColor Red
    exit 1
}

# 3. プロジェクトビルド
Write-Host "🏗️ プロジェクトをビルド中..." -ForegroundColor Yellow
try {
    .\elan\bin\lake.exe build
    Write-Host "✅ プロジェクトビルド完了！" -ForegroundColor Green
} catch {
    Write-Host "❌ プロジェクトビルド失敗: $_" -ForegroundColor Red
    exit 1
}

# 4. テスト実行
Write-Host "🧪 テスト実行中..." -ForegroundColor Yellow
try {
    .\elan\bin\lake.exe exe ai_proofs-lean
    Write-Host "✅ テスト実行完了！" -ForegroundColor Green
} catch {
    Write-Host "❌ テスト実行失敗: $_" -ForegroundColor Red
    exit 1
}

Write-Host "🎉 Lean 4 セットアップ完了！" -ForegroundColor Green
Write-Host "🚀 次は定理ガチャを回すで〜" -ForegroundColor Cyan

# 5. 環境変数の確認
Write-Host "📋 環境変数確認:" -ForegroundColor Yellow
Write-Host "PATH: $env:PATH" -ForegroundColor Gray
Write-Host "LEAN_SRC_PATH: $env:LEAN_SRC_PATH" -ForegroundColor Gray

Write-Host "✨ セットアップ完了！Cursorで開いて定理鑑定士になるで〜" -ForegroundColor Green 