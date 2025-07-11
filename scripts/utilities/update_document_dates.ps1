<#
.SYNOPSIS
    Updates date stamps in project documents.
.DESCRIPTION
    This script automatically finds and updates date stamps in various document files
    (e.g., Markdown, TeX) to the current date. It searches specified directories
    recursively.
.PARAMETER TargetDirectories
    An array of directories to search for documents.
.PARAMETER FileExtensions
    An array of file extensions to target (e.g., "*.md", "*.tex").
.EXAMPLE
    .\scripts\utilities\update_document_dates.ps1
    (Uses default directories and extensions)
.EXAMPLE
    .\scripts\utilities\update_document_dates.ps1 -TargetDirectories @("docs", "reports") -FileExtensions @("*.md")
    (Specifies target directories and extensions)
#>
param(
    [string[]]$TargetDirectories = @("docs", "_docs", "arxiv_submission", "main", "appendix"),
    [string[]]$FileExtensions = @("*.md", "*.tex")
)

$currentDate = Get-Date -Format "yyyy-MM-dd"
# Regex to find dates in formats like "Date: YYYY-MM-DD" or "**Date**: YYYY-MM-DD", case-insensitive
$dateRegex = "((?i)Date\s*:\s*|(?i)\*\*Date\*\*:\s*)(\d{4}-\d{2}-\d{2})"

Write-Host "🚀 Starting date update process..." -ForegroundColor Cyan
Write-Host "Today's Date: $currentDate"
Write-Host "Target Directories: $($TargetDirectories -join ', ')"
Write-Host "File Extensions: $($FileExtensions -join ', ')"
Write-Host "--------------------------------------------------"

$updatedFilesCount = 0

foreach ($dir in $TargetDirectories) {
    if (Test-Path $dir) {
        Get-ChildItem -Path $dir -Include $FileExtensions -Recurse -ErrorAction SilentlyContinue | ForEach-Object {
            $file = $_
            try {
                $content = Get-Content -Path $file.FullName -Raw -Encoding UTF8
                if ($content -match $dateRegex) {
                    $oldDate = $matches[2]
                    if ($oldDate -ne $currentDate) {
                        Write-Host "📝 Updating date in '$($file.FullName)' from $oldDate to $currentDate" -ForegroundColor Yellow
                        $newContent = $content -replace $dateRegex, "${1}$currentDate"
                        
                        if($file.IsReadOnly) {
                            $file.IsReadOnly = $false
                        }
                        
                        Set-Content -Path $file.FullName -Value $newContent -Encoding UTF8 -Force
                        $updatedFilesCount++
                    }
                }
            } catch {
                Write-Warning "Could not process file '$($file.FullName)': $_"
            }
        }
    } else {
        Write-Warning "Directory not found: '$dir'"
    }
}

Write-Host "--------------------------------------------------"
if ($updatedFilesCount -gt 0) {
    Write-Host "✅ Update complete. $updatedFilesCount file(s) updated." -ForegroundColor Green
} else {
    Write-Host "✅ No files required updating. All dates are current." -ForegroundColor Green
} 