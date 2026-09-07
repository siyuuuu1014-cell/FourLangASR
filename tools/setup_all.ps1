param(
    [string]$AsrRoot = "D:\dev\projects\fourlang_translation\asr_exp1",
    [string]$UzModelDir = ""
)

$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "FourLang ASR Mobile - setup"
Write-Host "============================================================"

& (Join-Path $PSScriptRoot "bootstrap_gradle.ps1")

& (Join-Path $PSScriptRoot "download_sherpa_aar.ps1")

if ([string]::IsNullOrWhiteSpace($UzModelDir)) {
    & (Join-Path $PSScriptRoot "prepare_models.ps1") -AsrRoot $AsrRoot
} else {
    & (Join-Path $PSScriptRoot "prepare_models.ps1") -AsrRoot $AsrRoot -UzModelDir $UzModelDir
}

Write-Host ""
Write-Host "============================================================"
Write-Host "SETUP COMPLETE"
Write-Host "============================================================"
Write-Host ""
Write-Host "Next:"
Write-Host "1. Open this project folder in Android Studio."
Write-Host "2. Use JDK 17."
Write-Host "3. Install Android SDK 34."
Write-Host "4. Connect an arm64 Android phone with USB debugging."
Write-Host "5. Click Run."
