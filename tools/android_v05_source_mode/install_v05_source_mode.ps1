param(
    [string]$ProjectRoot = "D:\dev\projects\FourLangAsrMobile"
)

$ErrorActionPreference = "Stop"

$PackageRoot =
    Split-Path -Parent $MyInvocation.MyCommand.Path

$ProjectRoot =
    (Resolve-Path $ProjectRoot).Path

$main =
    Join-Path $ProjectRoot "app\src\main\java\com\example\fourlangasr\MainActivity.kt"

$lidResult =
    Join-Path $ProjectRoot "app\src\main\java\com\example\fourlangasr\lid\LidResult.kt"

if (-not (Test-Path $main)) {
    throw "Missing MainActivity.kt: $main"
}

if (-not (Test-Path $lidResult)) {
    throw "Missing LidResult.kt: $lidResult"
}

$stamp =
    Get-Date -Format "yyyyMMdd_HHmmss"

$backupDir =
    Join-Path $ProjectRoot "backup_before_v05_source_mode_$stamp"

New-Item `
    -ItemType Directory `
    -Force `
    $backupDir |
    Out-Null

Copy-Item `
    $main `
    (Join-Path $backupDir "MainActivity.kt") `
    -Force

Copy-Item `
    $lidResult `
    (Join-Path $backupDir "LidResult.kt") `
    -Force

Copy-Item `
    (Join-Path $PackageRoot "MainActivity.kt") `
    $main `
    -Force

Copy-Item `
    (Join-Path $PackageRoot "LidResult.kt") `
    $lidResult `
    -Force

Write-Host ""
Write-Host "v0.5 source-mode switch installed."
Write-Host ""
Write-Host "Backup:"
Write-Host "  $backupDir"
Write-Host ""
Write-Host "Behavior:"
Write-Host "  AUTO   -> Speaker + TFLite LID + auto ASR router"
Write-Host "  MANUAL -> Speaker + fixed selected ASR; LID completely skipped"
Write-Host ""
Write-Host "UI:"
Write-Host "  old hidden btnLoad is reused as:"
Write-Host "    源语言模式：自动检测"
Write-Host "    源语言模式：手动锁定"
Write-Host ""
Write-Host "No XML, Gradle, model, sherpa AAR, or native library changes."
Write-Host ""
Write-Host "Next:"
Write-Host "  cd `"$ProjectRoot`""
Write-Host "  .\gradlew.bat clean"
Write-Host "  .\gradlew.bat assembleDebug"
