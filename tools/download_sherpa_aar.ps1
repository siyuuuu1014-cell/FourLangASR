param(
    [string]$Version = "1.13.7"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$LibDir = Join-Path $ProjectRoot "app\libs"
New-Item -ItemType Directory -Force -Path $LibDir | Out-Null

$FileName = "sherpa-onnx-$Version.aar"
$Destination = Join-Path $LibDir $FileName
$Url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/v$Version/$FileName"

Write-Host ""
Write-Host "Downloading sherpa-onnx Android AAR"
Write-Host "URL: $Url"
Write-Host "TO : $Destination"
Write-Host ""

Invoke-WebRequest -Uri $Url -OutFile $Destination

if (-not (Test-Path $Destination)) {
    throw "AAR download failed: $Destination"
}

$SizeMb = [math]::Round((Get-Item $Destination).Length / 1MB, 2)
Write-Host "[OK] $FileName - $SizeMb MB"
