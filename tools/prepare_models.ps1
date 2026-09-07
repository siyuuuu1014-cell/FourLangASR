param(
    [string]$AsrRoot = "D:\dev\projects\fourlang_translation\asr_exp1",
    [string]$UzModelDir = ""
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$AssetsRoot = Join-Path $ProjectRoot "app\src\main\assets\models"

function Need-File([string]$Path) {
    if (-not (Test-Path $Path -PathType Leaf)) {
        throw "Missing file: $Path"
    }
}

function Need-Dir([string]$Path) {
    if (-not (Test-Path $Path -PathType Container)) {
        throw "Missing directory: $Path"
    }
}

function Reset-Dir([string]$Path) {
    if (Test-Path $Path) {
        Remove-Item -Recurse -Force $Path
    }
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

Write-Host ""
Write-Host "Preparing Android ASR model assets"
Write-Host "PC ASR root : $AsrRoot"
Write-Host "Android dst : $AssetsRoot"
Write-Host ""

# ------------------------------------------------------------
# Chinese
# ------------------------------------------------------------
$ZhSrc = Join-Path $AsrRoot "models\sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01"
$ZhDst = Join-Path $AssetsRoot "zh"

Need-File (Join-Path $ZhSrc "model.int8.onnx")
Need-File (Join-Path $ZhSrc "tokens.txt")

Reset-Dir $ZhDst

Copy-Item (Join-Path $ZhSrc "model.int8.onnx") (Join-Path $ZhDst "model.int8.onnx")
Copy-Item (Join-Path $ZhSrc "tokens.txt") (Join-Path $ZhDst "tokens.txt")

$Bbpe = Join-Path $ZhSrc "bbpe.model"
if (Test-Path $Bbpe) {
    Copy-Item $Bbpe (Join-Path $ZhDst "bbpe.model")
}

Write-Host "[OK] ZH"

# ------------------------------------------------------------
# English
# ------------------------------------------------------------
$EnSrc = Join-Path $AsrRoot "models\sherpa-onnx-zipformer-gigaspeech-2023-12-12"
$EnDst = Join-Path $AssetsRoot "en"

$EnEncoder = Join-Path $EnSrc "encoder-epoch-30-avg-1.int8.onnx"
$EnDecoder = Join-Path $EnSrc "decoder-epoch-30-avg-1.onnx"
$EnJoiner = Join-Path $EnSrc "joiner-epoch-30-avg-1.int8.onnx"
$EnTokens = Join-Path $EnSrc "tokens.txt"

Need-File $EnEncoder
Need-File $EnDecoder
Need-File $EnJoiner
Need-File $EnTokens

Reset-Dir $EnDst

Copy-Item $EnEncoder (Join-Path $EnDst "encoder.int8.onnx")
Copy-Item $EnDecoder (Join-Path $EnDst "decoder.onnx")
Copy-Item $EnJoiner (Join-Path $EnDst "joiner.int8.onnx")
Copy-Item $EnTokens (Join-Path $EnDst "tokens.txt")

Write-Host "[OK] EN"

# ------------------------------------------------------------
# Russian Vosk
# ------------------------------------------------------------
$RuSrc = Join-Path $AsrRoot "models\vosk-model-small-ru-0.22"
$RuDstRoot = Join-Path $AssetsRoot "ru"
$RuDst = Join-Path $RuDstRoot "vosk-model-small-ru-0.22"

Need-Dir $RuSrc

Reset-Dir $RuDstRoot
Copy-Item -Recurse -Force $RuSrc $RuDstRoot

Need-File (Join-Path $RuDst "am\final.mdl")

Write-Host "[OK] RU"

# ------------------------------------------------------------
# Uzbek sherpa model
# ------------------------------------------------------------
if ([string]::IsNullOrWhiteSpace($UzModelDir)) {
    $Candidates = @(
        (Join-Path $AsrRoot "models\vosk-model-small-streaming-uz"),
        (Join-Path $AsrRoot "models\alphacep-vosk-model-small-streaming-uz")
    )

    foreach ($Candidate in $Candidates) {
        if (Test-Path $Candidate -PathType Container) {
            $UzModelDir = $Candidate
            break
        }
    }
}

if ([string]::IsNullOrWhiteSpace($UzModelDir)) {
    throw @"
Uzbek model directory was not found automatically.

Run again with:
  .\tools\prepare_models.ps1 -UzModelDir "FULL_PATH_TO_vosk-model-small-streaming-uz"

The directory must contain:
  am-onnx\encoder.onnx
  am-onnx\decoder.onnx
  am-onnx\joiner.onnx
  lang\tokens.txt
"@
}

$UzEncoder = Join-Path $UzModelDir "am-onnx\encoder.onnx"
$UzDecoder = Join-Path $UzModelDir "am-onnx\decoder.onnx"
$UzJoiner = Join-Path $UzModelDir "am-onnx\joiner.onnx"
$UzTokens = Join-Path $UzModelDir "lang\tokens.txt"

Need-File $UzEncoder
Need-File $UzDecoder
Need-File $UzJoiner
Need-File $UzTokens

$UzDst = Join-Path $AssetsRoot "uz"
Reset-Dir $UzDst

Copy-Item $UzEncoder (Join-Path $UzDst "encoder.onnx")
Copy-Item $UzDecoder (Join-Path $UzDst "decoder.onnx")
Copy-Item $UzJoiner (Join-Path $UzDst "joiner.onnx")
Copy-Item $UzTokens (Join-Path $UzDst "tokens.txt")

Write-Host "[OK] UZ"

Write-Host ""
Write-Host "All model assets are ready."
Write-Host ""

Get-ChildItem -Recurse -File $AssetsRoot |
    Select-Object FullName, @{N="MB";E={[math]::Round($_.Length / 1MB, 2)}} |
    Format-Table -AutoSize
