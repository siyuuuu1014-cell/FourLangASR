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

# ============================================================
# Dual-ORT setup:
#  - sherpa 的 libsherpa-onnx-jni.so 只能与它自带的
#    libonnxruntime.so（隐藏符号构建）配套加载；
#  - onnxruntime-java 桥只能与官方 onnxruntime-android 1.27.0
#    的 libonnxruntime.so 配套加载；
#  两者交叉组合在设备上实测 dlopen 失败，因此双库并存：
#  - AAR 保持原样（sherpa 原配 libonnxruntime.so）；
#  - 官方库改名 libort4jrt.so（SONAME 同长度补丁）；
#  - 桥 libonnxruntime4j_jni.so 的 DT_NEEDED 改为 libort4jrt.so。
# 产物 libort4jrt.so 写入 app/src/main/jniLibs/arm64-v8a/（gitignored，
# 可由本脚本重建）。
# ============================================================

$OrtVersion = "1.27.0"
$OrtAar = Join-Path $LibDir "onnxruntime-android-$OrtVersion.aar"
$OrtUrl = "https://maven.aliyun.com/repository/central/com/microsoft/onnxruntime/onnxruntime-android/$OrtVersion/onnxruntime-android-$OrtVersion.aar"
$BridgeSo = Join-Path $ProjectRoot "app\src\main\jniLibsrm64-v8a\libonnxruntime4j_jni.so"
$JniDir = Join-Path $ProjectRoot "app\src\main\jniLibsrm64-v8a"

Write-Host ""
Write-Host "Downloading official onnxruntime-android $OrtVersion AAR (dual-ORT setup)"
Invoke-WebRequest -Uri $OrtUrl -OutFile $OrtAar

python (Join-Path $PSScriptRoot "setup_dual_ort.py") $Destination $OrtAar $BridgeSo $Destination $JniDir

Remove-Item -Force $OrtAar
Write-Host "[OK] dual-ORT setup done (libort4jrt.so + patched bridge)"
