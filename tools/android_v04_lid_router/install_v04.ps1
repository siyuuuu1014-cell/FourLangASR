param(
    [string]$ProjectRoot = "D:\dev\projects\FourLangAsrMobile"
)

$ErrorActionPreference = "Stop"

$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path $ProjectRoot).Path

Write-Host "ProjectRoot: $ProjectRoot"
Write-Host "PackageRoot: $PackageRoot"

$appDir = Join-Path $ProjectRoot "app"
$javaRoot = Join-Path $appDir "src\main\java\com\example\fourlangasr"
$assetsDir = Join-Path $appDir "src\main\assets"
$libsDir = Join-Path $appDir "libs"
$jniDir = Join-Path $appDir "src\main\jniLibs\arm64-v8a"

if (-not (Test-Path (Join-Path $appDir "build.gradle"))) {
    throw "app\build.gradle not found under $ProjectRoot"
}

New-Item -ItemType Directory -Force $libsDir | Out-Null
New-Item -ItemType Directory -Force $jniDir | Out-Null
New-Item -ItemType Directory -Force (Join-Path $javaRoot "speaker") | Out-Null
New-Item -ItemType Directory -Force (Join-Path $javaRoot "lid") | Out-Null

# -----------------------------------------------------------------
# Backup current Android source/config
# -----------------------------------------------------------------
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backupDir = Join-Path $ProjectRoot "backup_v04_$stamp"
New-Item -ItemType Directory -Force $backupDir | Out-Null

$mainActivity = Join-Path $javaRoot "MainActivity.kt"
if (Test-Path $mainActivity) {
    Copy-Item $mainActivity (Join-Path $backupDir "MainActivity.kt")
}

Copy-Item (Join-Path $appDir "build.gradle") (Join-Path $backupDir "build.gradle")

Write-Host "Backup: $backupDir"

# -----------------------------------------------------------------
# Install Kotlin files + build.gradle
# -----------------------------------------------------------------
Copy-Item (Join-Path $PackageRoot "MainActivity.kt") $mainActivity -Force
Copy-Item (Join-Path $PackageRoot "build.gradle") (Join-Path $appDir "build.gradle") -Force

Get-ChildItem (Join-Path $PackageRoot "speaker\*.kt") | ForEach-Object {
    Copy-Item $_.FullName (Join-Path $javaRoot "speaker") -Force
}

Get-ChildItem (Join-Path $PackageRoot "lid\*.kt") | ForEach-Object {
    Copy-Item $_.FullName (Join-Path $javaRoot "lid") -Force
}

# -----------------------------------------------------------------
# Copy the two ONNX models already tested on PC
# -----------------------------------------------------------------
$speakerModel = Join-Path $ProjectRoot "experiments\speaker_exp1\models\speaker_embedding.onnx"
$lidModel = Join-Path $ProjectRoot "experiments\lid_exp1\models\voxlingua107\voxlingua107.onnx"

if (-not (Test-Path $speakerModel)) {
    throw "Speaker model not found: $speakerModel"
}

if (-not (Test-Path $lidModel)) {
    throw "LID model not found: $lidModel"
}

Copy-Item $speakerModel (Join-Path $assetsDir "speaker_embedding.onnx") -Force
Copy-Item $lidModel (Join-Path $assetsDir "voxlingua107.onnx") -Force

# -----------------------------------------------------------------
# Install EXACT ONNX Runtime Java bridge matching sherpa-onnx 1.13.7
#
# sherpa-onnx v1.13.7 Android build uses ORT 1.27.1.
# We reuse sherpa's libonnxruntime.so and add only:
#   - classes.jar
#   - libonnxruntime4j_jni.so
#
# This avoids packaging a second libonnxruntime.so.
# -----------------------------------------------------------------
$cacheDir = Join-Path $ProjectRoot "tools\onnxruntime_1_27_1"
New-Item -ItemType Directory -Force $cacheDir | Out-Null

$zipPath = Join-Path $cacheDir "onnxruntime-android-1.27.1.zip"
$extractDir = Join-Path $cacheDir "extracted"

if (-not (Test-Path $zipPath)) {
    Write-Host "Downloading ONNX Runtime 1.27.1 Android bridge..."
    & curl.exe -L -C - `
        --retry 20 `
        --retry-delay 2 `
        --retry-all-errors `
        "https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.27.1/onnxruntime-android-1.27.1.zip" `
        -o $zipPath

    if ($LASTEXITCODE -ne 0) {
        throw "curl download failed."
    }
}

if (Test-Path $extractDir) {
    Remove-Item $extractDir -Recurse -Force
}

Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force

$ortClasses = Join-Path $extractDir "classes.jar"
$ortJni = Join-Path $extractDir "jni\arm64-v8a\libonnxruntime4j_jni.so"

if (-not (Test-Path $ortClasses)) {
    throw "classes.jar missing from ORT archive: $ortClasses"
}

if (-not (Test-Path $ortJni)) {
    throw "libonnxruntime4j_jni.so missing from ORT archive: $ortJni"
}

Copy-Item $ortClasses (Join-Path $libsDir "onnxruntime-1.27.1-classes.jar") -Force
Copy-Item $ortJni (Join-Path $jniDir "libonnxruntime4j_jni.so") -Force

Write-Host ""
Write-Host "Installed:"
Write-Host "  app\src\main\assets\speaker_embedding.onnx"
Write-Host "  app\src\main\assets\voxlingua107.onnx"
Write-Host "  app\libs\onnxruntime-1.27.1-classes.jar"
Write-Host "  app\src\main\jniLibs\arm64-v8a\libonnxruntime4j_jni.so"
Write-Host "  speaker\*.kt"
Write-Host "  lid\*.kt"
Write-Host "  MainActivity.kt"
Write-Host "  app\build.gradle"
Write-Host ""
Write-Host "Next:"
Write-Host "  cd `"$ProjectRoot`""
Write-Host "  .\gradlew.bat assembleDebug"
