param(
    [string]$ProjectRoot = "D:\dev\projects\FourLangAsrMobile"
)

$ErrorActionPreference = "Stop"

$PackageRoot =
    Split-Path -Parent $MyInvocation.MyCommand.Path

$ProjectRoot =
    (Resolve-Path $ProjectRoot).Path

$appDir =
    Join-Path $ProjectRoot "app"

$appGradle =
    Join-Path $appDir "build.gradle"

$javaRoot =
    Join-Path $appDir "src\main\java\com\example\fourlangasr"

$assetsDir =
    Join-Path $appDir "src\main\assets"

$mainActivity =
    Join-Path $javaRoot "MainActivity.kt"

$lidDir =
    Join-Path $javaRoot "lid"

$modelSource =
    Join-Path $ProjectRoot "experiments\lid_exp4_tflite\converted_tf\voxlingua107_float32.tflite"

$modelDest =
    Join-Path $assetsDir "voxlingua107_float32.tflite"

if (-not (Test-Path $appGradle)) {
    throw "Missing app/build.gradle: $appGradle"
}

if (-not (Test-Path $mainActivity)) {
    throw "Missing MainActivity.kt: $mainActivity"
}

if (-not (Test-Path $modelSource)) {
    throw "Missing tested TFLite model: $modelSource"
}

$modelBytes =
    (Get-Item $modelSource).Length

if ($modelBytes -lt 80000000) {
    throw "TFLite model is unexpectedly small: $modelBytes bytes"
}

# ============================================================
# Backup stable Android v0.3 before touching it.
# ============================================================

$stamp =
    Get-Date -Format "yyyyMMdd_HHmmss"

$backupDir =
    Join-Path $ProjectRoot "backup_before_v04_tflite_$stamp"

New-Item `
    -ItemType Directory `
    -Force `
    $backupDir |
    Out-Null

Copy-Item `
    $mainActivity `
    (Join-Path $backupDir "MainActivity.kt") `
    -Force

Copy-Item `
    $appGradle `
    (Join-Path $backupDir "build.gradle") `
    -Force

Write-Host "Backup:"
Write-Host "  $backupDir"
Write-Host ""

# ============================================================
# Remove ONLY leftovers from the failed ORT-Java LID attempt.
#
# Do NOT touch:
#   sherpa-onnx-1.13.7.aar
#   libonnxruntime.so
# ============================================================

$oldOrtFiles = @(
    (Join-Path $appDir "libs\onnxruntime-1.27.0-classes.jar"),
    (Join-Path $appDir "libs\onnxruntime-1.27.1-classes.jar"),
    (Join-Path $appDir "src\main\jniLibs\arm64-v8a\libonnxruntime4j_jni.so")
)

foreach ($old in $oldOrtFiles) {
    if (Test-Path $old) {
        Write-Host "Removing old ORT-Java LID leftover:"
        Write-Host "  $old"
        Remove-Item $old -Force
    }
}

$oldOnnxAsset =
    Join-Path $assetsDir "voxlingua107.onnx"

if (Test-Path $oldOnnxAsset) {
    Write-Host "Removing unused old VoxLingua ONNX asset:"
    Write-Host "  $oldOnnxAsset"
    Remove-Item $oldOnnxAsset -Force
}

# ============================================================
# Install new Kotlin files + tested TFLite model.
# ============================================================

New-Item `
    -ItemType Directory `
    -Force `
    $lidDir |
    Out-Null

Copy-Item `
    (Join-Path $PackageRoot "lid\LidEngine.kt") `
    $lidDir `
    -Force

Copy-Item `
    (Join-Path $PackageRoot "lid\LidResult.kt") `
    $lidDir `
    -Force

Copy-Item `
    (Join-Path $PackageRoot "MainActivity.kt") `
    $mainActivity `
    -Force

Copy-Item `
    $modelSource `
    $modelDest `
    -Force

# ============================================================
# Patch app/build.gradle.
# ============================================================

$gradle =
    [System.IO.File]::ReadAllText(
        $appGradle
    )

# Remove failed ORT Java local-jar dependencies if they remain.
$gradle =
    [regex]::Replace(
        $gradle,
        "(?m)^\s*implementation\s+files\(['""]libs/onnxruntime-[^'""]+['""]\)\s*\r?\n?",
        ""
    )

# Normalize any existing LiteRT dependency.
$gradle =
    [regex]::Replace(
        $gradle,
        "(?m)^\s*implementation\s+['""]com\.google\.ai\.edge\.litert:litert:[^'""]+['""]\s*$",
        "    implementation 'com.google.ai.edge.litert:litert:2.1.6'"
    )

if (
    $gradle -notmatch
    "com\.google\.ai\.edge\.litert:litert:"
) {

    $match =
        [regex]::Match(
            $gradle,
            "(?m)^\s*dependencies\s*\{"
        )

    if (-not $match.Success) {
        throw "Could not find dependencies { in app/build.gradle"
    }

    $insertAt =
        $match.Index +
        $match.Length

    $dependencyText =
        "`r`n    // VoxLingua107 LID runtime; separate from sherpa ONNX Runtime.`r`n" +
        "    implementation 'com.google.ai.edge.litert:litert:2.1.6'"

    $gradle =
        $gradle.Insert(
            $insertAt,
            $dependencyText
        )
}

# IMPORTANT: write UTF-8 WITHOUT BOM.
$utf8NoBom =
    New-Object System.Text.UTF8Encoding($false)

[System.IO.File]::WriteAllText(
    $appGradle,
    $gradle,
    $utf8NoBom
)

Write-Host ""
Write-Host "Android v0.4 TFLite LID installed."
Write-Host ""
Write-Host "Installed:"
Write-Host "  app\src\main\assets\voxlingua107_float32.tflite"
Write-Host "  app\src\main\java\com\example\fourlangasr\lid\LidEngine.kt"
Write-Host "  app\src\main\java\com\example\fourlangasr\lid\LidResult.kt"
Write-Host "  app\src\main\java\com\example\fourlangasr\MainActivity.kt"
Write-Host "  LiteRT dependency: com.google.ai.edge.litert:litert:2.1.6"
Write-Host ""
Write-Host "NOT modified:"
Write-Host "  app\libs\sherpa-onnx-1.13.7.aar"
Write-Host "  sherpa libonnxruntime.so"
Write-Host ""
Write-Host "Next:"
Write-Host "  cd `"$ProjectRoot`""
Write-Host "  .\gradlew.bat clean"
Write-Host "  .\gradlew.bat assembleDebug"
