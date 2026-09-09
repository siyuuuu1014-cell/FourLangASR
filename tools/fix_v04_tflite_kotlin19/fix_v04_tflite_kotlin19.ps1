param(
    [string]$ProjectRoot = "D:\dev\projects\FourLangAsrMobile"
)

$ErrorActionPreference = "Stop"

$ProjectRoot =
    (Resolve-Path $ProjectRoot).Path

$PackageRoot =
    Split-Path -Parent $MyInvocation.MyCommand.Path

$appGradle =
    Join-Path $ProjectRoot "app\build.gradle"

$lidEngine =
    Join-Path $ProjectRoot "app\src\main\java\com\example\fourlangasr\lid\LidEngine.kt"

if (-not (Test-Path $appGradle)) {
    throw "Missing app/build.gradle: $appGradle"
}

if (-not (Test-Path $lidEngine)) {
    throw "Missing current LidEngine.kt: $lidEngine"
}

$stamp =
    Get-Date -Format "yyyyMMdd_HHmmss"

Copy-Item `
    $appGradle `
    "$appGradle.before_tflite_compat_$stamp" `
    -Force

Copy-Item `
    $lidEngine `
    "$lidEngine.before_tflite_compat_$stamp" `
    -Force

# ------------------------------------------------------------
# 1. Replace the LiteRT 2.1.6 Kotlin-based API.
#
# The project uses Kotlin 1.9.x, while LiteRT 2.1.6's Kotlin
# metadata is newer. For this prototype we use the classic
# Java Interpreter runtime from TensorFlow Lite 2.16.1.
#
# It is a separate native runtime (libtensorflowlite_jni.so),
# so it still does NOT touch sherpa's libonnxruntime.so.
# ------------------------------------------------------------

$gradle =
    [System.IO.File]::ReadAllText(
        $appGradle
    )

$gradle =
    [regex]::Replace(
        $gradle,
        "(?m)^\s*implementation\s+['""]com\.google\.ai\.edge\.litert:litert:[^'""]+['""]\s*\r?\n?",
        ""
    )

$gradle =
    [regex]::Replace(
        $gradle,
        "(?m)^\s*implementation\s+['""]org\.tensorflow:tensorflow-lite:[^'""]+['""]\s*\r?\n?",
        ""
    )

$dependenciesMatch =
    [regex]::Match(
        $gradle,
        "(?m)^\s*dependencies\s*\{"
    )

if (-not $dependenciesMatch.Success) {
    throw "Could not find dependencies { in app/build.gradle"
}

$insertAt =
    $dependenciesMatch.Index +
    $dependenciesMatch.Length

$dependencyText =
    "`r`n    // VoxLingua LID runtime. Java Interpreter API; compatible with Kotlin 1.9 project.`r`n" +
    "    implementation 'org.tensorflow:tensorflow-lite:2.16.1'"

$gradle =
    $gradle.Insert(
        $insertAt,
        $dependencyText
    )

# Write UTF-8 WITHOUT BOM.
$utf8NoBom =
    New-Object System.Text.UTF8Encoding($false)

[System.IO.File]::WriteAllText(
    $appGradle,
    $gradle,
    $utf8NoBom
)

# ------------------------------------------------------------
# 2. Install corrected LidEngine.kt using
#    org.tensorflow.lite.Interpreter.
# ------------------------------------------------------------

Copy-Item `
    (Join-Path $PackageRoot "LidEngine.kt") `
    $lidEngine `
    -Force

Write-Host ""
Write-Host "TFLite Kotlin-1.9 compatibility fix installed."
Write-Host ""
Write-Host "Dependency:"
Write-Host "  org.tensorflow:tensorflow-lite:2.16.1"
Write-Host ""
Write-Host "LidEngine API:"
Write-Host "  org.tensorflow.lite.Interpreter"
Write-Host ""
Write-Host "Not touched:"
Write-Host "  sherpa-onnx-1.13.7.aar"
Write-Host "  libonnxruntime.so"
Write-Host "  MainActivity.kt"
Write-Host "  VoxLingua TFLite model"
Write-Host ""
Write-Host "Next:"
Write-Host "  cd `"$ProjectRoot`""
Write-Host "  .\gradlew.bat clean"
Write-Host "  .\gradlew.bat assembleDebug"
