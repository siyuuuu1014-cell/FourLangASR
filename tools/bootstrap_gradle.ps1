param(
    [string]$GradleVersion = "8.2.1"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$WrapperDir = Join-Path $ProjectRoot "gradle\wrapper"
New-Item -ItemType Directory -Force -Path $WrapperDir | Out-Null

$Base = "https://raw.githubusercontent.com/gradle/gradle/v$GradleVersion"

$Files = @{
    "$Base/gradlew" = (Join-Path $ProjectRoot "gradlew")
    "$Base/gradlew.bat" = (Join-Path $ProjectRoot "gradlew.bat")
    "$Base/gradle/wrapper/gradle-wrapper.jar" = (Join-Path $WrapperDir "gradle-wrapper.jar")
}

foreach ($Item in $Files.GetEnumerator()) {
    Write-Host "Downloading $($Item.Key)"
    Invoke-WebRequest -Uri $Item.Key -OutFile $Item.Value
}

Write-Host ""
Write-Host "[OK] Gradle wrapper files are ready."
Write-Host "Run:"
Write-Host "  .\gradlew.bat --version"
