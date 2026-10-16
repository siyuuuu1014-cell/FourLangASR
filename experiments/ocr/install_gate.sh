#!/usr/bin/env bash
# 绕开 Gradle connected 反复重装触发 ColorOS 安装闸：push + pm install + 自动点"继续安装"。
# 用法: ./install_gate.sh <apk-path>
set -e
export MSYS_NO_PATHCONV=1
APK="$1"
TMP="/data/local/tmp/gate_install.apk"
adb push "$APK" "$TMP" >/dev/null
adb shell input keyevent KEYCODE_WAKEUP >/dev/null 2>&1 || true
adb shell wm dismiss-keyguard >/dev/null 2>&1 || true
( adb shell pm install -r "$TMP" >/tmp/pmout 2>&1 & )
for i in $(seq 1 12); do
  sleep 2
  F=$(adb shell dumpsys window 2>/dev/null | grep -m1 mFocusedApp || true)
  case "$F" in
    *InstallGuideActivity*) adb shell input tap 300 1978 >/dev/null 2>&1; echo "[gate] 继续安装" ;;
    *InstallFinishActivity*) adb shell input tap 534 2158 >/dev/null 2>&1; echo "[gate] 完成"; break ;;
    *) : ;;
  esac
done
echo "=== result ==="; cat /tmp/pmout 2>/dev/null || true
