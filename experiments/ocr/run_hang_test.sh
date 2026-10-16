#!/usr/bin/env bash
# 阶段2B-0 挂起取证运行器：跑单个 OcrHangIsolationTest 方法，独立超时；
# 若超时（挂起），对进程发 SIGQUIT 让 ART dump 全线程(含 native)栈，并尝试 debuggerd -b，
# 再把 logcat 里的 PROBE/HANGDUMP/JAVASTACKS/native 栈抓下来。
# 用法: ./run_hang_test.sh <method> [iters] [timeout_s]
set -u
export MSYS_NO_PATHCONV=1
METHOD="$1"
ITERS="${2:-60}"
TMO="${3:-150}"
PKG=com.example.fourlangasr
CLASS=com.example.fourlangasr.ocr.OcrHangIsolationTest
OUT="experiments/ocr/out/hang_${METHOD}.log"

adb shell am force-stop "$PKG" >/dev/null 2>&1
adb shell input keyevent KEYCODE_WAKEUP >/dev/null 2>&1
adb logcat -c >/dev/null 2>&1
# 不过滤：SIGQUIT 触发的 ART 线程栈（含 native 回溯）走 tag "art"/"DEBUG"，被 *:S 会丢掉。
adb logcat -v brief > "$OUT" 2>&1 &
LPID=$!

timeout "$TMO" adb shell am instrument -w \
  -e class "$CLASS#$METHOD" -e iters "$ITERS" \
  "$PKG.test/androidx.test.runner.AndroidJUnitRunner" >/dev/null 2>&1
RC=$?
sleep 1

if [ "$RC" -ne 0 ]; then
  PID=$(adb shell pidof "$PKG" 2>/dev/null | tr -d '\r' | awk '{print $1}')
  if [ -n "$PID" ]; then
    echo "[runner] timeout/hang rc=$RC pid=$PID -> SIGQUIT + debuggerd" >> "$OUT"
    adb shell "kill -3 $PID" 2>/dev/null
    sleep 5
    adb shell "debuggerd -b $PID" >> "$OUT" 2>&1 || echo "[runner] debuggerd failed (needs root)" >> "$OUT"
    adb shell "cat /proc/$PID/status" 2>/dev/null | grep -E "^Threads" >> "$OUT"
    sleep 1
    adb shell am force-stop "$PKG" >/dev/null 2>&1
  fi
fi

kill "$LPID" 2>/dev/null
echo "=== $METHOD rc=$RC  (log: $OUT) ==="
grep -E "PROBE|HANGDUMP|JSTACK|OK \(|FAILURES|Process crashed|rc=" "$OUT" | tail -25
