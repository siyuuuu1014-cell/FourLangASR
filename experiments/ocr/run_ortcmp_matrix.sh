#!/usr/bin/env bash
# C1 配置对照：依次对官方 ORT 跑单变量配置，每次独立进程 + 独立超时 + 独立日志。
# 通过监控日志：出现 DONE 视为通过->结束；出现 HANGDUMP 视为挂起->记录后强制停止。
# 用法: ./run_ortcmp_matrix.sh [iters] [stallMs] [timeout_s] [config ...]
set -u
export MSYS_NO_PATHCONV=1
ITERS="${1:-200}"; STALL="${2:-12000}"; TMO="${3:-150}"; shift 3 || true
CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then CONFIGS=(baseline intra2 intra1 nospin noarena nomempattern seq); fi
PKG=com.ortcmp.stress
RUNNER="$PKG.test/androidx.test.runner.AndroidJUnitRunner"
CLASS=com.ortcmp.stress.OrtOfficialStressTest
OUT=experiments/ocr/out
mkdir -p "$OUT"
SUMMARY="$OUT/ortcmp_c1_summary.txt"; : > "$SUMMARY"

for cfg in "${CONFIGS[@]}"; do
  LOG="$OUT/ortcmp_c1_${cfg}.log"
  adb shell am force-stop "$PKG" >/dev/null 2>&1
  adb logcat -c >/dev/null 2>&1
  adb shell input keyevent KEYCODE_WAKEUP >/dev/null 2>&1
  adb logcat OrtStress:V AndroidRuntime:E '*:S' > "$LOG" 2>&1 &
  LPID=$!
  ( timeout "$TMO" adb shell am instrument -w -e class "$CLASS#officialReuseSessionInference" \
      -e config "$cfg" -e iters "$ITERS" -e stallMs "$STALL" "$RUNNER" >/dev/null 2>&1 ) &
  APID=$!
  # 监控：DONE(通过) 或 HANGDUMP(挂起) 提前结束
  RESULT="unknown"
  for i in $(seq 1 "$TMO"); do
    sleep 1
    if grep -q "DONE reuse" "$LOG" 2>/dev/null; then RESULT="PASS"; break; fi
    if grep -q "HANGDUMP" "$LOG" 2>/dev/null; then RESULT="HANG"; sleep 2; break; fi
    if ! kill -0 "$APID" 2>/dev/null; then break; fi
  done
  kill "$APID" 2>/dev/null; adb shell am force-stop "$PKG" >/dev/null 2>&1; sleep 1; kill "$LPID" 2>/dev/null
  COMPLETED=$(grep -oE "lastCompletedIter=[0-9]+" "$LOG" | tail -1 | grep -oE "[0-9]+")
  [ -z "$COMPLETED" ] && COMPLETED=$(grep -oE "ROUND END reuse config=$cfg iter=[0-9]+" "$LOG" | grep -oE "[0-9]+$" | sort -n | tail -1)
  STEP=$(grep -oE "lastStep='[^']*'" "$LOG" | tail -1)
  echo "$cfg -> $RESULT completed=${COMPLETED:-?}/$ITERS ${STEP:-}" | tee -a "$SUMMARY"
done
echo "=== summary ==="; cat "$SUMMARY"
