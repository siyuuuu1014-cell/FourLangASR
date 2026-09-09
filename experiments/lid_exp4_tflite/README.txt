LID EXP4 - VoxLingua ONNX -> LiteRT/TFLite
(状态: 已完成，转换模型已集成进 Android App)

Goal
----
Keep the stable Android sherpa stack untouched.
Convert only VoxLingua LID to LiteRT/TFLite, benchmark it on PC,
then integrate it into Android only if fidelity is good.

Why
---
The direct ONNX Runtime Java approach conflicted with sherpa's ORT native runtime.
LiteRT/TFLite uses a separate runtime, avoiding libonnxruntime.so conflicts.

Files (current)
---------------
run_lid_exp4_tfbackend.py
  PC benchmark: TFLite backend vs ONNX on the same 80 test clips
  (reads the ONNX model from ../lid_exp1-3/models/voxlingua107 and the
  converted TFLite from ./converted_tf, audio from ../speaker_exp4).

lid_exp4_tfbackend_results.csv / lid_exp4_tfbackend_summary.csv
  Benchmark output. Overall 79/80 = 98.75%, TFLite/ONNX agreement 100%,
  TFLite RTF (5 s clip, PC) ~0.16.

The original conversion scripts (convert_voxlingua_*.ps1,
run_lid_exp4_tflite.py, run_lid_exp4_static5s.py) were removed after the
conversion completed. The converted model now ships as
  app/src/main/assets/voxlingua107_float32.tflite
and is loaded by
  app/src/main/java/com/example/fourlangasr/lid/LidEngine.kt
(true static 5 s model, input [1, 80000], output [1, 107];
 the four target class indices live in lid/LidTarget.kt).

Pass criteria (met)
-------------------
- TFLite 4-way overall >= 95%            -> 98.75%
- ONNX/TFLite prediction agreement >= 95% -> 100%
