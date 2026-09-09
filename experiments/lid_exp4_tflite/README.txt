LID EXP4 - VoxLingua ONNX -> LiteRT/TFLite

Goal
----
Keep the stable Android sherpa stack untouched.
Convert only VoxLingua LID to LiteRT/TFLite, benchmark it on PC,
then integrate it into Android only if fidelity is good.

Why
---
The direct ONNX Runtime Java approach conflicted with sherpa's ORT native runtime.
LiteRT/TFLite uses a separate runtime, avoiding libonnxruntime.so conflicts.

Files
-----
convert_voxlingua_to_tflite.ps1
run_lid_exp4_tflite.py

Install these two files into:
D:\dev\projects\FourLangAsrMobile\experiments\lid_exp4_tflite\

Run
---
1.
cd D:\dev\projects\FourLangAsrMobile\experiments\lid_exp4_tflite

2.
& .\convert_voxlingua_to_tflite.ps1 `
  -ProjectRoot "D:\dev\projects\FourLangAsrMobile"

3. If conversion says CONVERSION DONE:
.\.venv_convert\Scripts\python.exe .\run_lid_exp4_tflite.py

Pass criteria
-------------
Prefer:
- TFLite 4-way overall >= 95%
- Uzbek >= 90%
- ONNX/TFLite prediction agreement >= 95%

Best case:
- TFLite stays near ONNX 79/80 = 98.75%

Important
---------
Do NOT change the Android project yet.
Do NOT add LiteRT dependencies yet.
First prove the converted TFLite model on the same 80 test clips.
