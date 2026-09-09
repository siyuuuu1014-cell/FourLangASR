FourLangAsrMobile Android v0.4 - TFLite LID Router
=================================================

PC LID candidate
----------------
VoxLingua107 TensorFlow-backend TFLite float32

Same 80 clips:
ZH 19/20 = 95%
EN 20/20 = 100%
RU 20/20 = 100%
UZ 20/20 = 100%

Overall TFLite: 79/80 = 98.75%
ONNX/TFLite prediction agreement: 80/80 = 100%

Architecture
------------
Mic
 -> Silero VAD (sherpa)
 -> Speaker (sherpa)
 -> VoxLingua107 TFLite (LiteRT)
 -> 4-way zh/en/ru/uz
 -> auto AsrManager.switchLanguage() if required
 -> ASR (sherpa)
 -> [Speaker N][LANG] text

Critical runtime rule
---------------------
DO NOT add ONNX Runtime Java.
DO NOT add libonnxruntime4j_jni.so.
DO NOT replace sherpa-onnx-1.13.7.aar or libonnxruntime.so.

LID uses LiteRT, so it is isolated from sherpa's ONNX Runtime.

Model input
-----------
Static:
  1 x 80000 float32
  16000 Hz
  5.0 seconds

Preprocessing:
  <5s: symmetric zero pad
   5s: unchanged
  >5s: center crop

Very short utterances:
  <1.0s: keep current/manual ASR language; do not run LID.

Install
-------
Extract this package, then:

& "<package>\install_v04_tflite.ps1" `
  -ProjectRoot "D:\dev\projects\FourLangAsrMobile"

Build:
cd D:\dev\projects\FourLangAsrMobile
.\gradlew.bat clean
.\gradlew.bat assembleDebug

Install:
D:\dev\tools\Android_SDK\platform-tools\adb.exe install -r `
.\app\build\outputs\apk\debug\app-debug.apk

First mobile test
-----------------
1. Open app and wait 10 seconds.
2. Start conversation.
3. Speak 2-5 seconds Chinese.
4. Speak 2-5 seconds English.
5. Speak Russian.
6. Speak Uzbek.

Expected transcript:
[Speaker 1][ZH] ...
[Speaker 1][EN] ...
[Speaker 1][RU] ...
[Speaker 1][UZ] ...

Expected metrics:
LID          : ZH/EN/RU/UZ
LID mode     : VOXLINGUA_TFLITE_4WAY
LID window   : zero_pad / exact / center_crop
LID conf4    : ...
LID margin   : ...
LID infer    : ... ms
ASR route    : REUSE XX / SWITCH -> XX
ASR load     : ... ms
