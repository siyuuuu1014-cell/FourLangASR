Fix for FourLangAsrMobile v0.4 TFLite build
================================================

Observed error
--------------
LiteRT 2.1.6 contains Kotlin metadata 2.3.0.
The project root build.gradle uses Kotlin plugin 1.9.22.
Therefore Kotlin compilation fails before LidEngine can resolve the API.

Fix
---
Use the classic Java TensorFlow Lite Interpreter runtime:

  implementation 'org.tensorflow:tensorflow-lite:2.16.1'

and:

  import org.tensorflow.lite.Interpreter

This keeps the LID runtime separate from sherpa ONNX Runtime:
  LID    -> libtensorflowlite_jni.so
  sherpa -> libonnxruntime.so

It does NOT restore the old ORT-Java experiment.

Run
---
& "<folder>\fix_v04_tflite_kotlin19.ps1" `
  -ProjectRoot "D:\dev\projects\FourLangAsrMobile"

Then:
cd D:\dev\projects\FourLangAsrMobile
.\gradlew.bat clean
.\gradlew.bat assembleDebug

If build succeeds, install and first test app startup, then Start Conversation.
