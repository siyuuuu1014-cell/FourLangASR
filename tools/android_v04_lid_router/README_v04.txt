FourLangAsrMobile Android v0.4
====================================

Goal
----
Mic
 -> Silero VAD
 -> Speaker
 -> VoxLingua107 LID (4-way zh/en/ru/uz)
 -> auto AsrManager.switchLanguage() only when language changes
 -> ASR
 -> transcript: [Speaker N][LANG] text

PC benchmark used for this candidate
------------------------------------
VoxLingua107 ECAPA ONNX:
ZH 19/20 = 95%
EN 20/20 = 100%
RU 20/20 = 100%
UZ 20/20 = 100%
Overall 4-way = 79/80 = 98.75%

Important Android runtime detail
--------------------------------
The project already uses:
  app/libs/sherpa-onnx-1.13.7.aar

sherpa-onnx v1.13.7's Android build script uses ONNX Runtime 1.27.1.

Do NOT add a random com.microsoft.onnxruntime:onnxruntime-android version,
because that can package a second/incompatible libonnxruntime.so.

This package instead installs the matching ORT 1.27.1 Java bridge only:
  app/libs/onnxruntime-1.27.1-classes.jar
  app/src/main/jniLibs/arm64-v8a/libonnxruntime4j_jni.so

The existing sherpa AAR continues to provide libonnxruntime.so.

Install
-------
1. Extract this ZIP anywhere.

2. In PowerShell:
   & "<extracted-folder>\install_v04.ps1" `
     -ProjectRoot "D:\dev\projects\FourLangAsrMobile"

3. Build:
   cd D:\dev\projects\FourLangAsrMobile
   .\gradlew.bat assembleDebug

4. Install:
   D:\dev\tools\Android_SDK\platform-tools\adb.exe install -r `
   .\app\build\outputs\apk\debug\app-debug.apk

Expected UI behavior
--------------------
The existing source-language spinner remains as a debug/fallback control.
During a conversation it is automatically updated to the detected language.

Transcript examples:
  [Speaker 1][ZH] 你好
  [Speaker 2][RU] ...
  [Speaker 1][EN] hello

LID behavior
------------
- >= 1.0 s utterance:
    run VoxLingua 4-way zh/en/ru/uz
- < 1.0 s:
    keep currently loaded ASR language (avoids unstable short-clip LID)
- normal LID:
    if detected language == loaded ASR:
        reuse ASR
    else:
        switchLanguage(detected)
        then transcribe
- first MVP version does not reject low-confidence LID.
  It displays 4-way confidence/margin for debugging.

Files added
-----------
app/src/main/java/com/example/fourlangasr/lid/
  LidEngine.kt
  LidResult.kt

app/src/main/java/com/example/fourlangasr/speaker/
  SpeakerEngine.kt
  SpeakerManager.kt
  SpeakerProfile.kt
  SpeakerResult.kt

assets:
  speaker_embedding.onnx
  voxlingua107.onnx

native/java bridge:
  app/libs/onnxruntime-1.27.1-classes.jar
  app/src/main/jniLibs/arm64-v8a/libonnxruntime4j_jni.so

No activity_main.xml change is required.
