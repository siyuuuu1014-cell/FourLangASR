# FourLang ASR Mobile Prototype v0.1

This Android project integrates the current four-language ASR candidates:

| Language | Engine | Model |
|---|---|---|
| Chinese | sherpa-onnx online CTC | streaming Zipformer2 CTC INT8 |
| English | sherpa-onnx offline transducer | GigaSpeech Zipformer INT8 |
| Russian | Vosk Android | vosk-model-small-ru-0.22 |
| Uzbek | sherpa-onnx online transducer | alphacep streaming Uzbek Zipformer2 |

The app is a **push-to-talk prototype**:
1. Choose source language.
2. Load the language model.
3. Tap Start recording.
4. Speak.
5. Tap Stop and recognize.
6. Read offline ASR text, RTF, and Android PSS.

Translation is intentionally a placeholder. The interface is already present so your
offline translation model can be connected later without changing the ASR layer.

---

## 0. Gradle wrapper

The source zip includes a bootstrap script. Run `tools/setup_all.ps1` and it
downloads `gradlew`, `gradlew.bat`, and `gradle-wrapper.jar` automatically.

Use **JDK 17**.

---

## 1. Extract project

Example:

```powershell
cd D:\dev\projects
Expand-Archive .\FourLangAsrMobile.zip -DestinationPath .
cd .\FourLangAsrMobile
```

---

## 2. Download sherpa-onnx Android AAR and copy your existing models

Your PC ASR experiment root is assumed to be:

```text
D:\dev\projects\fourlang_translation\asr_exp1
```

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\setup_all.ps1
```

This will:

- Download:
  `sherpa-onnx-1.13.7.aar`
- Copy your tested ZH/EN/RU/UZ models into `app/src/main/assets/models/`

If Uzbek is in a different directory:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\setup_all.ps1 `
  -UzModelDir "D:\your\path\vosk-model-small-streaming-uz"
```

---

## 3. Expected model layout

```text
app/src/main/assets/models/
├─ zh/
│  ├─ model.int8.onnx
│  └─ tokens.txt
├─ en/
│  ├─ encoder.int8.onnx
│  ├─ decoder.onnx
│  ├─ joiner.int8.onnx
│  └─ tokens.txt
├─ uz/
│  ├─ encoder.onnx
│  ├─ decoder.onnx
│  ├─ joiner.onnx
│  └─ tokens.txt
└─ ru/
   └─ vosk-model-small-ru-0.22/
      ├─ am/
      ├─ conf/
      ├─ graph/
      └─ ...
```

---

## 4. Android Studio settings

Install:

- Android Studio
- Android SDK Platform 34
- JDK 17 (Android Studio embedded JDK is fine)

This prototype builds only:

```text
arm64-v8a
```

Use a real modern Android phone.

For an x86_64 emulator, edit `app/build.gradle`:

```gradle
ndk {
    abiFilters 'arm64-v8a', 'x86_64'
}
```

---

## 5. Run on phone

On the phone:

1. Enable Developer options.
2. Enable USB debugging.
3. Connect with USB.
4. Accept the RSA debugging prompt.

In Android Studio:

1. Open the `FourLangAsrMobile` folder.
2. Wait for Gradle Sync.
3. Select your phone.
4. Click **Run ▶**.

The first debug APK is large because all four ASR models are bundled.
That is expected for this prototype.

---

## 6. Test sequence

Start with this order:

### Chinese
- Source: Chinese
- Load model
- Record 5-15 seconds
- Stop & recognize
- Record:
  - model load ms
  - PSS load delta
  - inference ms
  - RTF
  - text quality

### English
Repeat.

### Uzbek
Repeat.

### Russian
The first Russian load copies the Vosk model from APK assets into internal app storage.
That first load can therefore take longer. Later loads reuse the copied model.

---

## 7. Important architecture

Only **one source-language ASR model** is loaded at a time.

```text
Mic
 ↓
16 kHz mono PCM
 ↓
AsrManager
 ├─ zh -> sherpa-onnx
 ├─ en -> sherpa-onnx
 ├─ ru -> Vosk
 └─ uz -> sherpa-onnx
 ↓
ASR text
 ↓
TranslationEngine        <-- your model goes here later
 ↓
Target text
```

The translation integration point is:

```text
app/src/main/java/com/example/fourlangasr/translation/TranslationEngine.kt
```

Replace `PlaceholderTranslationEngine` later.

---

## 8. What to report from the first phone test

For each language send:

```text
Phone model:
Android version:

ZH:
load:
PSS delta:
audio:
inference:
RTF:
prediction:

EN:
...

RU:
...

UZ:
...
```

Also test:

```text
EN load -> recognize -> release
ZH load -> recognize -> release
EN load again
```

and tell us whether PSS keeps rising.

---

## Notes

- The app performs no network ASR.
- It does not yet implement VAD.
- It does not yet display streaming partial text.
- It does not yet connect translation or TTS.
- Those should come after the first real-device ASR/RAM benchmark is stable.
