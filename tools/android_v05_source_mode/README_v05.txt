FourLangAsrMobile v0.5 - Source Language Mode
=================================================

Goal
----
Add two source-language modes without touching activity_main.xml:

1. AUTO
   Speaker -> VoxLingua TFLite LID -> auto ASR router -> ASR

2. MANUAL
   Speaker -> selected fixed ASR -> ASR
   LID is completely skipped.

UI
--
The old hidden Debug "load" button is reused as a visible mode toggle:

  源语言模式：自动检测
  源语言模式：手动锁定

AUTO:
- source-language spinner is disabled for user input
- spinner automatically shows latest LID route
- currently selected source remains only a fallback for <1s clips / LID failure

MANUAL:
- source-language spinner is enabled before conversation
- user selects ZH/EN/RU/UZ
- selecting a language loads that ASR
- conversation locks to that language
- no LID inference, no automatic language switching

No XML change is needed.
No Gradle/dependency change is needed.
No model/native library change is needed.

Install
-------
& "<folder>\install_v05_source_mode.ps1" `
  -ProjectRoot "D:\dev\projects\FourLangAsrMobile"

Then:
cd D:\dev\projects\FourLangAsrMobile
.\gradlew.bat clean
.\gradlew.bat assembleDebug

Test
----
AUTO:
- Chinese -> [ZH]
- English -> SWITCH -> EN
- same English again -> REUSE EN

MANUAL:
- select 中文
- speak English
- it MUST stay Chinese ASR
- metrics should show:
  LID mode : MANUAL_LOCK
  LID infer: 0 ms
  ASR route: REUSE ZH

This proves manual mode truly bypasses LID.
