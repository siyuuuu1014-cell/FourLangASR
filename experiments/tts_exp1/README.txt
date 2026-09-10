TTS 路 B 可行性验证 - sherpa-onnx Piper 俄语离线合成
=====================================================

目的
----
验证"App 内自带离线神经 TTS、不依赖系统引擎"（路 B）是否可行，
数据达标后再决定是否集成进 Android（tts/TtsEngine 的替代实现）。

模型
----
vits-piper-ru_RU-irina-medium（sherpa-onnx tts-models release）
  ru_RU-irina-medium.onnx   63 MB
  espeak-ng-data            18 MB（G2P 词典，Piper 必需）
  合计约 81 MB
样本率 22050 Hz。

PC 结果（与 App 相同的 sherpa-onnx 1.13.7，CPU 4 线程）
--------------------------------------------------------
  句子（0.6s / 2.8s / 3.3s / 6.4s 音频）:
  合成耗时 0.05 / 0.11 / 0.13 / 0.24 s
  平均 RTF 0.050（合成比实时快约 20 倍）
  全部输出非静音（RMS 0.04~0.06），时长随文本合理增长
  wav 产物：out_wav/sentence_*.wav（听感请人工试听）

结论
----
1. 路 B 可行：同一 sherpa-onnx 运行时已在 App 内，
   TTS 是官方支持的模型格式（VITS/Piper），无新增运行时。
2. 体积：+81 MB 资产（APK 现约 465 MB），可接受。
3. 速度：PC RTF 0.05；手机 CPU 预计高 3~8 倍（RTF 0.15~0.4），
   仍低于 1.0，可实时。真机数值待集成后测。
4. 音质：开源 Piper medium 为"可用但机械"，需人工试听确认可接受度。
5. 语言覆盖：zh/en/ru 均有 sherpa 官方 Piper/VITS 模型；
   uz 无开源模型——路 B 也解决不了乌兹别克语（数据问题）。

若集成：新增 OfflineNeuralTtsEngine : TtsEngine
（模型合成 PCM -> AudioTrack 播放），上层 pipeline /
translation seam / 自动朗读 / 防回灌 gate 全部不动。

运行
----
  & "D:\dev\projects\fourlang_translation\asr_exp1\.venv_sherpa\Scripts\Activate.ps1"
  cd experiments\tts_exp1
  python .\run_piper_ru_tts_probe.py

（模型在 models/ 下，已被 .gitignore 排除，不入库。）

乌兹别克 MMS TTS 验证（EXP1.2，2026-09-10）
------------------------------------------
结论：可行，需要自转换 + 西里尔转写步骤。

模型源：
  facebook/mms-tts/models/uzb-script_cyrillic（西里尔字母版，
  sherpa tts-models release 没有预导出的乌兹别克模型，需自转换）

转换（本地已执行，产物被 gitignore）：
  1. 从 hf-mirror 下载 config.json / vocab.txt / G_100000.pth（145MB）
  2. 官方 vits-mms.py 流程（MMS 源码 + torch + onnx）
  3. 本机适配（均已验证可行）：
     - 无 C 编译器 -> monotonic_align 换纯 numpy 实现
     - torch 2.14 需要 pip install onnxscript
     - torch.onnx.export 必须加 dynamo=False（旧导出器，
       新 dynamo 导出器无法追踪 RQS 的动态分支）
  产物：model.onnx 114MB + tokens.txt（16k，add_blank，
  frontend=characters）

PC 结果：
  句子（1.5s / 3.2s / 4.2s）：合成 0.23 / 0.47 / 0.53 s
  平均 RTF 0.14（比实时快约 7 倍），输出全部非静音、时长合理
  听感 wav：out_wav_uzb/（转换目录已清理，可用脚本重生成）

重要产品事实：
  MMS 乌兹别克模型只吃西里尔字母；App 的 uz ASR 输出拉丁字母，
  安卓集成前必须加 Latin->Cyrillic 转写步骤（规则映射，
  无第三方依赖即可实现）。
