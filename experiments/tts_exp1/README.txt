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

乌兹别克 Android 集成结果（2026-09-10）：回退
--------------------------------------------
PC 侧全部通过（模型转换、ORT 1.27 加载、sherpa 合成、听感验收），
但安卓集成失败：sherpa-onnx AAR 1.13.7 原生层构造该模型时静默崩溃
（App 直接退出，无 Java 异常、无 fatal 日志、dropbox 无记录、
16GB 设备内存充足）。

已排除（逐一实验验证）：
  asset 压缩 / assetManager 非 null / espeak 递归复制 /
  加载与流式识别并发（PC 无法复现）/ AudioTrack 跨线程释放 /
  内存峰值（单模型驻留）/ n_speakers 元数据 / ORT 版本
  （PC 1.27.0 正常）/ 导出工具链（torch 2.14 vs 2.0.1 图一致）/
  IR/opset 画像（IR9/opset17 转换后仍崩溃）

结论：AAR 1.13.7 原生层与 MMS 系模型不兼容（piper 系正常）。
已回退：uz 从 MODELS 移除、assets 删除、回系统引擎降级路径。
多模型引擎架构与 PC 转换产物保留在 tts_exp1；UzbekScript 转写器
与 uz 集成代码存于 git 历史（提交 12f0953），重试时可恢复。

重试清单（升级 sherpa AAR / ORT 后）：
  1. 从 git 历史恢复 UzbekScript.kt（提交 12f0953）
  2. assets/tts/uz/ 放回 model.onnx + tokens.txt
  3. 在 MODELS 中加回 uz ModelSpec + bump VERSION

sherpa AAR 升级风险评估（2026-09-11）
----------------------------------
最新版 v1.13.8（2026-09-10 发布，仅比当前 1.13.7 高一个补丁版）。

版本 diff（v1.13.7..v1.13.8，25 commits）：
  - 原生代码改动集中在 fast-clustering（说话人聚类）和 canary/cohere/
    funasr-nano/moonshine-v2 识别器——全部不是本项目用的模型；
  - **TTS 原生代码（offline-tts-*.cc）零改动** → uz MMS 崩溃在 1.13.8
    中被修复的概率低；
  - Android 构建的 ORT 从 1.27.1 升到 1.28.2（唯一可能影响 uz 崩溃
    的变量——崩溃发生在 Android ORT session 创建期，不排除是 1.27.1
    的 Android 专用 kernel bug）。

关键事实：当前 App 已无 ORT Java bridge 依赖（LID 换 TFLite 时已移除），
升级 AAR = 单文件替换，不再有 v0.4 时代的 so 配对风险。

PC 回归（sherpa 1.13.8，本机实测 run_sherpa138_regression.py）：
  uz mms TTS / ru piper TTS / zh zipformer2 CTC / en transducer / silero VAD
  全部 PASS——模型层面无回归风险。

结论：升级本身安全（补丁版 + PC 全绿 + 无配对问题），
但对解锁 uz 是低概率赌注（无 TTS 代码改动，仅 ORT 小版本提升）。
建议：可作为低成本实验直接尝试；若 uz 仍崩溃则维持现状等待
明确的 TTS/MMS 修复版本，升级本身依然保留（安全补丁）。
