# FourLang ASR Mobile Prototype v0.5

全离线四语种（中/英/俄/乌兹别克）连续对话安卓原型：

```text
Mic (16 kHz mono, 512 samples/frame)
 ↓
Silero VAD                0.6 s 静音断句，单句最长 15 s
 ↓ 每个完整句子（串行队列，逐句处理）
Speaker (3D-Speaker ERes2Net)   会话内最多 4 个说话人档案
 ↓
LID (VoxLingua107 TFLite)       固定 5 s / 80000 样本输入，4-way softmax
 ↓
AsrManager                      仅当语种变化时才切换/重载 ASR 模型
 ↓
"[Speaker N][LANG] text"        追加到会话转录
 ↓
TranslationEngine               本机 M2M100/SMaLL-100 int8 ONNX，路由：专项→总模型→服务器
 ↓
TTS                             朗读译文（翻译不可用时回退原文）；系统离线语音优先，内置 Piper 俄语兜底
```

| Language | Engine | Model |
|---|---|---|
| Chinese | sherpa-onnx online CTC | streaming Zipformer2 CTC INT8 |
| English | sherpa-onnx offline transducer | GigaSpeech Zipformer INT8 |
| Russian | Vosk Android | vosk-model-small-ru-0.22 |
| Uzbek | sherpa-onnx online transducer | alphacep streaming Uzbek Zipformer2 |

---

## 1. 源语言模式

旧的"加载"按钮复用为模式切换（对话中不可切换）：

- **AUTO（默认）**：每句跑 LID 自动路由，spinner 只显示最近一次路由结果。
  句长 < 1 s 或 LID 出错时沿用当前已加载语言，保证 LID 失败不中断 ASR。
- **MANUAL**：完全不加载 LID（省约 82 MB 内存与推理时间），锁定手选语言。

## 2. 构建

- JDK 17、Android SDK Platform 34。
- 只构建 `arm64-v8a`，用真机测试。
- 模型不在仓库里，先运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\setup_all.ps1
```

脚本从 `D:\dev\projects\fourlang_translation\asr_exp1` 复制四个 ASR 模型，
并下载 `sherpa-onnx-1.13.7.aar`。LID / Speaker / VAD 模型需要手动放入：

```text
app/src/main/assets/
├─ models/{zh,en,ru,uz}/…          四语种 ASR 模型
├─ silero_vad.onnx
├─ speaker_embedding.onnx          3D-Speaker ERes2Net
└─ voxlingua107_float32.tflite     ~82 MB，LiteRT
```

## 3. 指标面板

- 每次识别后显示：Speaker 分数/margin、LID conf4/margin/4-mass/推理耗时、
  ASR 路由（SWITCH/REUSE）、模型加载耗时、RTF、PSS 前后与增量。
- 单句管线逻辑在 `app/src/main/java/com/example/fourlangasr/pipeline/`
  （`ConversationPipeline` + `MetricsFormatter`），阈值等调参数值
  以各引擎 companion 常量为唯一来源，UI 文案直接引用。
- LID 的四个 VoxLingua 类别索引集中在 `lid/LidTarget.kt`，换模型前必读其注释。

## 4. 真机测试关注点

- ZH/EN/RU/UZ 混合连续说话：观察 SWITCH/REUSE 行为是否正确、PSS 是否持续上涨。
- `EN load → recognize → release → EN load again`：确认 PSS 能回落。
- AUTO 与 MANUAL 两种模式的内存占用对比。

## 5. 已知边界 / 下一步

- 无流式 partial 输出。翻译已接入本机 int8 ONNX（M2M100 总模型 + 六组 SMaLL-100 专项），
  路由为「专项（启用+就绪）→ 总模型 → 服务器 API」；翻译模型不在 APK 里，
  由 Wi-Fi 下的清单同步 + 前台服务下载（约 734 MB）到 `filesDir/translation/models/`。
- TTS 已支持自动朗读开关与语速调整；播放及结束后的 900 ms 内丢弃识别输入。
  乌兹别克内置 MMS TTS 因 Android 原生兼容性问题已回退，仍依赖系统语音。
- 中文句内夹英文词已接入 `ZH <unk> → EN 局部识别 → 原位回填`；
  音频窗口可重叠，但 token 归属沿用原始分组。回填失败保留 ASR 原文（剥掉 `<unk>` 占位符），
  并在指标中显示 FALLBACK（见 `experiments/codeswitch_unk/README.txt`）。
- 说话人采用 v0.3 观察模式（低置信即新建档案），真机验证后再考虑
  tentative-speaker 阶段。

## 6. 回归检查

```powershell
.\gradlew.bat :app:testDebugUnitTest :app:assembleDebug
```

- JVM 用例使用假的 ASR / 系统 TTS，不加载原生模型：覆盖窗口重叠丢词、
  回填失败降级、识别与释放互斥、备用引擎停止及迟到的播放回调。
- 真机仍需检查：识别中退出页面、最后一句未识别完就结束对话、
  使用备用系统引擎时关闭朗读，以及停止后是否出现回声自触发。
- 若 Windows JDK 报 `UnixDomainSockets.connect0: Invalid argument`，
  可为本次构建设置 `-Djdk.net.unixdomain.tmpdir` 指向现有的项目内临时目录；
  不需要修改应用代码或系统全局环境。
