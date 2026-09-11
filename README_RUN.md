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
TranslationEngine               目前是 Placeholder，接口已预留
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

- 无流式 partial 输出；无 TTS；翻译仍是 Placeholder。
- 短外语词（"嗯 hello 我的名字叫……"）检测路线已从 VoxLingua 短窗口
  转向"主语言 ASR 不确定度触发候选验证"（见
  `experiments/codeswitch_ascend/README.txt` 的结论）。
- 说话人采用 v0.3 观察模式（低置信即新建档案），真机验证后再考虑
  tentative-speaker 阶段。
