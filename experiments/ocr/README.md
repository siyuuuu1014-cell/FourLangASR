# OCR 阶段 0：可行性验证报告（隔离实验，未触碰生产代码）

分支：`feat/ocr-phase0-validation`。本目录只含实验脚本与结果，权重/数据集/虚拟环境全部 gitignore。
所有结论按 **已执行 / 未执行 / 推测** 三档标注，理论可行一律不写成已验证。

---

## 1. 实际使用的模型（已执行：下载并逐文件哈希）

来源 Hugging Face `PaddlePaddle/*_onnx`，license 标签均为 `apache-2.0`（HF API 实测）。
完整机器可读记录见 `model_assets.json`（含 URL、HTTP 状态、字节数、全量 sha256）。

| 模型 | 文件 | 体积 | sha256(前16) |
|---|---|---|---|
| PP-OCRv6_tiny_det_onnx | inference.onnx | 1.70 MB | `193bab7a04fca699` |
| PP-OCRv6_tiny_rec_onnx | inference.onnx | 4.26 MB | `9ef676d6ed3c8825` |
| PP-OCRv6_small_det_onnx | inference.onnx | 9.42 MB | `d73e0058b7a8086b` |
| PP-OCRv6_small_rec_onnx | inference.onnx | 20.18 MB | `5435fd747c9e0efe` |
| PP-OCRv5_mobile_det_onnx | inference.onnx | 4.60 MB | `a431985659dc9219` |

版本：PaddleOCR 3.7.0 / paddlex 对应发行、PP-OCRv6 三档（tiny 1.5M 参数、49 语言不含日文；small 7.7M、50 语言）。
官方文档确认 **乌兹别克文（uz）在 v6 的 50 语言表内**（`docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.md`）。

模型 I/O 契约（已执行，`ort_compat_check.py` 打印）：

- det：输入 `x` `[N,3,H,W]`（H/W 动态，需 32 的倍数），输出单通道概率图；预处理 `img_mode=BGR`、
  `scale=1/255`、`mean=[0.485,0.456,0.406]`、`std=[0.229,0.224,0.225]`。
- rec：输入 `x` `[N,3,48,W]`，tiny 输出 6906 类 / small 输出 18710 类；预处理 `RecResizeImg image_shape=[3,48,320]`。
- 后处理：det 为 `DBPostProcess`，rec 为 `CTCLabelDecode`，字典在包内 `inference.yml`（tiny 6904 条）。

## 2. ONNX Runtime 兼容性（已执行，PC CPU EP）

用两个隔离环境跑同一批模型：`onnxruntime==1.31.0`（系统）与 `onnxruntime==1.27.0`
（与本 App 的 `onnxruntime-1.27.0-classes.jar` + `libort4jrt.so` 同一核心版本）。

| 优化级别 | 5 个模型 1.31 vs 1.27 逐字节一致 | argmax 序列一致 |
|---|---|---|
| DISABLE_ALL | 5/5 | 5/5 |
| **BASIC**（App 翻译引擎所用） | **5/5** | 5/5 |
| EXTENDED | 4/5 | 5/5 |
| ALL | 4/5 | 5/5 |

唯一差异是 `PP-OCRv6_small_rec` 在 EXTENDED/ALL：`max|Δ|=2.2e-6`、`mean|Δ|=8e-11`，
**argmax 序列完全相同**（即解码文本不变）。结论：

1. 真实 PP-OCRv6 模型能在 ORT 1.27.0 上加载并推理（PC 侧已验证）。
2. **Android 侧必须把 OCR session 固定为 `ORT_ENABLE_BASIC`**，这样 PC↔Android 的数值输出可逐字节复现，
   与 `KitTranslationEngine.makeSession` 现有约束一致。
3. PC CPU 单次耗时（非设备数据）：tiny det 33.9ms@640×640、tiny rec 13.1ms、small det 59.6ms、small rec 18.9ms。

**未执行**：Android 真机上用打过补丁的 `libort4jrt.so` 加载 OCR 模型的 dlopen/推理验证（需阶段 1 的 APK）。

## 3. 样本与结果（已执行 32 张；口径缺陷已标明）

`select_samples.py` 产出两层：

- `handwritten_photo` 24 张 —— **真实拍摄**的乌兹别克语手写作业页（蓝黑钢笔、横线纸、含透视/阴影/模糊）。
  来源 HF `abduazizovanozima7/uzbek-ocr-dataset`（528 张 + `final_dataset.csv`，**未声明许可证**，
  仅用于内部可行性判断，不入库、不分发）。
- `printed_synth` 8 张 —— 用同一批真实乌兹别克语句子渲染的印刷体，再加温和相机降质（旋转±2.5°、
  错切、模糊 0.3–0.9、光照渐变、噪声）。**合成样本**，只用于覆盖印刷体层级。

评测（`eval_ocr.py`，去空格去标点后的字符 CER + 字母数字词元 recall/precision + 词元对齐的撇号矩阵）：

| 层 | 模型 | CER 中位 | 词召回 | 平均置信 | PC 耗时/图 |
|---|---|---|---|---|---|
| printed_synth | tiny | **0.006** | 0.971 | 0.968 | 276 ms |
| printed_synth | small | 0.013 | 0.913 | 0.983 | 503 ms |
| handwritten_photo | tiny | 0.378 | 0.189 | 0.815 | 696 ms |
| handwritten_photo | small | 0.274 | 0.326 | 0.859 | 1185 ms |

必须写清的三点口径限制：

1. 手写层 GT 是**整页片段级**标注（无行框、部分页只标了段落），页级 CER 有离群值（1 张 CER>1.5，
   最高 15.6，因输出比 GT 长得多）。因此**手写层的绝对 CER 不可信**，只能作方向性判断：
   24 张里只有 2–3 张词召回 >0.5。
2. `行内匹配`（OCR 行文本是否为 GT 子串）对行切分方式敏感，不等同于准确率：small 该值低于 tiny
   主要是行合并/切分不同，其文本 CER 其实只有 0.013。
3. **置信度与准确率无关**：实测出现 `conf 0.86 / 词召回 0.18` 与 `conf 0.89 / 词召回 0.60`。
   → 不能拿置信度当质量闸门，用户校对步骤是强制项，不是可选优化。

**未执行**：真实拍摄的印刷文档/路牌/屏幕照片（数据集中没有）、≥100 张分层集、人工行级标注。

## 4. 乌兹别克撇号全链路（已执行）

字典层（实测 `inference.yml`）：tiny 6904 条、small 18708 条，
**两者都不含 U+02BB `ʻ` 与 U+02BC `ʼ`**，只有 `U+0027 '`、`U+2018 ‘`、`´`。
→ 模型在原理上不可能输出官方 `oʻ/gʻ` 写法。

OCR 层（印刷层词元对齐统计，`out/eval_*_32.json`）：

| GT 写法 | tiny 输出 | small 输出 |
|---|---|---|
| `U+02BB ʻ` | → `U+0027` 16/18，丢失 2/18 | → **丢失 17/18**，→`U+0027` 1/18 |
| `U+2018 ‘` | → `U+0027` 4/6，丢失 2/6 | → 丢失 4/6 |
| `U+02BC ʼ` | → `U+0027` 2/2 | → `U+0027` 2/2 |

**tiny 保留撇号位置（换成 ASCII），small 直接吞掉** —— 这是选 tiny 的实测理由，不是体积理由。

分词层（`sentencepiece` 加载 App 同一个 `translation_pkg/sentencepiece.bpe.model`，128000 pieces，`unk_id=0`）：
`ʻ / ‘ / '` 三种写法 **都不产生 `<unk>`**，各自成独立 piece；去掉撇号会让词融合
（`▁O|ʻ|zbekiston` → `▁O|zbekiston`，19 pieces → 17 pieces）。

翻译层（`apostrophe_chain_test.py`，加载 App 实际使用的 int8 ONNX 包，`ORT_ENABLE_BASIC` + greedy，
与 `KitTranslationEngine` 同契约）同一句 uz→zh / uz→en 四种写法：

| 输入写法 | uz→zh | 与官方写法是否相同 |
|---|---|---|
| `Oʻ…koʻ…`（U+02BB） | 乌兹别克斯坦的经济依赖于工业和服务网络. | 基准 |
| `O‘…ko‘…`（U+2018） | 乌兹别克斯坦的经济依赖于工业和服务网络. | 相同 |
| `O'…ko'…`（ASCII） | 乌兹别克斯坦经济依赖于工业和服务网络. | 轻微差异（zh）/相同（en） |
| **丢掉撇号** | 乌兹别克斯坦经济依赖于工业和服务**交互**网络. | **语义漂移** |

uz→en 同样：丢撇号时 `service networks` → `service cross-country networks`。

**结论（已验证）**：撇号**绝对不能删**；`ʻ/‘/'` 三者对翻译基本等价。因此规范化方向是
"把各种撇号统一到一种写法并**保留符号**"，而不是"去掉特殊字符"。
三态文本（OCR 原文 / 用户校对原文 / 送翻译的规范化文本）中，规范化只做：
撇号族 → 统一为 `ʻ`（或统一为 `'`，二选一固定）、NFC、连续空白折叠、行尾连字符处理。
具体规则表与单测在阶段 1 落地（`U+0027/U+02BB/U+02BC/U+2018/U+2019` 全覆盖，禁止任何删除式映射）。

## 5. 官方 Android SDK 源码研读（已执行，源码级）

`deploy/ppocr-android/ppocr-sdk`：23 个 Kotlin 文件约 1760 行，**无 JNI/C++/CMake**。

- 依赖：`com.microsoft.onnxruntime:onnxruntime-android:1.21.1` + `com.quickbirdstudios:opencv:4.5.3`
  （实测该 AAR：`libopencv_java4.so` arm64 17.1MB + `libc++_shared.so` 6.2MB，单一 fat so，其 manifest `minSdkVersion=21`）。
- `ORTSessionManager.kt:40-43` 用 **`ALL_OPT`**（与 PC/Android 逐字节复现的要求冲突，移植时必须改成 BASIC）。
- `minSdk=26` 是项目自设（demo 只有 `mipmap-anydpi-v26` 自适应图标），SDK 代码里
  `Build.VERSION_CODES/RequiresApi/java.time` 零命中 → **不是 API 硬要求**。
- 纯 Kotlin、无 OpenCV 依赖可直接搬：`PolygonUnclip`（unclip 用圆角采样，Python pyclipper 用直角，**已知差异**）、
  `QuadGeometry`、`BoxSorter`、`CTCDecoder`（blank=0、`charIdx=idx-1`、置信度=保留步均值）、
  `MathUtils`、`YamlUtils`、`ModelConfig`（手写 YAML 解析，只读 `character_dict` 并在末尾补一个空格）。
- 必须重写的 OpenCV 调用点：`findContours`（最难）、`minAreaRect`、`fillPoly+Core.mean`、
  `warpPerspective(INTER_CUBIC, BORDER_REPLICATE)`、`threshold/dilate`、`resize(INTER_LINEAR)`、
  `imdecode`、`bitmapToMat`。
- 与 Python 官方产线的**已知行为差异**（对拍时必须预期）：候选框截断取 `findContours` 原始顺序前 3000
  （Python 按轮廓点数降序）、竖排裁剪触发阈值 1.5（PaddleX 为 2）、`boxType` 只实现 quad 未实现 polygon、
  rec 最大宽 3200（不是 320）、det 归一化按 BGR 通道顺序施加 mean/std。
- 该 SDK 只跑 det+rec，无整页/单行方向分类器；`OCRResult.wordBoxes` 字段存在但从未赋值。
- benchmark 脚手架可借鉴：`OCRBenchmarkTest` 的 `testLatencyBenchmark`（warmup+iterations，
  mean/stdev/p90/min，分 det/rec 的 pre/infer/post）与 `testMemoryBenchmark`（`Debug.MemoryInfo` 的
  totalPss/nativePss/dalvikPss 前后差）、`testOCRExportJSON`（导出与 PC 比对的 JSON）。

## 6. 三条 Android 后处理路径的评估

| | A 移植官方 Kotlin + 复用我们 ORT | B 复用官方后处理 + 保留 OpenCV | C 完全自研 Kotlin |
|---|---|---|---|
| 正确性 | 中：搬过来的部分与 Python 有 5 处已知差异，但都是官方已实现逻辑 | **高**：与官方 Android 行为一致，可直接用官方 benchmark 对拍 | 低：`findContours/minAreaRect/warp` 全要自己写对 |
| 维护成本 | 中（要维护 4 个几何原语的替代实现） | 低（跟随上游） | 高 |
| APK 增量 | ~0 | **+23.3 MB**（arm64 的 opencv fat so + libc++_shared） | 0 |
| minSdk 23 | 可行（26 非硬要求） | OpenCV AAR manifest 为 21 → 可行 | 可行 |
| 运行内存 | 低 | 低（OpenCV 主要是 so 常驻） | 低 |
| 与现有双 ORT 冲突 | 无（改依赖为我们自己的桥） | 需验证 `libc++_shared.so` 与 sherpa/vosk/ORT 现有 so 共存 | 无 |
| Kotlin 1.9.22 可编译 | 需以**源码引入**而非 AAR（官方用 Kotlin 2.1.0 编译，metadata 版本不兼容） | 同样需源码引入 | 原生 |

**推荐：B 为首选，但设一道 30 分钟量级的准入门槛**——阶段 1 第一步先做一个不含业务的最小 APK，
验证 `libopencv_java4.so` + `libc++_shared.so` 与我们打过补丁的 `libort4jrt.so`/sherpa 自带
`libonnxruntime.so` 能否共存、dlopen 成功、OCR 与翻译各自能推理。

- 共存通过 → 走 B：搬官方后处理，只改两处：ORT 依赖换成我们现有的 `onnxruntime-1.27.0-classes.jar`
  桥、`ALL_OPT` 改 `BASIC_OPT`；正确性最高且能复用官方 benchmark 对拍。
- 共存失败（`libc++_shared` 版本冲突或符号污染）→ 退到 A：把 `findContours/minAreaRect/warpPerspective/
  fillPoly+mean/threshold/resize` 这 6 个原语用 Kotlin 实现，其余（unclip/几何/排序/CTC/YAML）照搬官方，
  并用"同一张 det 概率图 → 同一组框"的逐像素/逐顶点 PC 对拍锁死。
- C（完全自研）不作为首选：撇号实验已经说明，精度瓶颈在模型与后处理细节，不在我们能自由发挥的地方。

## 7. 修订后的实施顺序（对应你的第 1 条）

阶段 0（本目录，已完成印刷+手写两层实测与 ORT 兼容验证）→
阶段 1（共存冒烟 + 后处理路径定案 + Kotlin 侧 rec-only 作为**推理通路验证任务**，不对外当功能）→
阶段 2（完整 Det+Rec 落地：相册导入 → 检测 → 裁剪识别 → 校对 → 现有 `TranslationService` 翻译 → 译文显示，
这才是第一版功能）→ 阶段 3（`TextArtifact` JSON 与四角坐标/原图尺寸/EXIF 方向落盘）→
阶段 4（Markdown 仅导出、文字转图片）。

## 8. 本目录文件

| 文件 | 作用 |
|---|---|
| `ort_compat_check.py` | ORT 版本 × 优化级别的输出一致性与耗时（`--compare` 比对两份 JSON） |
| `run_pc_oracle.py` | 官方 PaddleOCR 产线（`engine=onnxruntime` + 我们下载的 ONNX）产出 golden JSON |
| `select_samples.py` | 分层样本构建（真实手写照片 + 合成印刷体，含 GT 与撇号统计） |
| `eval_ocr.py` | CER / 词级 recall·precision / 行内匹配 / 撇号混淆矩阵 / 失败样例 |
| `apostrophe_chain_test.py` | 撇号在分词与翻译模型上的全链路影响 |
| `model_assets.json` | 模型来源 URL、HTTP 状态、字节数、sha256 |
| `samples/manifest.json` | 样本清单（tier/origin/GT/GT 行/哈希） |
| `out/golden_*.json`, `out/eval_*.json` | golden 与评测结果 |

`models/`、`.cache/`、`.tools/`、`samples/raw/`、`out/npy_*/` 已在根 `.gitignore` 中排除。

## 9. 尚未执行 / 需要补的（不写成结论）

- Android 真机：dlopen 共存、模型加载耗时、PSS 峰值、端到端延迟（**未执行**）。
- Kotlin 后处理与 PC golden 的逐字段对拍（**未执行**，依赖阶段 1 的路径定案）。
- 真实拍摄的印刷文档/路牌/屏幕照片 20–30 张（**缺样本**，需要你用真机提供）。
- 手写页的行级人工标注（当前手写层指标只能作方向性参考）。
- ≥100 张分层集的抽样方案与标注规范（下一步单独出文档）。
- 俄语西里尔 OCR：`cyrillic/eslav_PP-OCRv5_mobile_rec_onnx` 的存在与许可证已核实，
  但**未下载、未实测**；且按你的要求不做 Unicode 自动判语种，改由 App 源语言选择驱动模型切换。
