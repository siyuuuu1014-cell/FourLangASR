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

---

# 阶段 1：真机验证（OPPO PLW110 / Android 16 / arm64-v8a）

分支 `feat/ocr-phase0-validation`。范围严格限定在：native 共存、官方后处理移植与对拍、
撇号三态与单测、性能与内存。**未改动任何 ASR / 翻译 / TTS 生产逻辑**，
OCR 全部代码在 `androidTest` 源集里（生产 APK 的 dex 与 jniLibs 不受影响）。

## 设备

| 项 | 值 |
|---|---|
| 机型 | OPPO PLW110，Android 16 (SDK 36)，arm64-v8a |
| SoC / RAM / CPU | MT6899 / 15.1 GB / 8 核 |
| Java 堆上限 | 256 MB（未开 largeHeap） |

## 任务 A：native 共存 —— 已执行，通过

| 用例 | 结果 | 关键数据 |
|---|---|---|
| a1 OpenCV 加载并可用 | 通过 | findContours 1 个轮廓，minAreaRect 30×40 |
| a2 打过补丁的 ORT 桥加载 PP-OCRv6 并推理 | 通过 | 冷加载 **48ms**；det 预处理 20ms / 推理 88ms / 后处理 6ms；rec 推理 167ms；整链 **351ms**；PSS 加载后 **251.5MB** |
| a3 zh/en/uz 共用同一引擎 + 冒烟识别 | 通过 | `liveEngineCount=1`，cacheKey 只有一条 `ppocrv6_tiny_det\|ppocrv6_tiny_rec`；中文路牌 4 行、置信 0.998、命中"上海/打浦路/酒店" |
| a4a sherpa-onnx 中文 ASR（OCR 存活时） | 通过 | 见 `a4a_sherpa_zh.json` |
| a4b Vosk 俄语 ASR（自带另一份 ORT + JNA） | 通过 | 见 `a4b_vosk_ru.json` |
| a4c LiteRT LID | 通过 | 见 `a4c_lid.json` |
| a5 OCR 后翻译引擎可用性 | **未验证** | 设备上没有 734MB 翻译包，`Assume` 跳过 |
| a6 反复加载/释放内存回收 | 通过 | 5 轮：释放后 PSS 224→171→160→158→161MB，增长非单调，无泄漏 |

**结论**：OpenCV + sherpa 自带 `libonnxruntime.so` + 我们补丁的 `libort4jrt.so`/桥 +
Vosk/JNA + LiteRT 可在同一进程共存，OCR 推理与原有语音链路互不影响。

### OpenCV 依赖选型（实测踩坑，别再用 quickbird 那个 AAR）

官方 SDK 用的 `com.quickbirdstudios:opencv:4.5.3` 在这台机器上 **dlopen 失败**：
`cannot locate symbol "__sfp_handle_exceptions"`。解析 ELF 确认是该 AAR 自身缺陷——
它的 `libopencv_java4.so` 把该符号作为 `SHN_UNDEF` 导入，而它自带的 `libc++_shared.so`
只在 `.symtab` 里定义、**没有导出到 `.dynsym`**。
换成 OpenCV 官方发布在 Maven Central 的 `org.opencv:opencv:4.11.0` 后加载即通过
（arm64 `libopencv_java4.so` 21.6MB + `libc++_shared.so` 1.3MB，主 APK 本来就没有
`libc++_shared.so`，因此没有重复库冲突，构建也不需要 `pickFirst`）。

## 任务 B：官方后处理移植与 PC golden 对拍 —— 已执行，通过

移植方式：`deploy/ppocr-android/ppocr-sdk` 的 26 个 Kotlin 文件原样复制进
`app/src/androidTest/java/com/example/fourlangasr/ocr/paddle/`（保留 Apache 头），
只改两处：包名、`ORTSessionManager` 的 `ALL_OPT → BASIC_OPT`。
Kotlin 1.9.22 编译通过（只有 2 条上游自身警告），**不需要 AAR 也不需要升级 Kotlin**。

| 层 | 用例 | 结果 |
|---|---|---|
| 数值 | b1 检测概率图 vs PC | 通过：12 张图 **最大逐元素误差 1.8e-5，均值 4e-8**；`>0.3` 阈值穿越像素数最多差 1（约 3.6 万分之一） |
| 几何 | b2 四角框 IoU | 通过：**平均 IoU 0.964**，golden 行匹配率 100%（6 图 ×5 行全部配对） |
| 文本 | b2 整行 CER | 通过：**印刷层平均行 CER 0.009**，逐图 0.000–0.020 |
| 顺序 | b2 阅读顺序 | 通过：6 张图逆序对数为 0 |

按用户要求**没有把逐字节一致当门槛**：两侧都是 float32 裸数据比较，误差量级与阈值穿越差都被单独记录。

## 任务 C：三态文本与撇号 —— 已执行，通过

`ocr/text/UzbekText.kt`（main，纯新增）+ 9 个 JVM 单测全绿：
`rawText / displayText / translationInput` 三态；词内撇号族统一为 U+02BB，
词外引号不动；不变量"撇号数量只换写法不减少"由测试锁死。

过程中发现并修掉一个真实陷阱：**U+02BB 的 Unicode 类别是 Lm**，
`Char.isLetter()` 返回 true，最初用 `isLetter()` 做词内判定时撇号把自己也算成字母
（实测计数 21→24）。现在用 `isWordLetter()` 排除 Lm/Sk。

## 任务 D：性能与内存 —— 部分完成

已拿到：冷加载 48ms、det/rec 分阶段耗时、整链 351ms、加载后 PSS 251.5MB、
5 轮加载释放的 PSS 曲线（见 a2/a6）。
`OcrPerfBenchmarkTest`（warm 重载 + mean/p50/p90 聚合）**未能跑出报告**，
已提交但标记为待定位，不猜数字。

## 已复现的性能缺陷（阶段 2 必须先修）

真实照片与横线纸手写页会让 DB 后处理耗时爆炸：中文路牌照片、手写页在 400s 内都没跑完
（印刷层同尺寸图只要 0.3s）。原因是 `findContours` 在复杂背景/横线上返回海量轮廓，
而每个轮廓都要走一遍 `PolygonUnclip` 圆弧采样。
阶段 2 需要：轮廓数上限 + 按面积提前剪 + 行数上限 + 可中断，并把 `detMaxCandidates`
从 3000 降到与手机场景匹配的值。

## 未验证清单（不得当作结论）

- OCR + 翻译顺序执行的峰值内存（设备无翻译包）
- 真实街景（招牌/路牌/包装）准确率 —— 接口已备好，等实拍数据（见 `scene_eval/`）
- 手写页 OCR 延迟与准确率
- warm 重载与 p90 统计（d1 未跑出报告）
- 俄语西里尔识别模型（未下载、未实测）

---

# 阶段 2A：真实照片卡顿的定位与最小修复（OPPO PLW110 / Android 16 / arm64-v8a）

本轮目标只有一个：**查清"真实照片 DB 后处理 >400s"到底成不成立、根因在哪，并做最小修复**。
不改生产 ASR/翻译/TTS，不做 OCR 产品化迁移。所有新代码仍在 androidTest 内，可独立回滚。

## 结论先行（推翻阶段 1 的猜测）

**在现有全部资产上，DB 后处理根本复现不出 >400s。** 阶段 1 记的"中文路牌/手写页 400s 没跑完、
根因是 findContours 海量轮廓 + PolygonUnclip 圆弧采样"这个归因，被本轮分段计时**证伪**：
- 那些图（含上采样到 3200px、阈值下调到 0.03）DB 全程 ≤ 39ms，轮廓数只有 40~130，正像素比 ~2%。
- 连造的 3600 连通块合成最坏图，DB 也只要 578ms（且 minAreaRect/boxScore/unclip 合计 ~2ms，
  大头是 findContours 本身 + contour.toArray）。
- 真正会随规模线性增长的是**逐行识别（rec）**：实测 ~22~30ms/行，框数才是总耗时主因。

阶段 1 那次"400s"极可能是当时 logcat 过滤造成的**误判**（摘要里已记过一次同类误判），
叠加 d1 基准"跑不出报告"——而 d1 跑不出报告的**真正原因本轮查清了**（见任务 4 的卡死缺陷）。

## 任务 1：分段计时定位（已执行，e1~e4 + g1）

给 `DBPostProcessor.process` 加了 opt-in 探针 `DbPostProcessProfile`（默认 null，null 时逐字节走官方路径，
对拍不受影响），逐候选累加 threshold / findContours / contour→point / minAreaRect / boxScore(fillPoly+mean)
/ unclip / 放缩 的耗时，并记录二值正像素比、轮廓总数、进入各阶段的候选数、最终框数、boxScore 外接面积和。
`OCREngine` 另把"文字区域裁剪 crop"从 rec 里单独计时。报告经 logcat 分块转储后由
`harvest_logcat_reports.py` 重组（Android 16 上 adb 读不到 app 外部私有目录）。

整链分段（cap1280，ms）：解码 4~12 / det 推理 48~197 / **DB 后处理 2~15** / crop 3~30 / rec 19~541。
det 输入尺寸扫描（真实照片上采样，cap4000）：

| 长边 | 概率图 | 轮廓数 | 采纳框 | 正像素比 | DB 外推总耗时 |
|---|---|---|---|---|---|
| 1280 | 1280×960 | 40 | 4 | 1.85% | 21ms |
| 1920 | 1920×1440 | 89 | 4 | 2.06% | 13ms |
| 2560 | 2560×1920 | 118 | 3 | 2.17% | 21ms |
| 3200 | 3200×2400 | 130 | 3 | 2.20% | **39ms** |

阈值扫描（0.3→0.03）与 3600 块合成图（g1）同样证明 DB 有界。
源码定位：`maxCandidates=3000` 确如用户所说**只在 findContours 返回之后**才截断
（`DBPostProcessor.kt` 的 `numContours = minOf(contours.size, maxCandidates)`），
所以爆炸风险在候选循环，不在 findContours 调用本身——只是真实照片产生的候选根本没到那个量级。

## 任务 2：最小修复 + 安全降级（已执行，g1 验证）

只动 androidTest 内的 OCR 代码，加两道**对正常图完全惰性**的护栏：
1. **DB 工作预算** `detPostBudgetMs`（默认 2500ms）：逐候选循环超预算立即返回**已识别的部分框**，
   并置 `truncated`。g1 实测：3600 块图无预算 578ms 跑完 3000 框，设 300ms 预算则在 301ms 截断、
   返回 2046 框、`skippedByBudget=954`——证明"极复杂图也绝不无限期挂起"。
2. **识别框上限** `maxRecBoxes`（默认 120）：按面积保留最可能是正文的前 N 框再识别，
   超出丢弃并置 `degraded`。因为 rec 才是线性主因，这一条直接封顶总时长。
3. `OCREngineResult` 新增 `degraded` / `detectedBoxCount`，供调用方决定"提示用户手动裁剪"。
4. 协程取消 / 超时返回 / 原生不可中断**分开处理**：OpenCV/ORT 原生调用不可被中断，
   所以不依赖 `future.cancel`，而靠 DB 预算 + rec 上限这两个**内部**有界点保证返回；
   test 外层再套硬截止只为不挂死测试，不作为生产保证。
未提高全局阈值、未改轮廓模式、未删小面积文字——正常图结果与官方逐字节一致（见任务 3）。

## 任务 3：精度对拍 + 性能分布（已执行，b2 + f2）

- **Golden 对拍（b2，printed 层）与阶段 1 完全一致**：meanIoU 0.964、行 CER 0.009、
  整行命中 1.0、阅读顺序单调 true。护栏对正常图零影响。
- 撇号：JVM 单测 `testDebugUnitTest` 全绿（9 项，含"撇号数只统一不删除"不变量）。
- **耗时分布（f2，8 张印刷 golden）**：total p50=207 / p90=244 / max=244ms；
  DB p90=9ms；rec 主导（~134ms）。`degraded` 全 false。印刷/常规自然图 p95 远低于 3s 目标。
- 复杂/极端图：受限于资产里没有真正的 pathological 原图，"复杂图 ≤10s"只能在合成图（g1）上验证
  有界性，真实 12MP 街景原图仍待用户提供后补测。

## 任务 4：内存与共存（部分通过 + 一个新发现的关键缺陷）

- **共存（a4a/a4b/a4c）重跑全绿**：OCR 引擎在场时 sherpa-zh / Vosk-ru / LiteRT-LID 均正常，
  本轮改动没影响 ASR/LID 运行时。
- **单轮内存回收（f1，8 轮重建/释放）**：native 堆释放后稳定 ~15.4MB（drift +0.2MB），
  PSS 回到 ~114MB 基线，冷加载 ~40~85ms/轮。f3（池化复用 15 次）单帧 p50=235 / p90=248 / max=287ms，
  平稳；`releaseAll` 后 native 堆 231.3→15.3MB，确实回收（OS 级 PSS 只降 ~3MB 是 malloc arena 保留，正常）。
- ⚠️ **关键缺陷（未通过，交阶段 2B）**：**单进程内反复做模型操作会卡死**——
  f1 紧循环"重建 OcrRuntime+释放"在约 8~32 次之间把进程挂死（线程停 S 态、无崩溃无 OOM）；
  f3 对同一引擎连续 `run()` 到第 ~17 次也挂死，而挂死前 PSS/帧耗时全程平稳（**不是内存泄漏**，
  是 ORT 会话 create/close 与推理累积的**原生句柄/线程/fd 未回收**，耗尽后死锁）。
  这正是阶段 1 d1 基准"跑不出报告"的根因。生产绝不能"每张照片重建一次引擎/无上限连跑"，
  必须：池化复用 + 每 N 次主动回收 + 可中断释放，或定位并修补 ORT 会话释放路径。
- **未验证**：OCR + 本地翻译顺序执行峰值内存（设备无 734MB 翻译包，不得声称已通过）。

## 本轮改动文件

- `app/src/androidTest/.../ocr/paddle/postprocess/DBPostProcessor.kt`：加 `profile` 探针 + `workBudgetMs`
  预算 + `truncatedOut`（默认参数，官方路径不变）。
- `.../postprocess/DbPostProcessProfile.kt`（新）：分段计账器。
- `.../paddle/PaddleOCRConfig.kt`：加 `detPostBudgetMs=2500`、`maxRecBoxes=120`。
- `.../engine/DetectionEngine.kt`：透传预算 + `truncated`。
- `.../engine/OCREngine.kt` / `OCREngineResult.kt`：rec 框上限 + `degraded`/`detectedBoxCount` + crop 计时。
- `.../ocr/Harness.kt`：`writeReport` 增加 logcat 分块转储（Android 16 读不到外部私有目录）。
- 新测试：`OcrDbProfileTest`(e1~e4)、`OcrDbGuardTest`(g1)、`OcrPhase2RegressionTest`(f1~f3)。
- 工具：`experiments/ocr/harvest_logcat_reports.py`、`install_gate.sh`（绕开 ColorOS adb 安装闸）。
- 报告：`experiments/ocr/out/{a_db_profile,a_db_profile_4000,a_db_threshold_sweep,a_det_size_sweep,g1_db_budget_guard,f1_memory_cycles,f2_timing_distribution,f3_pooled_reuse,b2_box_text_parity}.json`。

## 阶段 2A 未通过 / 未验证（不得当作结论）

- 30+ 轮模型重建/释放稳定性：**未通过**（约 8~17 次原生操作即死锁）。
- 真实 12MP 街景原图的卡顿复现与准确率：等用户实拍数据。
- OCR + 本地翻译峰值内存：设备无翻译包，未测。

## 阶段 2B 建议

1. **优先修死锁**：查 `ORTSessionManager` 会话 create/close 与 `session.run` 的原生资源回收
   （线程池 / mmap / OrtValue），确立"可长期存活"的引擎生命周期；生产用池化单例，不逐张重建。
2. 把这套 OCR runtime（含护栏）迁入生产 `main`，接相册导入 + 校对 UI + `TranslationService`。
3. 用用户实拍 ≥100 张分层街景补真实准确率与"复杂图 ≤10s"验收。
4. 下载并实测俄语 PP-OCRv5 Cyrillic rec，验证预留的 profile 切换。

---

# 阶段 2B-0：重复推理挂起 —— 定位与证据（OPPO PLW110 / Android 16 / arm64-v8a）

本轮只做稳定性定位，不改生产 ASR/翻译/TTS，不接 UI。结论：**挂起被定位到 OCR 自带的
ONNX Runtime 原生库 `libort4jrt.so`（ORT 1.27）内部线程池，在"反复大张量推理"后进入 futex 等待而卡死；
与 OpenCV、与 ASR/双 runtime、与内存/句柄泄漏都无关。** 这是证据支持的定位，不是猜测；
但"为什么 ORT 线程池会卡"的最后一层因设备无 root/无 NDK 调试器而无法符号化到函数名（见"取证受限"）。

## 复现矩阵（同一进程内反复操作，OcrHangIsolationTest，非 root 真机）

| 代号 | 内容 | 结果 | 线程/FD/native |
|---|---|---|---|
| A | 只反复 Det 推理（小张量 64×64） | 60 次 ✅ | 27 / 114 / 24.8MB 全平 |
| B | 只反复 Rec 推理（小张量 48×320） | 60 次 ✅ | 27 / 114 / 33.8MB 全平 |
| G | Det+Rec 交替（纯 ORT，小张量，240 次 run） | 40 轮 ✅ | 27 / 114 / 33.8MB 全平 |
| H | 只跑 OpenCV（解码/前处理/DB/裁剪，无 ORT） | 40 轮 ✅ | 28 / 114 / 15.9MB 全平 |
| **N** | **只 ORT，真实尺寸大张量(1×3×960×1280)，无 OpenCV** | **❌ ~14 轮挂** | 挂前 24 / 114 / 23.6MB 全平 |
| C | 完整管线（OpenCV+ORT 真实图） | ❌ ~18 轮挂 | 35 / 114 / 232MB(arena) 全平 |
| I | 完整管线，ORT intra-op=1 | ❌ ~8 轮挂 | — |
| L | 完整管线，OpenCV 线程=1 | ❌ ~21 轮挂 | — |
| M | 完整管线，两者都=1 | ❌ ~9 轮挂 | — |

**决定性对照**：G（小 ORT 张量）不挂、N（大 ORT 张量、无 OpenCV）挂 → 触发条件是
**ORT 处理"大到真正启用 intra-op 线程池并行"的张量、且反复多次**；OpenCV 不是必要条件
（C 只是恰好也用了大图）。限线程数（I/L/M）、关 arena、独立复制缓冲都不能规避。

## 任务 A：原生依赖 ELF 审计（experiments/ocr/elf_audit.py）

- `libort4jrt.so`(OCR ORT 1.27) 与 `libonnxruntime.so`(sherpa)：DT_NEEDED 无 libc++_shared，
  导出 C++ 符号=0 → **各自静态内嵌一份隐藏 libc++**。
- `libopencv_java4.so`：DT_NEEDED **含 libc++_shared.so** → 用共享 libc++。
- 全部共用同一 `libc.so`（同一 malloc）。
- 结论：进程里确实有两套 C++ 运行时（ORT 静态隐藏 + OpenCV 共享），**但无导出符号冲突、malloc 统一**，
  且 N 证明不加载 OpenCV 也挂 → **双 libc++/双 runtime 不是本挂起的原因**（推翻上一轮的猜测）。

## 任务 B：JNI 与数据生命周期审查

- `ORTSessionManager.runSession`：OnnxTensor 与 OrtSession.Result 均在 finally 关闭；
  输出经 `copyFloatBuffer` 拷成 Java float[] 再返回，不持有原生张量。
- OpenCV Mat（Det/Rec 前处理、DB、QuadTextCrop、BitmapUtils）正常与异常路径均 release。
- OpenCV 与 ORT 之间**不共享原生缓冲**（经 Java float[] 拷贝桥接）。
- 结合 N（无 OpenCV 也挂）：挂起**不是**对象提前/重复释放或跨库缓冲破坏。

## 任务 C：最小交叉实验

见上表 A/B/G/H/N/C/I/L/M。已把"大张量并行"与"OpenCV 交错"两个变量分离，锁定前者。

## 任务 D：独立 OCR 进程可行性

- 用 `/proc/<pid>/maps`（经 run-as，非 root 可读自身）确认：挂起进程里**只映射了
  `libort4jrt.so` + `libonnxruntime4j_jni.so`**，sherpa 的 `libonnxruntime.so`/`libvosk`/TFLite **未加载**。
- 即：N/C 已经等价于"只含 OCR 原生库的独立进程"，仍挂 → **拆独立 OCR 进程不能规避**该挂起。
- 因此不建议把"独立 OCR Service + IPC"当作本问题的解法（它绕不开 libort4jrt.so 自身的 bug）。

## 取证受限（如实记录，不推测替代）

- `debuggerd -b <pid>`：`root is required`（非 root 不可用）。
- `simpleperf record`：`cpu-cycles` 与 `cpu-clock` 均报 `not supported on the device`（ColorOS 关闭了 perf_event）。
- 本机未装 NDK → 无 `lldb-server` 可经 run-as 附加；Android Studio GUI 调试不可用。
- SIGQUIT 的 ART 线程栈写进 `/data/anr`，非 root 读不到。
- **拿到的替代证据**：经 `/proc/<pid>/task/<tid>/syscall` 采 PC + `/proc/maps` 反查，
  确认挂起时测试线程与约 10 个 ORT worker 线程全部停在 `libc.so` 的 futex 等待地址
  （PC 0x7ad1f08d9c→libc+0x56d9c），另有线程 PC 落在 `libort4jrt.so` 段内（base.apk pgoff 0x21487000）。
  即"卡在 ORT 原生线程池的 futex 等待"，但**函数名级别**的栈需要 root/NDK 才能给出。

## 候选根因（按可能性，未定论）

1. **`libort4jrt.so` 是被 patchelf 改名(SONAME libonnxruntime.so→libort4jrt.so)的 ORT 1.27**，
   改名/重定位破坏了其线程池或 `dlopen` 自检，反复大推理后触发等待竞态。**最可疑**。
2. ORT 1.27 本身在该机型/该模型上的 intra-op 线程池 bug。
3. 与 `OrtEnvironment.getEnvironment()` 单例的全局状态累积有关（D 每轮新建 session 仍挂，env 未换）。

## 推荐修复路线（需你拍板，涉及原生库的要先做兼容评估）

- 首选：**做一份 ORT 运行时兼容性评估**（对应你 E 条）——比较 (a) 用未改名的官方 ORT Java AAR 跑 OCR、
  (b) 让 OCR 复用 sherpa 的 `libonnxruntime.so`（去掉双 runtime hack）、(c) 升级/降级 ORT 版本。
  评估含 ABI、minSdk、包体、与 ASR/翻译共存、Golden 对拍。**在你批准前不替换任何生产原生库。**
- 在根因（尤其候选 1 的改名问题）用 NDK 调试器/root 设备拿到函数级栈确认前，**不建议进入正式 OCR 集成**。

## 2B-0 交付物

- 复现矩阵、ELF 审计、maps/PC 取证：`experiments/ocr/out/{hang_*.log,hung_maps.txt}`、`elf_audit.py`。
- 隔离测试：`OcrHangIsolationTest`(A/B/G/H/N/C/I/L/M/D) + `HangProbe`（线程/FD/内存计数 + 反射 dump）。
- Golden 回归：b2 在诊断开关复位后仍通过（IoU 0.964 / CER 0.009 / 顺序单调）。
- 结论：**未找到不改原生库即可稳定跑 >~14 次大推理的方案**；根因锁定在 libort4jrt.so 内部，
  需运行时兼容性评估 + 函数级栈取证后再修。

## 2B-0 方案(a) 追加：setup_dual_ort.py 的 ELF 改动审计（已执行）

工具：`experiments/ocr/ort_patch_diff.py`（pyelftools 逐字节 diff + section 定位）。
对照：官方 `onnxruntime-android:1.27.0` AAR 解出的 `libonnxruntime.so` / `libonnxruntime4j_jni.so`
vs 仓库里已打补丁的 `libort4jrt.so` / 改过 NEEDED 的桥。

- 官方 lib 里字符串 `libonnxruntime.so` 只出现 **1 次**，位于 `.dynstr` 的 SONAME（0x26a4）。
- 官方桥里字符串也只出现 **1 次**，是 DT_NEEDED（0x4d3b）。
- 两个文件被 `bytes.replace` 改动**各 13 字节、均在 .dynstr**，其余字节完全一致（size 相同）。
- 改后动态段正确：lib DT_SONAME=`libort4jrt.so`；bridge DT_NEEDED=`libort4jrt.so`。

**结论**：`patch_all()` 虽是对整文件 `bytes.replace`，但在当前这两个文件上每个只命中一处，
实际是外科式修改，**没有误伤其它字符串/数据**。因此"补丁破坏了二进制导致挂起"不成立。
（该写法仍脆弱：一旦上游哪天在 .rodata 里也出现同名字符串就会被误改。建议后续换成只改
DT_SONAME/DT_NEEDED 的 ELF 工具，但这属于健壮性改进，不是本 bug 的因。）

**兼容性发现**：官方 `com.microsoft.onnxruntime:onnxruntime-android:1.27.0` 的 manifest 要求
**minSdk 24**，而主 App 是 **minSdk 23**。这解释了为何项目当初没有直接引 AAR、而是抽
`classes.jar` + 自改 `.so` 接线；方案(a) 若要落到生产，会撞上 minSdk 抬到 24 的约束（需单独评估）。

## 2B-0 方案(a) 待补：官方原版 vs 修改版 200 次对照（未完成）

已建隔离模块 `ortcmp`（仅依赖官方 AAR，不含 sherpa/opencv/vosk/tflite/翻译），
压力测试 `OrtOfficialStressTest`（≥200 次大张量推理 + 反复建/释放 session + 线程/FD/PSS/native/耗时）
已就绪；目标 APK 已装真机，测试 APK 安装过程中设备 USB 掉线，**对照尚未跑出结果**。
设备恢复后按用户分支逻辑判定：官方也挂→进方案(c)；官方不挂→查补丁/加载链路/JNI 桥；两者都稳→复核复现条件。

## 2B-0 方案(a) 对照结果（已执行，真机 OPPO PLW110 / Android 16）

隔离模块 `ortcmp`（只含官方 onnxruntime-android:1.27.0，无 sherpa/opencv/vosk/tflite），
模型经 Gradle `syncOrtcmpModels`+`verifyOrtcmpModels` 校验与主 App 同源同 sha256；
输入、SessionOptions(BASIC,intra-op=4)、每轮 1 det[1,3,960,1280] + 5 rec[1,3,48,800] 完全一致。

| 构建 | 测试 | 结果 | 挂起前资源 |
|---|---|---|---|
| 修改版 libort4jrt.so | 复用 session 持续推理 | **挂起 @iter 15**（完成 0–14） | native ~372MB(arena) / threads 25→22 / fd 106 平 |
| 官方官方 AAR 1.27.0 | 复用 session 持续推理 | **挂起 @iter 13**（完成 0–12） | native ~372MB / threads 25→22 / fd 106 平 |
| 官方官方 AAR 1.27.0 | 反复建/销 session | **挂起 @iter 11**（完成 0–10） | native ~13.8MB / threads 16 / fd 106 平 |

**结论（证据支持，非预设）**：官方未改版与修改版**都挂**，且挂在同一迭代量级、资源曲线一致 →
- 不是双 ORT 改名/补丁的问题（官方原版一样挂）；
- 不是 OpenCV / sherpa 共存问题（ortcmp 不加载它们）；
- 不是进程隔离问题（已是最小 OCR-only 进程）；
- 挂起前线程/FD/native/PSS **无持续增长**（不是内存/FD/线程泄漏，是 ORT 内部线程池在反复大张量推理后停滞）。
→ 指向 **ORT 1.27.0 自身的线程池/会话在该模型+大张量负载下的问题**（与具体 ORT 版本/构建配置相关）。

**验收现状（如实）**：
- 官方与修改版**都有真机结果**；都**未跑满 200 轮**（分别卡在 ~13/~11），这不是测试没写好，而是被 ORT 挂起本身阻断——200 轮验收在本轮**未达成**。
- ORT 版本对照（方案 c）尚未做：需再引其它官方 ORT 版本 AAR 到 `ortcmp` 复测。
- ASR/Vosk/LID 共存回归（a4a/b/c）通过；翻译泄漏回归用例已写，但本机无 734MB 翻译包，**假定跳过、未验证**。

**下一步（方案 c）**：在 `ortcmp` 里依次替换官方 ORT 版本（如 1.22 / 1.23 / 1.26 等）与构建配置
（intra-op=1、disable spinning、不同 arena），用同一 200 轮脚本定位"哪个版本/配置不挂"，再决定生产路线。
