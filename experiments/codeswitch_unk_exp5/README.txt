Code-switch UNK-Trigger EXP5 - ASR 不确定度触发外语候选验证
============================================================

背景
----
Natural EXP4 shortwin 的结论（见 ../codeswitch_natural_exp4/README_shortwin.txt）：
VoxLingua 短窗口做词级外语检测不可行：
  最优配置 detected 86.7% 但 falseIsland 83%（目标 <=5%）。

因此本实验改走新路线：

  整句主语言仍由 VoxLingua LID 决定（不变）
  ↓
  主语言 ASR（ZH 流式 CTC）跑整句
  ↓
  输出中的 <unk> token（以及异常区域）作为触发信号
  ↓
  以 <unk> 时间戳为中心裁出局部片段（± pad 秒，相邻合并）
  ↓
  外语候选 ASR（EN GigaSpeech transducer）识别该片段
  ↓
  命中真实英文词 => 检出 code-switch，文本回填

直觉依据
--------
中文 CTC 词表只有 1000 个 byte-BPE token，英文单词在中文模型里
大概率落到 <unk> 或输出被吞。如果这个信号足够强，
我们就不需要任何额外检测模型——主语言 ASR 本身就是检测器。

数据
----
复用 natural EXP4 的 ASCEND 子集（不复制音频）：
  manifest: ../codeswitch_natural_exp4/ascend_natural_manifest.csv
  audio   : ../codeswitch_natural_exp4/audio/
    positive_zh_en 30 条（中文为主夹 1~3 个英文词，含英文词真值）
    negative_zh    15 条（纯中文，测误触发）
    negative_en    15 条（纯英文，诊断中文模型对英语的 unk 行为）

模型（与安卓 App 完全一致，PC 目录直接引用）
---------------------------------------------
ZH: sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01
    （App: models/zh/model.int8.onnx，流式 CTC，greedy，尾部补 0.66s 静音）
EN: sherpa-onnx-zipformer-gigaspeech-2023-12-12 (int8)
    （App: models/en/*.int8.onnx，离线 transducer，greedy）

指标定义
--------
detectedRate   : 正样本中，EN 验证输出命中 >=1 个真值英文词的比例
wordRecovery   : 真值英文词被 EN 验证找回的比例（词级）
falseIslandRate: 负样本(纯中文)中出现 >=1 个触发区域的比例
enWordFalseRate: 纯中文触发区域上 EN 输出了真实英文词的比例（更严格的误报）

通过标准（与 shortwin 相同）
---------------------------
detectedRate  >= 70~80%
falseIslandRate <= 5%

如果 ZH CTC 根本不吐 <unk>
---------------------------
那么"unk 触发"路线不成立，结果 CSV 里的 unk_count/zh_text
可以用来分析替代触发信号（空输出、吞词、置信度），
那是 EXP6 的事，本实验如实报告即可。

结果（60 条全量，zh/zh/en = 30/15/15）
--------------------------------------
[Stage 1] ZH CTC 触发信号
  positive_zh_en: 25/30 至少 1 个 <unk>
  negative_zh    :  0/15 <unk>            <- 纯中文零触发
  negative_en    : 14/15 <unk>            <- 纯英文几乎全 unk（诊断用）

[Stage 2] EN 验证找回（pad sweep）
  pad=0.30: detected 26.7%  wordRecovery 18.2%  falseIsland 0%
  pad=0.45: detected 60.0%  wordRecovery 45.5%  falseIsland 0%
  pad=0.60: detected 60.0%  wordRecovery 47.7%  falseIsland 0%
（判定为双向子串包含：EN "SMARTPHONE" 命中 gt {smart, phone}；
 单字母 gt 如缩写 C/R/P 只做精确 token 匹配。每 region 解码
 整窗 + 左/右半窗，任一命中即检出。）

关键结论
--------
1. unk 触发信号成立且极其干净：纯中文 0/15 误触发
   （对比 shortwin 最优配置 falseIsland 83%），
   25/30 正样本英文词会落成 <unk>。
2. 检出率 60% 未达 70% 门槛，miss 分两类（0.60 pad 下 12 条）：
   a) 有 <unk> 但 EN 验证失败（约 8 条）：区域含大量中文上下文，
      EN GigaSpeech 解码成乱英文/中文拟音。例：positive -> "AH STATE"、
      friends -> "CONFRONT FAMILY"、app -> "GO OUT ... MOBILE"。
   b) 无 <unk> 吞词（约 4 条）：CTC 把英文词硬拼成中文音节
      （topic -> "更疼的他配"、baseline -> "斯兰"），没有触发信号。
3. 产品含义：本路线"漏报不误报"——漏掉时英文词在中文文本里丢失，
   但不会像 shortwin 那样在纯中文句上幻觉外语/翻译。
   对"漏报 > 幻觉"的应用，60% 检出 + 0% 误报可能比
   86% 检出 + 83% 误报更可用。

下一步（EXP6）
-------------
a) 触发增强：吞词无 <unk> 的案例改用 token 级置信度
   （ZH CTC 的 ys_probs/上下文概率）找低置信中文音节区，同样送 EN 验证。
b) 验证增强：EN 输出与真值做编辑距离/词形模糊匹配；
   或对 region 内更细滑窗（当前整窗+半窗对孤词仍不稳定）。
c) 若 a+b 后仍 <70%，考虑把本路线与产品需求对齐：
   只检测"高频产品词表"（如 smart phone / app / postcard 已可检出
   这类 >=2 音节词），接受单音节/吞词漏报。

运行
----
环境（已有 sherpa venv）：

  & "D:\dev\projects\fourlang_translation\asr_exp1\.venv_sherpa\Scripts\Activate.ps1"
  cd D:\dev\projects\FourLangAsrMobile\experiments\codeswitch_unk_exp5
  python .\run_codeswitch_unk_exp5.py

可选参数：
  --limit N        只跑前 N 条（冒烟测试）
  --pads 0.3,0.45  自定义 pad sweep（默认 0.30/0.45/0.60 秒）

ZH 整句识别只跑一次并缓存，pad 变化只影响区域裁剪与 EN 验证。

输出
----
unk_exp5_utterances.csv   每条音频：zh 文本、unk 数、区域数、真值/找回词
unk_exp5_regions.csv      每个触发区域：时间范围、EN 验证文本
unk_exp5_summary.csv      每个 pad 配置的汇总指标

最后打印 RECOMMENDED CONFIG。
