Natural Code-switch EXP4 - ASCEND subset
===========================================

用途
----
自动从公开 ASCEND 中筛出一小批真实中英 code-switch 录音：

正样本：
  中文为主 + 内部夹 1~3 个英文词

负样本：
  纯中文
  纯英文

默认数量：
  positive_zh_en = 30
  negative_zh    = 15
  negative_en    = 15

为什么适合当前项目
----------------
我们当前只想解决：

  MAIN -> SHORT FOREIGN -> MAIN

比如：
  “嗯 hello 我的名字叫……”
  “我的 hobby 是读书”

不做：
  前半句中文、后半句英文的大块语言切换。

ASCEND 本身提供：
  audio
  transcription
  duration
  language
  speaker/session/topic

并且音频采样率为 16 kHz。

下载策略
--------
脚本使用：
  load_dataset(..., streaming=True)

不会主动把整个约 1.2 GB 数据集完整下载下来。

它按：
  validation -> test -> train

依次扫描，数量够了就停止。

另外：
  Audio(decode=False)

所以不会触发之前遇到的 torchcodec 音频解码错误。
音频由 soundfile 自己写成 PCM16 WAV。

环境
----
建议直接使用之前装过 datasets 的 sherpa 环境：

& "D:\dev\projects\fourlang_translation\asr_exp1\.venv_sherpa\Scripts\Activate.ps1"

需要：
  datasets
  numpy
  soundfile

检查：

python -c "import datasets,numpy,soundfile; print('OK')"

运行
----
新建/使用：

D:\dev\projects\FourLangAsrMobile\experiments\codeswitch_natural_exp4

将 prepare_ascend_natural_exp4.py 放进去。

然后：

cd "D:\dev\projects\FourLangAsrMobile\experiments\codeswitch_natural_exp4"

python .\prepare_ascend_natural_exp4.py

默认使用：
  https://hf-mirror.com

如果镜像有问题：

python .\prepare_ascend_natural_exp4.py `
  --endpoint https://huggingface.co

输出
----
audio\
  positive_zh_en\
  negative_zh\
  negative_en\

ascend_natural_manifest.csv

筛选规则
--------
正样本：
- ASCEND language == mixed
- 1.5~7.5 秒
- 至少 4 个中文字
- 只夹 1~3 个英文词
- 英文片段前后都必须有中文
- 英文片段内部不能再次出现中文
- 中文仍是主语言

因此会排除：
- 全英文
- 英文只出现在句首/句尾
- 一句话里多次大范围中英来回切换
- 很长英文短语

下一步
------
先人工快速看 ascend_natural_manifest.csv 的 transcription，
确认 30 条正样本确实符合我们要的产品场景。

确认后再做：
Natural EXP4 detector benchmark
  Sandwich detector
  Top-1 / Top-2 language
  foreign-region ASR candidate verification
