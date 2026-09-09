Code-switch Island EXP2
=========================

目标
----
在 EXP1 基础上解决两个问题：

1. 不再只看 hard top-1 标签
   改成 foreign probability - main probability 的 evidence。

2. 加纯单语 negative controls
   真正测“整句 False Island Rate”。

3. 不再整句 0.25s dense scan
   改成 coarse -> refine 两阶段扫描，降低手机计算量。

数据
----
正样本：
复用 EXP1 的 24 条：
main + [foreign 0.8/1.2s] + main

负样本：
新增 20 条纯单语：
ZH 5
EN 5
RU 5
UZ 5

检测
----
Stage A coarse:
  window 1.25s
  hop 0.75s

Stage B refine:
  只在 coarse 发现 foreign evidence 的候选附近
  window 1.25s
  hop 0.25s
  radius ±0.75s

final:
  至少 2 个相邻 fine windows 支持同一个 foreign language

阈值
----
foreign_prob / evidence：

0.40 / 0.05
0.45 / 0.10
0.50 / 0.15
0.55 / 0.20
0.60 / 0.25

其中：
  foreign_prob = 当前最强的非主语言概率
  evidence     = P(foreign) - P(main)

关键指标
--------
correct_foreign_rate
  正样本中，小岛找到且语言正确

wrong_foreign_rate
  找到，但判断成错误的第三语言

miss_rate
  小岛完全没找到

false_island_rate
  纯单语整句话被误报成“含外语小岛”的比例
  这是 EXP2 最重要的新指标

center_mae_sec
  外语小岛位置误差

triggered_sample_rate
  有多少句进入了 refine

avg_fine_windows_per_sample
  平均每句话额外跑多少个 fine LID window
  用来估算手机成本

建议通过线
----------
correct_foreign_rate >= 80%
wrong_foreign_rate <= 5%
false_island_rate <= 5%
center_mae_sec <= 0.50s

同时希望：
avg_fine_windows_per_sample 明显低于 EXP1 的 dense scan。

运行
----
放到：

D:\dev\projects\FourLangAsrMobile\experiments\codeswitch_island_exp2

进入环境：

& "D:\dev\projects\FourLangAsrMobile\experiments\lid_exp4_tflite\.venv_convert\Scripts\Activate.ps1"

然后：

cd D:\dev\projects\FourLangAsrMobile\experiments\codeswitch_island_exp2

python .\prepare_codeswitch_island_exp2.py

python .\run_codeswitch_island_exp2.py

输出
----
negative_manifest.csv
island_exp2_window_results.csv
island_exp2_utterance_results.csv
island_exp2_summary.csv

如果 EXP2 通过：
下一步不再继续合成数据。
直接录自然 code-switch 句子：

我今天有一个 meeting
这个 file 发给我
今天去 airport 接人
Сегодня у нас meeting
Men bugun meeting ga boraman

然后做 Natural Code-switch EXP3。
