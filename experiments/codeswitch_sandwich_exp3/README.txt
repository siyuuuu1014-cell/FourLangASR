Code-switch Sandwich EXP3
===========================

目标
----
把 EXP2 的：

false island = 25%
wrong        = 8.33%

压下来。

这轮不重新跑 TFLite。
直接复用 EXP2 已经计算好的概率：

codeswitch_island_exp2\island_exp2_window_results.csv

所以运行会比 EXP2 快很多。

核心思想
--------
目标场景限定为：

MAIN -> FOREIGN -> MAIN

例如：

ZH ZH ZH | EN EN | ZH ZH ZH

只有满足“三明治结构”才接受外语小岛。

规则
----
1. 外语小岛至少连续 2 个 fine windows。
2. 外语小岛必须在句子内部。
3. 小岛必须较短：
   relaxed  <= 1.8s
   balanced <= 1.7s
   strict   <= 1.6s
4. 小岛前必须有稳定主语言。
5. 小岛后必须重新回到稳定主语言。
6. 同时出现两个不同稳定外语岛时拒绝，避免乱猜。

主语言约束
----------
relaxed:
  main_prob >= 0.45
  main advantage >= 0.05

balanced:
  main_prob >= 0.50
  main advantage >= 0.10

strict:
  main_prob >= 0.55
  main advantage >= 0.15

外语阈值
--------
Pforeign / evidence:

0.45 / 0.10
0.50 / 0.15
0.55 / 0.20
0.60 / 0.25

运行
----
新建目录：

D:\dev\projects\FourLangAsrMobile\experiments\codeswitch_sandwich_exp3

把 run_codeswitch_sandwich_exp3.py 放进去。

然后：

cd D:\dev\projects\FourLangAsrMobile\experiments\codeswitch_sandwich_exp3

python .\run_codeswitch_sandwich_exp3.py

如果 base 环境没有 numpy，
继续用 .venv_convert 运行即可。

输入依赖
--------
暂时不能删除：

codeswitch_island_exp1
  -> island_manifest.csv

codeswitch_island_exp2
  -> negative_manifest.csv
  -> island_exp2_window_results.csv

输出
----
sandwich_exp3_utterance_results.csv
sandwich_exp3_summary.csv

重点
----
看 RECOMMENDED CONFIG：

目标：
correct >= 75~80%
wrong <= 5%
falseIsland <= 5%
center MAE <= 0.50s

如果 falseIsland 能从 25% 压到 <=5%，
即使 correct 暂时保持 75%左右，
也说明 sandwich 规则很有价值。

如果 EXP3 通过
-------------
下一步不再继续合成实验，
直接进入自然 code-switch 录音：

我今天有一个 meeting
这个 file 发给我
今天去 airport 接人
Сегодня у нас meeting
Men bugun meeting ga boraman
