Natural Code-switch EXP4 Benchmark
======================================

继续使用同一个目录：

D:\dev\projects\FourLangAsrMobile\experiments\codeswitch_natural_exp4

不要再创建新的 EXP5/EXP6 目录。

前提
----
先完成：

prepare_ascend_natural_exp4.py

并确认存在：

ascend_natural_manifest.csv
audio\
  positive_zh_en\
  negative_zh\
  negative_en\

运行环境
--------
使用已经跑通 VoxLingua TFLite 的环境：

& "D:\dev\projects\FourLangAsrMobile\experiments\lid_exp4_tflite\.venv_convert\Scripts\Activate.ps1"

然后：

cd "D:\dev\projects\FourLangAsrMobile\experiments\codeswitch_natural_exp4"

python .\run_natural_codeswitch_exp4.py

默认使用 4 个 TFLite CPU threads。

快速调试前 5 条：

python .\run_natural_codeswitch_exp4.py --limit 5

正式实验一定不要加 --limit。

算法
----
每条真实录音：

1. 主语言已知
   positive_zh_en -> ZH
   negative_zh    -> ZH
   negative_en    -> EN

2. 1.25 s window
3. 0.25 s hop
4. 每个局部窗对称补零到模型固定 5 秒输入
5. VoxLingua 4-way ZH/EN/RU/UZ 概率
6. MAIN -> FOREIGN -> MAIN sandwich detector
7. 所有阈值 sweep 复用同一轮窗口概率，不重复跑模型

为什么这次没有 MAE / IoU
-----------------------
ASCEND 当前筛出的 transcript 没有 word-level 时间戳。

因此我们知道：
  这句话里有 English 小岛

但不知道：
  English 精确从第几秒到第几秒

所以 Natural EXP4 不伪造时间真值，不计算：
  center MAE
  IoU

真正评估：

detected_rate
  正样本中有没有检测到外语小岛

foreign_top1_rate
  外语候选 Top-1 是否为 EN

foreign_top2_rate
  EN 是否至少进入候选 Top-2

wrong_top1_rate
  Top-1 被错误判断为 RU/UZ 等的比例

false_island_rate
  纯中文和纯英文整句话被错误判断含外语小岛的比例

建议目标
--------
false_island_rate <= 5%

foreign_top2_rate >= 80~85%

Top-1 暂时只观察。

原因：
短到 1~2 个词时，LID 很难做到和整句一样稳定。
如果真实语言能进入 Top-2，下一步就可以：

foreign island
-> LID Top-2
-> 两个候选 ASR 只识别这个小片段
-> 根据 ASR 输出质量选择最终词

输出
----
natural_exp4_utterance_results.csv
natural_exp4_summary.csv

下一步
------
如果 Natural EXP4 达标，再做最后一个 PC 验证：

Foreign-region ASR candidate verification

这一步才真正测试：
meeting / file / project 等外语词能不能被第二 ASR 恢复。

之后再决定是否接 Android。
