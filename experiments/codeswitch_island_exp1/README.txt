Code-switch Island EXP1
=========================

目标
----
只解决：
“主语言一句话里偶尔冒出 1~2 个其他语言词”。

不是解决：
“前半句中文，后半句俄语”的大块语言切换。

受控样本
--------
主语言 2.0s
+ 外语小岛 0.8s 或 1.2s
+ 主语言 2.0s

4 个主语言 × 3 个外语方向 × 2 个岛长度
= 24 条受控样本。

例如：
ZH + [EN 0.8s] + ZH
ZH + [RU 1.2s] + ZH
EN + [ZH 0.8s] + EN
...

注意：
这些是“声学语言小岛”受控样本，
外语片段是从真实外语句子中心裁出来的，
不保证刚好是完整单词。

所以 EXP1 只回答：
“LID 能不能发现一个很短的异语言区域？”

真正的 meeting / airport / file 等自然 code-switch
放到 EXP2 录音测试。

检测方法
--------
先假设主语言已经知道：

AUTO:
整句 LID -> 主语言

MANUAL:
用户选择 -> 主语言

然后：

一句话
-> 1.0s / 1.25s 滑窗
-> hop 0.25s
-> 每个局部窗 zero-pad 到 TFLite 固定 5s 输入
-> VoxLingua 4-way
-> 找连续 >= 2 个“非主语言”窗口
-> foreign island candidate

阈值 sweep
----------
conf / margin:

0.45 / 0.10
0.55 / 0.15
0.65 / 0.20
0.75 / 0.25

核心指标
--------
correct_foreign_lang_rate
  是否找到了外语小岛，并且语言种类也正确

wrong_foreign_lang_rate
  找到了小岛，但判断成了错误的第三种语言

false_alarm_window_rate
  真正纯主语言区域里，误报成其他语言窗口的比例

center_mae_sec
  预测小岛中心和真实中心相差多少秒

mean_iou
  预测小岛时间段和真实小岛的重合度

第一轮建议通过线
----------------
correct_foreign_lang_rate >= 80%
wrong_foreign_lang_rate <= 5%
false_alarm_window_rate <= 5%
center_mae_sec <= 0.50s

这里先定 80%，不是 90%，
因为 0.8~1.2 秒本来就是更难的短语言识别任务。

如果 EXP1 通过：
EXP2 直接录自然句子：

“我今天有一个 meeting”
“这个 file 发给我”
“今天去 airport 接人”
“Сегодня у нас meeting”
“Men bugun meeting ga boraman”

然后再决定：
1. 只标记 foreign word
2. 还是把这小段送第二个 ASR 补识别

运行
----
把两个 .py 文件放到：

D:\dev\projects\FourLangAsrMobile\experiments\codeswitch_island_exp1

进入已经可用的 TFLite 环境：

& "D:\dev\projects\FourLangAsrMobile\experiments\lid_exp4_tflite\.venv_convert\Scripts\Activate.ps1"

然后：

cd D:\dev\projects\FourLangAsrMobile\experiments\codeswitch_island_exp1

python .\prepare_codeswitch_island_exp1.py

python .\run_codeswitch_island_exp1.py

结果
----
island_manifest.csv
island_exp1_window_results.csv
island_exp1_utterance_results.csv
island_exp1_summary.csv
