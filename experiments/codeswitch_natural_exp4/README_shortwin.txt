Natural EXP4 - Short Window Sweep
=================================

这次不要建立新实验目录。

继续放在：

D:\dev\projects\FourLangAsrMobile\experiments\codeswitch_natural_exp4

目的
----
当前 1.25 秒窗口在真实 ASCEND code-switch 上：

detected = 10%
falseIsland = 3.33%

结论：
误报低，但真实单词被 1.25 秒中文上下文淹没。

所以本轮只测试“缩短局部窗口”是否能解决。

测试窗口
--------
0.50 s / hop 0.125 s
0.75 s / hop 0.125 s
1.00 s / hop 0.250 s
1.25 s / hop 0.250 s  <- baseline

每种窗口只跑一次 TFLite，
阈值和 support sweep 全部复用缓存概率。

support
-------
support = 1
  允许单个强 foreign window
  适合非常短的单词
  但可能增加误报

support = 2
  至少两个相邻 foreign windows
  更保守

仍然要求
--------
candidate 前面存在稳定主语言
candidate 后面重新出现稳定主语言

但比上一版 sandwich 放宽很多。

目标
----
希望找到至少一种配置：

detected / Top-2 >= 70~80%
falseIsland <= 5%

如果做不到
----------
停止继续调 VoxLingua 短词检测。

下一条路线改为：

主语言 ASR
-> <unk> / 空输出 / 异常 token / 低质量局部区域
-> 触发外语候选验证
-> EN/RU/UZ 候选 ASR 识别这个短片段

也就是说：
VoxLingua 继续负责整句主语言，
不再承担“单个 foreign word detector”。

运行
----
把 run_natural_codeswitch_shortwin.py 放到：

D:\dev\projects\FourLangAsrMobile\experiments\codeswitch_natural_exp4

进入 TFLite 环境：

& "D:\dev\projects\FourLangAsrMobile\experiments\lid_exp4_tflite\.venv_convert\Scripts\Activate.ps1"

然后：

cd "D:\dev\projects\FourLangAsrMobile\experiments\codeswitch_natural_exp4"

python .\run_natural_codeswitch_shortwin.py

第一次会花一些时间，因为需要真正跑四组短窗口。

不要反复执行。
所有结果都会保存：

natural_exp4_shortwin_windows.csv
natural_exp4_shortwin_utterances.csv
natural_exp4_shortwin_summary.csv

最后把：

RECOMMENDED SHORT-WINDOW CONFIG

截图发回来即可。
