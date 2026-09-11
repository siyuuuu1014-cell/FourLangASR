Code-switch 基准 - ASCEND 自然中英夹杂句
==========================================

背景
----
产品场景：中文为主、句中夹 1~3 个英文词的句子
（"嗯 hello 我的名字叫……"、"我的 hobby 是读书"）。
不做大段语言切换。

数据
----
从公开 ASCEND 数据集流式筛选（prepare_ascend.py）：

  正样本 positive_zh_en  30 条：中文为主 + 1~3 个英文词
  负样本 negative_zh      15 条：纯中文
  负样本 negative_en      15 条：纯英文

筛选规则：ASCEND language==mixed、1.5~7.5 秒、至少 4 个中文字、
英文词 1~3 个且前后都有中文。产出 ascend_natural_manifest.csv +
audio/ 目录（音频不入库，脚本可重建）。

基准脚本
--------
run_ascend_benchmark.py
  整句 LID（VoxLingua TFLite）+ 三明治式外语窗口检测的基线结果。
run_shortwin_sweep.py
  0.5/0.75/1.0/1.25 秒短窗口 × 阈值/support 扫描。
  产物：shortwin_summary.csv / shortwin_utterances.csv / shortwin_windows.csv

结论
----
VoxLingua 短窗口做"词级外语检测"不可行：

  最优配置 detected 86.7% 但 falseIsland 83%（目标 <=5%）

因此词级检测改用主语言 ASR 的 <unk> 触发路线，
见 ../codeswitch_unk/。

运行
----
使用安装了 sherpa-onnx + soundfile 的 Python 环境：

  python .\prepare_ascend.py
  python .\run_ascend_benchmark.py
  python .\run_shortwin_sweep.py
