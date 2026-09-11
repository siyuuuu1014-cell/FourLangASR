LID 选型与 TFLite 转换
======================

目标
----
从四语种测试集（zh/en/ru/uz 各 20 条）选出一个
四向（zh/en/ru/uz）准确率最高的整句语种识别方案，
并转换成与 sherpa ONNX Runtime 隔离的 TFLite 运行时
（避免与 App 内 sherpa 的 libonnxruntime.so 冲突）。

选型结果（80 条测试音频）
------------------------
| 方案 | 四向准确率 | 结论 |
|---|---|---|
| VoxLingua107 ECAPA（采用） | 79/80 = 98.75% | 唯一达标 |
| Whisper base（淘汰） | 59/80 = 73.75%，uz 0% | 不达标 |
| Whisper tiny（淘汰） | 58/80 = 72.5%，uz 0% | 不达标 |

淘汰脚本与结果 CSV 已从仓库移除（git 历史可查），
仅保留胜出方案与其 TFLite 转换。

TFLite 转换（run_lid_tflite.py）
-------------------------------
- 在 PC 上对 VoxLingua ONNX 与 TFLite 后端做同 80 条对比：
  TFLite 与 ONNX 预测一致率 100%，准确率同为 98.75%，
  5 秒静态输入 RTF ~0.16（PC）。
- 转换产物 converted_tf/（不入库）最终以
  app/src/main/assets/voxlingua107_float32.tflite 打包进 App，
  由 lid/LidEngine.kt 加载；四个目标类索引见 lid/LidTarget.kt。

脚本与数据
----------
run_lid_voxlingua.py    VoxLingua 四向基准（产物 lid_voxlingua_*.csv）
run_lid_tflite.py       TFLite 后端保真度对比（产物 lid_tflite_*.csv）
测试音频：../speaker/speaker_data（zh/en/ru/uz 各 20 条）
模型目录：models/（voxlingua107 等，不入库，可重新下载）
