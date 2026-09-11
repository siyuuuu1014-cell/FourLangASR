Code-switch 检测 - 主语言 ASR <unk> 触发路线
=============================================

背景
----
见 ../codeswitch_ascend/：VoxLingua 短窗口做词级检测不可行
（detected 86.7% 但 falseIsland 83%）。

本目录验证新路线：

  整句主语言由 LID 决定
  ↓
  主语言 ASR（ZH 流式 CTC）跑整句
  ↓
  <unk> token 时间戳作为外语触发信号
  ↓
  裁局部片段交给 EN（GigaSpeech）验证
  ↓
  命中英文实词则回填进中文文本

结果（ASCEND 30/15/15）
-----------------------
ZH CTC 触发信号：
  positive_zh_en 25/30 出现 <unk>；negative_zh 0/15（纯中文零误触发）

EN 验证回填（pad 0.6，run 分组，双向子串匹配 + 半窗多候选）：
  clip strict 12/25，relax 18/25（严格词级 48%，近似级 72%）
  falseIsland 0%

后续探索结论（diag_*.py 与 README 演化记录）：
- token 置信度触发不可行：流式 CTC greedy 无 ys_probs，
  modified_beam_search 不支持流式 CTC
- token-span 区域（unk 到下一 token）完全失败（0/25）：上下文太少
- EN GigaSpeech 对纯中文 15/15 幻觉英文实词 -> EN 文本内容
  不能做真假判定，精度完全依赖 ZH <unk> 触发（0 误触发）
- Whisper tiny/base 作验证器不优于 GigaSpeech（且越大越差）

产品形态与集成
--------------
"ZH <unk> -> EN 回填"已集成进 App（pipeline/CodeSwitchRecovery）。
预期：约半数夹词句回填正确词、另有约两成近似词、吞词句漏报、
纯中文零干扰。真机验收后阈值微调见 App 代码注释。

脚本
----
run_unk_benchmark.py        主基准（ZH 整句一次缓存，pad 扫描 + EN 验证）
diag_backfill_strategy.py   回填候选选择策略模拟
diag_token_confidence.py    token 置信度探测（结论：不可行）
diag_token_span_regions.py  token 边界区域方案（结论：失败）
diag_whisper_verifier.py    Whisper 验证器对比（结论：不如 GigaSpeech）
