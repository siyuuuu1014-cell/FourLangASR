m2m100 四语 ONNX 参考解码器
==========================

作用
----
验证 int8 移动端包的加载与解码契约，作为 App 端（Kotlin）
实现的对照 oracle。模型来自训练项目（m2m100_418M 微调，
translation_pkg/ 不入库）。

结论（2026-09-16 实测）
----------------------
greedy / beam5 在 6 个代表方向上输出正确的自然译文，
PC（4 线程）RTF：
  greedy ~1.3s/句；beam5 ~5.3~8.0s/句
手机端预计慢 3~6 倍，beam 宽度需在真机按延迟预算权衡。

关键契约（Kotlin 移植必须遵守，均踩坑验证）
--------------------------------------------
1. 模型词表 id ≠ sentencepiece id！
   sp.encode(..., out_type=int) 的 id 与模型 embedding 的 id
   不一致，必须经 vocab.json 做 piece -> model_id 重映射
   （encode 用 vocab[piece]，decode 用反向表）。
   spm 模型只提供 BPE 合并规则/规范化/byte-fallback。
2. 源序列 = [src_lang_token] + spm_pieces + [eos(2)]。
   缺源尾 </s> 时输出是乱码（已踩坑）。
3. 解码首步：input_ids=[eos(2), tgt_lang_token]，
   use_cache_branch=False（past 喂 48 个 [1,16,0,64] 零张量）。
4. 后续步：use_cache_branch=True，input_ids=[上一 token]。
   **encoder KV 必须冻结**：cache 分支输出的
   present.X.encoder.key/value 是 [0,16,1,64] 占位张量，
   喂回会 MatMul 广播报错；每步只更新 decoder 部分。
5. 图优化必须 ORT_ENABLE_BASIC（1.30 EXTENDED 有
   TransposeDQWeightsForMatMulNBits 跨图引用 bug，README 已写）。
6. 语言 token：en=128022 zh=128102 uz=128096 ru=128077；
   eos=2 pad=1 bos=0。vocab.json 值域 0..128003，
   语言 token 128004..128103，madeup 词 128104..128111。

运行
----
  experiments/tts/.venv_sherpa138/Scripts/python.exe run_m2m100_reference.py [--greedy|--beam N]
模型路径常量指向 ../../translation_pkg/。
