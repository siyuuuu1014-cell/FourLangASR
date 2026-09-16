m2m100 四语翻译：参考解码器 + tokenizer 资产提取
==================================================

作用
----
1. run_m2m100_reference.py：验证 int8 移动端包的加载与解码契约，
   作为 App 端（Kotlin）实现的对照 oracle。
2. extract_tokenizer_assets.py：从 sentencepiece.bpe.model + vocab.json
   提取 Kotlin tokenizer 所需资产（app/src/main/assets/m2m100/tokenizer.bin），
   并与 sentencepiece 0.2.2 官方编码器逐条对拍（337,138 条语料 0 不一致）。

模型来自训练项目（m2m100_418M 微调，translation_pkg/ 不入库）。

解码结论（2026-09-16 实测）
--------------------------
greedy / beam5 在 6 个代表方向上输出正确的自然译文，
PC（4 线程）RTF：
  greedy ~1.3s/句；beam5 ~5.3~8.0s/句
手机端预计慢 3~6 倍，beam 宽度需在真机按延迟预算权衡。

关键契约（Kotlin 移植必须遵守，均踩坑验证）
--------------------------------------------
1. 模型词表 id ≠ sentencepiece id！
   必须经 vocab.json 做 piece -> model_id 重映射
   （encode 用 vocab[piece]，缺失 piece -> unk id 3）。
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

Tokenizer 事实（对拍确认，Kotlin 实现照此）
--------------------------------------------
1. vocab.json = HF 官方 facebook/m2m100_418M/vocab.json（md5 一致，
   已与 HF 比对）。它是 frequency 序（非 spm 序），值域 0..128003。
   其中 390 个 spm pieces 不在 vocab.json —— 官方即如此（无 embedding），
   编码时映射为 unk id 3（与 HF M2M100Tokenizer._convert_token_to_id 一致）。
2. 归一化 = normalizer_spec.precompiled_charsmap（Darts double-array trie，
   nmt_nfkc 规则，含 e+U+0301 -> é 这类组合规则），加
   Normalizer::Normalize 管线：头部空格跳过 → 逐段最长 trie 匹配
   → 前段以空格结尾时吞本段头部空格（空白折叠）→ ' ' 转 ▁ →
   尾部空格去除 → 首部加 dummy prefix ▁。
   注意：不是完整 NFKC！(如缅甸文 virama 序列不重排)。
   Kotlin 需移植 darts-clone commonPrefixSearch（见 extract 脚本的 Darts 类）。
3. BPE 算法（sentencepiece>=0.1.99 bpe_model.cc SampleEncode）：
   初始符号 = 归一化文本逐字符（▁ 是独立基符号）；
   合并 = 按 piece score（词频）优先的 agenda（优先队列）：
   任意相邻对 (a,b) 只要 a+b 是词表 piece 就可合并，
   每次取 score 最高者（同分取左位置小者），合并后把新相邻对入队。
   **没有固定 merge 表**，也不按训练顺序应用。
4. 符号切分细节：未知字符（不在词表）会吞并其后紧邻的、
   同样不在词表的 Unicode 类别 M（Mn/Mc/Me）组合标记，作为一个符号。
   例：'Ϣ̌'（Ϣ 与 ̌ 都不在词表）-> 一个符号；'x̌'（x 在词表）-> 两个符号；
   'Ϣ̀'（̀ 在词表）-> 两个符号。该行为与源码注释的
   "user_defined_symbols matcher" 不完全一致（0.2.2 wheel 实测），
   以对拍结果为准（本仓库已按实测规则实现并通过全量对拍）。
5. 未知名符号以原字符串输出（piece 字符串不变，id 为 unk）。
6. detok：model_id -> piece 字符串，拼接后 ▁ -> 空格，strip。
   lang tokens（__xx__）与 madeup 词（<madeupwordN>）的字符串
   已在资产 detok 表中补齐。

资产格式（tokenizer.bin，little-endian，version 3）
---------------------------------------------------
  magic "M2M3" | version int32 | pieceCount int32
  | detokCount int32 | charsmapUnits int32 | normalizedBytes int32
  | markRangeCount int32 | poolBytes int32
  | pieces: pieceCount × {uint32 poolOff, uint32 byteLen, int32 modelId, float32 score}
  | detok: detokCount × uint32 poolOff
  | charsmap units: charsmapUnits × uint32（Darts double-array）
  | charsmap normalized: bytes（NUL 分隔的替换串）
  | mark ranges: markRangeCount × {uint32 begin, uint32 end}（Unicode 类别 M 码点区间）
  | stringPool: bytes
piece 数组按 spm id 顺序；score 用于 BPE agenda 优先级。

对拍
----
  /d/dev/projects/fourlang_translation/.venv/Scripts/python.exe extract_tokenizer_assets.py
语料：全部 128000 个 piece（含前导空格变体）+ 四语真实句 +
边界/空白/全角/emoji/组合标记压力（基数 × 组合标记）。
输出：
  app/src/main/assets/m2m100/tokenizer.bin
  tokenizer_test_vectors.json / normalizer_test_vectors.json（Kotlin 单测向量）

运行参考解码器
--------------
  experiments/tts/.venv_sherpa138/Scripts/python.exe run_m2m100_reference.py [--greedy|--beam N]
模型路径常量指向 ../../translation_pkg/。
