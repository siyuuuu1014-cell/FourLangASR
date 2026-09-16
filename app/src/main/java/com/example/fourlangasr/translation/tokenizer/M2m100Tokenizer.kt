package com.example.fourlangasr.translation.tokenizer

import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets
import java.util.PriorityQueue

/**
 * SentencePiece BPE 分词器（M2M100/SMaLL-100 家族，Kotlin 实现）。
 *
 * 算法与 experiments/translation/extract_tokenizer_assets.py 的 Python 参考实现
 * 完全一致，后者与 sentencepiece 0.2.2 官方编码器在 337,138 条语料上
 * 逐条对拍（0 不一致）。契约要点：
 *
 * 1. 归一化 = 模型内嵌的 precompiled_charsmap（Darts double-array trie，
 *    nmt_nfkc 规则）+ 空白折叠/空格→▁/dummy prefix 管线。
 * 2. BPE = 按 piece score（词频）优先的 agenda：任意相邻对 (a,b)
 *    只要 a+b 是词表 piece 就可合并，每次取 score 最高者（同分取左位置小者）。
 *    初始符号逐字符，▁ 是独立基符号。
 * 3. 未知字符（不在词表）吞并其后紧邻的、同样不在词表的
 *    Unicode 类别 M（Mn/Mc/Me）组合标记，作为一个符号。
 * 4. 模型词表 id ≠ sentencepiece id：piece → model_id 经 vocab.json
 *    重映射；缺失 piece → unk id。
 *
 * 两种构造：
 * - [fromAsset]：预构建的紧凑资产（APK assets / 测试资源）；
 * - [fromSpm]：运行时从模型包内的原始 sentencepiece.bpe.model + vocab.json
 *   构建（下载的模型包自包含，不依赖 APK 内资产）。
 *
 * 实例只读，可跨线程共享。
 */
class M2m100Tokenizer private constructor(
    private val pieces: Array<String>,
    private val pieceToIdx: HashMap<String, Int>,
    private val pieceScores: FloatArray,
    private val pieceModelIds: IntArray,
    private val unkModelId: Int,
    private val detokPoolOffs: IntArray,
    private val detokPool: ByteArray,
    private val units: IntArray,
    private val normalized: ByteArray,
    private val langTokens: Map<String, Long>,
) {

    companion object {
        const val DUMMY = "\u2581" // ▁

        private const val MAGIC = "M2M3"

        /** 从预构建紧凑资产加载（格式见 extract_tokenizer_assets.py）。 */
        fun fromAsset(input: InputStream, langTokens: Map<String, Long>): M2m100Tokenizer {
            val data = input.use { it.readBytes() }
            val buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN)

            val magicBytes = ByteArray(4)
            buf.get(magicBytes)
            check(String(magicBytes, StandardCharsets.US_ASCII) == MAGIC) { "bad tokenizer asset magic" }
            val version = buf.int
            check(version == 3) { "unsupported tokenizer asset version $version" }
            val pieceCount = buf.int
            val detokCount = buf.int
            val charsmapUnits = buf.int
            val normalizedBytes = buf.int
            val markRangeCount = buf.int
            buf.int // poolBytes（顺序读取用，未单独使用）

            val poolOffs = IntArray(pieceCount)
            val byteLens = IntArray(pieceCount)
            val modelIds = IntArray(pieceCount)
            val scores = FloatArray(pieceCount)
            for (i in 0 until pieceCount) {
                poolOffs[i] = buf.int
                byteLens[i] = buf.int
                modelIds[i] = buf.int
                scores[i] = buf.float
            }

            val detokOffs = IntArray(detokCount)
            for (i in 0 until detokCount) {
                detokOffs[i] = buf.int
            }

            val units = IntArray(charsmapUnits)
            for (i in 0 until charsmapUnits) {
                units[i] = buf.int
            }

            val normalized = ByteArray(normalizedBytes)
            buf.get(normalized)

            // mark ranges 段（UnicodeMarkRanges 常量与资产生成同源，此处仅消费跳过）
            for (i in 0 until markRangeCount) {
                buf.int
                buf.int
            }

            val pool = ByteArray(buf.remaining())
            buf.get(pool)

            val pieces = arrayOfNulls<String>(pieceCount)
            val pieceToIdx = HashMap<String, Int>(pieceCount * 2)
            for (i in 0 until pieceCount) {
                val s = String(pool, poolOffs[i], byteLens[i], StandardCharsets.UTF_8)
                pieces[i] = s
                pieceToIdx[s] = i
            }

            @Suppress("UNCHECKED_CAST")
            return M2m100Tokenizer(
                pieces = pieces as Array<String>,
                pieceToIdx = pieceToIdx,
                pieceScores = scores,
                pieceModelIds = modelIds,
                unkModelId = 3,
                detokPoolOffs = detokOffs,
                detokPool = pool,
                units = units,
                normalized = normalized,
                langTokens = langTokens,
            )
        }

        /**
         * 从模型包内的原始文件构建：sentencepiece.bpe.model + vocab.json。
         * [addedTokensJsonText] 可选（added_tokens.json 原文，用于精确的
         * 语言 token 字符串）；缺失时按 "__{code}__" 约定合成。
         */
        fun fromSpm(
            spmModelBytes: ByteArray,
            vocabJsonText: String,
            addedTokensJsonText: String?,
            langTokens: Map<String, Long>,
        ): M2m100Tokenizer {
            val spm = SpmModel.parse(spmModelBytes)
            val vocab = org.json.JSONObject(vocabJsonText)
            val unkId = vocab.optInt("<unk>", 3)

            val n = spm.pieces.size
            val pool = java.io.ByteArrayOutputStream()
            val piecePoolOffs = IntArray(n)

            fun intern(s: String): Int {
                val off = pool.size()
                pool.write(s.toByteArray(StandardCharsets.UTF_8))
                pool.write(0) // NUL 结尾
                return off
            }

            val pieces = arrayOfNulls<String>(n)
            val scores = FloatArray(n)
            val modelIds = IntArray(n)
            val pieceToIdx = HashMap<String, Int>(n * 2)
            for (i in 0 until n) {
                val p = spm.pieces[i]
                piecePoolOffs[i] = intern(p)
                pieces[i] = p
                scores[i] = spm.scores[i]
                modelIds[i] = vocab.optInt(p, unkId)
                pieceToIdx[p] = i
            }

            // detok 表：model id → pool 偏移；未覆盖 id = -1（detok 时跳过）
            var maxModelId = (langTokens.values.maxOrNull() ?: 0L).toInt()
            run {
                val keys = vocab.keys()
                while (keys.hasNext()) {
                    val v = vocab.optInt(keys.next(), -1)
                    if (v > maxModelId) maxModelId = v
                }
            }
            val detokOffs = IntArray(maxModelId + 1) { -1 }
            for (piece in pieces) {
                if (piece == null) continue
                val mid = vocab.optInt(piece, -1)
                if (mid in 0..maxModelId) detokOffs[mid] = piecePoolOffs[pieceToIdx[piece]!!]
            }
            val langStrings = HashMap<Int, String>()
            if (addedTokensJsonText != null) {
                val added = org.json.JSONObject(addedTokensJsonText)
                for (key in added.keys()) {
                    val v = added.optInt(key, -1)
                    if (v in 0..maxModelId) {
                        langStrings[v] = key
                    }
                }
            }
            for ((code, id) in langTokens) {
                val idInt = id.toInt()
                if (idInt in 0..maxModelId) {
                    detokOffs[idInt] = intern(langStrings[idInt] ?: "__${code}__")
                }
            }

            @Suppress("UNCHECKED_CAST")
            return M2m100Tokenizer(
                pieces = pieces as Array<String>,
                pieceToIdx = pieceToIdx,
                pieceScores = scores,
                pieceModelIds = modelIds,
                unkModelId = unkId,
                detokPoolOffs = detokOffs,
                detokPool = pool.toByteArray(),
                units = spm.charsmapUnits,
                normalized = spm.charsmapNormalized,
                langTokens = langTokens,
            )
        }
    }

    // ================================================================ 归一化

    private fun isMark(cp: Int): Boolean {
        var lo = 0
        var hi = UnicodeMarkRanges.STARTS.size - 1
        while (lo <= hi) {
            val mid = (lo + hi) ushr 1
            when {
                cp < UnicodeMarkRanges.STARTS[mid] -> hi = mid - 1
                cp > UnicodeMarkRanges.ENDS[mid] -> lo = mid + 1
                else -> return true
            }
        }
        return false
    }

    /** darts-clone commonPrefixSearch（字节级）。返回 (value, 匹配字节数) 列表。 */
    private fun commonPrefixSearch(key: ByteArray, offset: Int, length: Int): ArrayList<IntArray> {
        val results = ArrayList<IntArray>(4)
        if (units.isEmpty()) return results
        var nodePos = 0
        nodePos = nodePos xor offsetOf(units[0])
        for (i in 0 until length) {
            val b = key[offset + i].toInt() and 0xFF
            nodePos = nodePos xor b
            if (nodePos >= units.size) break
            val unit = units[nodePos]
            if (labelOf(unit) != b) break
            nodePos = nodePos xor offsetOf(unit)
            if (hasLeaf(unit)) {
                if (nodePos < units.size) {
                    results.add(intArrayOf(valueOf(units[nodePos]), i + 1))
                }
            }
        }
        return results
    }

    private fun offsetOf(unit: Int): Int =
        (unit ushr 10) shl ((unit and (1 shl 9)) ushr 6)

    private fun labelOf(unit: Int): Int = unit and (0x80000000.toInt() or 0xFF)

    private fun hasLeaf(unit: Int): Boolean = ((unit ushr 8) and 1) == 1

    private fun valueOf(unit: Int): Int = unit and 0x7FFFFFFF

    /** Normalizer::NormalizePrefix 等价实现。返回 (替换串, 消耗字符数)。 */
    private fun normalizePrefix(text: String): Pair<String, Int> {
        val key = text.toByteArray(StandardCharsets.UTF_8)
        val results = commonPrefixSearch(key, 0, key.size)
        var longest = 0
        var value = 0
        for (r in results) {
            if (r[1] > longest) {
                longest = r[1]
                value = r[0]
            }
        }
        if (longest == 0) {
            val cp = text.codePointAt(0)
            return String(Character.toChars(cp)) to Character.charCount(cp)
        }
        var end = value
        while (end < normalized.size && normalized[end].toInt() != 0) end++
        val rep = String(normalized, value, end - value, StandardCharsets.UTF_8)
        val matched = String(key, 0, longest, StandardCharsets.UTF_8)
        return rep to matched.codePointCount(0, matched.length)
    }

    /**
     * Normalizer::Normalize 等价实现（add_dummy_prefix / remove_extra_whitespaces /
     * escape_whitespaces 均取 proto 默认 true）。
     */
    fun normalize(text: String): String {
        var s = text
        while (s.isNotEmpty()) {
            val (rep, consumed) = normalizePrefix(s)
            if (rep != " ") break
            s = s.substring(consumed)
        }
        if (s.isEmpty()) return ""
        val sb = StringBuilder()
        sb.append(DUMMY)
        var isPrevSpace = true
        while (s.isNotEmpty()) {
            val (rep, consumed) = normalizePrefix(s)
            var sp = rep
            while (isPrevSpace && sp.startsWith(" ")) sp = sp.substring(1)
            if (sp.isNotEmpty()) {
                sb.append(sp.replace(" ", DUMMY))
                isPrevSpace = sp.endsWith(" ")
            }
            s = s.substring(consumed)
        }
        while (sb.isNotEmpty() && sb.last() == DUMMY[0]) sb.setLength(sb.length - 1)
        return sb.toString()
    }

    // ================================================================ BPE 编码

    /** 符号切分：逐字符；未知字符吞并其后紧邻的、不在词表的类别 M 组合标记。 */
    private fun splitSymbols(word: String): ArrayList<String> {
        val syms = ArrayList<String>(word.length)
        var i = 0
        while (i < word.length) {
            val cp = word.codePointAt(i)
            val start = i
            i += Character.charCount(cp)
            val piece = word.substring(start, i)
            if (!pieceToIdx.containsKey(piece)) {
                while (i < word.length) {
                    val cp2 = word.codePointAt(i)
                    if (!isMark(cp2)) break
                    val markStart = i
                    i += Character.charCount(cp2)
                    if (pieceToIdx.containsKey(word.substring(markStart, i))) {
                        i = markStart
                        break
                    }
                }
                syms.add(word.substring(start, i))
            } else {
                syms.add(piece)
            }
        }
        return syms
    }

    private class Cand(val score: Float, val left: Int, val size: Int, val merged: String)

    private fun encodeWord(word: String): ArrayList<String> {
        val syms = ArrayList<String>(word.length + 1)
        syms.add(DUMMY)
        syms.addAll(splitSymbols(word))
        val n = syms.size
        val alive = BooleanArray(n) { true }

        val heap = PriorityQueue<Cand>(
            compareByDescending<Cand> { it.score }.thenBy { it.left },
        )
        for (i in 0 until n - 1) {
            val merged = syms[i] + syms[i + 1]
            val idx = pieceToIdx[merged] ?: continue
            heap.add(Cand(pieceScores[idx], i, merged.length, merged))
        }

        while (heap.isNotEmpty()) {
            val c = heap.poll()
            if (!alive[c.left]) continue
            var right = c.left + 1
            while (right < n && !alive[right]) right++
            if (right >= n) continue
            if (syms[c.left].length + syms[right].length != c.size) continue

            syms[c.left] = c.merged
            alive[right] = false

            var prev = c.left - 1
            while (prev >= 0 && !alive[prev]) prev--
            if (prev >= 0) {
                val m = syms[prev] + syms[c.left]
                val idx = pieceToIdx[m]
                if (idx != null) heap.add(Cand(pieceScores[idx], prev, m.length, m))
            }
            var nxt = right + 1
            while (nxt < n && !alive[nxt]) nxt++
            if (nxt < n) {
                val m = syms[c.left] + syms[nxt]
                val idx = pieceToIdx[m]
                if (idx != null) heap.add(Cand(pieceScores[idx], c.left, m.length, m))
            }
        }

        val out = ArrayList<String>(n)
        for (i in 0 until n) {
            if (alive[i]) out.add(syms[i])
        }
        return out
    }

    /** 返回 spm pieces（未知名保留原字符串，与 sentencepiece 输出一致）。 */
    fun encodePieces(text: String): List<String> {
        val normalized = normalize(text)
        val pieces = ArrayList<String>()
        var start = 0
        while (start < normalized.length) {
            val end = normalized.indexOf(DUMMY, start + 1)
            val wordEnd = if (end < 0) normalized.length else end
            // 归一化文本以 ▁ 开头；每个 ▁ 之后的段是一个 word（空白折叠保证非空）
            val word = normalized.substring(start + 1, wordEnd)
            if (word.isNotEmpty()) {
                pieces.addAll(encodeWord(word))
            }
            if (end < 0) break
            start = end
        }
        return pieces
    }

    private fun pieceToModelId(piece: String): Long {
        val idx = pieceToIdx[piece] ?: return unkModelId.toLong()
        return pieceModelIds[idx].toLong()
    }

    /** pieces → model id 序列（未知名 → unk id）。 */
    fun piecesToModelIds(pieces: List<String>): LongArray {
        val ids = LongArray(pieces.size)
        for (i in pieces.indices) {
            ids[i] = pieceToModelId(pieces[i])
        }
        return ids
    }

    /**
     * 完整源序列：[src_lang_token] + pieces ids + [eos]，可直接喂给 encoder。
     * [eosId] 与语言 token 由模型包 MANIFEST 提供（调用方注入，避免硬编码）。
     */
    fun encodeSourceIds(text: String, srcLang: String, eosId: Long): LongArray {
        val pieces = encodePieces(text)
        val body = piecesToModelIds(pieces)
        val ids = LongArray(body.size + 2)
        ids[0] = langTokens[srcLang] ?: error("unsupported src lang: $srcLang")
        System.arraycopy(body, 0, ids, 1, body.size)
        ids[ids.size - 1] = eosId
        return ids
    }

    fun langToken(code: String): Long =
        langTokens[code] ?: error("unsupported lang: $code")

    // ================================================================ 解码

    /** model id → piece 字符串；越界/未覆盖 id 跳过。 */
    fun detok(ids: LongArray): String {
        val sb = StringBuilder()
        for (id in ids) {
            val i = id.toInt()
            if (i < 0 || i >= detokPoolOffs.size) continue
            val off = detokPoolOffs[i]
            if (off < 0) continue
            var end = off
            while (end < detokPool.size && detokPool[end].toInt() != 0) end++
            sb.append(String(detokPool, off, end - off, StandardCharsets.UTF_8))
        }
        return sb.toString().replace(DUMMY, " ").trim()
    }
}
