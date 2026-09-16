package com.example.fourlangasr.translation

import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets
import java.util.PriorityQueue

/**
 * M2M100 四语模型的 SentencePiece BPE 分词器（Kotlin 实现）。
 *
 * 实现完全对照 experiments/translation/extract_tokenizer_assets.py 的
 * Python 参考实现，后者已与 sentencepiece 0.2.2 官方编码器
 * 在 337,138 条语料上逐条对拍（0 不一致）。契约要点：
 *
 * 1. 归一化 = 模型内嵌的 precompiled_charsmap（Darts double-array trie，
 *    nmt_nfkc 规则）+ 空白折叠/空格→▁/dummy prefix 管线。
 * 2. BPE = 按 piece score（词频）优先的 agenda：任意相邻对 (a,b)
 *    只要 a+b 是词表 piece 就可合并，每次取 score 最高者（同分取左位置小者）。
 *    初始符号逐字符，▁ 是独立基符号。
 * 3. 未知字符（不在词表）吞并其后紧邻的、同样不在词表的
 *    Unicode 类别 M（Mn/Mc/Me）组合标记，作为一个符号。
 * 4. 模型词表 id ≠ sentencepiece id：piece → model_id 经 vocab.json
 *    重映射（资产内已固化）；缺失 piece → unk id 3。
 *
 * 内存占用约 25-35MB（词表 HashMap + 字符串池），实例应复用。
 * 线程安全：实例只读，可跨线程共享。
 */
class M2m100Tokenizer private constructor(
    // ---- 词表（spm id 空间） ----
    private val pieceStrings: Array<String>,
    private val pieceToIdx: HashMap<String, Int>,
    private val pieceScores: FloatArray,
    private val pieceModelIds: IntArray,
    // ---- detok（model id 空间，0..128111） ----
    private val modelIdToPoolOff: IntArray,
    private val pool: ByteArray,
    // ---- 归一化 charsmap（Darts double-array） ----
    private val units: IntArray,
    private val normalized: ByteArray,
    // ---- Unicode 类别 M 码点区间 ----
    private val markRangeStarts: IntArray,
    private val markRangeEnds: IntArray,
) {

    companion object {
        const val DUMMY = "\u2581" // ▁
        const val BOS_ID = 0L
        const val PAD_ID = 1L
        const val EOS_ID = 2L
        const val UNK_MODEL_ID = 3L

        val LANG_TOKENS = mapOf(
            "en" to 128022L,
            "zh" to 128102L,
            "uz" to 128096L,
            "ru" to 128077L,
        )

        private const val MAGIC = "M2M3"

        fun load(input: InputStream): M2m100Tokenizer {
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

            val pieceStrings = arrayOfNulls<String>(pieceCount)
            val pieceScores = FloatArray(pieceCount)
            val pieceModelIds = IntArray(pieceCount)
            val piecePoolOff = IntArray(pieceCount)
            val pieceByteLen = IntArray(pieceCount)
            for (i in 0 until pieceCount) {
                piecePoolOff[i] = buf.int
                pieceByteLen[i] = buf.int
                pieceModelIds[i] = buf.int
                pieceScores[i] = buf.float
            }

            val detokOff = IntArray(detokCount)
            for (i in 0 until detokCount) {
                detokOff[i] = buf.int
            }

            val units = IntArray(charsmapUnits)
            for (i in 0 until charsmapUnits) {
                units[i] = buf.int
            }

            val normalized = ByteArray(normalizedBytes)
            buf.get(normalized)

            val markStarts = IntArray(markRangeCount)
            val markEnds = IntArray(markRangeCount)
            for (i in 0 until markRangeCount) {
                markStarts[i] = buf.int
                markEnds[i] = buf.int
            }

            val pool = ByteArray(buf.remaining())
            buf.get(pool)

            val pieceToIdx = HashMap<String, Int>(pieceCount * 2)
            for (i in 0 until pieceCount) {
                val s = String(pool, piecePoolOff[i], pieceByteLen[i], StandardCharsets.UTF_8)
                pieceStrings[i] = s
                pieceToIdx[s] = i
            }

            @Suppress("UNCHECKED_CAST")
            return M2m100Tokenizer(
                pieceStrings = pieceStrings as Array<String>,
                pieceToIdx = pieceToIdx,
                pieceScores = pieceScores,
                pieceModelIds = pieceModelIds,
                modelIdToPoolOff = detokOff,
                pool = pool,
                units = units,
                normalized = normalized,
                markRangeStarts = markStarts,
                markRangeEnds = markEnds,
            )
        }
    }

    // ================================================================ 归一化

    private fun isMark(cp: Int): Boolean {
        var lo = 0
        var hi = markRangeStarts.size - 1
        while (lo <= hi) {
            val mid = (lo + hi) ushr 1
            when {
                cp < markRangeStarts[mid] -> hi = mid - 1
                cp > markRangeEnds[mid] -> lo = mid + 1
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
        val idx = pieceToIdx[piece] ?: return UNK_MODEL_ID
        return pieceModelIds[idx].toLong()
    }

    /** pieces → model id 序列（未知名 → unk id 3）。 */
    fun piecesToModelIds(pieces: List<String>): LongArray {
        val ids = LongArray(pieces.size)
        for (i in pieces.indices) {
            ids[i] = pieceToModelId(pieces[i])
        }
        return ids
    }

    /** 完整源序列：[src_lang_token] + pieces ids + [eos]，可直接喂给 encoder。 */
    fun encodeSourceIds(text: String, srcLang: String): LongArray {
        val pieces = encodePieces(text)
        val body = piecesToModelIds(pieces)
        val ids = LongArray(body.size + 2)
        ids[0] = LANG_TOKENS[srcLang] ?: error("unsupported src lang: $srcLang")
        System.arraycopy(body, 0, ids, 1, body.size)
        ids[ids.size - 1] = EOS_ID
        return ids
    }

    // ================================================================ 解码

    /** model id → piece 字符串；越界 id 跳过。 */
    fun detok(ids: LongArray): String {
        val sb = StringBuilder()
        for (id in ids) {
            val i = id.toInt()
            if (i < 0 || i >= modelIdToPoolOff.size) continue
            val off = modelIdToPoolOff[i]
            var end = off
            while (end < pool.size && pool[end].toInt() != 0) end++
            sb.append(String(pool, off, end - off, StandardCharsets.UTF_8))
        }
        return sb.toString().replace(DUMMY, " ").trim()
    }
}
