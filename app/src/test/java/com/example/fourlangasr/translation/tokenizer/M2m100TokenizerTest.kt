package com.example.fourlangasr.translation.tokenizer

import org.json.JSONArray
import org.junit.Assert.assertEquals
import org.junit.Assert.fail
import org.junit.BeforeClass
import org.junit.Test

/**
 * M2m100Tokenizer 与 sentencepiece 0.2.2 oracle 的对拍单测。
 *
 * 向量由 experiments/translation/extract_tokenizer_assets.py 生成
 * （全量对拍 337,138 条 0 不一致；此处为精简子集，用于守护 Kotlin 移植）。
 *
 * 覆盖两条构造路径：
 * - [fromAsset]：预构建紧凑资产；
 * - [fromSpm]：运行时从原始 sentencepiece.bpe.model + vocab.json 构建
 *   （下载模型包的自包含路径），用同一批 oracle 向量对拍。
 */
class M2m100TokenizerTest {

    companion object {
        /** 四语 m2m100 的语言 token（模型包 MANIFEST 的 fixtures；生产代码从包内读取）。 */
        private val LANG_TOKENS = mapOf(
            "en" to 128022L,
            "zh" to 128102L,
            "uz" to 128096L,
            "ru" to 128077L,
        )

        private const val EOS_ID = 2L

        private lateinit var fromAsset: M2m100Tokenizer
        private lateinit var fromSpm: M2m100Tokenizer

        @BeforeClass
        @JvmStatic
        fun load() {
            val cl = M2m100TokenizerTest::class.java.classLoader
            fromAsset = M2m100Tokenizer.fromAsset(
                cl.getResourceAsStream("m2m100/tokenizer.bin")!!,
                LANG_TOKENS,
            )
            fromSpm = M2m100Tokenizer.fromSpm(
                spmModelBytes = cl.getResourceAsStream("m2m100/sentencepiece.bpe.model")!!.readBytes(),
                vocabJsonText = cl.getResourceAsStream("m2m100/vocab.json")!!
                    .readBytes().toString(Charsets.UTF_8),
                addedTokensJsonText = cl.getResourceAsStream("m2m100/added_tokens.json")!!
                    .readBytes().toString(Charsets.UTF_8),
                langTokens = LANG_TOKENS,
            )
        }

        private fun readJsonArray(name: String): JSONArray {
            val text = M2m100TokenizerTest::class.java.classLoader
                .getResourceAsStream(name)!!.readBytes().toString(Charsets.UTF_8)
            return JSONArray(text)
        }
    }

    @Test
    fun normalize_matchesOracle() {
        val vectors = readJsonArray("m2m100/normalizer_test_vectors.json")
        for (i in 0 until vectors.length()) {
            val v = vectors.getJSONObject(i)
            val text = v.getString("text")
            val expected = v.getString("normalized")
            val actual = fromAsset.normalize(text)
            if (actual != expected) {
                fail("normalize(${text.repr()}) = ${actual.repr()}, expected ${expected.repr()} (vector $i)")
            }
        }
    }

    @Test
    fun encodePieces_matchesOracle() {
        val vectors = readJsonArray("m2m100/tokenizer_test_vectors.json")
        for (i in 0 until vectors.length()) {
            val v = vectors.getJSONObject(i)
            val text = v.getString("text")
            val expectedPieces = v.getJSONArray("pieces")
            val expected = (0 until expectedPieces.length()).map { expectedPieces.getString(it) }
            val actual = fromAsset.encodePieces(text)
            if (actual != expected) {
                fail("encodePieces(${text.repr()}) = $actual, expected $expected (vector $i)")
            }
        }
    }

    @Test
    fun encodeIds_matchesOracle() {
        val vectors = readJsonArray("m2m100/tokenizer_test_vectors.json")
        for (i in 0 until vectors.length()) {
            val v = vectors.getJSONObject(i)
            val text = v.getString("text")
            val expectedIds = v.getJSONArray("ids")
            val expected = LongArray(expectedIds.length()) { expectedIds.getLong(it) }
            val actual = fromAsset.piecesToModelIds(fromAsset.encodePieces(text))
            if (!actual.contentEquals(expected)) {
                fail("ids(${text.repr()}) = ${actual.toList()}, expected ${expected.toList()} (vector $i)")
            }
        }
    }

    /** fromSpm 构造与 fromAsset 构造在 oracle 向量上完全一致（验证 Kotlin 端构建移植）。 */
    @Test
    fun fromSpm_matchesFromAssetOnOracleVectors() {
        val vectors = readJsonArray("m2m100/tokenizer_test_vectors.json")
        for (i in 0 until vectors.length()) {
            val v = vectors.getJSONObject(i)
            val text = v.getString("text")
            val expected = fromAsset.encodePieces(text)
            val actual = fromSpm.encodePieces(text)
            if (actual != expected) {
                fail("fromSpm(${text.repr()}) = $actual, expected $expected (vector $i)")
            }
        }
    }

    @Test
    fun detok_roundTripsOraclePieces() {
        // pieces 取自 oracle（sentencepiece 0.2.2）对真实句子的切分
        val cases = listOf(
            listOf("▁今", "天", "天", "气", "怎么", "样", "?") to "今天天气怎么样?",
            listOf("▁При", "вет", "▁мир") to "Привет мир",
            listOf("▁Bug", "un", "▁ob", "-", "ha", "vo", "▁yaxshi") to "Bugun ob-havo yaxshi",
            listOf("▁What", "▁is", "▁the", "▁weather", "▁like", "▁today", "?") to
                "What is the weather like today?",
        )
        for ((pieces, expected) in cases) {
            val ids = fromAsset.piecesToModelIds(pieces)
            assertEquals(expected, fromAsset.detok(ids))
        }
    }

    @Test
    fun encodeSourceIds_buildsFullSequence() {
        val ids = fromAsset.encodeSourceIds("今天天气怎么样？", "zh", EOS_ID)
        assertEquals(LANG_TOKENS["zh"]!!, ids[0])
        assertEquals(EOS_ID, ids[ids.size - 1])
        val body = ids.copyOfRange(1, ids.size - 1)
        val expectedBody = fromAsset.piecesToModelIds(fromAsset.encodePieces("今天天气怎么样？"))
        assertEquals(expectedBody.toList(), body.toList())
    }

    private fun String.repr(): String {
        val sb = StringBuilder("\"")
        for (ch in this) {
            when {
                ch == '"' -> sb.append("\\\"")
                ch == '\\' -> sb.append("\\\\")
                ch == '\n' -> sb.append("\\n")
                ch == '\t' -> sb.append("\\t")
                ch.code < 0x20 -> sb.append(String.format("\\u%04x", ch.code))
                else -> sb.append(ch)
            }
        }
        sb.append('"')
        return sb.toString()
    }
}
