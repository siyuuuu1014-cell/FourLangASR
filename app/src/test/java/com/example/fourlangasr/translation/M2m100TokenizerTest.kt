package com.example.fourlangasr.translation

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
 */
class M2m100TokenizerTest {

    companion object {
        private lateinit var tok: M2m100Tokenizer

        @BeforeClass
        @JvmStatic
        fun load() {
            val input = M2m100TokenizerTest::class.java.classLoader
                .getResourceAsStream("m2m100/tokenizer.bin")
                ?: error("tokenizer.bin not on test classpath")
            tok = M2m100Tokenizer.load(input)
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
        var checked = 0
        for (i in 0 until vectors.length()) {
            val v = vectors.getJSONObject(i)
            val text = v.getString("text")
            val expected = v.getString("normalized")
            val actual = tok.normalize(text)
            if (actual != expected) {
                fail("normalize(${text.repr()}) = ${actual.repr()}, expected ${expected.repr()} (vector $i)")
            }
            checked++
        }
        assertEquals(vectors.length(), checked)
    }

    @Test
    fun encodePieces_matchesOracle() {
        val vectors = readJsonArray("m2m100/tokenizer_test_vectors.json")
        var checked = 0
        for (i in 0 until vectors.length()) {
            val v = vectors.getJSONObject(i)
            val text = v.getString("text")
            val expectedPieces = v.getJSONArray("pieces")
            val actual = tok.encodePieces(text)
            val expected = (0 until expectedPieces.length()).map { expectedPieces.getString(it) }
            if (actual != expected) {
                fail("encodePieces(${text.repr()}) = $actual, expected $expected (vector $i)")
            }
            checked++
        }
        assertEquals(vectors.length(), checked)
    }

    @Test
    fun encodeIds_matchesOracle() {
        val vectors = readJsonArray("m2m100/tokenizer_test_vectors.json")
        var checked = 0
        for (i in 0 until vectors.length()) {
            val v = vectors.getJSONObject(i)
            val text = v.getString("text")
            val expectedIds = v.getJSONArray("ids")
            val actual = tok.piecesToModelIds(tok.encodePieces(text))
            val expected = LongArray(expectedIds.length()) { expectedIds.getLong(it) }
            if (!actual.contentEquals(expected)) {
                fail("ids(${text.repr()}) = ${actual.toList()}, expected ${expected.toList()} (vector $i)")
            }
            checked++
        }
        assertEquals(vectors.length(), checked)
    }

    @Test
    fun detok_roundTripsOraclePieces() {
        // pieces 取自 oracle（sentencepiece 0.2.2）对真实句子的切分；
        // detok 语义与 PC 参考解码器一致：▁ → 空格 + strip
        val cases = listOf(
            listOf("▁今", "天", "天", "气", "怎么", "样", "?") to "今天天气怎么样?",
            listOf("▁При", "вет", "▁мир") to "Привет мир",
            listOf("▁Bug", "un", "▁ob", "-", "ha", "vo", "▁yaxshi") to "Bugun ob-havo yaxshi",
            listOf("▁What", "▁is", "▁the", "▁weather", "▁like", "▁today", "?") to
                "What is the weather like today?",
        )
        for ((pieces, expected) in cases) {
            val ids = tok.piecesToModelIds(pieces)
            assertEquals(expected, tok.detok(ids))
        }
    }

    @Test
    fun encodeSourceIds_buildsFullSequence() {
        val ids = tok.encodeSourceIds("今天天气怎么样？", "zh")
        assertEquals(M2m100Tokenizer.LANG_TOKENS["zh"]!!, ids[0])
        assertEquals(M2m100Tokenizer.EOS_ID, ids[ids.size - 1])
        // 中间段与 oracle 一致
        val body = ids.copyOfRange(1, ids.size - 1)
        val expectedBody = tok.piecesToModelIds(tok.encodePieces("今天天气怎么样？"))
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
