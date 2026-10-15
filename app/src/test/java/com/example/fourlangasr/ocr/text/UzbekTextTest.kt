package com.example.fourlangasr.ocr.text

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * 规范化规则的单元测试。
 *
 * 核心不变量：词内撇号**只换写法、不增不减**（阶段0 实测删除撇号会让
 * uz→en 的 "service networks" 变成 "service cross-country networks"）。
 */
class UzbekTextTest {

    private val canonical = UzbekText.CANONICAL_APOSTROPHE

    @Test
    fun threeApostropheSpellingsConvergeToOfficialForm() {
        for (variant in listOf("O'zbekiston", "O‘zbekiston", "Oʻzbekiston", "Oʼzbekiston", "O´zbekiston")) {
            assertEquals(variant, "O${canonical}zbekiston", UzbekText.toTranslationInput(variant))
        }
    }

    @Test
    fun gApostropheIsCanonicalizedToo() {
        assertEquals("g${canonical}alaba", UzbekText.toTranslationInput("g'alaba"))
        assertEquals("ko${canonical}rsatish", UzbekText.toTranslationInput("ko‘rsatish"))
    }

    @Test
    fun quotesWrappingAWholeWordAreLeftAlone() {
        // 词首/词尾的引号左右不都是字母，属于标点而不是乌兹别克正字法撇号
        assertEquals("'salom'", UzbekText.toTranslationInput("'salom'"))
        assertEquals("‘salom’", UzbekText.toTranslationInput("‘salom’"))
    }

    @Test
    fun apostropheCountIsNeverReduced() {
        val samples = listOf(
            "O'zbekiston iqtisodiyoti xizmat ko'rsatish tarmoqlariga tayanadi.",
            "To'liq ro'yxat o'qish uchun g'alaba kerak.",
            "O‘zbekiston — markaziy Osiyoda.",
            "Ma'lumot, inson huquqlari.",
        )
        for (s in samples) {
            val out = UzbekText.toTranslationInput(s)
            assertEquals(
                "撇号数量在规范化前后必须一致: $s",
                UzbekText.intraWordApostrophes(s),
                UzbekText.intraWordApostrophes(out),
            )
            assertTrue(out, out.isNotEmpty())
        }
    }

    @Test
    fun noLetterIsEverDropped() {
        val s = "O'rinli g'alaba ko'rsatish"
        val out = UzbekText.toTranslationInput(s)
        assertEquals(
            s.count { UzbekText.isWordLetter(it) },
            out.count { UzbekText.isWordLetter(it) },
        )
        assertFalse(out, out.contains("'"))
    }

    @Test
    fun dashesAndWhitespaceAreNormalized() {
        assertEquals("a-b c", UzbekText.toTranslationInput("a‐b   c"))
        assertEquals("O${canonical}zbekiston", UzbekText.toTranslationInput("  O'zbekiston  "))
    }

    @Test
    fun displayTextKeepsTheOriginalSpelling() {
        val raw = "O'zbekiston\tg'alaba"
        val states = OcrTextStates.of(raw)
        assertEquals(raw, states.rawText)
        // 校对层必须保留 OCR 实际给出的写法，否则用户看不到模型的真实输出
        assertTrue(states.displayText, states.displayText.contains("O'zbekiston"))
        assertEquals("O'zbekiston g'alaba", states.displayText)
        assertEquals("O${canonical}zbekiston g${canonical}alaba", states.translationInput)
    }

    @Test
    fun nfcIsAppliedBeforeMatching() {
        // "ü" 写成 u + 组合分音符时，撇号判定的 isLetter 依赖 NFC 后的码位
        assertEquals("küch", UzbekText.toTranslationInput("ku\u0308ch"))
    }

    @Test
    fun droppedApostropheCannotBeRestored() {
        // small 模型会直接丢撇号（阶段0 实测 17/18）。规范化不会凭空补，
        // 这正是"第一版选 tiny 而不是 small"的理由，写成测试固定住这个认知。
        assertEquals("Ozbekiston korsatish", UzbekText.toTranslationInput("Ozbekiston korsatish"))
    }
}
