package com.example.fourlangasr.ocr.text

/**
 * 乌兹别克语文本的三态表示与规范化。
 *
 * 三态不可合并，因为阶段0 实测到两件事：
 * 1. PP-OCRv6 的字典（tiny 6904 / small 18708 条）**不含 U+02BB `ʻ` 与 U+02BC `ʼ`**，
 *    模型永远无法原样输出官方写法：tiny 会替换成 ASCII `'`（16/18 次），
 *    small 会直接丢掉（17/18 次）。
 * 2. 撇号**删掉**会改变译文语义（uz→en 实测 "service networks" → "service cross-country
 *    networks"；uz→zh "服务网络" → "服务交互网络"），而 `ʻ / ‘ / '` 三种写法彼此等价。
 *
 * 因此规范化只做"统一写法"，绝不做"删除特殊字符"。
 * [toTranslationInput] 的输入若含词内撇号，输出必须保留同样数量的撇号（由单测锁死）。
 */
data class OcrTextStates(
    /** OCR 原样输出，一字不改，用于追溯与置信度对齐。 */
    val rawText: String,
    /** 用户校对的对象：只做显示层清理，不改正字法。 */
    val displayText: String,
    /** 送入 TranslationService 的文本：撇号统一为官方 U+02BB。 */
    val translationInput: String,
) {
    companion object {
        fun of(rawText: String): OcrTextStates = OcrTextStates(
            rawText = rawText,
            displayText = UzbekText.displayHygiene(rawText),
            translationInput = UzbekText.toTranslationInput(rawText),
        )
    }
}

object UzbekText {

    /** 乌兹别克语官方撇号（modifier letter turned comma）。 */
    const val CANONICAL_APOSTROPHE: Char = 'ʻ'

    /**
     * 各种会被写成撇号的字符：ASCII 直撇号、左右单引号、修饰撇号、
     * 尖撇号、重音符。规范化把它们统一成 [CANONICAL_APOSTROPHE]。
     */
    private val APOSTROPHE_FAMILY: Set<Char> = setOf(
        '\'',                 // U+0027 直撇号（OCR 最常见的替代写法）
        '‘',      // U+2018 左单引号（排版里常用来表示 o‘）
        '’',      // U+2019
        'ʻ',      // U+02BB 官方 turned comma，规范化的目标
        'ʼ',      // U+02BC
        'ʹ',      // U+02B9
        '´',      // U+00B4
    )

    /**
     * 判断"词内字母"时必须排除修饰字母/修饰符号：
     * U+02BB 的 Unicode 类别是 Lm（MODIFIER_LETTER），`Char.isLetter()` 会返回 true，
     * 直接把 isLetter 当依据会让撇号把自己也认成字母（实测会让撇号计数从 21 变 24）。
     */
    internal fun isWordLetter(c: Char): Boolean {
        if (!c.isLetter()) return false
        val cat = Character.getType(c)
        return cat != Character.MODIFIER_LETTER.toInt() && cat != Character.MODIFIER_SYMBOL.toInt()
    }

    /** 词内撇号 = 左右都是字母的撇号族字符；这是规范化必须保住的量。 */
    fun intraWordApostrophes(text: String): Int {
        val normalized = java.text.Normalizer.normalize(text, java.text.Normalizer.Form.NFC)
        var n = 0
        for (i in normalized.indices) {
            val c = normalized[i]
            if (c in APOSTROPHE_FAMILY &&
                i > 0 && i < normalized.length - 1 &&
                isWordLetter(normalized[i - 1]) && isWordLetter(normalized[i + 1])
            ) {
                n++
            }
        }
        return n
    }

    /**
     * 显示层清理：折叠空白、去掉行尾多余空格。
     * 刻意不碰任何字符本身——用户校对时看到的必须还是"识别出来的样子"。
     */
    fun displayHygiene(text: String): String =
        text.replace("\r\n", "\n")
            .replace('\r', '\n')
            .split('\n')
            .joinToString("\n") { it.trim().replace(Regex("[ \\t]+"), " ") }
            .trim()

    /**
     * 生成送翻译的规范化文本。规则按顺序执行，全部可测：
     * 1. NFC 归一化；
     * 2. 词内撇号族 → [CANONICAL_APOSTROPHE]（词外的引号一律不动，避免把
     *    `'salom'` 这种整词引号变成非法正字法）；
     * 3. U+2010..U+2015 各类连字符 → ASCII '-'；
     * 4. 连续空白折叠为单空格并 trim。
     *
     * 绝不删除撇号、绝不删除任何字母。
     */
    fun toTranslationInput(text: String): String {
        val normalized = java.text.Normalizer.normalize(text, java.text.Normalizer.Form.NFC)
        val sb = StringBuilder(normalized.length)
        for (i in normalized.indices) {
            val c = normalized[i]
            val isWordApostrophe = c in APOSTROPHE_FAMILY &&
                i > 0 && i < normalized.length - 1 &&
                isWordLetter(normalized[i - 1]) && isWordLetter(normalized[i + 1])
            sb.append(if (isWordApostrophe) CANONICAL_APOSTROPHE else c)
        }
        return sb.toString()
            .replace(Regex("[\\u2010-\\u2015]"), "-")
            .replace(Regex("\\s+"), " ")
            .trim()
    }
}
