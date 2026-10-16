package com.example.fourlangasr.ocr.text

/**
 * 文本规范化器：语言相关，与"用哪个识别模型"解耦。
 *
 * 存在这个接口的原因是 PP-OCRv6 Tiny 本身就是一个覆盖 49 种语言的多语模型：
 * 中文、英文、乌兹别克语共用同一份 det+rec 权重，语言之间只差
 * "用哪个识别模型（未来俄语换 PP-OCRv5 Cyrillic）"和"用哪套文本规则"，
 * 所以规范化能力必须可按语言替换，而不是写死一种。
 */
interface TextNormalizer {

    /** 显示层清理：不改变任何字素。 */
    fun display(text: String): String

    /** 送翻译模型的文本。 */
    fun forTranslation(text: String): String
}

/** 中文/英文等语言：只做空白与标点整理，不改字法。 */
object LatinCjkTextNormalizer : TextNormalizer {
    override fun display(text: String): String = UzbekText.displayHygiene(text)

    override fun forTranslation(text: String): String =
        text.trim().replace(Regex("\\s+"), " ")
}

/** 乌兹别克语：词内撇号统一为官方 U+02BB，绝不删除。 */
object UzbekTextNormalizer : TextNormalizer {
    override fun display(text: String): String = UzbekText.displayHygiene(text)

    override fun forTranslation(text: String): String = UzbekText.toTranslationInput(text)
}
