package com.example.fourlangasr.translation

import com.example.fourlangasr.model.AppLanguage

interface TranslationEngine {
    fun translate(
        text: String,
        source: AppLanguage,
        target: AppLanguage,
    ): String
}

/**
 * Replace this class later with your four-language offline translation model.
 * The ASR layer does not need to change.
 */
class PlaceholderTranslationEngine : TranslationEngine {
    override fun translate(
        text: String,
        source: AppLanguage,
        target: AppLanguage,
    ): String {
        if (text.isBlank()) {
            return ""
        }

        return "【翻译模型尚未接入】\n" +
            "${source.code} → ${target.code}\n" +
            "ASR 输入：$text"
    }
}
