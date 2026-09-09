package com.example.fourlangasr.translation

import com.example.fourlangasr.model.AppLanguage

interface TranslationEngine {
    fun translate(
        text: String,
        source: AppLanguage,
        target: AppLanguage,
    ): TranslationResult
}

/**
 * 占位实现：真实离线翻译模型接入前，原样返回 ASR 文本。
 *
 * passthrough 语义：
 * - outputText = sourceText
 * - speechLanguage = sourceLanguage（中文 ASR 必须用中文 voice 朗读，
 *   即使 targetLanguage 是 EN——避免英文 voice 念中文）
 * - translated = false
 *
 * 以后替换成真实翻译实现时，TTS 层与 ASR 主链路不需要改动：
 * TTS 只依赖 TranslationResult 的 outputText / speechLanguage。
 */
class PlaceholderTranslationEngine : TranslationEngine {

    override fun translate(
        text: String,
        source: AppLanguage,
        target: AppLanguage,
    ): TranslationResult {

        return TranslationResult(
            sourceText = text,
            outputText = text,
            sourceLanguage = source,
            targetLanguage = target,
            speechLanguage = source,
            translated = false,
        )
    }
}
