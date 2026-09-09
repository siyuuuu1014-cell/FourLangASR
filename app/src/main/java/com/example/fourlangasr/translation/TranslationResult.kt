package com.example.fourlangasr.translation

import com.example.fourlangasr.model.AppLanguage

/**
 * 一次翻译调用的结构化结果。
 *
 * 关键语义：
 * - [outputText] 是下一阶段（TTS）实际使用的文本；
 * - [speechLanguage] 是 [outputText] 实际所属语言，
 *   TTS 必须用该语言朗读，不能用 targetLanguage 猜；
 * - [translated] = false 表示当前是 passthrough
 *   （真实翻译模型尚未接入），此时
 *   outputText == sourceText、speechLanguage == sourceLanguage。
 */
data class TranslationResult(
    val sourceText: String,
    val outputText: String,
    val sourceLanguage: AppLanguage,
    val targetLanguage: AppLanguage,
    val speechLanguage: AppLanguage,
    val translated: Boolean,
)
