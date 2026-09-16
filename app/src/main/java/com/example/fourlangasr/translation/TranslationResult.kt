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
 *   outputText == sourceText、speechLanguage == sourceLanguage；
 * - [modelLabel] 标识本次实际使用的模型：
 *   "server" = 服务器 API；本地模型 id = 本地引擎（UI 负责渲染
 *   为"总模型/专项模型"文案）；null = passthrough。
 */
data class TranslationResult(
    val sourceText: String,
    val outputText: String,
    val sourceLanguage: AppLanguage,
    val targetLanguage: AppLanguage,
    val speechLanguage: AppLanguage,
    val translated: Boolean,
    val modelLabel: String? = null,
)
