package com.example.fourlangasr.pipeline

import com.example.fourlangasr.lid.LidResult
import com.example.fourlangasr.model.AppLanguage
import com.example.fourlangasr.model.AsrResult
import com.example.fourlangasr.speaker.SpeakerResult

/**
 * 一个 VAD 句子经过完整管线后的结果。
 */
data class PipelineResult(
    val speaker: SpeakerResult,
    val lid: LidResult,
    val routedLanguage: AppLanguage,
    val asr: AsrResult,
    val switchedAsr: Boolean,
    val asrLoadMs: Long?,

    /*
     * code-switch 回填（ZH 主语言且有 <unk> 时非空处理）。
     * recoveredText 优先用于显示。
     */
    val recovery: RecoveryResult,
) {

    /**
     * 显示用文本：有回填用回填结果，否则用 ASR 原文。
     */
    val displayText: String
        get() =
            recovery.recoveredText
                .ifEmpty {
                    asr.text
                }
}
