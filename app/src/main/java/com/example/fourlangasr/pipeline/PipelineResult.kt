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
)
