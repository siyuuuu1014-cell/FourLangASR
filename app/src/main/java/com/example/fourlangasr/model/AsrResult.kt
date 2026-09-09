package com.example.fourlangasr.model

data class LoadResult(
    val loadMs: Long,
    val pssBeforeMb: Float,
    val pssAfterMb: Float,
) {
    val pssDeltaMb: Float
        get() = pssAfterMb - pssBeforeMb
}

data class AsrResult(
    val text: String,
    val audioDurationSec: Float,
    val inferenceMs: Long,
    val rtf: Float,
    val pssBeforeMb: Float,
    val pssAfterMb: Float,

    /*
     * <unk> token 的起始时间（秒），仅流式 CTC（ZH）引擎填充。
     *
     * EXP5/EXP6 的 code-switch 触发信号：中文为主语言时句子里的
     * 英文词会落成 <unk>，CodeSwitchRecovery 用这些时间戳裁出
     * 音频片段交给 EN 验证回填。其它引擎恒为空数组。
     */
    val unkTimestampsSec: FloatArray = FloatArray(0),
)
