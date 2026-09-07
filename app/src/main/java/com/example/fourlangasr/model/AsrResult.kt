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
)
