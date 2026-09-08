package com.example.fourlangasr.speaker

data class SpeakerResult(
    val label: String,
    val speakerId: Int?,
    val bestScore: Float,
    val secondScore: Float,
    val margin: Float,
    val isNewSpeaker: Boolean,
    val acceptedExisting: Boolean,
    val skipped: Boolean,
    val profileCount: Int,
    val reason: String,
)
