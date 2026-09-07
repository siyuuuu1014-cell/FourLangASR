package com.example.fourlangasr.asr

import com.example.fourlangasr.model.AppLanguage
import com.example.fourlangasr.model.AsrResult
import com.example.fourlangasr.model.LoadResult

interface AsrEngine {
    val language: AppLanguage

    fun load(): LoadResult

    fun transcribe(
        samples: FloatArray,
        sampleRate: Int = 16000,
    ): AsrResult

    fun release()
}
