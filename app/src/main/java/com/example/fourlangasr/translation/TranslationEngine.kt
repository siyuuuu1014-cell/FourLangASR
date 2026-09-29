package com.example.fourlangasr.translation

import com.example.fourlangasr.model.AppLanguage

interface TranslationEngine {
    fun translate(
        text: String,
        source: AppLanguage,
        target: AppLanguage,
    ): TranslationResult
}
