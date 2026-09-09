package com.example.fourlangasr.lid

data class LidResult(
    val languageCode: String,
    val confidence: Float,
    val secondLanguageCode: String?,
    val secondConfidence: Float,
    val margin: Float,
    val targetMass: Float,
    val inferenceMs: Long,
    val windowMode: String,
    val skipped: Boolean,
    val reason: String,
) {

    companion object {

        fun manualLocked(
            languageCode: String,
        ): LidResult {

            return LidResult(
                languageCode = languageCode,
                confidence = Float.NaN,
                secondLanguageCode = null,
                secondConfidence = Float.NaN,
                margin = Float.NaN,
                targetMass = Float.NaN,
                inferenceMs = 0L,
                windowMode = "manual",
                skipped = true,
                reason = "MANUAL_LOCK",
            )
        }

        fun shortFallback(
            languageCode: String,
        ): LidResult {

            return LidResult(
                languageCode = languageCode,
                confidence = Float.NaN,
                secondLanguageCode = null,
                secondConfidence = Float.NaN,
                margin = Float.NaN,
                targetMass = Float.NaN,
                inferenceMs = 0L,
                windowMode = "skip",
                skipped = true,
                reason = "SHORT_KEEP_CURRENT",
            )
        }

        fun errorFallback(
            languageCode: String,
            error: Throwable,
        ): LidResult {

            return LidResult(
                languageCode = languageCode,
                confidence = Float.NaN,
                secondLanguageCode = null,
                secondConfidence = Float.NaN,
                margin = Float.NaN,
                targetMass = Float.NaN,
                inferenceMs = 0L,
                windowMode = "error",
                skipped = true,
                reason =
                    "ERROR_KEEP_CURRENT(" +
                            error.javaClass.simpleName +
                            ")",
            )
        }
    }
}
