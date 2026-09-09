package com.example.fourlangasr.pipeline

import com.example.fourlangasr.model.LoadResult
import com.example.fourlangasr.speaker.SpeakerManager
import java.util.Locale

/**
 * 把管线结果排版成 MainActivity 的指标面板文本。
 *
 * 纯字符串拼装：不接触模型、不接触 UI 状态，
 * 阈值等数值直接引用各引擎的常量，避免两处漂移。
 */
object MetricsFormatter {

    fun segmentMetrics(
        pipeline: PipelineResult,
    ): String {

        val speaker =
            pipeline.speaker

        val lid =
            pipeline.lid

        val routedLanguage =
            pipeline.routedLanguage

        val asr =
            pipeline.asr

        val speakerScoreText =
            formatMetric(
                speaker.bestScore
            )

        val speakerMarginText =
            formatMetric(
                speaker.margin
            )

        val lidConfidenceText =
            formatMetric(
                lid.confidence
            )

        val lidMarginText =
            formatMetric(
                lid.margin
            )

        val lidMassText =
            formatMetric(
                lid.targetMass
            )

        val routeText =
            if (
                pipeline.switchedAsr
            ) {

                "SWITCH -> " +
                        routedLanguage
                            .code
                            .uppercase(
                                Locale.US
                            )

            } else {

                "REUSE " +
                        routedLanguage
                            .code
                            .uppercase(
                                Locale.US
                            )
            }

        val loadText =
            if (
                pipeline.asrLoadMs != null
            ) {

                "${pipeline.asrLoadMs} ms"

            } else {

                "0 ms"
            }

        return String.format(
            Locale.US,
            """
            Speaker      : %s
            Speaker mode : %s
            Spk score    : %s
            Spk margin   : %s
            Profiles     : %d / %d
            LID          : %s
            LID mode     : %s
            LID window   : %s
            LID conf4    : %s
            LID margin   : %s
            LID 4-mass   : %s
            LID infer    : %d ms
            ASR route    : %s
            ASR load     : %s
            Language     : %s
            Audio        : %.3f s
            ASR infer    : %d ms
            ASR RTF      : %.4f
            ASR speed    : %.2fx realtime
            PSS before   : %.1f MB
            PSS after    : %.1f MB
            PSS delta    : %+.1f MB
            """.trimIndent(),
            speaker.label,
            speaker.reason,
            speakerScoreText,
            speakerMarginText,
            speaker.profileCount,
            SpeakerManager.MAX_SPEAKERS,
            lid.languageCode
                .uppercase(
                    Locale.US
                ),
            lid.reason,
            lid.windowMode,
            lidConfidenceText,
            lidMarginText,
            lidMassText,
            lid.inferenceMs,
            routeText,
            loadText,
            routedLanguage.code,
            asr.audioDurationSec,
            asr.inferenceMs,
            asr.rtf,
            if (
                asr.rtf > 0f
            ) {

                1.0f /
                        asr.rtf

            } else {

                0f
            },
            asr.pssBeforeMb,
            asr.pssAfterMb,
            asr.pssAfterMb -
                    asr.pssBeforeMb,
        )
    }

    fun loadMetrics(
        result: LoadResult,
    ): String {

        return String.format(
            Locale.US,
            """
            Model load   : %d ms
            PSS before   : %.1f MB
            PSS after    : %.1f MB
            Load delta   : %+.1f MB
            """.trimIndent(),
            result.loadMs,
            result.pssBeforeMb,
            result.pssAfterMb,
            result.pssDeltaMb,
        )
    }

    private fun formatMetric(
        value: Float,
    ): String {

        return if (
            value.isFinite()
        ) {

            String.format(
                Locale.US,
                "%.4f",
                value,
            )

        } else {

            "N/A"
        }
    }
}
