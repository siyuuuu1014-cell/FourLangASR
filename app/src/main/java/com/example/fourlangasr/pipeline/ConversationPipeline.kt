package com.example.fourlangasr.pipeline

import com.example.fourlangasr.asr.AsrManager
import com.example.fourlangasr.lid.LidEngine
import com.example.fourlangasr.lid.LidResult
import com.example.fourlangasr.model.AppLanguage
import com.example.fourlangasr.speaker.SpeakerManager
import java.util.Locale

/**
 * 单个 VAD 句子的识别管线。
 *
 * Speaker -> LID -> 语种路由 -> ASR
 *
 * 调用方（MainActivity 的 recognition worker）通过串行 Channel
 * 保证同一时间只有一个句子进入本管线，因此内部无需加锁。
 */
class ConversationPipeline(
    private val speakerManager: SpeakerManager,
    private val lidEngine: LidEngine,
    private val asrManager: AsrManager,
) {

    fun processSegment(
        samples: FloatArray,
        sourceMode: SourceMode,
        manualLanguage: AppLanguage,
        fallbackLanguage: AppLanguage,
    ): PipelineResult {

        val durationSec =
            samples.size / 16000.0f

        /*
         * 1. Speaker -> 这句话是谁说的。
         */
        val speakerResult =
            speakerManager.identify(
                samples = samples,
                sampleRate = 16000,
            )

        /*
         * 2. LID -> zh / en / ru / uz。
         *
         * 手动模式完全不做 LID 推理；
         * 短句和 LID 异常都回退到 fallbackLanguage。
         * LID 失败不允许影响 ASR。
         */
        val lidResult =
            if (
                sourceMode ==
                SourceMode.MANUAL
            ) {

                LidResult.manualLocked(
                    languageCode =
                        manualLanguage.code,
                )

            } else if (
                durationSec >=
                LidEngine.MIN_AUDIO_SEC
            ) {

                try {

                    lidEngine.identify(
                        samples = samples,
                        sampleRate = 16000,
                    )

                } catch (
                    e: Throwable
                ) {

                    LidResult.errorFallback(
                        languageCode =
                            fallbackLanguage.code,
                        error = e,
                    )
                }

            } else {

                LidResult.shortFallback(
                    languageCode =
                        fallbackLanguage.code,
                )
            }

        /*
         * 3. 路由 -> 只在语种变化时才切换 ASR 模型。
         */
        val routedLanguage =
            if (
                sourceMode ==
                SourceMode.MANUAL
            ) {

                manualLanguage

            } else {

                appLanguageFromLidCode(
                    lidResult.languageCode
                )
            }

        val switchedAsr =
            asrManager.loadedLanguage !=
                    routedLanguage

        val asrLoadMs =
            if (
                switchedAsr
            ) {

                asrManager
                    .switchLanguage(
                        routedLanguage
                    )
                    .loadMs

            } else {

                null
            }

        /*
         * 4. ASR -> 文本。
         */
        val asrResult =
            asrManager.transcribe(
                samples = samples,
                sampleRate = 16000,
            )

        return PipelineResult(
            speaker = speakerResult,
            lid = lidResult,
            routedLanguage = routedLanguage,
            asr = asrResult,
            switchedAsr = switchedAsr,
            asrLoadMs = asrLoadMs,
        )
    }

    companion object {

        /**
         * LID 输出的语种代码 -> AppLanguage。
         *
         * LidTarget 只映射 zh/en/ru/uz 四个类别，
         * 其余 103 类不会出现在 four-way 结果里。
         */
        fun appLanguageFromLidCode(
            code: String,
        ): AppLanguage {

            return when (
                code.lowercase(
                    Locale.US
                )
            ) {

                "zh" -> AppLanguage.ZH

                "en" -> AppLanguage.EN

                "ru" -> AppLanguage.RU

                "uz" -> AppLanguage.UZ

                else ->
                    error(
                        "Unsupported LID code: $code"
                    )
            }
        }
    }
}
