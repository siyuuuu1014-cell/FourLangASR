package com.example.fourlangasr.tts

import com.example.fourlangasr.model.AppLanguage

/**
 * TTS 路由管理器：对外仍是单个 [TtsEngine]。
 *
 * 策略：
 * - 系统引擎（[systemEngine]，含多系统引擎 fallback 池）
 *   有该语言离线 voice -> 用系统引擎（音质/生态最好）；
 * - 系统没有 -> 交给离线神经引擎（[offlineEngine]，
 *   如内置 Piper ru_RU，解决系统 voice 缺失的语言）。
 *
 * MainActivity 只依赖 TtsEngine 接口，不感知内部有两个引擎。
 */
class TtsManager(
    private val systemEngine: TtsEngine,
    private val offlineEngine:
            OfflineNeuralTtsEngine,
) : TtsEngine {

    override val isReady: Boolean
        get() =
            systemEngine.isReady ||
                    offlineEngine.isReady

    override val isSpeaking: Boolean
        get() =
            systemEngine.isSpeaking ||
                    offlineEngine.isSpeaking

    override fun speak(
        text: String,
        language: AppLanguage,
    ): Boolean {

        val engine =
            when {

                /*
                 * 系统引擎有该语言 voice（实测缓存）-> 系统优先。
                 */
                systemAvailable(
                    language
                ) ->

                    systemEngine

                /*
                 * 系统没有、离线引擎声明支持 -> 离线。
                 */
                offlineEngine
                    .supportsLanguage(
                        language
                    ) ->

                    offlineEngine

                /*
                 * 都不支持/系统未就绪：
                 * 走系统引擎路径，由它给出可诊断错误。
                 */
                else ->

                    systemEngine
            }

        return engine.speak(
            text = text,
            language = language,
        )
    }

    /*
     * 系统 voice 可用性实测较贵（每次对 4 语言 setLanguage），
     * 做短 TTL 缓存；装新语音包后最多 CACHE_MS 内生效。
     */
    private var cachedSystemAvailability:
            Map<String, Boolean>? =
        null

    private var cachedAtMs =
        0L

    private fun systemAvailable(
        language: AppLanguage,
    ): Boolean {

        return systemAvailability()[
            language.code
        ] ==
            true
    }

    private fun systemAvailability():
            Map<String, Boolean> {

        val now =
            System.currentTimeMillis()

        if (
            cachedSystemAvailability ==
            null ||
            now -
            cachedAtMs >
            CACHE_MS
        ) {

            cachedSystemAvailability =
                systemEngine
                    .voiceAvailability()

            cachedAtMs =
                now
        }

        return cachedSystemAvailability
            ?: emptyMap()
    }

    override fun setSpeechRate(
        rate: Float,
    ) {

        systemEngine.setSpeechRate(
            rate
        )

        offlineEngine.setSpeechRate(
            rate
        )
    }

    override val speechRate: Float
        get() = systemEngine.speechRate

    override fun voiceAvailability():
            Map<String, Boolean> {

        val result =
            systemAvailability()
                .toMutableMap()

        for (
        language
        in AppLanguage.entries
        ) {

            if (
                result[
                    language.code
                ] !=
                true &&
                offlineEngine
                    .supportsLanguage(
                        language
                    )
            ) {

                result[
                    language.code
                ] =
                    true
            }
        }

        return result
    }

    override fun stop() {

        systemEngine.stop()
        offlineEngine.stop()
    }

    override fun release() {

        systemEngine.release()
        offlineEngine.release()
    }

    companion object {

        private const val CACHE_MS =
            30_000L
    }
}
