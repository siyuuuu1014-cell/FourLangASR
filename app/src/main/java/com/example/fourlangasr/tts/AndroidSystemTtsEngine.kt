package com.example.fourlangasr.tts

import android.content.Context
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import com.example.fourlangasr.model.AppLanguage
import java.util.Locale
import java.util.concurrent.atomic.AtomicInteger

/**
 * Android 系统 TextToSpeech 实现（TTS v0.1）。
 *
 * - 使用 applicationContext，不持有 Activity；
 * - 初始化异步（OnInitListener），未 ready 时 speak 失败并给出原因；
 * - 每次 speak 用独立 utteranceId + QUEUE_FLUSH；
 * - speak 前 setLanguage 并检查 LANG_MISSING_DATA /
 *   LANG_NOT_SUPPORTED / voice.isNetworkConnectionRequired；
 * - onStart/onDone/onError 维护 isSpeaking（防 TTS 声音回灌 VAD），
 *   onDone/onError/stop/release 都能把状态复位，不会永久卡 true。
 *
 * 未来替换为 OfflineNeuralTtsEngine 时，上层 pipeline 无需改动。
 */
class AndroidSystemTtsEngine(
    context: Context,
    private val listener: TtsEngine.Listener,
) : TtsEngine {

    private val appContext =
        context.applicationContext

    @Volatile
    override var isSpeaking =
        false
        private set

    @Volatile
    override var speechRate =
        1.0f
        private set

    private var tts:
            TextToSpeech? =
        null

    @Volatile
    private var ready =
        false

    @Volatile
    private var initError:
            String? =
        null

    private val utteranceCounter =
        AtomicInteger(0)

    /*
     * 共享进度监听：主引擎与 Google fallback 引擎都挂同一个实例，
     * isSpeaking 状态机全局唯一，utterance 事件统一上报。
     */
    private val progressListener =
        object :
            UtteranceProgressListener() {

            override fun onStart(
                utteranceId: String?
            ) {

                isSpeaking =
                    true

                Log.d(
                    TAG,
                    "onStart: $utteranceId"
                )

                listener.onSpeakStart()
            }

            override fun onDone(
                utteranceId: String?
            ) {

                isSpeaking =
                    false

                Log.d(
                    TAG,
                    "onDone: $utteranceId"
                )

                listener.onSpeakDone()
            }

            /*
             * QUEUE_FLUSH 打断旧 utterance 时回调的是
             * onStop 而不是 onDone：必须复位 isSpeaking，
             * 否则极端情况下 VAD gate 会永久卡住。
             * 不通知 UI（若随后有新 utterance，onStart 会通知）。
             */
            override fun onStop(
                utteranceId: String?,
                interrupted: Boolean,
            ) {

                isSpeaking =
                    false
            }

            override fun onError(
                utteranceId: String?
            ) {

                isSpeaking =
                    false

                Log.w(
                    TAG,
                    "onError: $utteranceId"
                )

                listener.onSpeakError(
                    "TTS playback error"
                )
            }

            override fun onError(
                utteranceId: String?,
                errorCode: Int,
            ) {

                isSpeaking =
                    false

                Log.w(
                    TAG,
                    "onError: $utteranceId code=$errorCode"
                )

                listener.onSpeakError(
                    "TTS playback error ($errorCode)"
                )
            }
        }

    /*
     * Google TTS fallback 引擎（懒创建）：
     * 系统默认引擎缺少某语言语音（如俄语）时，
     * 若已安装 Google TTS 则用它朗读。
     */
    private var fallbackTts:
            TextToSpeech? =
        null

    @Volatile
    private var fallbackReady =
        false

    override val isReady: Boolean
        get() = ready

    init {
        tts =
            TextToSpeech(
                appContext,
            ) { status ->

                val engine =
                    tts

                if (
                    engine == null ||
                    status !=
                    TextToSpeech.SUCCESS
                ) {

                    initError =
                        "TextToSpeech init failed (status $status)"

                    ready =
                        false

                    Log.w(
                        TAG,
                        "init failed: status=$status"
                    )

                    listener.onEngineReady(
                        emptyMap()
                    )

                    return@TextToSpeech
                }

                Log.d(
                    TAG,
                    "init OK, engine=${engine.voice?.name ?: "?"}"
                )

                engine.setOnUtteranceProgressListener(
                    progressListener
                )

                ready =
                    true

                initError =
                    null

                listener.onEngineReady(
                    voiceAvailability()
                )
            }
    }

    override fun setSpeechRate(
        rate: Float,
    ) {

        speechRate =
            rate
                .coerceIn(
                    0.5f,
                    2.0f
                )
    }

    override fun voiceAvailability():
            Map<String, Boolean> {

        val engine =
            tts
                ?: return emptyMap()

        if (
            !ready
        ) {

            return emptyMap()
        }

        /*
         * 诊断：ColorOS 等引擎的 getVoices() 列表可能与实际
         * setLanguage 能力不一致（曾出现全 ✗ 但 zh/en 能播）。
         * 保留日志用于对比，判断以 setLanguage 实测为准。
         */
        try {

            val voices =
                engine.voices

            Log.d(
                TAG,
                "availability voices=${voices?.size ?: "null"}: " +
                        (voices
                            ?.joinToString(
                                "; "
                            ) { v ->
                                "${v.name}(${v.locale})" +
                                        "net=${v.isNetworkConnectionRequired}"
                            }
                            ?: "")
            )

        } catch (
            e: Throwable
        ) {

            Log.w(
                TAG,
                "getVoices failed: ${e.message}"
            )
        }

        /*
         * 与 speak 完全一致的实测路径：
         * setLanguage 返回 >= 0（AVAILABLE/COUNTRY_AVAILABLE/
         * COUNTRY_VAR_AVAILABLE）且激活 voice 不依赖网络。
         * 默认引擎不支持的语种再试 Google fallback。
         *
         * setLanguage 会切走引擎当前语音，但每次 speak 前都会
         * 重新 setLanguage，因此没有副作用。
         */
        val result =
            mutableMapOf<
                    String,
                    Boolean
                    >()

        for (
        language
        in AppLanguage.entries
        ) {

            val locale =
                localeFor(
                    language
                )

            var activeEngine:
                    TextToSpeech =
                engine

            var langResult =
                engine.setLanguage(
                    locale
                )

            if (
                langResult <
                TextToSpeech.LANG_AVAILABLE
            ) {

                ensureGoogleFallback()
                    ?.let { fallback ->

                        if (
                            fallbackReady
                        ) {

                            val fbResult =
                                fallback.setLanguage(
                                    locale
                                )

                            if (
                                fbResult >=
                                TextToSpeech.LANG_AVAILABLE
                            ) {

                                activeEngine =
                                    fallback

                                langResult =
                                    fbResult
                            }
                        }
                    }
            }

            val offline =
                activeEngine
                    .voice
                    ?.isNetworkConnectionRequired !=
                    true

            val available =
                langResult >=
                        TextToSpeech.LANG_AVAILABLE &&
                        offline

            Log.d(
                TAG,
                "availability ${language.code}: " +
                        "setLanguage=$langResult " +
                        "fallback=${activeEngine !== engine} " +
                        "voice=${activeEngine.voice?.name ?: "none"} " +
                        "offline=$offline -> $available"
            )

            result[
                language.code
            ] =
                available
        }

        return result
    }

    override fun speak(
        text: String,
        language: AppLanguage,
    ): Boolean {

        val primary =
            tts

        if (
            primary == null ||
            !ready
        ) {

            listener.onSpeakError(
                initError
                    ?: "TTS is not ready yet"
            )

            return false
        }

        if (
            text.isBlank()
        ) {

            return false
        }

        /*
         * 引擎选择：默认引擎优先；它不支持该语言时，
         * 若已安装 Google TTS（支持俄语等多语言语音包）则 fallback。
         */
        val locale =
            localeFor(
                language
            )

        var engine:
                TextToSpeech =
            primary

        var langResult =
            primary.setLanguage(
                locale
            )

        if (
            langResult <
            TextToSpeech.LANG_AVAILABLE
        ) {

            ensureGoogleFallback()
                ?.let { fallback ->

                    if (
                        fallbackReady
                    ) {

                        val fbResult =
                            fallback.setLanguage(
                                locale
                            )

                        if (
                            fbResult >=
                            TextToSpeech.LANG_AVAILABLE
                        ) {

                            engine =
                                fallback

                            langResult =
                                fbResult
                        }
                    }
                }
        }

        Log.d(
            TAG,
            "speak lang=${language.code} locale=$locale " +
                    "setLanguage=$langResult " +
                    "fallback=${engine !== primary} " +
                    "voice=${engine.voice?.name ?: "none"} " +
                    "network=${engine.voice?.isNetworkConnectionRequired}"
        )

        if (
            langResult ==
            TextToSpeech.LANG_MISSING_DATA ||
            langResult ==
            TextToSpeech.LANG_NOT_SUPPORTED
        ) {

            Log.w(
                TAG,
                "speak rejected: no offline voice for ${language.code}"
            )

            listener.onSpeakError(
                "TTS voice missing/unsupported for " +
                        language.code +
                        " (install an offline voice)"
            )

            return false
        }

        engine.voice
            ?.let { voice ->

                if (
                    voice.isNetworkConnectionRequired
                ) {

                    listener.onSpeakError(
                        "Current voice requires network. " +
                                "Install an offline voice first."
                    )

                    return false
                }
            }

        engine.setSpeechRate(
            speechRate
        )

        engine.setPitch(
            1.0f
        )

        val utteranceId =
            "fourlang_" +
                    utteranceCounter
                        .incrementAndGet()

        val result =
            engine.speak(
                text,
                TextToSpeech.QUEUE_FLUSH,
                null,
                utteranceId,
            )

        if (
            result ==
            TextToSpeech.ERROR
        ) {

            isSpeaking =
                false

            Log.e(
                TAG,
                "speak() returned ERROR for $utteranceId"
            )

            listener.onSpeakError(
                "TTS speak failed"
            )

            return false
        }

        /*
         * 提交成功即视为播放中：gate 从这一刻生效，
         * 不等异步 onStart（避免极端情况下漏帧）。
         */
        isSpeaking =
            true

        Log.d(
            TAG,
            "speak submitted: $utteranceId"
        )

        return true
    }

    override fun stop() {

        isSpeaking =
            false

        tts
            ?.stop()
    }

    override fun release() {

        stop()

        tts
            ?.shutdown()

        tts =
            null

        ready =
            false

        fallbackTts
            ?.shutdown()

        fallbackTts =
            null

        fallbackReady =
            false
    }

    /**
     * 懒创建 Google TTS 实例（fallback 引擎）。
     *
     * 仅当 Google TTS 已安装时可用；init 异步完成，
     * 完成前 fallbackReady == false（本句会走缺失提示，
     * 下一句自动重试即可）。
     */
    private fun ensureGoogleFallback():
            TextToSpeech? {

        fallbackTts
            ?.let {
                return it
            }

        val installed =
            try {

                appContext
                    .packageManager
                    .getPackageInfo(
                        GOOGLE_TTS_PACKAGE,
                        0,
                    )

                true

            } catch (
                _: Throwable
            ) {

                false
            }

        if (
            !installed
        ) {

            Log.d(
                TAG,
                "google tts not installed, skip fallback"
            )

            return null
        }

        val engine =
            TextToSpeech(
                appContext,
                { status ->

                    if (
                        status ==
                        TextToSpeech.SUCCESS
                    ) {

                        Log.d(
                            TAG,
                            "google fallback init OK"
                        )

                        fallbackTts
                            ?.setOnUtteranceProgressListener(
                                progressListener
                            )

                        fallbackReady =
                            true

                    } else {

                        Log.w(
                            TAG,
                            "google fallback init failed: $status"
                        )

                        fallbackReady =
                            false
                    }
                },
                GOOGLE_TTS_PACKAGE,
            )

        fallbackTts =
            engine

        return engine
    }

    private fun localeFor(
        language: AppLanguage,
    ): Locale {

        return when (
            language.code
        ) {

            "zh" ->
                Locale.forLanguageTag(
                    "zh-CN"
                )

            "en" ->
                Locale.forLanguageTag(
                    "en-US"
                )

            "ru" ->
                Locale.forLanguageTag(
                    "ru-RU"
                )

            "uz" ->
                Locale.forLanguageTag(
                    "uz-UZ"
                )

            else ->
                Locale.US
        }
    }

    companion object {

        /*
         * logcat 过滤：adb logcat -s FourLangTTS
         */
        private const val TAG =
            "FourLangTTS"

        /*
         * Google TTS 引擎包名：支持俄语等多语言离线语音包，
         * 作为默认引擎的语言 fallback。
         */
        const val GOOGLE_TTS_PACKAGE =
            "com.google.android.tts"
    }
}
