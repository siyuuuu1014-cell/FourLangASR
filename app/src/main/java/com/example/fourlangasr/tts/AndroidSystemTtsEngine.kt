package com.example.fourlangasr.tts

import android.content.Context
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
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

                    return@TextToSpeech
                }

                engine.setOnUtteranceProgressListener(
                    object :
                        UtteranceProgressListener() {

                        override fun onStart(
                            utteranceId: String?
                        ) {

                            isSpeaking =
                                true

                            listener.onSpeakStart()
                        }

                        override fun onDone(
                            utteranceId: String?
                        ) {

                            isSpeaking =
                                false

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

                            listener.onSpeakError(
                                "TTS playback error ($errorCode)"
                            )
                        }
                    }
                )

                ready =
                    true

                initError =
                    null
            }
    }

    override fun speak(
        text: String,
        language: AppLanguage,
    ): Boolean {

        val engine =
            tts

        if (
            engine == null ||
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
         * 切换 voice 并检查离线可用性。
         */
        val locale =
            localeFor(
                language
            )

        val langResult =
            engine.setLanguage(
                locale
            )

        if (
            langResult ==
            TextToSpeech.LANG_MISSING_DATA ||
            langResult ==
            TextToSpeech.LANG_NOT_SUPPORTED
        ) {

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
            1.0f
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
}
