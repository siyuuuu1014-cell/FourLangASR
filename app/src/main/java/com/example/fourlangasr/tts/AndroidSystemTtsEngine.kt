package com.example.fourlangasr.tts

import android.content.Context
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import com.example.fourlangasr.model.AppLanguage
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap
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
    private val createEngine: (Context, TextToSpeech.OnInitListener, String?) -> TextToSpeech =
        { appContext, onInit, packageName ->
            if (packageName == null) TextToSpeech(appContext, onInit)
            else TextToSpeech(appContext, onInit, packageName)
        },
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

    private val playbackLock = Any()
    private var activeUtteranceId: String? = null

    @Volatile
    private var released = false

    private fun finishUtterance(utteranceId: String?): Boolean = synchronized(playbackLock) {
        if (utteranceId == null || utteranceId != activeUtteranceId) return@synchronized false
        activeUtteranceId = null
        isSpeaking = false
        cancelWatchdog()
        true
    }

    /*
     * 看门狗：部分 vendor 引擎在异常路径上一个回调都不给
     * （既不 onStart，也不 onDone/onStop），isSpeaking 会永久停在 true，
     * MainActivity 的麦克风控门就此一直静音，只能靠手动"停止朗读"救回。
     */
    private val watchdogHandler =
        android.os.Handler(android.os.Looper.getMainLooper())

    private var watchdogTask: Runnable? = null

    private fun cancelWatchdog() {
        watchdogTask?.let { watchdogHandler.removeCallbacks(it) }
        watchdogTask = null
    }

    private fun scheduleWatchdog(utteranceId: String, textLength: Int) {
        cancelWatchdog()

        // 约 250ms/字、15s 起步：宁可宽估，也不要在正常播报中途插队。
        val timeoutMs = 15_000L + textLength * 250L

        val task = Runnable {
            if (finishUtterance(utteranceId)) {
                Log.w(
                    TAG,
                    "watchdog: $utteranceId ${timeoutMs}ms 内无任何回调，按播完收尾",
                )
                listener.onSpeakDone()
            }
        }

        watchdogTask = task

        watchdogHandler.postDelayed(
            task,
            timeoutMs,
        )
    }

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

                synchronized(playbackLock) {
                    if (utteranceId == null || utteranceId != activeUtteranceId) return
                    isSpeaking = true
                }

                Log.d(
                    TAG,
                    "onStart: $utteranceId"
                )

                listener.onSpeakStart()
            }

            override fun onDone(
                utteranceId: String?
            ) {

                if (!finishUtterance(utteranceId)) return

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

                finishUtterance(utteranceId)
            }

            override fun onError(
                utteranceId: String?
            ) {

                if (!finishUtterance(utteranceId)) return

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

                if (!finishUtterance(utteranceId)) return

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
     * Fallback 引擎池（懒创建，按包名缓存）：
     * 系统默认引擎缺少某语言语音（如俄语）时，
     * 依次尝试手机里所有已安装的 TTS 引擎
     * （Google TTS、讯飞等自带多语言 voice 的引擎）。
     */
    private val fallbackEngines =
        ConcurrentHashMap<
                String,
                FallbackEngine
                >()

    /**
     * 一个 fallback 引擎实例（init 异步完成，ready 前不可用）。
     */
    private class FallbackEngine(
        val tts: TextToSpeech,
    ) {

        @Volatile
        var ready =
            false
    }

    override val isReady: Boolean
        get() = ready

    init {
        tts =
            createEngine(
                appContext,
                TextToSpeech.OnInitListener { status ->
                    if (released) return@OnInitListener

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

                        return@OnInitListener
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
                },
                null,
            )
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

            val (activeEngine, langResult) =
                resolveEngineForLocale(
                    primary = engine,
                    locale = locale,
                )

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

    @Synchronized
    override fun speak(
        text: String,
        language: AppLanguage,
    ): Boolean {

        val primary =
            tts

        if (
            released || primary == null ||
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
         * 自动尝试所有已安装的其它 TTS 引擎
         * （Google TTS / 讯飞等自带多语言 voice 的引擎）。
         */
        val locale =
            localeFor(
                language
            )

        val (engine, langResult) =
            resolveEngineForLocale(
                primary = primary,
                locale = locale,
            )

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

        // 在提交前登记 ID，旧任务的 onStop/onDone 不得清掉新任务状态。
        synchronized(playbackLock) {
            activeUtteranceId = utteranceId
            isSpeaking = true
        }
        val result = try {
            engine.speak(
                text,
                TextToSpeech.QUEUE_FLUSH,
                null,
                utteranceId,
            )
        } catch (e: Exception) {
            finishUtterance(utteranceId)
            listener.onSpeakError("TTS speak failed: ${e.message}")
            return false
        }

        if (
            result ==
            TextToSpeech.ERROR
        ) {

            finishUtterance(utteranceId)

            Log.e(
                TAG,
                "speak() returned ERROR for $utteranceId"
            )

            listener.onSpeakError(
                "TTS speak failed"
            )

            return false
        }

        Log.d(
            TAG,
            "speak submitted: $utteranceId"
        )

        scheduleWatchdog(utteranceId, text.length)

        return true
    }

    @Synchronized
    override fun stop() {
        cancelWatchdog()
        synchronized(playbackLock) {
            activeUtteranceId = null
            isSpeaking = false
        }
        // 当前播放可能来自任意备用引擎。先作废回调，再停止所有实例。
        val engines = listOfNotNull(tts) + fallbackEngines.values.map { it.tts }
        for (engine in engines) {
            try {
                engine.stop()
            } catch (e: Exception) {
                Log.w(TAG, "TTS stop failed", e)
            }
        }
    }

    @Synchronized
    override fun release() {
        released = true

        stop()

        tts
            ?.shutdown()

        tts =
            null

        ready =
            false

        synchronized(
            fallbackEngines
        ) {

            fallbackEngines
                .values
                .forEach {
                    engine ->

                    engine
                        .tts
                        .shutdown()
                }

            fallbackEngines.clear()
        }
    }

    /**
     * 为某个 Locale 选择可用的引擎。
     *
     * @return (引擎, setLanguage 结果)；结果 >= 0 才可用。
     */
    private fun resolveEngineForLocale(
        primary: TextToSpeech,
        locale: Locale,
    ): Pair<TextToSpeech, Int> {

        var engine =
            primary

        var result =
            primary.setLanguage(
                locale
            )

        if (
            result >=
            TextToSpeech.LANG_AVAILABLE
        ) {

            return engine to result
        }

        /*
         * 默认引擎不支持该语言：逐个尝试已安装的其它 TTS 引擎
         * （Google TTS / 讯飞等自带多语言 voice 的引擎）。
         * init 是异步的，本句可能全部未 ready 而失败，
         * 下一句会自动重试。
         */
        for (
        fallback
        in ensureFallbackEngines()
        ) {

            if (
                !fallback.ready
            ) {

                continue
            }

            val fbResult =
                fallback
                    .tts
                    .setLanguage(
                        locale
                    )

            if (
                fbResult >=
                TextToSpeech.LANG_AVAILABLE
            ) {

                Log.d(
                    TAG,
                    "fallback engine picked: " +
                            fallback.tts.voice
                                ?.name
                                ?: "?"
                )

                return fallback.tts to fbResult
            }
        }

        return engine to result
    }

    /**
     * 懒创建所有已安装的其它 TTS 引擎（排除系统默认引擎）。
     */
    @Synchronized
    private fun ensureFallbackEngines():
            List<FallbackEngine> {

        if (released) return emptyList()

        val primary =
            tts
                ?: return emptyList()

        val defaultEngine =
            primary
                .getDefaultEngine()

        val installed =
            primary
                .getEngines()

        for (
        info
        in installed
        ) {

            val pkg =
                info.name

            if (
                pkg ==
                defaultEngine ||
                fallbackEngines.containsKey(
                    pkg
                )
            ) {

                continue
            }

            val engine =
                createEngine(
                    appContext,
                    TextToSpeech.OnInitListener { status ->
                        if (released) return@OnInitListener

                        val fb =
                            fallbackEngines[
                                pkg
                            ]
                                ?: return@OnInitListener

                        if (
                            status ==
                            TextToSpeech.SUCCESS
                        ) {

                            Log.d(
                                TAG,
                                "fallback engine ready: $pkg " +
                                        "(${info.label})"
                            )

                            fb
                                .tts
                                .setOnUtteranceProgressListener(
                                    progressListener
                                )

                            fb.ready =
                                true

                        } else {

                            Log.w(
                                TAG,
                                "fallback engine init failed: " +
                                        "$pkg status=$status"
                            )

                            fb.ready =
                                false
                        }
                    },
                    pkg,
                )

            fallbackEngines[
                pkg
            ] =
                FallbackEngine(
                    engine
                )

            Log.d(
                TAG,
                "fallback engine creating: $pkg (${info.label})"
            )
        }

        synchronized(
            fallbackEngines
        ) {

            return fallbackEngines
                .values
                .toList()
        }
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
