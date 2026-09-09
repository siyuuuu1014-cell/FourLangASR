package com.example.fourlangasr.tts

import com.example.fourlangasr.model.AppLanguage

/**
 * TTS 抽象：post-ASR pipeline 只依赖本接口，
 * 不知道底层是 Android 系统 TTS 还是未来的离线神经 TTS。
 *
 * 生命周期事件通过 [Listener] 上报（回调线程不保证是主线程，
 * 实现方负责线程切换）。[isSpeaking] 供麦克风侧做防回灌 gate。
 */
interface TtsEngine {

    val isReady: Boolean

    /**
     * 当前是否有语音正在播放。
     *
     * @Volatile 语义：由 AudioRecord 回调线程读取，
     * 由 TTS 回调线程写入。
     */
    val isSpeaking: Boolean

    /**
     * 朗读一段文本。
     *
     * @return true = 已提交播放；false = 失败
     * （未初始化 / 语言无离线 voice / 需要网络），
     * 失败原因通过 [Listener.onSpeakError] 上报。
     */
    fun speak(
        text: String,
        language: AppLanguage,
    ): Boolean

    /**
     * 设置朗读语速（0.5 ~ 2.0，默认 1.0）。
     * 对后续 speak 生效。
     */
    fun setSpeechRate(
        rate: Float,
    )

    /**
     * 当前语速。
     */
    val speechRate: Float

    /**
     * 系统 voice 可用性预检：AppLanguage.code -> 是否存在
     * 不依赖网络的离线 voice。未初始化完成时返回空 map。
     */
    fun voiceAvailability():
            Map<String, Boolean>

    /**
     * 停止当前播放并复位状态。
     */
    fun stop()

    /**
     * 停止并释放底层引擎（shutdown）。
     * 之后不能再 speak。
     */
    fun release()

    interface Listener {

        fun onSpeakStart()

        fun onSpeakDone()

        fun onSpeakError(
            reason: String,
        )

        /**
         * 底层引擎异步初始化完成，附当时的 voice 可用性。
         */
        fun onEngineReady(
            availability:
                Map<String, Boolean>,
        )
    }
}
