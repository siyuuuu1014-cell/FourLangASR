package com.example.fourlangasr.vad

import android.content.res.AssetManager
import com.k2fsa.sherpa.onnx.SileroVadModelConfig
import com.k2fsa.sherpa.onnx.Vad
import com.k2fsa.sherpa.onnx.VadModelConfig

class VadEngine(
    assetManager: AssetManager
) {

    companion object {

        const val SAMPLE_RATE = 16000

        /*
         * 512 / 16000
         * = 0.032 秒
         * = 32 ms
         */
        const val WINDOW_SIZE = 512

        /*
         * Silero VAD 参数。
         * MainActivity 的指标面板直接引用这些常量。
         */
        const val SPEECH_PROB_THRESHOLD = 0.5F

        const val END_SILENCE_SEC = 0.6F

        const val MIN_SPEECH_SEC = 0.25F

        const val MAX_SPEECH_SEC = 15.0F
    }

    private val vad: Vad

    init {

        val sileroConfig =
            SileroVadModelConfig(
                model = "silero_vad.onnx",

                /*
                 * 语音概率阈值
                 */
                threshold = SPEECH_PROB_THRESHOLD,

                /*
                 * 连续静音约 0.6 秒：
                 * 判断当前一句结束。
                 */
                minSilenceDuration = END_SILENCE_SEC,

                /*
                 * 小于约 0.25 秒的声音，
                 * 不作为有效语音。
                 */
                minSpeechDuration = MIN_SPEECH_SEC,

                windowSize = WINDOW_SIZE,

                /*
                 * 防止一句话无限增长。
                 */
                maxSpeechDuration = MAX_SPEECH_SEC,
            )

        val config =
            VadModelConfig(
                sileroVadModelConfig = sileroConfig,
                sampleRate = SAMPLE_RATE,
                numThreads = 1,
                provider = "cpu",
                debug = false,
            )

        vad =
            Vad(
                assetManager = assetManager,
                config = config,
            )
    }

    fun accept(samples: FloatArray) {

        vad.acceptWaveform(samples)
    }

    fun isSpeechDetected(): Boolean {

        return vad.isSpeechDetected()
    }

    fun hasSegment(): Boolean {

        return !vad.empty()
    }

    /**
     * 取得一整个 VAD 语音段。
     */
    fun takeSegment(): FloatArray? {

        if (vad.empty()) {
            return null
        }

        val segment = vad.front()

        /*
         * copyOf()：
         * 先复制 Java/Kotlin 侧数组，
         * 再 pop native segment，
         * 更安全。
         */
        val samples =
            segment.samples.copyOf()

        vad.pop()

        return samples
    }

    /**
     * 强制输出 VAD 内还没有结束的语音。
     */
    fun flush() {

        vad.flush()
    }

    fun reset() {

        vad.reset()
    }

    fun release() {

        vad.release()
    }
}