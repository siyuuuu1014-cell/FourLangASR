package com.example.fourlangasr.tts

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.util.Log
import com.example.fourlangasr.model.AppLanguage
import com.k2fsa.sherpa.onnx.GeneratedAudio
import com.k2fsa.sherpa.onnx.OfflineTts
import com.k2fsa.sherpa.onnx.OfflineTtsConfig
import com.k2fsa.sherpa.onnx.OfflineTtsModelConfig
import com.k2fsa.sherpa.onnx.OfflineTtsVitsModelConfig
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger

/**
 * 离线神经 TTS 引擎（路 B 可行性验证的 Android 实现）。
 *
 * 用与 ASR 相同的 sherpa-onnx 运行时合成语音，不依赖系统 TTS 引擎。
 * 当前内置模型：Piper ru_RU（assets/tts/ru），约 81 MB。
 *
 * 首次 speak 时把 onnx / tokens / espeak-ng-data 从 assets 复制到
 * filesDir（原生层对 espeak 数据目录按文件读取，asset 不支持目录
 * 枚举），之后复用；合成与播放都在后台线程，speak 立即返回。
 *
 * 设计为 tts/TtsManager 的语言 fallback：
 * 系统引擎有 voice 的语言（zh/en）仍走系统；系统没有的（ru）走这里。
 */
class OfflineNeuralTtsEngine(
    context: Context,
    private val listener: TtsEngine.Listener,
) : TtsEngine {

    private val appContext =
        context.applicationContext

    private val executor =
        Executors.newSingleThreadExecutor()

    @Volatile
    override var isSpeaking =
        false
        private set

    @Volatile
    override var speechRate =
        1.0f
        private set

    @Volatile
    private var modelReady =
        false

    private var loadError:
            String? =
        null

    private var tts:
            OfflineTts? =
        null

    private var audioTrack:
            AudioTrack? =
        null

    /*
     * 代际计数：speak 新句子会让旧任务过期，
     * 旧任务合成/播放前检查代际后丢弃。
     */
    private val generation =
        AtomicInteger(0)

    override val isReady: Boolean
        get() = modelReady

    /**
     * 本引擎内置模型支持的语言（当前仅 ru）。
     */
    fun supportsLanguage(
        language: AppLanguage,
    ): Boolean {

        return language.code ==
                "ru"
    }

    override fun speak(
        text: String,
        language: AppLanguage,
    ): Boolean {

        if (
            !supportsLanguage(
                language
            )
        ) {

            listener.onSpeakError(
                "Offline engine has no voice for " +
                        language.code
            )

            return false
        }

        if (
            text.isBlank()
        ) {

            return false
        }

        /*
         * 打断上一句（QUEUE_FLUSH 语义），gate 立即生效。
         */
        stopPlayback()

        val myGeneration =
            generation.incrementAndGet()

        isSpeaking =
            true

        executor.execute {

            try {

                val engine =
                    ensureLoaded()

                if (
                    engine == null
                ) {

                    isSpeaking =
                        false

                    listener.onSpeakError(
                        loadError
                            ?: "Offline TTS load failed"
                    )

                    return@execute
                }

                if (
                    myGeneration !=
                    generation.get()
                ) {

                    return@execute
                }

                val audio =
                    engine.generate(
                        text,
                        0,
                        speechRate,
                    )

                if (
                    myGeneration !=
                    generation.get()
                ) {

                    return@execute
                }

                play(
                    audio,
                    myGeneration,
                )

            } catch (
                e: Throwable
            ) {

                Log.w(
                    TAG,
                    "synthesis failed: ${e.message}"
                )

                isSpeaking =
                    false

                listener.onSpeakError(
                    "Offline TTS: ${e.message}"
                )
            }
        }

        return true
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

        return mapOf(
            "ru" to true,
        )
    }

    override fun stop() {

        generation.incrementAndGet()
        stopPlayback()
    }

    override fun release() {

        stop()

        executor.shutdownNow()

        tts =
            null

        modelReady =
            false
    }

    /**
     * 懒加载：assets -> filesDir 复制（仅首次），再建 OfflineTts。
     */
    private fun ensureLoaded():
            OfflineTts? {

        if (
            modelReady
        ) {

            return tts
        }

        try {

            val modelDir =
                ensureModelFiles()

            val config =
                OfflineTtsConfig(
                    model =
                        OfflineTtsModelConfig(
                            vits =
                                OfflineTtsVitsModelConfig(
                                    model =
                                        File(
                                            modelDir,
                                            MODEL_ONNX,
                                        )
                                            .absolutePath,
                                    tokens =
                                        File(
                                            modelDir,
                                            MODEL_TOKENS,
                                        )
                                            .absolutePath,
                                    dataDir =
                                        File(
                                            modelDir,
                                            ESPEAK_DIR,
                                        )
                                            .absolutePath,
                                ),
                            numThreads = 2,
                            debug = false,
                            provider = "cpu",
                        ),
                )

            tts =
                OfflineTts(
                    appContext.assets,
                    config,
                )

            Log.d(
                TAG,
                "loaded: sampleRate=${tts?.sampleRate()}"
            )

            modelReady =
                true

            return tts

        } catch (
            e: Throwable
        ) {

            loadError =
                e.message

            Log.e(
                TAG,
                "load failed: ${e.message}",
                e,
            )

            return null
        }
    }

    private fun ensureModelFiles():
            File {

        val modelDir =
            File(
                appContext.filesDir,
                FILES_SUBDIR,
            )

        if (
            File(
                modelDir,
                MODEL_ONNX,
            ).isFile
        ) {

            return modelDir
        }

        check(
            modelDir.mkdirs() ||
                    modelDir.isDirectory
        ) {
            "Unable to create ${modelDir.absolutePath}"
        }

        copyAsset(
            ASSET_ROOT + MODEL_ONNX,
            File(
                modelDir,
                MODEL_ONNX,
            ),
        )

        copyAsset(
            ASSET_ROOT + MODEL_TOKENS,
            File(
                modelDir,
                MODEL_TOKENS,
            ),
        )

        val espeakDir =
            File(
                modelDir,
                ESPEAK_DIR,
            )

        /*
         * espeak-ng-data 是深目录（lang/、voices/ 等数百文件），
         * 必须递归复制；只拷第一层会导致 espeak 加载失败。
         */
        copyAssetDir(
            ASSET_ROOT + ESPEAK_DIR,
            espeakDir,
        )

        Log.d(
            TAG,
            "model files ready in ${modelDir.absolutePath}"
        )

        return modelDir
    }

    /**
     * 递归复制 assets 目录到 filesDir。
     */
    private fun copyAssetDir(
        assetDir: String,
        targetDir: File,
    ) {

        val children =
            appContext
                .assets
                .list(
                    assetDir
                )

        if (
            children == null
        ) {

            return
        }

        check(
            targetDir.mkdirs() ||
                    targetDir.isDirectory
        ) {
            "Unable to create ${targetDir.absolutePath}"
        }

        for (
        name
        in children
        ) {

            val assetChild =
                "$assetDir/$name"

            val fileChild =
                File(
                    targetDir,
                    name,
                )

            /*
             * AssetManager.list 对目录返回子项数组，对文件返回 null。
             */
            val grandchildren =
                appContext
                    .assets
                    .list(
                        assetChild
                    )

            if (
                grandchildren ==
                null
            ) {

                copyAsset(
                    assetChild,
                    fileChild,
                )

            } else if (
                grandchildren.isNotEmpty()
            ) {

                copyAssetDir(
                    assetChild,
                    fileChild,
                )
            }
        }
    }

    private fun copyAsset(
        assetPath: String,
        target: File,
    ) {

        appContext
            .assets
            .open(
                assetPath
            )
            .use { input ->

                target
                    .outputStream()
                    .use { output ->

                        input.copyTo(
                            output,
                            bufferSize =
                                256 * 1024,
                        )
                    }
            }
    }

    /**
     * 播放合成音频（阻塞至播完或被打断）。
     */
    private fun play(
        audio: GeneratedAudio,
        myGeneration: Int,
    ) {

        val samples =
            audio.samples

        val sampleRate =
            audio.sampleRate

        if (
            samples.isEmpty()
        ) {

            isSpeaking =
                false

            listener.onSpeakDone()

            return
        }

        val track =
            AudioTrack.Builder()
                .setAudioAttributes(
                    AudioAttributes
                        .Builder()
                        .setUsage(
                            AudioAttributes
                                .USAGE_MEDIA
                        )
                        .setContentType(
                            AudioAttributes
                                .CONTENT_TYPE_SPEECH
                        )
                        .build()
                )
                .setAudioFormat(
                    AudioFormat
                        .Builder()
                        .setSampleRate(
                            sampleRate
                        )
                        .setEncoding(
                            AudioFormat
                                .ENCODING_PCM_FLOAT
                        )
                        .setChannelMask(
                            AudioFormat
                                .CHANNEL_OUT_MONO
                        )
                        .build()
                )
                .setTransferMode(
                    AudioTrack
                        .MODE_STATIC
                )
                .setBufferSizeInBytes(
                    samples.size *
                            4
                )
                .build()

        audioTrack =
            track

        try {

            track.write(
                samples,
                0,
                samples.size,
                AudioTrack.WRITE_BLOCKING,
            )

            if (
                myGeneration !=
                generation.get()
            ) {

                return
            }

            track.play()

            while (
                track.playState ==
                AudioTrack.PLAYSTATE_PLAYING &&
                myGeneration ==
                generation.get()
            ) {

                Thread.sleep(
                    50
                )
            }

        } finally {

            /*
             * stopPlayback() 可能已释放本 track：
             * 只在仍归属本引擎时释放，避免二次 release。
             */
            if (
                audioTrack ===
                track
            ) {

                audioTrack =
                    null

                try {

                    track.release()

                } catch (_: Throwable) {
                }
            }
        }

        if (
            myGeneration ==
            generation.get()
        ) {

            isSpeaking =
                false

            listener.onSpeakDone()
        }
    }

    private fun stopPlayback() {

        audioTrack
            ?.let { track ->

                try {

                    track.stop()

                } catch (_: Throwable) {
                }

                track.release()
            }

        audioTrack =
            null

        isSpeaking =
            false
    }

    companion object {

        private const val TAG =
            "FourLangTTS"

        private const val ASSET_ROOT =
            "tts/ru/"

        private const val FILES_SUBDIR =
            "tts_ru"

        private const val MODEL_ONNX =
            "ru_RU-irina-medium.onnx"

        private const val MODEL_TOKENS =
            "tokens.txt"

        private const val ESPEAK_DIR =
            "espeak-ng-data"
    }
}
