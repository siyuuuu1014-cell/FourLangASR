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
 * 当前内置模型：Piper ru_RU（assets/tts/ru），约 81 MB，
 * 以 asset 相对路径直接加载（与 sherpa-onnx 官方 Android
 * Piper TTS 引擎 APK 的用法一致）。
 *
 * 合成与播放都在后台线程，speak 立即返回。
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
         * 打断上一句（QUEUE_FLUSH 语义）：只递增代际，
         * 播放线程的循环会在 50ms 内自行退出并释放自己的
         * AudioTrack——AudioTrack 不是线程安全的，绝不在
         * 其它线程直接 stop/release（曾导致 destroyed-mutex
         * 原生崩溃）。
         */
        generation.incrementAndGet()

        val myGeneration =
            generation.get()

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

        /*
         * 只改代际：播放循环自行退出并释放 AudioTrack。
         * isSpeaking 立即复位，麦克风 gate 马上生效。
         */
        generation.incrementAndGet()

        isSpeaking =
            false
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
     * 预加载模型（幂等）。
     *
     * 在对话开始前调用（无 VAD/ASR 等 sherpa native 并发活动），
     * 把模型加载从"识别完成后的 speak 路径"挪出，
     * 规避加载期与流式识别并发的原生崩溃窗口。
     */
    fun preload() {

        executor.execute {

            ensureLoaded()
        }
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

        /*
         * 完整 = onnx + 关键 G2P 文件 + 版本标记一致。
         * 旧版本 APK 可能已复制过不完整/旧结构的 espeak-ng-data，
         * 检测到不完整就整体删除重拷，避免 native 层崩溃。
         */
        val versionFile =
            File(
                modelDir,
                VERSION_FILE,
            )

        val espeakCheck =
            File(
                modelDir,
                ESPEAK_DIR,
            )

        val complete =
            File(
                modelDir,
                MODEL_ONNX,
            ).isFile &&
                    File(
                        espeakCheck,
                        "phondata",
                    ).isFile &&
                    versionFile
                        .takeIf {
                            it.isFile
                        }
                        ?.readText()
                        ?.trim() ==
                    VERSION

        if (
            complete
        ) {

            return modelDir
        }

        Log.d(
            TAG,
            "model files incomplete or outdated, re-copying"
        )

        modelDir
            .deleteRecursively()

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

        /*
         * 加载前校验关键 G2P 数据文件存在；
         * 缺失时给出明确错误而不是让 native 层崩溃。
         */
        val required =
            listOf(
                "phondata",
                "phonindex",
                "phontab",
                "intonations",
                "ru_dict",
            )

        val missing =
            required.filter { name ->

                !File(
                    espeakDir,
                    name,
                ).isFile
            }

        if (
            missing.isNotEmpty()
        ) {

            throw IllegalStateException(
                "espeak-ng-data incomplete, missing: " +
                        missing
            )
        }

        /*
         * 复制完整后写入版本标记。
         */
        versionFile.writeText(
            VERSION
        )

        Log.d(
            TAG,
            "model files ready in ${modelDir.absolutePath}"
        )

        return modelDir
    }

    /**
     * 递归复制 assets 目录到 filesDir。
     *
     * AssetManager.list 对文件路径的返回值在不同版本语义不一
     * （null 或空数组都可能），两种都按文件尝试复制；
     * 对目录递归。复制失败只记日志，由后续关键文件校验兜底。
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

            val grandchildren =
                appContext
                    .assets
                    .list(
                        assetChild
                    )

            val isDirectory =
                grandchildren !=
                        null &&
                        grandchildren
                            .isNotEmpty()

            if (
                isDirectory
            ) {

                copyAssetDir(
                    assetChild,
                    fileChild,
                )

            } else {

                /*
                 * null = 文件（标准语义）；空数组 = 部分版本的
                 * 文件语义，同样按文件复制（open 失败则跳过，
                 * 关键文件由 ensureModelFiles 校验）。
                 */
                try {

                    copyAsset(
                        assetChild,
                        fileChild,
                    )

                } catch (
                    e: Throwable
                ) {

                    Log.w(
                        TAG,
                        "copy failed (may be empty dir): " +
                                "$assetChild - ${e.message}"
                    )
                }
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

                try {

                    Thread.sleep(
                        50
                    )

                } catch (
                    _: InterruptedException
                ) {

                    break
                }
            }

        } finally {

            /*
             * AudioTrack 只在播放线程释放（单线程约束）。
             */
            try {

                track.release()

            } catch (_: Throwable) {
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

    companion object {

        private const val TAG =
            "FourLangTTS"

        private const val ASSET_ROOT =
            "tts/ru/"

        private const val MODEL_ONNX =
            "ru_RU-irina-medium.onnx"

        private const val MODEL_TOKENS =
            "tokens.txt"

        private const val ESPEAK_DIR =
            "espeak-ng-data"

        /*
         * filesDir 模型副本的版本标记：
         * 资产或复制逻辑变化时递增，强制重拷。
         */
        private const val FILES_SUBDIR =
            "tts_ru"

        private const val VERSION =
            "2"

        private const val VERSION_FILE =
            ".version"
    }
}
