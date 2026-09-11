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
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger

/**
 * 离线神经 TTS 引擎（路 B 可行性验证的 Android 实现）。
 *
 * 用与 ASR 相同的 sherpa-onnx 运行时合成语音，不依赖系统 TTS 引擎。
 * 按语言挂载内置模型（MODELS 表）：
 *   ru -> Piper irina-medium（espeak G2P 数据）
 *
 * 模型从 assets 复制到 filesDir（espeak 数据需真实未压缩文件，
 * aapt 会压缩无扩展名的 asset），每次复制带版本标记与关键文件
 * 校验。加载 OfflineTts 时 assetManager 必须传 null
 * （sherpa file-utils.cc 的显式要求，否则 FATAL abort）。
 *
 * 合成与播放都在单一后台线程；打断只改代际计数，
 * 播放循环自行退出并释放自己的 AudioTrack（线程安全约束）。
 *
 * 设计为 tts/TtsManager 的语言 fallback：
 * 系统引擎有 voice 的语言（zh/en）仍走系统；系统没有的走这里。
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

    private val engines =
        ConcurrentHashMap<
                String,
                OfflineTts
                >()

    @Volatile
    private var loadError:
            String? =
        null

    /*
     * 代际计数：speak 新句子会让旧任务过期，
     * 旧任务合成/播放前检查代际后丢弃。
     */
    private val generation =
        AtomicInteger(0)

    override val isReady: Boolean
        get() = engines.isNotEmpty()

    /**
     * 本引擎内置模型支持的语言（ru / uz）。
     */
    fun supportsLanguage(
        language: AppLanguage,
    ): Boolean {

        return MODELS.containsKey(
            language.code
        )
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
                    ensureLoaded(
                        language.code
                    )

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

        return MODELS
            .keys
            .associateWith {
                true
            }
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

        engines.clear()
    }

    /**
     * 预加载 ru 模型（幂等）。
     *
     * 在对话开始前调用，把最常用模型的加载从
     * "识别完成后的 speak 路径"挪出。uz 保持懒加载
     * （114MB，只在真正说乌兹别克语时占用内存）。
     */
    fun preload() {

        executor.execute {

            ensureLoaded(
                "ru"
            )
        }
    }

    /**
     * 按语言懒加载：assets -> filesDir 复制（仅首次），
     * 再建 OfflineTts 并缓存。
     */
    private fun ensureLoaded(
        code: String,
    ): OfflineTts? {

        engines[
            code
        ]
            ?.let {
                return it
            }

        val spec =
            MODELS[
                code
            ]
                ?: return null

        /*
         * 内存防御：同一时刻只驻留一个离线 TTS 模型
         * （ru 63+18MB / uz 114MB）。加载另一个语言的模型前
         * 释放当前模型，避免与识别管线叠加触发 native OOM
         * 或 LMK。代价是双语交替时重新加载（数秒）。
         */
        if (
            engines.isNotEmpty()
        ) {

            Log.d(
                TAG,
                "unloading ${engines.keys} " +
                        "before loading $code"
            )

            /*
             * 注意：这里绝不能动 generation 计数器！
             * speak() 已把当前代际记为 myGeneration，
             * 若此处递增，后面 myGeneration != generation
             * 的检查会把待播任务自我作废（曾导致"模型加载了
             * 但永远不朗读"）。卸载在单线程 executor 上执行，
             * 不可能与 play() 并发，无需靠代际打断。
             */
            engines.forEach {
                (oldCode, engine) ->

                Log.d(
                    TAG,
                    "unload: release[$oldCode] start"
                )

                engine
                    .release()

                Log.d(
                    TAG,
                    "unload: release[$oldCode] done"
                )
            }

            engines.clear()

            Log.d(
                TAG,
                "unload: map cleared"
            )
        }

        try {

            Log.d(
                TAG,
                "load[$code]: ensureModelFiles start"
            )

            val modelDir =
                ensureModelFiles(
                    spec
                )

            Log.d(
                TAG,
                "load[$code]: model files OK, " +
                        "building recognizer"
            )

            val config =
                OfflineTtsConfig(
                    model =
                        OfflineTtsModelConfig(
                            vits =
                                OfflineTtsVitsModelConfig(
                                    model =
                                        File(
                                            modelDir,
                                            spec.modelName,
                                        )
                                            .absolutePath,
                                    tokens =
                                        File(
                                            modelDir,
                                            spec.tokensName,
                                        )
                                            .absolutePath,
                                    dataDir =
                                        spec
                                            .dataDirName
                                            ?.let {
                                                File(
                                                    modelDir,
                                                    it,
                                                )
                                                    .absolutePath
                                            }
                                            ?: "",
                                ),
                            numThreads = 2,
                            debug = false,
                            provider = "cpu",
                        ),
                )

            val engine =
                OfflineTts(
                    /*
                     * 关键：从文件系统（filesDir 副本）加载模型时
                     * assetManager 必须传 null——sherpa 会按 asset
                     * 相对路径解析非空 assetManager 下的所有路径，
                     * 绝对路径加载会直接 FATAL abort。
                     * （sherpa-onnx file-utils.cc 的显式要求）
                     */
                    null,
                    config,
                )

            Log.d(
                TAG,
                "loaded[$code]: sampleRate=${engine.sampleRate()}"
            )

            engines[
                code
            ] =
                engine

            return engine

        } catch (
            e: Throwable
        ) {

            loadError =
                e.message

            Log.e(
                TAG,
                "load failed[$code]: ${e.message}",
                e,
            )

            return null
        }
    }

    private fun ensureModelFiles(
        spec: ModelSpec,
    ): File {

        val modelDir =
            File(
                appContext.filesDir,
                spec.filesSubdir,
            )

        val versionFile =
            File(
                modelDir,
                VERSION_FILE,
            )

        /*
         * 完整 = onnx + tokens +（有 G2P 数据的模型）关键
         * espeak 文件 + 版本标记一致。
         * 旧版本 APK 可能已复制过不完整/旧结构的数据，
         * 检测到不完整就整体删除重拷，避免 native 层崩溃。
         */
        val complete =
            File(
                modelDir,
                spec.modelName,
            ).isFile &&
                    File(
                        modelDir,
                        spec.tokensName,
                    ).isFile &&
                    (
                        spec.dataDirName ==
                            null ||
                            File(
                                File(
                                    modelDir,
                                    spec.dataDirName,
                                ),
                                "phondata",
                            ).isFile
                        ) &&
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
            "model files incomplete or outdated " +
                    "for ${spec.code}, re-copying"
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
            spec.assetRoot +
                    spec.modelName,
            File(
                modelDir,
                spec.modelName,
            ),
        )

        copyAsset(
            spec.assetRoot +
                    spec.tokensName,
            File(
                modelDir,
                spec.tokensName,
            ),
        )

        if (
            spec.dataDirName !=
            null
        ) {

            val dataDir =
                File(
                    modelDir,
                    spec.dataDirName,
                )

            /*
             * espeak-ng-data 是深目录（lang/、voices/ 等数百
             * 文件），必须递归复制；只拷第一层会导致加载失败。
             */
            copyAssetDir(
                spec.assetRoot +
                        spec.dataDirName,
                dataDir,
            )

            if (
                !File(
                    dataDir,
                    "phondata",
                ).isFile
            ) {

                throw IllegalStateException(
                    "G2P data incomplete for ${spec.code}"
                )
            }
        }

        versionFile.writeText(
            VERSION
        )

        Log.d(
            TAG,
            "model files ready[${spec.code}] in " +
                    modelDir.absolutePath
        )

        return modelDir
    }

    /**
     * 递归复制 assets 目录到 filesDir。
     *
     * AssetManager.list 对文件路径的返回值在不同版本语义不一
     * （null 或空数组都可能），两种都按文件尝试复制；
     * 对目录递归。复制失败只记日志，由关键文件校验兜底。
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

            /*
             * 实际开始出声时上报 onSpeakStart：
             * 与系统引擎的异步 onStart 对齐，
             * 状态栏从"继续监听…"切换为"TTS 播放中…"。
             */
            listener.onSpeakStart()

            /*
             * 完成判定不能依赖 playState 轮询：
             * 部分设备上 MODE_STATIC 播完后 playState 不跳变，
             * 曾导致循环永不退出（状态卡"播放中"、
             * isSpeaking 卡 true、VAD gate 永久锁死）。
             * 改用 marker 通知 + 超时兜底。
             */
            val done =
                java.util.concurrent.atomic
                    .AtomicBoolean(
                        false
                    )

            track.setNotificationMarkerPosition(
                samples.size - 1
            )

            track.setPlaybackPositionUpdateListener(
                object :
                    AudioTrack
                        .OnPlaybackPositionUpdateListener {

                    override fun onMarkerReached(
                        track: AudioTrack
                    ) {

                        done.set(
                            true
                        )
                    }

                    override fun onPeriodicNotification(
                        track: AudioTrack
                    ) {
                    }
                }
            )

            track.play()

            val timeoutMs =
                (
                    samples.size *
                            1000L /
                            sampleRate
                    ) +
                        1500L

            val deadline =
                System.currentTimeMillis() +
                        timeoutMs

            while (
                !done.get() &&
                myGeneration ==
                generation.get() &&
                System.currentTimeMillis() <
                deadline
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

            Log.d(
                TAG,
                "offline onSpeakDone"
            )

            listener.onSpeakDone()
        }
    }

    companion object {

        private const val TAG =
            "FourLangTTS"

        private const val VERSION =
            "4"

        private const val VERSION_FILE =
            ".version"

        /**
         * 一个内置 TTS 模型的描述。
         */
        private data class ModelSpec(
            val code: String,
            val assetRoot: String,
            val modelName: String,
            val tokensName: String,
            val dataDirName: String?,
            val filesSubdir: String,
        )

        private val MODELS =
            mapOf(
                "ru" to
                    ModelSpec(
                        code = "ru",
                        assetRoot = "tts/ru/",
                        modelName =
                            "ru_RU-irina-medium.onnx",
                        tokensName = "tokens.txt",
                        dataDirName =
                            "espeak-ng-data",
                        filesSubdir = "tts_ru",
                    ),

                /*
                 * uz（MMS uzb-script_cyrillic）曾被加入本表，
                 * 但 sherpa AAR 1.13.7 原生层构造该模型时静默
                 * 崩溃，已回退到系统引擎降级路径。
                 * 重试清单见 experiments/tts/README.txt
                 * （模型、转写器与多模型结构均在 git 历史中）。
                 */
            )
    }
}
