package com.example.fourlangasr

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.View
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.Spinner
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.example.fourlangasr.asr.AsrManager
import com.example.fourlangasr.audio.AudioRecorder
import com.example.fourlangasr.model.AppLanguage
import com.example.fourlangasr.model.AsrResult
import com.example.fourlangasr.model.LoadResult
import com.example.fourlangasr.translation.PlaceholderTranslationEngine
import com.example.fourlangasr.vad.VadEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale

class MainActivity : AppCompatActivity() {

    private lateinit var spinnerSource: Spinner
    private lateinit var spinnerTarget: Spinner

    private lateinit var btnLoad: Button
    private lateinit var btnRecord: Button
    private lateinit var btnRelease: Button
    private lateinit var btnTranslate: Button

    private lateinit var textStatus: TextView
    private lateinit var textResult: TextView
    private lateinit var textMetrics: TextView
    private lateinit var textTranslation: TextView

    private lateinit var asrManager: AsrManager
    private lateinit var audioRecorder: AudioRecorder

    /*
     * Silero VAD。
     *
     * 第一次开始对话时加载，
     * 后续重复使用。
     */
    private var vadEngine: VadEngine? = null

    /*
     * VAD 检测出的完整句子进入这里。
     *
     * 这样多个句子不会同时抢 ASR 模型。
     */
    private var segmentChannel:
            Channel<FloatArray>? = null

    private var recognitionJob:
            Job? = null

    /*
     * 当前是否处于连续对话模式。
     */
    @Volatile
    private var conversationActive =
        false

    /*
     * 当前 VAD 是否已经检测到人在讲话。
     */
    @Volatile
    private var vadSpeechActive =
        false

    /*
     * 防止语言模型重复切换。
     */
    @Volatile
    private var languageSwitchInProgress =
        false

    private val translationEngine =
        PlaceholderTranslationEngine()

    private var lastAsrText: String = ""

    /**
     * 麦克风权限请求。
     */
    private val requestMicPermission =
        registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { granted ->

            if (granted) {

                startConversationNow()

            } else {

                showStatus(
                    "麦克风权限被拒绝。"
                )
            }
        }

    override fun onCreate(
        savedInstanceState: Bundle?
    ) {

        super.onCreate(savedInstanceState)

        setContentView(
            R.layout.activity_main
        )

        bindViews()

        asrManager =
            AsrManager(
                applicationContext
            )

        audioRecorder =
            AudioRecorder(
                applicationContext,
                sampleRate = 16000,
            )

        /*
         * ========================================================
         * 语言 Spinner
         * ========================================================
         */

        val languages =
            AppLanguage.entries.toTypedArray()

        val adapter =
            ArrayAdapter(
                this,
                android.R.layout.simple_spinner_item,
                languages,
            ).apply {

                setDropDownViewResource(
                    android.R.layout
                        .simple_spinner_dropdown_item
                )
            }

        spinnerSource.adapter =
            adapter

        spinnerTarget.adapter =
            adapter

        spinnerSource.setSelection(
            AppLanguage.ZH.ordinal
        )

        spinnerTarget.setSelection(
            AppLanguage.EN.ordinal
        )

        /*
         * ========================================================
         * 源语言变化：
         *
         * 自动释放旧模型
         * ↓
         * 自动加载新模型
         * ========================================================
         */

        spinnerSource.onItemSelectedListener =
            SimpleItemSelectedListener {

                if (conversationActive) {

                    return@SimpleItemSelectedListener
                }

                if (languageSwitchInProgress) {

                    return@SimpleItemSelectedListener
                }

                val newLanguage =
                    selectedSource()

                /*
                 * 已经是当前语言，不重复加载。
                 */
                if (
                    asrManager.loadedLanguage ==
                    newLanguage
                ) {

                    return@SimpleItemSelectedListener
                }

                lifecycleScope.launch {

                    languageSwitchInProgress =
                        true

                    setBusy(true)

                    showStatus(
                        "正在切换到 " +
                                "${newLanguage.displayName}…"
                    )

                    try {

                        val result =
                            withContext(
                                Dispatchers.IO
                            ) {

                                /*
                                 * switchLanguage()
                                 *
                                 * 内部已经：
                                 * 1. release 旧模型
                                 * 2. 创建新模型
                                 * 3. load 新模型
                                 */
                                asrManager
                                    .switchLanguage(
                                        newLanguage
                                    )
                            }

                        showStatus(
                            "${newLanguage.displayName} " +
                                    "模型已就绪。"
                        )

                        showLoadMetrics(
                            result
                        )

                        btnRecord.isEnabled =
                            true

                        btnRelease.isEnabled =
                            true

                    } catch (
                        e: Throwable
                    ) {

                        withContext(
                            Dispatchers.IO
                        ) {

                            asrManager.release()
                        }

                        btnRecord.isEnabled =
                            false

                        btnRelease.isEnabled =
                            false

                        showStatus(
                            "模型切换失败：\n" +
                                    "${e.javaClass.simpleName}: " +
                                    "${e.message}"
                        )

                    } finally {

                        languageSwitchInProgress =
                            false

                        setBusy(false)
                    }
                }
            }

        /*
         * ========================================================
         * 原来的加载按钮。
         *
         * 你如果 XML 中已经 gone，
         * 用户看不到它。
         *
         * 这里保留只是方便 Debug。
         * ========================================================
         */

        btnLoad.setOnClickListener {

            loadSelectedLanguage()
        }

        /*
         * ========================================================
         * 主按钮：
         *
         * 开始对话
         * ↕
         * 结束对话
         * ========================================================
         */

        btnRecord.text =
            "开始对话"

        btnRecord.setOnClickListener {

            if (conversationActive) {

                stopConversation()

            } else {

                ensurePermissionThenConversation()
            }
        }

        /*
         * ========================================================
         * 手动释放按钮。
         *
         * 正式 UI 可以隐藏。
         * ========================================================
         */

        btnRelease.setOnClickListener {

            if (conversationActive) {

                showStatus(
                    "请先结束当前对话。"
                )

                return@setOnClickListener
            }

            lifecycleScope.launch {

                setBusy(true)

                withContext(
                    Dispatchers.IO
                ) {

                    asrManager.release()
                }

                btnRecord.isEnabled =
                    false

                btnRelease.isEnabled =
                    false

                showStatus(
                    "ASR 模型已释放。"
                )

                setBusy(
                    busy = false,
                    keepRecordDisabled = true,
                )
            }
        }

        /*
         * ========================================================
         * 翻译接口暂时仍为 Placeholder。
         * ========================================================
         */

        btnTranslate.setOnClickListener {

            val source =
                selectedSource()

            val target =
                selectedTarget()

            textTranslation.text =
                translationEngine.translate(
                    text = lastAsrText,
                    source = source,
                    target = target,
                )
        }
    }

    /**
     * ============================================================
     * View 绑定
     * ============================================================
     */

    private fun bindViews() {

        spinnerSource =
            findViewById(
                R.id.spinnerSource
            )

        spinnerTarget =
            findViewById(
                R.id.spinnerTarget
            )

        btnLoad =
            findViewById(
                R.id.btnLoad
            )

        btnRecord =
            findViewById(
                R.id.btnRecord
            )

        btnRelease =
            findViewById(
                R.id.btnRelease
            )

        btnTranslate =
            findViewById(
                R.id.btnTranslate
            )

        textStatus =
            findViewById(
                R.id.textStatus
            )

        textResult =
            findViewById(
                R.id.textResult
            )

        textMetrics =
            findViewById(
                R.id.textMetrics
            )

        textTranslation =
            findViewById(
                R.id.textTranslation
            )
    }

    private fun selectedSource():
            AppLanguage =

        spinnerSource.selectedItem
                as AppLanguage

    private fun selectedTarget():
            AppLanguage =

        spinnerTarget.selectedItem
                as AppLanguage

    /**
     * ============================================================
     * Debug：
     * 手动加载当前语言。
     *
     * 正常用户流程不需要这个按钮。
     * ============================================================
     */

    private fun loadSelectedLanguage() {

        if (conversationActive) {

            showStatus(
                "请先结束当前对话。"
            )

            return
        }

        val language =
            selectedSource()

        lifecycleScope.launch {

            setBusy(true)

            showStatus(
                "正在加载 " +
                        "${language.displayName} 模型…"
            )

            try {

                val loadResult =
                    withContext(
                        Dispatchers.IO
                    ) {

                        asrManager
                            .switchLanguage(
                                language
                            )
                    }

                showStatus(
                    "${language.displayName} " +
                            "模型已加载。"
                )

                showLoadMetrics(
                    loadResult
                )

                btnRecord.isEnabled =
                    true

                btnRelease.isEnabled =
                    true

            } catch (
                e: Throwable
            ) {

                withContext(
                    Dispatchers.IO
                ) {

                    asrManager.release()
                }

                btnRecord.isEnabled =
                    false

                btnRelease.isEnabled =
                    false

                showStatus(
                    "模型加载失败：\n" +
                            "${e.javaClass.simpleName}: " +
                            "${e.message}"
                )

            } finally {

                setBusy(false)
            }
        }
    }

    /**
     * ============================================================
     * 麦克风权限
     * ============================================================
     */

    private fun ensurePermissionThenConversation() {

        val granted =
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.RECORD_AUDIO,
            ) ==
                    PackageManager.PERMISSION_GRANTED

        if (granted) {

            startConversationNow()

        } else {

            requestMicPermission.launch(
                Manifest.permission.RECORD_AUDIO
            )
        }
    }

    /**
     * ============================================================
     * 开始连续对话
     * ============================================================
     */

    private fun startConversationNow() {

        if (conversationActive) {
            return
        }

        if (
            asrManager.loadedLanguage ==
            null
        ) {

            showStatus(
                "当前语言模型尚未加载完成。"
            )

            return
        }

        lifecycleScope.launch {

            btnRecord.isEnabled =
                false

            btnLoad.isEnabled =
                false

            btnRelease.isEnabled =
                false

            spinnerSource.isEnabled =
                false

            showStatus(
                "正在启动语音检测…"
            )

            try {

                /*
                 * 第一次开始对话时加载 Silero VAD。
                 *
                 * 后面开始第二次对话时直接复用。
                 */
                val vad =
                    withContext(
                        Dispatchers.IO
                    ) {

                        vadEngine
                            ?: VadEngine(
                                applicationContext.assets
                            ).also {

                                vadEngine = it
                            }
                    }

                withContext(
                    Dispatchers.IO
                ) {

                    vad.reset()
                }

                /*
                 * 创建串行 ASR Worker。
                 */
                startRecognitionWorker()

                conversationActive =
                    true

                vadSpeechActive =
                    false

                textResult.text =
                    "等待讲话…"

                textMetrics.text =
                    """
                    VAD          : Silero
                    Sample rate  : 16000 Hz
                    Window       : 512 samples
                    Frame        : 32 ms
                    End silence  : ~0.6 s
                    """.trimIndent()

                /*
                 * 真正开始持续读取麦克风。
                 */
                audioRecorder
                    .startStreaming(
                        frameSize =
                            VadEngine.WINDOW_SIZE
                    ) { frame ->

                        processVadFrame(
                            vad = vad,
                            frame = frame,
                        )
                    }

                btnRecord.text =
                    "结束对话"

                btnRecord.isEnabled =
                    true

                showStatus(
                    "正在监听，请开始讲话。"
                )

            } catch (
                e: Throwable
            ) {

                conversationActive =
                    false

                vadSpeechActive =
                    false

                try {

                    audioRecorder
                        .stopStreaming()

                } catch (_: Throwable) {
                }

                segmentChannel
                    ?.close()

                segmentChannel =
                    null

                recognitionJob
                    ?.cancel()

                recognitionJob =
                    null

                btnRecord.text =
                    "开始对话"

                btnRecord.isEnabled =
                    asrManager.loadedLanguage !=
                            null

                spinnerSource.isEnabled =
                    true

                spinnerTarget.isEnabled =
                    true

                btnRelease.isEnabled =
                    asrManager.loadedLanguage !=
                            null

                showStatus(
                    "VAD 启动失败：\n" +
                            "${e.javaClass.simpleName}: " +
                            "${e.message}"
                )
            }
        }
    }

    /**
     * ============================================================
     * AudioRecorder 每送来 512 samples，
     * 都进入这里。
     *
     * 注意：
     * 这个方法运行在 AudioRecord 后台线程，
     * 不是 UI 线程。
     * ============================================================
     */

    private fun processVadFrame(
        vad: VadEngine,
        frame: FloatArray,
    ) {

        if (!conversationActive) {
            return
        }

        try {

            /*
             * 512 samples
             * ↓
             * Silero VAD
             */
            vad.accept(
                frame
            )

            val speechDetected =
                vad.isSpeechDetected()

            /*
             * 首次检测到讲话。
             */
            if (
                speechDetected &&
                !vadSpeechActive
            ) {

                vadSpeechActive =
                    true

                runOnUiThread {

                    if (
                        conversationActive
                    ) {

                        showStatus(
                            "检测到讲话，正在聆听…"
                        )
                    }
                }
            }

            /*
             * VAD 可能一次产生一个或多个
             * 已完成 speech segment。
             */
            while (
                vad.hasSegment()
            ) {

                val segment =
                    vad.takeSegment()
                        ?: break

                vadSpeechActive =
                    false

                if (
                    segment.isNotEmpty()
                ) {

                    /*
                     * 不在录音线程直接跑 ASR。
                     *
                     * 只扔进 Channel。
                     */
                    segmentChannel
                        ?.trySend(
                            segment
                        )
                }
            }

        } catch (
            e: Throwable
        ) {

            runOnUiThread {

                if (
                    conversationActive
                ) {

                    showStatus(
                        "VAD 处理失败：" +
                                "${e.message}"
                    )
                }
            }
        }
    }

    /**
     * ============================================================
     * 创建一个串行 ASR 队列。
     *
     * VAD：
     * 句1
     * 句2
     * 句3
     *
     * ↓
     *
     * Channel
     *
     * ↓
     *
     * ASR 句1
     * ASR 句2
     * ASR 句3
     *
     * 不会三个 ASR 同时运行。
     * ============================================================
     */

    private fun startRecognitionWorker() {

        /*
         * 防止意外存在旧 Worker。
         */
        segmentChannel
            ?.close()

        recognitionJob
            ?.cancel()

        val channel =
            Channel<FloatArray>(
                capacity =
                    Channel.UNLIMITED
            )

        segmentChannel =
            channel

        recognitionJob =
            lifecycleScope.launch {

                for (
                samples
                in channel
                ) {

                    recognizeVadSegment(
                        samples
                    )
                }
            }
    }

    /**
     * ============================================================
     * 自动识别一个 VAD 句子。
     * ============================================================
     */

    private suspend fun recognizeVadSegment(
        samples: FloatArray,
    ) {

        if (
            samples.isEmpty()
        ) {
            return
        }

        val duration =
            samples.size /
                    16000.0f

        showStatus(
            String.format(
                Locale.US,
                "检测到 %.2f 秒语音，正在离线识别…",
                duration,
            )
        )

        try {

            val result =
                withContext(
                    Dispatchers.IO
                ) {

                    asrManager.transcribe(
                        samples = samples,
                        sampleRate = 16000,
                    )
                }

            showAsrResult(
                result
            )

            btnTranslate.isEnabled =
                result.text.isNotBlank()

            if (
                conversationActive
            ) {

                showStatus(
                    "识别完成，继续监听下一句…"
                )
            }

        } catch (
            e: Throwable
        ) {

            showStatus(
                "识别失败：\n" +
                        "${e.javaClass.simpleName}: " +
                        "${e.message}"
            )
        }
    }

    /**
     * ============================================================
     * 用户点击“结束对话”
     * ============================================================
     */

    private fun stopConversation() {

        if (
            !conversationActive &&
            !audioRecorder.isRecording()
        ) {

            return
        }

        /*
         * 先禁止新的 VAD frame。
         */
        conversationActive =
            false

        vadSpeechActive =
            false

        btnRecord.isEnabled =
            false

        showStatus(
            "正在结束对话…"
        )

        lifecycleScope.launch {

            try {

                /*
                 * 1.
                 * 停止 AudioRecord，
                 * 等待录音线程退出。
                 */
                withContext(
                    Dispatchers.IO
                ) {

                    audioRecorder
                        .stopStreaming()
                }

                /*
                 * 2.
                 * flush VAD。
                 *
                 * 比如用户说：
                 *
                 * “你好”
                 *
                 * 然后马上点结束，
                 * 没有等够 0.6 秒静音。
                 *
                 * flush() 尽量把这最后一句输出。
                 */
                val remainingSegments =
                    withContext(
                        Dispatchers.IO
                    ) {

                        val result =
                            mutableListOf<
                                    FloatArray
                                    >()

                        vadEngine
                            ?.let { vad ->

                                vad.flush()

                                while (
                                    vad.hasSegment()
                                ) {

                                    val segment =
                                        vad
                                            .takeSegment()
                                            ?: break

                                    if (
                                        segment
                                            .isNotEmpty()
                                    ) {

                                        result.add(
                                            segment
                                        )
                                    }
                                }

                                vad.reset()
                            }

                        result
                    }

                /*
                 * 3.
                 * 最后剩余句子也送给 ASR。
                 */
                for (
                segment
                in remainingSegments
                ) {

                    segmentChannel
                        ?.trySend(
                            segment
                        )
                }

                /*
                 * 4.
                 * 不会再产生新句子。
                 */
                segmentChannel
                    ?.close()

                /*
                 * 5.
                 * 等待队列里的最后一句
                 * 识别完成。
                 */
                recognitionJob
                    ?.join()

                showStatus(
                    "对话已结束。"
                )

            } catch (
                e: Throwable
            ) {

                showStatus(
                    "结束对话时出错：\n" +
                            "${e.javaClass.simpleName}: " +
                            "${e.message}"
                )

            } finally {

                segmentChannel =
                    null

                recognitionJob =
                    null

                btnRecord.text =
                    "开始对话"

                btnRecord.isEnabled =
                    asrManager.loadedLanguage !=
                            null

                spinnerSource.isEnabled =
                    true

                spinnerTarget.isEnabled =
                    true

                btnLoad.isEnabled =
                    true

                btnRelease.isEnabled =
                    asrManager.loadedLanguage !=
                            null
            }
        }
    }

    /**
     * ============================================================
     * ASR 结果 UI
     * ============================================================
     */

    private fun showAsrResult(
        result: AsrResult
    ) {

        lastAsrText =
            result.text

        textResult.text =
            if (
                result.text.isBlank()
            ) {

                "【空输出】"

            } else {

                result.text
            }

        textMetrics.text =
            String.format(
                Locale.US,
                """
                Language     : %s
                Audio        : %.3f s
                Inference    : %d ms
                RTF          : %.4f
                Speed        : %.2fx realtime
                PSS before   : %.1f MB
                PSS after    : %.1f MB
                PSS delta    : %+.1f MB
                """.trimIndent(),
                asrManager
                    .loadedLanguage
                    ?.code
                    ?: "?",
                result.audioDurationSec,
                result.inferenceMs,
                result.rtf,
                if (
                    result.rtf > 0f
                ) {
                    1.0f /
                            result.rtf
                } else {
                    0f
                },
                result.pssBeforeMb,
                result.pssAfterMb,
                result.pssAfterMb -
                        result.pssBeforeMb,
            )
    }

    /**
     * ============================================================
     * 模型加载指标
     * ============================================================
     */

    private fun showLoadMetrics(
        result: LoadResult
    ) {

        textMetrics.text =
            String.format(
                Locale.US,
                """
                Model load   : %d ms
                PSS before   : %.1f MB
                PSS after    : %.1f MB
                Load delta   : %+.1f MB
                """.trimIndent(),
                result.loadMs,
                result.pssBeforeMb,
                result.pssAfterMb,
                result.pssDeltaMb,
            )
    }

    private fun showStatus(
        message: String
    ) {

        textStatus.text =
            "状态：$message"
    }

    /**
     * ============================================================
     * UI Busy 状态
     * ============================================================
     */

    private fun setBusy(
        busy: Boolean,
        keepRecordDisabled: Boolean =
            false,
    ) {

        btnLoad.isEnabled =
            !busy

        spinnerSource.isEnabled =
            !busy &&
                    !conversationActive

        spinnerTarget.isEnabled =
            !busy

        if (busy) {

            btnRecord.isEnabled =
                false

            btnRelease.isEnabled =
                false

        } else {

            if (
                !keepRecordDisabled
            ) {

                btnRecord.isEnabled =
                    asrManager.loadedLanguage !=
                            null
            }

            btnRelease.isEnabled =
                asrManager.loadedLanguage !=
                        null
        }
    }

    /**
     * ============================================================
     * Activity 销毁
     * ============================================================
     */

    override fun onDestroy() {

        conversationActive =
            false

        vadSpeechActive =
            false

        try {

            segmentChannel
                ?.close()

        } catch (_: Throwable) {
        }

        segmentChannel =
            null

        recognitionJob
            ?.cancel()

        recognitionJob =
            null

        /*
         * 先停止录音线程。
         */
        audioRecorder.cancel()

        /*
         * 再释放 VAD native 模型。
         */
        try {

            vadEngine
                ?.release()

        } catch (_: Throwable) {
        }

        vadEngine =
            null

        /*
         * 最后释放当前 ASR。
         */
        asrManager.release()

        super.onDestroy()
    }
}