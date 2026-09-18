package com.example.fourlangasr

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Log
import android.view.View
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.example.fourlangasr.asr.AsrManager
import com.example.fourlangasr.audio.AudioRecorder
import com.example.fourlangasr.model.AppLanguage
import com.example.fourlangasr.model.LoadResult
import com.example.fourlangasr.pipeline.ConversationPipeline
import com.example.fourlangasr.pipeline.CodeSwitchRecovery
import com.example.fourlangasr.pipeline.MetricsFormatter
import com.example.fourlangasr.pipeline.PipelineResult
import com.example.fourlangasr.pipeline.SourceMode
import com.example.fourlangasr.translation.TranslationResult
import com.example.fourlangasr.translation.ui.TranslationUiController
import com.example.fourlangasr.tts.AndroidSystemTtsEngine
import com.example.fourlangasr.tts.OfflineNeuralTtsEngine
import com.example.fourlangasr.tts.TtsEngine
import com.example.fourlangasr.tts.TtsManager
import com.example.fourlangasr.vad.VadEngine
import com.example.fourlangasr.speaker.SpeakerManager
import com.example.fourlangasr.lid.LidEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancel
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.joinAll
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.Locale

class MainActivity : AppCompatActivity() {

    private lateinit var spinnerSource: Spinner
    private lateinit var spinnerTarget: Spinner

    private lateinit var btnLoad: Button
    private lateinit var btnRecord: Button
    private lateinit var btnRelease: Button
    private lateinit var btnAutoSpeak: Button
    private lateinit var btnSpeechRate: Button
    private lateinit var textTtsAvailability: TextView

    private lateinit var textStatus: TextView
    private lateinit var textResult: TextView
    private lateinit var textMetrics: TextView

    private lateinit var translationUi: TranslationUiController

    private lateinit var asrManager: AsrManager
    private lateinit var audioRecorder: AudioRecorder
    private lateinit var speakerManager: SpeakerManager
    private lateinit var lidEngine: LidEngine
    private lateinit var codeSwitchRecovery:
            CodeSwitchRecovery

    /*
     * 单句识别管线（Speaker -> LID -> 路由 -> ASR -> code-switch 回填）。
     * 逻辑在 pipeline 包中，Activity 只负责 UI 与生命周期。
     */
    private lateinit var conversationPipeline:
            ConversationPipeline

    /*
     * v0.4 自动语言路由。
     *
     * source spinner 暂时保留：
     * - 对话开始前仍可手动选择初始/Debug 语言
     * - 对话中会自动显示最近一次 LID 路由语言
     */

    /*
     * v0.5:
     * AUTO   -> run VoxLingua LID and route zh/en/ru/uz automatically.
     * MANUAL -> completely skip LID and lock to spinnerSource.
     */
    @Volatile
    private var sourceMode =
        SourceMode.AUTO

    /*
     * v0.3 观察模式：
     * 每次“开始对话”都会清空本次会话的 Speaker 1~4，
     * 但 Speaker ONNX 模型本身会复用，不会每句重新加载。
     */
    private val conversationTranscript =
        StringBuilder()

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

    @Volatile
    private var activityDestroyed = false

    /*
     * TTS 播放结束后的回声冷却截止时间（毫秒时间戳）。
     *
     * 系统 TTS 的 onDone 常在扬声器实际停止前触发，
     * 播完瞬间的残留回声会被麦克风拾取、被 VAD 判成"新语音"，
     * 造成 识别 → 朗读 → 再识别 的自触发循环。
     * 冷却期内音频帧直接丢弃。
     */
    @Volatile
    private var ttsCooldownUntil =
        0L

    /*
     * 冷却结束时在音频线程重置一次 VAD，
     * 丢弃播放期间任何已进入 VAD 的残留状态。
     * （VAD 不是线程安全的，reset 必须由音频线程执行）
     */
    @Volatile
    private var pendingVadReset =
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

    /*
     * TTS v0.1（Android 系统 TextToSpeech）。
     *
     * post-ASR pipeline：
     * ASR 文本 → Translation abstraction → TTS
     * 朗读的是 translation result 的 outputText / speechLanguage。
     *
     * 回调线程不保证是主线程，UI 更新经 runOnUiThread；
     * 销毁后 conversationActive == false，回调不再触碰 UI 文案。
     */
    private val ttsListener =
        object : TtsEngine.Listener {

            override fun onSpeakStart() {

                runOnUiThread {
                    if (activityDestroyed) return@runOnUiThread

                    if (
                        conversationActive
                    ) {

                        showStatus(
                            "TTS 播放中…"
                        )
                    }
                }
            }

            override fun onSpeakDone() {

                runOnUiThread {
                    if (activityDestroyed) return@runOnUiThread

                    /*
                     * 播完进入回声冷却期，并标记 VAD 重置：
                     * 扬声器残响不再触发"识别→朗读"自循环。
                     */
                    ttsCooldownUntil =
                        System.currentTimeMillis() +
                                TTS_COOLDOWN_MS

                    pendingVadReset =
                        true

                    if (
                        conversationActive
                    ) {

                        showStatus(
                            "继续监听…"
                        )
                    }
                }
            }

            override fun onSpeakError(
                reason: String,
            ) {

                runOnUiThread {
                    if (activityDestroyed) return@runOnUiThread

                    showStatus(
                        "TTS: $reason"
                    )

                    /*
                     * 缺离线语音时首次弹窗引导下载，
                     * 之后只显示状态提示，不重复打扰。
                     */
                    if (
                        !voiceMissingDialogShown &&
                        (
                            reason.contains(
                                "voice missing"
                            ) ||
                                reason.contains(
                                    "not supported"
                                )
                            )
                    ) {

                        voiceMissingDialogShown =
                            true

                        showVoiceMissingDialog(
                            reason
                        )
                    }
                }
            }

            override fun onEngineReady(
                availability:
                    Map<String, Boolean>,
            ) {

                runOnUiThread {
                    if (activityDestroyed) return@runOnUiThread

                    /*
                     * ttsEngine 是 TtsManager：查询合并后的完整可用性
                     * （系统 voice + 离线引擎支持的语言）。
                     */
                    updateTtsAvailability(
                        ttsEngine
                            .voiceAvailability()
                    )
                }
            }
        }

    private lateinit var ttsEngine:
            TtsEngine

    /*
     * 自动朗读开关（默认开）。
     * 关闭后识别照常，只是不再自动 TTS；
     * 关闭动作同时停止正在播放的语音。
     */
    private var autoSpeakEnabled =
        true

    /*
     * 语速选项（0.75x / 1.0x / 1.25x），默认 1.0x。
     */
    private val speechRateOptions =
        floatArrayOf(
            0.75f,
            1.0f,
            1.25f,
        )

    private var speechRateIndex =
        1

    /*
     * 缺语音对话框只弹一次（Activity 生命周期内），
     * 避免每句 RU/UZ 都打断用户。
     */
    private var voiceMissingDialogShown =
        false

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

        /*
         * 翻译 UI 编排（翻译按钮/引擎指示/译文显示/模型同步）
         * 独立在 TranslationUiController，MainActivity 只负责注入视图与数据源。
         */
        translationUi =
            TranslationUiController(
                context = this,
                scope = lifecycleScope,
                btnTranslate =
                    findViewById(R.id.btnTranslate),
                btnSpecialistModels =
                    findViewById(R.id.btnSpecialistModels),
                textTranslation =
                    findViewById(R.id.textTranslation),
                textEngineHint =
                    findViewById(R.id.textEngineHint),
                lastAsrText = { lastAsrText },
                selectedSource = { selectedSource() },
                selectedTarget = { selectedTarget() },
            )

        asrManager =
            AsrManager(
                applicationContext
            )

        audioRecorder =
            AudioRecorder(
                applicationContext,
                sampleRate = 16000,
            )

        speakerManager =
            SpeakerManager(
                applicationContext.assets
            )

        /*
         * IMPORTANT:
         * LID uses LiteRT/TFLite, not ONNX Runtime Java.
         * This keeps it isolated from sherpa-onnx's native ORT.
         */
        lidEngine =
            LidEngine(
                applicationContext
            )

        /*
         * EXP5/EXP6 code-switch 回填：EN 模型懒加载，
         * 首次遇到中文句中的 <unk> 才真正加载。
         */
        codeSwitchRecovery =
            CodeSwitchRecovery(
                applicationContext.assets
            )

        conversationPipeline =
            ConversationPipeline(
                speakerManager = speakerManager,
                lidEngine = lidEngine,
                asrManager = asrManager,
                codeSwitchRecovery =
                    codeSwitchRecovery,
            )

        /*
         * TTS v0.1：系统 TextToSpeech 为主引擎；
         * 离线神经引擎（内置 Piper ru_RU）补齐系统 voice 缺失的语言。
         * 生命周期跟随 Activity（onDestroy 统一 release）。
         */
        val systemEngine =
            AndroidSystemTtsEngine(
                applicationContext,
                ttsListener,
            )

        val offlineEngine =
            OfflineNeuralTtsEngine(
                applicationContext,
                ttsListener,
            )

        /*
         * 对话开始前预加载离线 TTS 模型：
         * 避免模型首次加载与 VAD/ASR 等 sherpa native 活动并发
         * （该并发窗口曾导致 destroyed-mutex 原生崩溃）。
         */
        offlineEngine.preload()

        ttsEngine =
            TtsManager(
                systemEngine =
                    systemEngine,
                offlineEngine =
                    offlineEngine,
            )

        translationUi.syncModelsIfUnmetered()

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

        translationUi.updateEngineHint()

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

                translationUi.updateEngineHint()

                if (conversationActive) {

                    return@SimpleItemSelectedListener
                }

                if (languageSwitchInProgress) {

                    return@SimpleItemSelectedListener
                }

                /*
                 * AUTO mode keeps spinner disabled for the user.
                 * Programmatic LID updates must not trigger model switching here.
                 */
                if (
                    sourceMode ==
                    SourceMode.AUTO
                ) {

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

        spinnerTarget.onItemSelectedListener =
            SimpleItemSelectedListener {

                translationUi.updateEngineHint()
            }

        /*
         * AUTO mode still needs one initial ASR loaded as a safe fallback
         * before the first LID result arrives. Start with the current
         * spinner selection (ZH by default).
         */
        lifecycleScope.launch {

            try {

                val initialLanguage =
                    selectedSource()

                if (
                    asrManager.loadedLanguage !=
                    initialLanguage
                ) {

                    val loadResult =
                        withContext(
                            Dispatchers.IO
                        ) {

                            asrManager.switchLanguage(
                                initialLanguage
                            )
                        }

                    showLoadMetrics(
                        loadResult
                    )
                }

                btnRecord.isEnabled =
                    true

                btnRelease.isEnabled =
                    true

                showStatus(
                    "自动检测模式已就绪；当前备用源语言：" +
                            selectedSource().displayName
                )

            } catch (
                e: Throwable
            ) {

                btnRecord.isEnabled =
                    false

                btnRelease.isEnabled =
                    false

                showStatus(
                    "初始 ASR 模型加载失败：\n" +
                            "${e.javaClass.simpleName}: " +
                            "${e.message}"
                )
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

        /*
         * v0.5:
         * Reuse the old hidden Debug load button as the source-mode switch.
         * This avoids changing activity_main.xml.
         */
        btnLoad.visibility =
            View.VISIBLE

        updateSourceModeUi()

        btnLoad.setOnClickListener {

            if (conversationActive) {

                showStatus(
                    "请先结束当前对话再切换源语言模式。"
                )

                return@setOnClickListener
            }

            toggleSourceMode()
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
                    speakerManager.release()
                    lidEngine.release()
                    codeSwitchRecovery.release()
                }

                btnRecord.isEnabled =
                    false

                btnRelease.isEnabled =
                    false

                showStatus(
                    "ASR + Speaker + LID + Code-switch 模型已释放。"
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

        /*
         * ========================================================
         * 自动朗读开关（TTS v0.1）。
         * ========================================================
         */

        updateAutoSpeakButton()

        btnAutoSpeak.setOnClickListener {

            autoSpeakEnabled =
                !autoSpeakEnabled

            updateAutoSpeakButton()

            if (
                !autoSpeakEnabled
            ) {

                /*
                 * 关闭时立刻停掉正在播放的语音。
                 */
                stopSpeech()

                showStatus(
                    "自动朗读已关闭。"
                )

            } else {

                showStatus(
                    "自动朗读已开启。"
                )
            }
        }

        /*
         * ========================================================
         * 语速控制（0.75x / 1.0x / 1.25x 循环）。
         * ========================================================
         */

        updateSpeechRateButton()

        btnSpeechRate.setOnClickListener {

            speechRateIndex =
                (speechRateIndex + 1) %
                        speechRateOptions.size

            ttsEngine.setSpeechRate(
                speechRateOptions[
                    speechRateIndex
                ]
            )

            updateSpeechRateButton()
        }

        /*
         * 专项模型页入口：六组语言对模型的按需下载/更新/启用。
         */
        /*
         * 朗读语音可用性行：点击打开系统语音安装/设置页，
         * 用于下载缺失的离线语音（如 Google TTS 的俄语包）。
         */
        textTtsAvailability.setOnClickListener {

            openTtsVoiceSettings()
        }
    }

    /**
     * 打开系统 TTS 语音入口。
     *
     * 优先级：
     * 1. Google TTS 的语音数据安装页（支持俄语等多语言语音包；
     *    系统默认引擎如 ColorOS 只提供中文语音，装不了俄语）；
     * 2. 默认引擎的 ACTION_INSTALL_TTS_DATA；
     * 3. 系统"文字转语音"设置页。
     *
     * 系统 voice 无法由 App 注入——只能引导用户安装/切换引擎语音包。
     */
    private fun openTtsVoiceSettings() {

        val googleIntent =
            Intent(
                TextToSpeech.Engine
                    .ACTION_INSTALL_TTS_DATA
            )
                .setPackage(
                    AndroidSystemTtsEngine
                        .GOOGLE_TTS_PACKAGE
                )

        val installIntent =
            Intent(
                TextToSpeech.Engine
                    .ACTION_INSTALL_TTS_DATA
            )

        /*
         * 系统 TTS 设置页没有公开 SDK 常量，
         * action 是固定字符串 com.android.settings.TTS_SETTINGS。
         */
        val settingsIntent =
            Intent(
                "com.android.settings.TTS_SETTINGS"
            )

        val target =
            if (
                googleIntent
                    .resolveActivity(
                        packageManager
                    ) !=
                null
            ) {

                googleIntent

            } else if (
                installIntent
                    .resolveActivity(
                        packageManager
                    ) !=
                null
            ) {

                installIntent

            } else if (
                settingsIntent
                    .resolveActivity(
                        packageManager
                    ) !=
                null
            ) {

                settingsIntent

            } else {

                showStatus(
                    "未找到语音设置入口，请到系统设置 → 文字转语音中下载离线语音。"
                )

                return
            }

        try {

            startActivity(
                target
            )

        } catch (
            e: Throwable
        ) {

            showStatus(
                "打开语音设置失败：${e.message}"
            )
        }
    }

    /**
     * Google TTS 是否已安装（它能提供俄语等多语言离线语音包）。
     */
    private fun isGoogleTtsInstalled():
            Boolean {

        return Intent(
            TextToSpeech.Engine
                .ACTION_INSTALL_TTS_DATA
        )
            .setPackage(
                AndroidSystemTtsEngine
                    .GOOGLE_TTS_PACKAGE
            )
            .resolveActivity(
                packageManager
            ) !=
            null
    }

    private fun updateSpeechRateButton() {

        btnSpeechRate.text =
            "语速：" +
                    speechRateOptions[
                        speechRateIndex
                    ] +
                    "x"
    }

    /**
     * 显示四语离线 voice 可用性（✓ 可朗读 / ✗ 缺 voice）。
     * 空 map = 引擎未就绪或不可用。
     * 行本身可点击：打开系统语音下载/设置页。
     */
    private fun updateTtsAvailability(
        availability:
            Map<String, Boolean>,
    ) {

        if (
            availability.isEmpty()
        ) {

            textTtsAvailability.text =
                "朗读语音：不可用或初始化中"

            return
        }

        val builder =
            StringBuilder(
                "朗读语音："
            )

        var hasMissing =
            false

        AppLanguage.entries
            .forEachIndexed {
                index,
                language ->

                if (
                    index > 0
                ) {

                    builder.append(
                        "  "
                    )
                }

                val available =
                    availability[
                        language.code
                    ] ==
                    true

                if (
                    !available
                ) {

                    hasMissing =
                        true
                }

                builder
                    .append(
                        language.code
                    )
                    .append(
                        if (
                            available
                        ) {

                            "✓"

                        } else {

                            "✗"
                        }
                    )
            }

        if (
            hasMissing
        ) {

            builder.append(
                "（点此下载缺失语音）"
            )
        }

        textTtsAvailability.text =
            builder.toString()
    }

    /**
     * 缺离线语音时的引导对话框。
     */
    private fun showVoiceMissingDialog(
        reason: String,
    ) {

        if (
            isFinishing ||
            isDestroyed
        ) {

            return
        }

        val guide =
            if (
                isGoogleTtsInstalled()
            ) {

                "系统默认引擎可能缺少该语言语音。\n" +
                        "Google TTS 支持俄语等语音包——" +
                        "点击下方按钮前往下载。"

            } else {

                "系统默认引擎缺少该语言语音（如俄语）。\n" +
                        "请先安装 Google TTS（应用商店搜索 " +
                        "\"Google 文字转语音\"），安装后回到本页\n" +
                        "1. 系统设置 → 文字转语音 → 把引擎切换为 Google TTS\n" +
                        "2. 再点这里下载对应语言的离线语音包。"
            }

        androidx.appcompat.app
            .AlertDialog
            .Builder(
                this
            )
            .setTitle(
                "缺少离线朗读语音"
            )
            .setMessage(
                "$reason\n\n" +
                        "$guide\n\n" +
                        "语音包由 TTS 引擎提供，App 无法自带安装。"
            )
            .setPositiveButton(
                "去下载语音"
            ) { _, _ ->

                openTtsVoiceSettings()
            }
            .setNegativeButton(
                "取消",
                null,
            )
            .show()
    }

    private fun updateAutoSpeakButton() {

        btnAutoSpeak.text =
            if (
                autoSpeakEnabled
            ) {

                "自动朗读：开"

            } else {

                "自动朗读：关"
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

        btnAutoSpeak =
            findViewById(
                R.id.btnAutoSpeak
            )

        btnSpeechRate =
            findViewById(
                R.id.btnSpeechRate
            )

        textTtsAvailability =
            findViewById(
                R.id.textTtsAvailability
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

    }

    private fun selectedSource():
            AppLanguage =

        spinnerSource.selectedItem
                as AppLanguage

    private fun selectedTarget():
            AppLanguage =

        spinnerTarget.selectedItem
                as AppLanguage

    private fun toggleSourceMode() {

        if (
            conversationActive
        ) {
            return
        }

        sourceMode =
            if (
                sourceMode ==
                SourceMode.AUTO
            ) {
                SourceMode.MANUAL
            } else {
                SourceMode.AUTO
            }

        updateSourceModeUi()

        if (
            sourceMode ==
            SourceMode.MANUAL
        ) {

            /*
             * If AUTO loaded the TFLite LID, release it when switching
             * to MANUAL so manual mode truly saves memory.
             */
            lifecycleScope.launch {

                withContext(
                    Dispatchers.IO
                ) {

                    lidEngine.release()
                }

                showStatus(
                    "已切换为手动源语言。请选择语言后开始对话。"
                )
            }

        } else {

            showStatus(
                "已切换为自动检测源语言。当前源语言仅作为短句/LID失败备用。"
            )
        }
    }

    private fun updateSourceModeUi() {

        if (
            sourceMode ==
            SourceMode.AUTO
        ) {

            btnLoad.text =
                "源语言模式：自动检测"

            spinnerSource.isEnabled =
                false

        } else {

            btnLoad.text =
                "源语言模式：手动锁定"

            spinnerSource.isEnabled =
                !conversationActive &&
                        !languageSwitchInProgress
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
                if (
                    sourceMode ==
                    SourceMode.AUTO
                ) {
                    "正在启动 VAD + Speaker + LID…"
                } else {
                    "正在启动 VAD + Speaker（手动源语言）…"
                }
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

                    /*
                     * Speaker 模型第一次开始对话时加载，
                     * 后续会话复用模型；每次新会话只清空 Speaker profiles。
                     */
                    speakerManager.load()
                    speakerManager.resetProfiles()

                    /*
                     * AUTO:
                     *   load/reuse the ~82 MB TFLite LID model.
                     *
                     * MANUAL:
                     *   do not load LID at all. This saves inference time and
                     *   allows the user to lock one ASR for the whole session.
                     */
                    if (
                        sourceMode ==
                        SourceMode.AUTO
                    ) {

                        lidEngine.load()
                    }
                }

                /*
                 * 创建串行 Speaker + ASR Worker。
                 */
                startRecognitionWorker()

                conversationActive =
                    true

                vadSpeechActive =
                    false

                conversationTranscript.clear()

                translationUi.resetTranslationLog()

                textResult.text =
                    "等待讲话…"

                textMetrics.text =
                    """
                    VAD          : Silero
                    Speaker      : 3D-Speaker ERes2Net
                    Source mode  : ${sourceMode.name}
                    LID          : ${if (sourceMode == SourceMode.AUTO) "VoxLingua107 TFLite" else "SKIPPED (manual lock)"}
                    LID input    : ${if (sourceMode == SourceMode.AUTO) "fixed ${LidEngine.FIXED_SECONDS} s / ${LidEngine.FIXED_SAMPLES}" else "N/A"}
                    Auto route   : ${if (sourceMode == SourceMode.AUTO) "ZH / EN / RU / UZ" else "OFF"}
                    Sample rate  : ${VadEngine.SAMPLE_RATE} Hz
                    Window       : ${VadEngine.WINDOW_SIZE} samples
                    Frame        : ${1000 * VadEngine.WINDOW_SIZE / VadEngine.SAMPLE_RATE} ms
                    End silence  : ~${VadEngine.END_SILENCE_SEC} s
                    Spk score    : >= ${SpeakerManager.SCORE_THRESHOLD}
                    Spk margin   : >= ${SpeakerManager.MARGIN_THRESHOLD}
                    Max speakers : ${SpeakerManager.MAX_SPEAKERS}
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
                    sourceMode ==
                        SourceMode.MANUAL

                spinnerTarget.isEnabled =
                    true

                btnLoad.isEnabled =
                    true

                updateSourceModeUi()

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

        /*
         * 回声冷却期：播完瞬间的扬声器残响直接丢弃，
         * 防止自触发循环。
         */
        if (
            System.currentTimeMillis() <
            ttsCooldownUntil
        ) {
            return
        }

        /*
         * TTS 防回灌 gate：
         * 播放期间麦克风仍在采集，但这些帧不得进入 VAD——
         * 否则扬声器声音会触发 VAD → ASR → 又朗读，形成循环。
         * AudioRecord 本身继续运行，不反复启停。
         */
        if (ttsEngine.isSpeaking) {
            return
        }

        /*
         * 冷却结束后由音频线程重置一次 VAD，
         * 清掉播放期间进入的任何残留状态。
         */
        if (pendingVadReset) {
            pendingVadReset = false
            vad.reset()
            vadSpeechActive = false
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

        /*
         * Capture this on the main coroutine before Dispatchers.IO.
         * It is used only when the utterance is too short for reliable LID
         * or if LID inference fails.
         */
        val modeForSegment =
            sourceMode

        val fallbackLanguage =
            asrManager.loadedLanguage
                ?: selectedSource()

        val manualLanguage =
            selectedSource()

        showStatus(
            String.format(
                Locale.US,
                if (
                    modeForSegment ==
                    SourceMode.AUTO
                ) {
                    "检测到 %.2f 秒语音，正在执行 Speaker + LID + ASR…"
                } else {
                    "检测到 %.2f 秒语音，手动源语言模式：Speaker + ASR…"
                },
                duration,
            )
        )

        try {

            /*
             * 串行 Channel 保证同一时间只有一个句子
             * 进入 Speaker/LID/ASR 原生管线。
             */
            val pipeline =
                withContext(
                    Dispatchers.IO
                ) {

                    conversationPipeline
                        .processSegment(
                            samples = samples,
                            sourceMode =
                                modeForSegment,
                            manualLanguage =
                                manualLanguage,
                            fallbackLanguage =
                                fallbackLanguage,
                        )
                }

            /*
             * Show the detected language in the existing source spinner.
             * Its listener returns immediately while conversationActive=true,
             * so this does NOT trigger a second switch.
             */
            if (
                modeForSegment ==
                    SourceMode.AUTO &&
                spinnerSource
                    .selectedItemPosition !=
                    pipeline.routedLanguage.ordinal
            ) {

                spinnerSource.setSelection(
                    pipeline.routedLanguage.ordinal
                )
            }

            showAsrResult(
                pipeline
            )

            translationUi.setTranslateEnabled(
                pipeline.asr.text
                    .isNotBlank()
            )

            if (
                conversationActive
            ) {

                showStatus(
                    "识别完成：" +
                            pipeline
                                .routedLanguage
                                .code
                                .uppercase(
                                    Locale.US
                                ) +
                            "，继续监听下一句…"
                )
            }

            /*
             * post-ASR pipeline（放在识别完成状态之后触发）：
             * 识别文本 → Translation abstraction → TTS。
             * TTS 的失败提示由此成为最后写入的状态，
             * 不会被"识别完成…"覆盖。
             */
            speakPipelineResult(
                pipeline
            )

        } catch (e: CancellationException) {
            throw e
        } catch (
            e: Throwable
        ) {

            showStatus(
                "Speaker / LID / ASR 处理失败：\n" +
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

        /*
         * 结束对话即停止正在播放的朗读。
         */
        stopSpeech()

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
                    sourceMode ==
                        SourceMode.MANUAL

                spinnerTarget.isEnabled =
                    true

                btnLoad.isEnabled =
                    true

                updateSourceModeUi()

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
        pipeline: PipelineResult,
    ) {

        /*
         * code-switch 回填后文本优先。
         */
        lastAsrText =
            pipeline.displayText

        val spokenText =
            if (
                pipeline.asr.text.isBlank()
            ) {

                "【空输出】"

            } else {

                pipeline.displayText
            }

        /*
         * v0.4 transcript:
         * [Speaker N][ZH/EN/RU/UZ] text
         */
        conversationTranscript
            .append("[")
            .append(
                pipeline.speaker.label
            )
            .append("]")
            .append("[")
            .append(
                pipeline.routedLanguage
                    .code
                    .uppercase(
                        Locale.US
                    )
            )
            .append("] ")
            .append(
                spokenText
            )
            .append("\n")

        textResult.text =
            conversationTranscript
                .toString()
                .trimEnd()

        textMetrics.text =
            MetricsFormatter.segmentMetrics(
                pipeline = pipeline,
            )
    }

    /**
     * ============================================================
     * post-ASR pipeline：
     * ASR 最终文本 → Translation abstraction → TTS
     * ============================================================
     *
     * actual source language 取 pipeline.routedLanguage：
     * AUTO  模式 = LID 路由最终选择的语言；
     * MANUAL 模式 = 用户锁定语言。
     * 两者在 ConversationPipeline.processSegment 里统一折叠为
     * routedLanguage，这里不重新从 UI spinner 猜测。
     *
     * 朗读语言一律取 translation result 的 speechLanguage：
     * 真实翻译接入前 passthrough 保证 speechLanguage == sourceLanguage
     * （中文 ASR 永远用中文 voice，即使 targetLanguage 选了 EN）。
     */
    private suspend fun speakPipelineResult(
        pipeline: PipelineResult,
    ) {

        Log.d(
            "FourLangTTS",
            "speakPipelineResult lang=${pipeline.routedLanguage.code} " +
                    "text='${pipeline.displayText.take(40)}' " +
                    "autoSpeak=$autoSpeakEnabled"
        )

        /*
         * 自动朗读开关关闭时跳过 TTS，
         * 识别与文本显示不受影响。
         */
        if (
            !conversationActive || activityDestroyed || !autoSpeakEnabled
        ) {

            return
        }

        val text =
            pipeline.displayText

        if (
            text.isBlank()
        ) {

            return
        }

        val target =
            selectedTarget()

        translationUi.showTranslating()

        val translation =
            try {

                withContext(Dispatchers.IO) {

                    translationUi.service.translate(
                        text = text,
                        source =
                            pipeline.routedLanguage,
                        target = target,
                    )
                }

            } catch (e: Throwable) {

                /*
                 * 翻译不可用（无网络且本地模型未就绪等）：
                 * 回退朗读 ASR 原文，并在翻译结果区明确显示失败原因。
                 */
                Log.w(
                    "FourLangTTS",
                    "翻译失败，回退朗读 ASR 原文",
                    e,
                )

                translationUi.showAutoSpeakFallback(e)

                null
            }

        if (translation != null) {

            translationUi.showAutoSpeakTranslation(
                sourceText = text,
                result = translation,
            )
        }

        ttsEngine.speak(
            text =
                translation?.outputText
                    ?: text,
            language =
                translation?.speechLanguage
                    ?: pipeline.routedLanguage,
        )

        /*
         * 每次朗读后刷新 voice 可用性显示：
         * 用户中途安装/更换语音包后能及时反映。
         */
        updateTtsAvailability(
            ttsEngine.voiceAvailability()
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
            MetricsFormatter.loadMetrics(
                result
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
                    !conversationActive &&
                    sourceMode ==
                    SourceMode.MANUAL

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

    override fun onResume() {
        super.onResume()

        // 从专项模型页返回后刷新引擎指示（启用/停用状态可能已变化）
        translationUi.updateEngineHint()
    }

    override fun onDestroy() {
        activityDestroyed = true
        conversationActive = false
        vadSpeechActive = false
        segmentChannel?.close()
        segmentChannel = null

        // cancel 不会中断正在执行的 native 推理。连同初始化/切换任务一起
        // 等待结束，才能释放模型；清理使用独立 scope，不阻塞主线程。
        val pendingJobs = lifecycleScope.coroutineContext[Job]
            ?.children?.toList().orEmpty()
        lifecycleScope.cancel()
        recognitionJob = null
        ttsEngine.release()

        CoroutineScope(Dispatchers.IO).launch {
            pendingJobs.joinAll()
            // 停止对话任务也可能正在操作 AudioRecord，先等它退出，避免双重停止。
            audioRecorder.cancel()

            val releases = listOf<Pair<String, () -> Unit>>(
                "VAD" to { vadEngine?.release(); vadEngine = null },
                "Speaker" to { speakerManager.release() },
                "LID" to { lidEngine.release() },
                "Code-switch" to { codeSwitchRecovery.release() },
                "ASR" to { asrManager.release() },
            )
            for ((name, release) in releases) {
                try {
                    release()
                } catch (e: Exception) {
                    Log.e("FourLangASR", "Failed to release $name", e)
                }
            }
        }
        super.onDestroy()
    }

    private fun stopSpeech() {
        // 手动停止也会留下扬声器残响，和自然播完共用冷却机制。
        ttsCooldownUntil = System.currentTimeMillis() + TTS_COOLDOWN_MS
        pendingVadReset = true
        ttsEngine.stop()
    }

    companion object {

        /*
         * TTS 播完后的回声冷却时长。
         * 覆盖系统引擎 onDone 提前触发 + 扬声器残响的时间窗。
         */
        private const val TTS_COOLDOWN_MS =
            900L
    }
}
