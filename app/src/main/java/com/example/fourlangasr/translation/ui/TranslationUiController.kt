package com.example.fourlangasr.translation.ui

import android.content.Context
import android.content.Intent
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.util.Log
import android.widget.Button
import android.widget.TextView
import com.example.fourlangasr.model.AppLanguage
import com.example.fourlangasr.translation.TranslationModules
import com.example.fourlangasr.translation.TranslationResult
import com.example.fourlangasr.translation.engine.ServerTranslationEngine
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * 翻译相关的 UI 编排（从 MainActivity 抽取）：
 * - 翻译按钮 / 专项模型入口按钮的点击流程；
 * - 实时引擎指示（当前方向使用哪个引擎）；
 * - 译文显示（含模型标签，供翻译按钮与自动朗读两条路径共用）；
 * - 模型同步触发（Wi-Fi 下：清单拉取 + 总模型首装 + 过期重下）。
 *
 * 视图由构造注入，业务数据（ASR 文本、语言选择）由 provider 注入，
 * 与 MainActivity 解耦（不反向依赖 Activity 内部状态）。
 */
class TranslationUiController(
    private val context: Context,
    private val scope: CoroutineScope,
    private val btnTranslate: Button,
    btnTextTranslate: Button,
    private val editManualText: android.widget.EditText,
    private val textSourceSpinner: android.widget.Spinner,
    private val textTargetSpinner: android.widget.Spinner,
    private val textTextEngineHint: TextView,
    btnSpecialistModels: Button,
    private val textTranslation: TextView,
    private val textEngineHint: TextView,
    private val lastAsrText: () -> String,
    private val selectedSource: () -> AppLanguage,
    private val selectedTarget: () -> AppLanguage,
) {

    val modules: TranslationModules = TranslationModules.getInstance(context)

    val service
        get() = modules.service

    init {
        btnTranslate.setOnClickListener { onTranslateClick() }
        btnTextTranslate.setOnClickListener { onTextTranslateClick() }
        btnSpecialistModels.setOnClickListener { openSpecialistModels() }

        // 文字翻译模块：独立的源/目标语言选择（与语音链路完全解耦）
        val languages = AppLanguage.entries.toTypedArray()
        textSourceSpinner.adapter =
            android.widget.ArrayAdapter(
                context,
                android.R.layout.simple_spinner_item,
                languages,
            ).apply {
                setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
            }
        textTargetSpinner.adapter =
            android.widget.ArrayAdapter(
                context,
                android.R.layout.simple_spinner_item,
                languages,
            ).apply {
                setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
            }
        textSourceSpinner.setSelection(AppLanguage.ZH.ordinal)
        textTargetSpinner.setSelection(AppLanguage.EN.ordinal)
        textSourceSpinner.onItemSelectedListener =
            com.example.fourlangasr.SimpleItemSelectedListener { updateTextEngineHint() }
        textTargetSpinner.onItemSelectedListener =
            com.example.fourlangasr.SimpleItemSelectedListener { updateTextEngineHint() }
        updateTextEngineHint()

        // 旋转/重建后把进程里留存的记录重新画出来（不重绘就看不见历史）。
        renderTranslationLog()
    }

    private fun selectedTextSource(): AppLanguage =
        textSourceSpinner.selectedItem as AppLanguage

    private fun selectedTextTarget(): AppLanguage =
        textTargetSpinner.selectedItem as AppLanguage

    /** 文字翻译模块的引擎指示（与语音模块分开显示）。 */
    fun updateTextEngineHint() {
        val label = try {
            service.engineLabelFor(selectedTextSource(), selectedTextTarget())
        } catch (e: Exception) {
            android.util.Log.w("FourLangTranslation", "text engineLabelFor 异常", e)
            "服务器 API"
        }
        textTextEngineHint.text = "文字翻译引擎：$label"
    }

    /** 主界面 ASR 出文本后开放翻译按钮（由 MainActivity 的识别流程调用）。 */
    fun setTranslateEnabled(enabled: Boolean) {
        btnTranslate.isEnabled = enabled
    }

    // ================================================================ 翻译按钮

    private fun onTranslateClick() {
        val source = selectedSource()
        val target = selectedTarget()
        val text = lastAsrText()
        if (text.isBlank()) return

        showTranslating()
        scope.launch {
            val result = try {
                withContext(Dispatchers.IO) {
                    service.translate(text = text, source = source, target = target)
                }
            } catch (e: Throwable) {
                appendFailure(e, null)
                return@launch
            }
            appendEntry(
                source = text,
                body = result.outputText,
                label = friendlyModelLabel(result.modelLabel),
                sourceLang = result.sourceLanguage.code,
                targetLang = result.targetLanguage.code,
            )
        }
    }

    /**
     * 翻译记录（与 ASR 记录同构）：每条 = 原文 + 译文（模型），按时间累积，
     * 由 [resetTranslationLog] 随对话开始清空。
     * 实体存在进程单例 [TranslationModules] 里——本控制器随 Activity 重建，
     * 放在自己字段里旋转就全丢。
     */
    private val translationLog = modules.translationLog

    /** 对话开始时清空翻译记录（MainActivity 与 ASR transcript 同步调用）。 */
    fun resetTranslationLog() {
        translationLog.setLength(0)
        renderTranslationLog()
    }

    /** 翻译开始（本地模型要几秒到几十秒，给用户明确进度感）。 */
    fun showTranslating() {
        textTranslation.text =
            translationLog.toString().trimEnd() + "\n翻译中…"
        scrollToTranslation()
    }

    /** 自动朗读路径：翻译成功后追加一条记录（原文/译文成对，带方向标签）。 */
    fun showAutoSpeakTranslation(sourceText: String, result: TranslationResult) {
        if (!result.translated) return
        appendEntry(
            source = sourceText,
            body = result.outputText,
            label = friendlyModelLabel(result.modelLabel),
            sourceLang = result.sourceLanguage.code,
            targetLang = result.targetLanguage.code,
        )
    }

    /** 自动朗读路径：翻译失败但回退朗读原文 —— 记录失败原因。 */
    fun showAutoSpeakFallback(e: Throwable) {
        appendFailure(e, "已朗读原文")
    }

    private fun appendEntry(
        source: String,
        body: String,
        label: String,
        sourceLang: String,
        targetLang: String,
    ) {
        val text = if (body.isBlank()) "（未生成译文）" else body
        translationLog
            .append("原文[")
            .append(sourceLang)
            .append("→")
            .append(targetLang)
            .append("]：")
            .append(source)
            .append("\n")
        if (label.isEmpty()) {
            translationLog.append("译文：\n")
        } else {
            translationLog.append("译文（").append(label).append("）：\n")
        }
        translationLog.append(text).append("\n\n")
        renderTranslationLog()
    }

    private fun appendFailure(e: Throwable, note: String?) {
        translationLog.append("翻译失败")
        if (note != null) {
            translationLog.append("（").append(note).append("）")
        }
        translationLog
            .append("：")
            .append(e.javaClass.simpleName)
            .append(": ")
            .append(e.message)
            .append("\n\n")
        renderTranslationLog()
    }

    private fun renderTranslationLog() {
        textTranslation.text =
            translationLog.toString().trimEnd().ifBlank { "—" }
        scrollToTranslation()
    }

    /** 把翻译结果区滚进可视范围（页面较长，结果在下方）。 */
    private fun scrollToTranslation() {
        textTranslation.post {
            textTranslation.requestRectangleOnScreen(
                android.graphics.Rect(0, 0, textTranslation.width, textTranslation.height),
                false,
            )
        }
    }

    /** 手动输入翻译：按文字模块的源/目标语言（与语音链路独立）翻译，结果追加到记录。 */
    private fun onTextTranslateClick() {
        val text = editManualText.text.toString().trim()
        if (text.isEmpty()) {
            android.widget.Toast.makeText(context, "请先输入要翻译的文字", android.widget.Toast.LENGTH_SHORT).show()
            return
        }
        val source = selectedTextSource()
        val target = selectedTextTarget()
        showTranslating()
        scope.launch {
            val result = try {
                withContext(Dispatchers.IO) {
                    service.translate(text = text, source = source, target = target)
                }
            } catch (e: Throwable) {
                appendFailure(e, null)
                return@launch
            }
            // 成功才清空：失败时用户输入被抹掉就只能重新打字。
            editManualText.text.clear()
            appendEntry(
                source = text,
                body = result.outputText,
                label = friendlyModelLabel(result.modelLabel),
                sourceLang = source.code,
                targetLang = target.code,
            )
        }
    }

    // ================================================================ 引擎指示

    /**
     * 实时引擎指示：与 TranslationService 的路由规则一致，
     * 显示当前源/目标方向会使用哪个翻译引擎（专项 / 总模型 / 服务器）。
     */
    fun updateEngineHint() {
        val label = try {
            service.engineLabelFor(selectedSource(), selectedTarget())
        } catch (e: Exception) {
            android.util.Log.w("FourLangTranslation", "engineLabelFor 异常", e)
            "服务器 API"
        }
        android.util.Log.d(
            "FourLangTranslation",
            "engine hint: ${selectedSource().code}->${selectedTarget().code} => $label",
        )
        textEngineHint.text = "当前翻译引擎：$label"

        // 文字模块的提示走同一批事件源（同步完成 / onResume 返回专项页）：
        // 只刷新语音提示会让文字区一直停在"本地模型下载中"。
        updateTextEngineHint()
    }

    /** TranslationResult.modelLabel → 展示文案。 */
    fun friendlyModelLabel(modelLabel: String?): String =
        when {
            modelLabel.isNullOrBlank() -> ""
            modelLabel == ServerTranslationEngine.MODEL_LABEL_SERVER -> "服务器 API"
            else -> {
                val generalistId = modules.sync.latestManifest?.generalist()?.modelId
                if (modelLabel == generalistId) "总模型 $modelLabel" else "专项模型 $modelLabel"
            }
        }

    // ================================================================ 同步与入口

    /**
     * 模型同步（Wi-Fi 下）：拉取远端清单 → 总模型首装 → 过期包重下。
     * 六组专项不在启动时自动下载，由专项模型页按需触发。
     */
    fun syncModelsIfUnmetered() {
        val sync = modules.sync
        scope.launch {
            if (!isOnUnmeteredNetwork()) return@launch
            val lastLogPct = HashMap<String, Int>()
            val ok = sync.sync(installGeneralistIfMissing = true) { modelId, phase, done, total ->
                if (phase == "error") {
                    Log.w("FourLangTranslation", "$modelId 同步失败")
                    return@sync
                }
                if (total > 0) {
                    val pct = (done * 100 / total).toInt()
                    if (pct != lastLogPct[modelId] && pct % 10 == 0) {
                        lastLogPct[modelId] = pct
                        Log.d("FourLangTranslation", "$modelId $phase $pct%")
                    }
                }
            }
            if (ok) {
                Log.d(
                    "FourLangTranslation",
                    "manifest synced: ${sync.latestManifest?.models?.size ?: 0} models",
                )
                updateEngineHint()
            }
        }
    }

    private fun openSpecialistModels() {
        context.startActivity(Intent(context, SpecialistModelsActivity::class.java))
    }

    private fun isOnUnmeteredNetwork(): Boolean {
        val cm = context.getSystemService(ConnectivityManager::class.java) ?: return false
        val caps = cm.getNetworkCapabilities(cm.activeNetwork) ?: return false
        return caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_NOT_METERED)
    }
}
