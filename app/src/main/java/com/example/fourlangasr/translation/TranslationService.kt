package com.example.fourlangasr.translation

import com.example.fourlangasr.model.AppLanguage

/**
 * 翻译路由：本地 m2m100（离线，首选）→ 服务器六组 SMaLL-100（回退）。
 *
 * 路由规则（v1，可调整）：
 * 1. 本地模型已下载且就绪 → KitTranslationEngine（离线可用、无网络延迟）；
 * 2. 本地未就绪或本地方案失败 → ServerTranslationEngine；
 * 3. 两者都失败 → 异常上抛（调用方决定 UI 提示）。
 *
 * 必须在后台线程调用。
 */
class TranslationService(
    private val kitEngine: KitTranslationEngine,
    private val serverEngine: ServerTranslationEngine,
    private val downloadManager: TranslationDownloadManager,
) : TranslationEngine {

    override fun translate(
        text: String,
        source: AppLanguage,
        target: AppLanguage,
    ): TranslationResult {
        if (downloadManager.isModelReady()) {
            return try {
                kitEngine.translate(text, source, target)
            } catch (e: Exception) {
                serverEngine.translate(text, source, target)
            }
        }
        return serverEngine.translate(text, source, target)
    }

    /** 本地 m2m100 模型已就绪（可直接离线翻译）。 */
    fun isLocalModelReady(): Boolean = downloadManager.isModelReady()

    /** 下载 + 解包本地模型（幂等；模型已就绪时立即返回 true）。 */
    fun ensureLocalModel(
        onProgress: (phase: String, doneBytes: Long, totalBytes: Long) -> Unit = { _, _, _ -> },
    ): Boolean = downloadManager.ensureModel(onProgress)

    /** 关闭本地 ORT session（退出时调用）。 */
    fun close() {
        kitEngine.close()
    }
}
