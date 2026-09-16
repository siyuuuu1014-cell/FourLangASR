package com.example.fourlangasr.translation

import com.example.fourlangasr.model.AppLanguage
import com.example.fourlangasr.translation.engine.KitTranslationEngine
import com.example.fourlangasr.translation.engine.ServerTranslationEngine
import com.example.fourlangasr.translation.model.ManifestModel
import com.example.fourlangasr.translation.model.ModelPackageManager
import com.example.fourlangasr.translation.model.ModelRegistry
import com.example.fourlangasr.translation.model.ModelRuntimeConfig
import com.example.fourlangasr.translation.tokenizer.M2m100Tokenizer
import java.io.File

/**
 * 翻译路由：专项模型 → 总模型 → 服务器 API。
 *
 * 选型规则（全部由远端 manifest + 本地登记驱动，不写死模型）：
 * 1. 目标方向存在"已下载且已启用"的专项模型 → 本地专项（用户在新页面开启）；
 * 2. 总模型（覆盖方向最多的模型）已下载 → 本地总模型；
 * 3. 都没有 / 本地失败 → 服务器六组模型 API 兜底。
 *
 * 本地引擎内存约 1GB/实例：单槽缓存，切换模型时释放旧引擎
 * （分词器与运行时配置轻量，按模型常驻缓存）。
 * 必须在后台线程调用。
 */
class TranslationService(
    private val registry: ModelRegistry,
    private val packageManager: ModelPackageManager,
    private val manifestProvider: () -> com.example.fourlangasr.translation.model.ModelsManifest?,
    private val serverEngine: ServerTranslationEngine,
) : TranslationEngine {

    private val engineLock = Any()
    private var loadedModelId: String? = null
    private var loadedEngine: KitTranslationEngine? = null
    private val configCache = HashMap<String, ModelRuntimeConfig>()
    private val tokenizerCache = HashMap<String, M2m100Tokenizer>()

    private fun engineFor(model: ManifestModel): KitTranslationEngine {
        synchronized(engineLock) {
            val cached = loadedEngine
            if (cached != null && loadedModelId == model.modelId) return cached
            // 单槽：切换模型先释放旧引擎（避免两个 ~1GB 引擎并存 OOM）
            cached?.close()
            loadedEngine = null
            loadedModelId = null

            val config = configCache.getOrPut(model.modelId) {
                packageManager.loadRuntimeConfig(model.modelId, model.family)
            }
            val tokenizer = tokenizerCache.getOrPut(model.modelId) {
                packageManager.loadTokenizer(model.modelId, config.languages)
            }
            val dir = packageManager.modelDir(model.modelId)
            val engine = KitTranslationEngine(
                config = config,
                tokenizer = tokenizer,
                encoderFile = File(dir, ModelPackageManager.ENCODER_FILE),
                decoderFile = File(dir, ModelPackageManager.DECODER_FILE),
            )
            loadedEngine = engine
            loadedModelId = model.modelId
            return engine
        }
    }

    /** 本地可用模型：启用且就绪的专项（覆盖该方向）优先于就绪的总模型。 */
    private fun localModelFor(srcCode: String, tgtCode: String): ManifestModel? {
        val manifest = manifestProvider() ?: return null
        val generalist = manifest.generalist()
        val specialist = manifest.specialists().firstOrNull { model ->
            model.covers(srcCode, tgtCode) &&
                registry.get(model.modelId)?.enabled == true &&
                packageManager.isModelReady(model.modelId)
        }
        if (specialist != null) return specialist
        return generalist.takeIf { packageManager.isModelReady(it.modelId) }
    }

    /**
     * 该方向当前会使用的引擎的可读标识（供 UI 实时显示，
     * 与 [translate] 的路由规则完全一致）。
     */
    fun engineLabelFor(source: AppLanguage, target: AppLanguage): String {
        val local = localModelFor(source.code, target.code)
        if (local != null) {
            val generalistId = manifestProvider()?.generalist()?.modelId
            return if (local.modelId == generalistId) {
                "总模型 ${local.modelId}"
            } else {
                "专项模型 ${local.modelId}"
            }
        }
        return "服务器 API"
    }

    override fun translate(
        text: String,
        source: AppLanguage,
        target: AppLanguage,
    ): TranslationResult {
        val local = localModelFor(source.code, target.code)
        if (local != null) {
            return try {
                engineFor(local).translate(text, source, target)
            } catch (e: Exception) {
                serverEngine.translate(text, source, target)
            }
        }
        return serverEngine.translate(text, source, target)
    }

    /** 关闭当前加载的本地引擎（退出时调用）。 */
    fun close() {
        synchronized(engineLock) {
            loadedEngine?.close()
            loadedEngine = null
            loadedModelId = null
        }
    }
}
