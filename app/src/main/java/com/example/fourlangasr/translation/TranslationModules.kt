package com.example.fourlangasr.translation

import android.content.Context
import com.example.fourlangasr.translation.engine.ServerTranslationEngine
import com.example.fourlangasr.translation.model.ModelManifestClient
import com.example.fourlangasr.translation.model.ModelPackageManager
import com.example.fourlangasr.translation.model.ModelRegistry
import com.example.fourlangasr.translation.model.ModelSyncManager
import java.io.File

/**
 * 翻译子系统模块装配（进程内单例，MainActivity 与专项模型页共享）。
 *
 * 各模块职责：
 * - [registry]：本地已装模型的版本指纹 + 启用开关（JSON 持久化）；
 * - [packages]：单模型包的下载/解包/校验与资源加载；
 * - [manifestClient]：远端模型清单客户端；
 * - [sync]：清单拉取 + 版本对齐（过期重下）+ 总模型首装；
 * - [service]：翻译路由（专项 → 总模型 → 服务器）。
 */
class TranslationModules private constructor(
    applicationContext: Context,
) {
    private val baseDir = File(applicationContext.filesDir, "translation")

    val registry = ModelRegistry(File(baseDir, "model_registry.json"))
    val packages = ModelPackageManager(
        baseUrlProvider = { ServerConfig.baseUrl(applicationContext) },
        modelsRoot = File(baseDir, "models"),
    )
    val manifestClient = ModelManifestClient(
        baseUrlProvider = { ServerConfig.baseUrl(applicationContext) },
    )
    val sync = ModelSyncManager(
        manifestClient = manifestClient,
        packageManager = packages,
        registry = registry,
        cacheFile = File(baseDir, "model_manifest_cache.json"),
    )
    val service = TranslationService(
        registry = registry,
        packageManager = packages,
        manifestProvider = { sync.latestManifest },
        serverEngine = ServerTranslationEngine(
            baseUrlProvider = { ServerConfig.baseUrl(applicationContext) },
        ),
    )

    companion object {
        @Volatile
        private var instance: TranslationModules? = null

        fun getInstance(context: Context): TranslationModules {
            return instance ?: synchronized(this) {
                instance ?: TranslationModules(context.applicationContext).also { instance = it }
            }
        }
    }
}
