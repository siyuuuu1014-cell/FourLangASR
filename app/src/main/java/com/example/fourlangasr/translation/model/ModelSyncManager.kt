package com.example.fourlangasr.translation.model

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/**
 * 模型同步：拉取远端清单 + 版本对齐 + 按需下载。
 *
 * 版本更新机制（与后端契约一致）：
 * - GET /models/manifest 取每个模型的 version + source_sha256；
 * - 本地已安装模型（[ModelRegistry]）的 sha 与远端不一致 → 重新下载该包；
 * - [installGeneralistIfMissing] 时，总模型未安装 → 自动下载（首次使用流程，
 *   由调用方按"非计费网络"条件决定是否开启）。
 *
 * [latestManifest] 供路由/UI 读取（内存缓存，不落盘）。
 */
class ModelSyncManager(
    private val manifestClient: ModelManifestClient,
    private val packageManager: ModelPackageManager,
    private val registry: ModelRegistry,
    private val cacheFile: File,
) {

    @Volatile
    var latestManifest: ModelsManifest? = readCache()
        private set

    private fun readCache(): ModelsManifest? =
        try {
            if (cacheFile.isFile) {
                ModelManifestParser.parse(cacheFile.readText(Charsets.UTF_8))
            } else {
                null
            }
        } catch (e: Exception) {
            null
        }

    private fun writeCache(manifest: ModelsManifest) {
        try {
            val tmp = File(cacheFile.parentFile, cacheFile.name + ".tmp")
            cacheFile.parentFile?.mkdirs()
            // 只持久化必需字段（与远端清单一致的最小集）
            val arr = org.json.JSONArray()
            for (m in manifest.models) {
                arr.put(
                    org.json.JSONObject()
                        .put("model_id", m.modelId)
                        .put("family", m.family)
                        .put("version", m.version)
                        .put("source_sha256", m.sourceSha256)
                        .put("size_bytes", m.sizeBytes)
                        .put("directions", org.json.JSONArray(m.directions))
                        .put("download_url", m.downloadUrl),
                )
            }
            val text = org.json.JSONObject().put("models", arr).toString()
            tmp.writeText(text, Charsets.UTF_8)
            if (!tmp.renameTo(cacheFile)) {
                tmp.copyTo(cacheFile, overwrite = true)
                tmp.delete()
            }
        } catch (e: Exception) {
            // 缓存失败不影响内存状态
        }
    }

    /** 仅拉取清单（成功则写入本地缓存，供离线路由使用）。失败抛异常。 */
    suspend fun refreshManifest() {
        val manifest = withContext(Dispatchers.IO) {
            manifestClient.fetchManifest()
        }
        latestManifest = manifest
        writeCache(manifest)
        // 旧布局迁移是纯本地文件操作（零流量），清单就绪即可执行
        migrateLegacyIfNeeded(manifest)
    }

    private suspend fun migrateLegacyIfNeeded(manifest: ModelsManifest) {
        val generalist = manifest.generalist()
        if (packageManager.isModelReady(generalist.modelId) ||
            registry.get(generalist.modelId) != null
        ) {
            return
        }
        val migrated = withContext(Dispatchers.IO) {
            packageManager.migrateLegacyLayout(generalist.modelId)
        }
        if (migrated) {
            registry.markInstalled(
                generalist.modelId,
                generalist.version,
                generalist.sourceSha256,
            )
        }
    }

    /**
     * 拉取清单并同步本地包。
     * 返回 true 表示清单拉取成功（下载失败不影响返回，逐模型容错）。
     */
    suspend fun sync(
        installGeneralistIfMissing: Boolean,
        onModelProgress: (modelId: String, phase: String, doneBytes: Long, totalBytes: Long) -> Unit,
    ): Boolean {
        return withContext(Dispatchers.IO) {
            val manifest = try {
                manifestClient.fetchManifest()
            } catch (e: Exception) {
                null
            } ?: return@withContext false
            latestManifest = manifest
            writeCache(manifest)

            // 旧布局迁移（一次性）：旧 tar 解包复用，平铺文件清理后重新下载
            migrateLegacyIfNeeded(manifest)

            for (model in manifest.models) {
                val installed = registry.get(model.modelId)
                val stale = installed != null &&
                    (installed.sourceSha256 != model.sourceSha256 ||
                        installed.version != model.version)
                val isGeneralist = model.modelId == manifest.generalist().modelId
                val shouldInstall =
                    stale ||
                        (isGeneralist && installed == null && installGeneralistIfMissing)

                if (!shouldInstall) continue

                try {
                    packageManager.ensureModel(model) { phase, done, total ->
                        onModelProgress(model.modelId, phase, done, total)
                    }
                    registry.markInstalled(model.modelId, model.version, model.sourceSha256)
                } catch (e: Exception) {
                    // 单个模型失败不阻断其它模型同步；保留旧包（registry 未更新 = 仍视为过期）
                    onModelProgress(model.modelId, "error", -1, -1)
                }
            }
            true
        }
    }
}
