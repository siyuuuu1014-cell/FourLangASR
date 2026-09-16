package com.example.fourlangasr.translation.model

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

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
) {

    @Volatile
    var latestManifest: ModelsManifest? = null
        private set

    /** 仅拉取清单。失败抛异常，由调用方决定重试/提示。 */
    suspend fun refreshManifest() {
        latestManifest = withContext(Dispatchers.IO) {
            manifestClient.fetchManifest()
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

            // 旧版布局迁移（一次性）：旧 tar 解包复用，平铺文件清理后重新下载
            val generalist = manifest.generalist()
            if (!packageManager.isModelReady(generalist.modelId) &&
                registry.get(generalist.modelId) == null
            ) {
                if (packageManager.migrateLegacyLayout(generalist.modelId)) {
                    registry.markInstalled(
                        generalist.modelId,
                        generalist.version,
                        generalist.sourceSha256,
                    )
                }
            }

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
