package com.example.fourlangasr.translation.model

import com.example.fourlangasr.translation.tokenizer.M2m100Tokenizer
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.ConcurrentHashMap

/**
 * 单模型包管理：下载（断点续传）→ 解包 → 完整性校验 → 运行时配置/分词器加载。
 *
 * 目录布局（modelsRoot 下按 model_id 隔离）：
 *   modelsRoot/<model_id>/
 *     package.tar(.part)              下载的 tar（可续传，完成后保留以便重装）
 *     encoder_model.onnx / decoder_model_merged.onnx   模型权重
 *     sentencepiece.bpe.model / vocab.json / added_tokens.json  分词器
 *     MANIFEST.json                   包内清单（文件大小 + 语言 token + 结构参数）
 *
 * 模型 id、下载 URL、期望大小全部来自 [ManifestModel]，本类不写死任何模型。
 * 必须在后台线程调用。
 */
class ModelPackageManager(
    private val baseUrlProvider: () -> String,
    private val modelsRoot: File,
) {

    companion object {
        const val ENCODER_FILE = "encoder_model.onnx"
        const val DECODER_FILE = "decoder_model_merged.onnx"
        const val SPM_FILE = "sentencepiece.bpe.model"
        const val VOCAB_FILE = "vocab.json"
        const val ADDED_TOKENS_FILE = "added_tokens.json"
        const val PACKAGE_MANIFEST_FILE = "MANIFEST.json"
        const val TAR_FILE = "package.tar"
        const val PART_FILE = "package.tar.part"

        const val PHASE_DOWNLOAD = "download"
        const val PHASE_UNPACK = "unpack"
    }

    fun modelDir(modelId: String): File = File(modelsRoot, modelId)

    /**
     * 同一 model_id 的下载/解包必须串行：App 启动的 Wi-Fi 同步与前台下载服务
     * 共享同一个 ModelPackageManager 实例，两路并发会交错写同一个 package.tar.part。
     */
    private val modelLocks = ConcurrentHashMap<String, Any>()

    private fun lockFor(modelId: String): Any =
        modelLocks.computeIfAbsent(modelId) { Any() }

    /** 包已解包且关键文件齐全（可加载运行时配置 + 分词器 + 引擎）。 */
    fun isModelReady(modelId: String): Boolean {
        val dir = modelDir(modelId)
        return listOf(ENCODER_FILE, DECODER_FILE, SPM_FILE, VOCAB_FILE, PACKAGE_MANIFEST_FILE)
            .all { File(dir, it).isFile }
    }

    fun isPackageDownloaded(modelId: String): Boolean =
        File(modelDir(modelId), TAR_FILE).isFile

    /**
     * 解包结果与包内 MANIFEST 声明的文件大小一致。
     * 清单没声明 size 的文件不参与判定（老包不能因此被误判损坏而重下 734MB）。
     */
    private fun extractedSizesMatch(modelId: String): Boolean {
        val dir = modelDir(modelId)
        val manifestFile = File(dir, PACKAGE_MANIFEST_FILE)
        if (!manifestFile.isFile) return false
        val files = runCatching {
            JSONObject(manifestFile.readText(Charsets.UTF_8)).getJSONObject("files")
        }.getOrNull() ?: return true
        for (name in listOf(ENCODER_FILE, DECODER_FILE, SPM_FILE, VOCAB_FILE)) {
            val expected = files.optJSONObject(name)?.optLong("size_bytes", -1L) ?: -1L
            if (expected > 0 && File(dir, name).length() != expected) return false
        }
        return true
    }

    /**
     * 包内 MANIFEST 的 source_sha256 与服务器清单声明的是否同一个模型。
     * 这是唯一独立于「包自声明大小」的校验锚点：服务器重建包/发错包/续传拼出
     * 混血文件时，大小可能恰好对上，权重指纹不会。任一侧缺失则视为无法判定（放过）。
     */
    private fun sourceShaMatches(modelId: String, model: ManifestModel): Boolean {
        val expected = model.sourceSha256.trim()
        if (expected.isEmpty()) return true
        val manifestFile = File(modelDir(modelId), PACKAGE_MANIFEST_FILE)
        if (!manifestFile.isFile) return false
        val actual = runCatching {
            JSONObject(manifestFile.readText(Charsets.UTF_8)).optString("source_sha256", "")
        }.getOrNull()?.trim().orEmpty()
        if (actual.isEmpty()) return true
        return actual.equals(expected, ignoreCase = true)
    }

    /**
     * 已就绪 + 大小自洽 + 指纹匹配，才算真正可用（避免半损坏包被当成安装完成）。
     * 四语总模型的包内 MANIFEST 没有 source_sha256 字段，[sourceShaMatches] 对它是空操作，
     * 只有专项包（两处都声明了指纹）才真正参与判定。
     */
    private fun isInstalledUsable(modelId: String, model: ManifestModel): Boolean =
        isModelReady(modelId) && extractedSizesMatch(modelId) && sourceShaMatches(modelId, model)

    /**
     * 下载模型包（Range 断点续传；total 未知时为 -1）。
     * [retriedFresh] 仅在 416 校验失败后丢弃 .part 重下时为 true，防止递归。
     */
    fun download(
        model: ManifestModel,
        onProgress: (downloadedBytes: Long, totalBytes: Long) -> Unit = { _, _ -> },
        retriedFresh: Boolean = false,
    ) {
        val dir = modelDir(model.modelId)
        dir.mkdirs()
        val part = File(dir, PART_FILE)
        val existing = if (part.isFile) part.length() else 0L

        val conn = URL(baseUrlProvider() + model.downloadUrl).openConnection() as HttpURLConnection
        conn.connectTimeout = 15_000
        conn.readTimeout = 60_000
        if (existing > 0) {
            conn.setRequestProperty("Range", "bytes=$existing-")
        }
        val code = conn.responseCode
        val append: Boolean
        val total: Long
        var downloaded = existing
        when (code) {
            206 -> {
                append = true
                total = parseTotalFromContentRange(conn.getHeaderField("Content-Range"))
            }
            200 -> {
                append = false
                downloaded = 0
                total = conn.contentLengthLong
            }
            416 -> {
                // 416 只表示 Range 起点越界，不代表 .part 内容完整（服务器换过更短的包同样回 416）。
                // 用 Content-Range 的总长与本地 .part 比对，合不上就丢弃重下，不能无条件转正。
                val remoteTotal = parseTotalFromContentRange(conn.getHeaderField("Content-Range"))
                conn.disconnect()
                val sizeOk = remoteTotal > 0 && part.length() == remoteTotal &&
                    (model.sizeBytes <= 0 || remoteTotal == model.sizeBytes)
                if (sizeOk) {
                    check(part.renameTo(File(dir, TAR_FILE))) { "无法将已完成的 part 改名" }
                    return
                }
                part.delete()
                if (!retriedFresh) {
                    download(model, onProgress, retriedFresh = true)
                    return
                }
                throw IOException("下载 ${model.modelId} 失败：Range 越界且 .part 大小与远端不符")
            }
            else -> {
                conn.disconnect()
                throw IOException("下载 ${model.modelId} 失败：HTTP $code")
            }
        }
        try {
            conn.inputStream.use { input ->
                FileOutputStream(part, append).use { out ->
                    val buf = ByteArray(256 * 1024)
                    while (true) {
                        val n = input.read(buf)
                        if (n < 0) break
                        out.write(buf, 0, n)
                        downloaded += n
                        onProgress(downloaded, total)
                    }
                }
            }
        } finally {
            conn.disconnect()
        }
        if (total > 0 && downloaded < total) {
            throw IOException("下载不完整：$downloaded/$total，可再次调用续传")
        }
        if (model.sizeBytes > 0 && downloaded != model.sizeBytes) {
            // manifest 声明的 size_bytes 可能与实际 tar 大小有出入（后端打包后未同步声明），
            // 真正的完整性由解包 + 包内 MANIFEST 校验保证，这里只告警不阻断。
            android.util.Log.w(
                "FourLangTranslation",
                "${model.modelId}: 实际大小 $downloaded != manifest 声明 ${model.sizeBytes}",
            )
        }
        check(part.renameTo(File(dir, TAR_FILE))) { "无法将 part 改名为 $TAR_FILE" }
    }

    private fun parseTotalFromContentRange(header: String?): Long {
        val h = header ?: return -1
        val slash = h.lastIndexOf('/')
        return if (slash > 0) h.substring(slash + 1).trim().toLongOrNull() ?: -1 else -1
    }

    /** 解包 tar，并按包内 MANIFEST.json 校验各文件大小。 */
    fun unpack(modelId: String, onProgress: (fileName: String, doneBytes: Long) -> Unit = { _, _ -> }) {
        val dir = modelDir(modelId)
        val tar = File(dir, TAR_FILE)
        check(tar.isFile) { "模型包 tar 不存在，请先下载" }
        tar.inputStream().buffered().use { input ->
            TarExtractor.extract(input, dir, onProgress)
        }
        verifyPackage(modelId)
        check(isModelReady(modelId)) { "解包后仍缺少关键文件" }
    }

    /**
     * 包内 MANIFEST.json 的 files.size_bytes 与解包结果比对。
     * 只校验运行时必需文件（辅助文件如 vocab.txt 缺失不影响使用，
     * 不应导致整个包被判定损坏）。
     */
    fun verifyPackage(modelId: String) {
        val dir = modelDir(modelId)
        val manifestFile = File(dir, PACKAGE_MANIFEST_FILE)
        check(manifestFile.isFile) { "缺少 $PACKAGE_MANIFEST_FILE" }
        val files = JSONObject(manifestFile.readText(Charsets.UTF_8)).getJSONObject("files")
        for (name in listOf(ENCODER_FILE, DECODER_FILE, SPM_FILE, VOCAB_FILE)) {
            val entry = files.optJSONObject(name)
                ?: throw IllegalStateException("$modelId: MANIFEST 缺少 $name 条目")
            val expected = entry.getLong("size_bytes")
            val actual = File(dir, name).length()
            check(actual == expected) { "包校验失败：$name 实际 $actual != 期望 $expected" }
        }
    }

    /** 一步到位：缺包/未解包/校验不过则下载 + 解包 + 校验（同模型串行）。 */
    fun ensureModel(
        model: ManifestModel,
        onProgress: (phase: String, doneBytes: Long, totalBytes: Long) -> Unit = { _, _, _ -> },
    ): Boolean {
        val id = model.modelId
        synchronized(lockFor(id)) {
            if (isInstalledUsable(id, model)) return true

            val tar = File(modelDir(id), TAR_FILE)

            // 文件在但校验不过（解包被中断/旧包）：先用现成 tar 重解，省一次 734MB 下载。
            if (isModelReady(id) && tar.isFile) {
                runCatching { unpack(id) }.onFailure { tar.delete() }
                if (isInstalledUsable(id, model)) return true
            }

            if (!tar.isFile) {
                download(model, { d, t -> onProgress(PHASE_DOWNLOAD, d, t) })
            }
            val total = tar.length()
            try {
                unpack(id) { _, done -> onProgress(PHASE_UNPACK, done, total) }
            } catch (e: Exception) {
                // 坏 tar 必须删掉：否则下次见 tar 还在就跳过下载，解包又失败，永远修不好。
                tar.delete()
                throw e
            }
            if (!sourceShaMatches(id, model)) {
                val declared = runCatching {
                    JSONObject(File(modelDir(id), PACKAGE_MANIFEST_FILE).readText(Charsets.UTF_8))
                        .optString("source_sha256", "")
                }.getOrNull().orEmpty()
                removePackage(id)
                throw IOException(
                    "$id: 包内权重指纹与服务器清单不符（包内 ${declared.take(12)}… / " +
                        "清单 ${model.sourceSha256.take(12)}…），已清除以便重新下载",
                )
            }
            return isModelReady(id)
        }
    }

    /** 删除模型目录（含 tar 与解包文件）。 */
    fun removePackage(modelId: String) {
        modelDir(modelId).deleteRecursively()
    }

    /**
     * 旧版布局一次性迁移：旧下载器把包放在 modelsRoot 的上一级。
     * - 有旧 tar → 移入 models/<generalistId>/ 并解包（免重新下载）；
     * - 平铺文件齐全（onnx + spm + vocab + MANIFEST）→ 移入模型目录复用；
     * - 平铺但不齐全（无 spm/vocab/MANIFEST，无法补全）→ 删除释放空间，
     *   由 sync 重新下载完整包。
     * 返回迁移后总模型是否就绪。
     */
    fun migrateLegacyLayout(generalistId: String): Boolean {
        val legacyRoot = modelsRoot.parentFile ?: return false
        val dir = modelDir(generalistId)
        val oldTar = File(legacyRoot, "model-package.tar")
        if (oldTar.isFile) {
            dir.mkdirs()
            oldTar.renameTo(File(dir, TAR_FILE))
            return try {
                unpack(generalistId)
                true
            } catch (e: Exception) {
                // 解包失败视为 tar 损坏：删除 tar 以便重新下载（不销毁其它内容）
                File(dir, TAR_FILE).delete()
                false
            }
        }
        val required = listOf(
            ENCODER_FILE,
            DECODER_FILE,
            SPM_FILE,
            VOCAB_FILE,
            PACKAGE_MANIFEST_FILE,
        )
        if (required.all { File(legacyRoot, it).isFile }) {
            dir.mkdirs()
            for (f in required + ADDED_TOKENS_FILE) {
                val src = File(legacyRoot, f)
                if (src.isFile) src.renameTo(File(dir, f))
            }
            // 校验失败不销毁已迁移的文件：缺的只是辅助文件时模型仍可用，
            // 由 sync 按需重新下载完整包补齐（下载会覆盖/补全目录）。
            return try {
                verifyPackage(generalistId)
                true
            } catch (e: Exception) {
                false
            }
        }
        File(legacyRoot, ENCODER_FILE).delete()
        File(legacyRoot, DECODER_FILE).delete()
        File(legacyRoot, "model-package.tar.part").delete()
        return false
    }

    // ---------------------------------------------------------------- 运行时资源加载

    /** 包内 MANIFEST.json（+ config.json / generation_config.json）→ 引擎运行时配置。 */
    fun loadRuntimeConfig(modelId: String, family: String): ModelRuntimeConfig {
        val dir = modelDir(modelId)
        val manifestJson = File(dir, PACKAGE_MANIFEST_FILE).readText(Charsets.UTF_8)
        val configFile = File(dir, "config.json")
        val generationFile = File(dir, "generation_config.json")
        return ModelRuntimeConfigParser.parse(
            packageManifestJson = manifestJson,
            configJson = configFile.takeIf { it.isFile }?.readText(Charsets.UTF_8),
            generationConfigJson = generationFile.takeIf { it.isFile }?.readText(Charsets.UTF_8),
            modelId = modelId,
            family = family,
        )
    }

    /** 包内 sentencepiece 原始文件 → 分词器（首次构建，之后缓存于内存）。 */
    fun loadTokenizer(modelId: String, languages: Map<String, Long>): M2m100Tokenizer {
        val dir = modelDir(modelId)
        val spmBytes = File(dir, SPM_FILE).readBytes()
        val vocabText = File(dir, VOCAB_FILE).readText(Charsets.UTF_8)
        val addedTokensFile = File(dir, ADDED_TOKENS_FILE)
        val addedTokensText = if (addedTokensFile.isFile) {
            addedTokensFile.readText(Charsets.UTF_8)
        } else {
            null
        }
        return M2m100Tokenizer.fromSpm(spmBytes, vocabText, addedTokensText, languages)
    }
}
