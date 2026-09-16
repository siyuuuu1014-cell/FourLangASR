package com.example.fourlangasr.translation.model

import com.example.fourlangasr.translation.tokenizer.M2m100Tokenizer
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

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
    private val baseUrl: String,
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

    /** 包已解包且关键文件齐全（可加载运行时配置 + 分词器 + 引擎）。 */
    fun isModelReady(modelId: String): Boolean {
        val dir = modelDir(modelId)
        return listOf(ENCODER_FILE, DECODER_FILE, SPM_FILE, VOCAB_FILE, PACKAGE_MANIFEST_FILE)
            .all { File(dir, it).isFile }
    }

    fun isPackageDownloaded(modelId: String): Boolean =
        File(modelDir(modelId), TAR_FILE).isFile

    /** 下载模型包（Range 断点续传；total 未知时为 -1）。 */
    fun download(
        model: ManifestModel,
        onProgress: (downloadedBytes: Long, totalBytes: Long) -> Unit = { _, _ -> },
    ) {
        val dir = modelDir(model.modelId)
        dir.mkdirs()
        val part = File(dir, PART_FILE)
        val existing = if (part.isFile) part.length() else 0L

        val conn = URL(baseUrl + model.downloadUrl).openConnection() as HttpURLConnection
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
                // Range 越界：.part 已是完整文件（服务器已重建包）
                conn.disconnect()
                check(part.renameTo(File(dir, TAR_FILE))) { "无法将已完成的 part 改名" }
                return
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
            throw IOException("下载大小与 manifest 不符：$downloaded != ${model.sizeBytes}")
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

    /** 包内 MANIFEST.json 的 files.size_bytes 与解包结果比对。 */
    fun verifyPackage(modelId: String) {
        val dir = modelDir(modelId)
        val manifestFile = File(dir, PACKAGE_MANIFEST_FILE)
        check(manifestFile.isFile) { "缺少 $PACKAGE_MANIFEST_FILE" }
        val files = JSONObject(manifestFile.readText(Charsets.UTF_8)).getJSONObject("files")
        for (name in files.keys()) {
            val expected = files.getJSONObject(name).getLong("size_bytes")
            val actual = File(dir, name).length()
            check(actual == expected) { "包校验失败：$name 实际 $actual != 期望 $expected" }
        }
    }

    /** 一步到位：缺包/未解包则下载 + 解包 + 校验。 */
    fun ensureModel(
        model: ManifestModel,
        onProgress: (phase: String, doneBytes: Long, totalBytes: Long) -> Unit = { _, _, _ -> },
    ): Boolean {
        if (isModelReady(model.modelId)) return true
        if (!isPackageDownloaded(model.modelId)) {
            download(model) { d, t -> onProgress(PHASE_DOWNLOAD, d, t) }
        }
        val total = File(modelDir(model.modelId), TAR_FILE).length()
        unpack(model.modelId) { _, done -> onProgress(PHASE_UNPACK, done, total) }
        return isModelReady(model.modelId)
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
                dir.deleteRecursively()
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
            return try {
                verifyPackage(generalistId)
                true
            } catch (e: Exception) {
                dir.deleteRecursively()
                false
            }
        }
        File(legacyRoot, ENCODER_FILE).delete()
        File(legacyRoot, DECODER_FILE).delete()
        File(legacyRoot, "model-package.tar.part").delete()
        return false
    }

    // ---------------------------------------------------------------- 运行时资源加载

    /** 包内 MANIFEST.json → 引擎运行时配置（结构参数 + 语言 token + 特殊 token）。 */
    fun loadRuntimeConfig(modelId: String, family: String): ModelRuntimeConfig {
        val dir = modelDir(modelId)
        val json = File(dir, PACKAGE_MANIFEST_FILE).readText(Charsets.UTF_8)
        return ModelRuntimeConfigParser.parse(json, modelId, family)
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
