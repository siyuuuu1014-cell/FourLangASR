package com.example.fourlangasr.translation

import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URL

/**
 * 四语 m2m100 模型包首次下载 + 解包（约 734MB，不打包进 APK）。
 *
 * 服务器端点：GET {baseUrl}/download/model-package（tar 流，Starlette
 * FileResponse 原生支持 HTTP Range → 断点续传）。
 *
 * 流程：分块续传下载到 model-package.tar.part → 完成后改名 .tar →
 * 按 tar 格式解包到 modelDir → 用包内 MANIFEST.json 校验各文件大小。
 * 必须在后台线程调用。
 */
class TranslationDownloadManager(
    private val baseUrl: String = ServerTranslationEngine.DEFAULT_BASE_URL,
    private val modelDir: File,
) {

    companion object {
        const val PACKAGE_ENDPOINT = "/download/model-package"
        const val TAR_FILE = "model-package.tar"
        const val PART_FILE = "model-package.tar.part"
        const val MANIFEST_FILE = "MANIFEST.json"
        const val ENCODER_FILE = "encoder_model.onnx"
        const val DECODER_FILE = "decoder_model_merged.onnx"

        const val PHASE_DOWNLOAD = "download"
        const val PHASE_UNPACK = "unpack"
    }

    /** KitTranslationEngine 可以直接开始翻译。 */
    fun isModelReady(): Boolean =
        File(modelDir, ENCODER_FILE).exists() && File(modelDir, DECODER_FILE).exists()

    fun isPackageDownloaded(): Boolean = File(modelDir, TAR_FILE).exists()

    private fun tarFile(): File = File(modelDir, TAR_FILE)
    private fun partFile(): File = File(modelDir, PART_FILE)

    /**
     * 下载模型包（断点续传）。total 未知时为 -1。
     */
    fun download(onProgress: (downloadedBytes: Long, totalBytes: Long) -> Unit = { _, _ -> }) {
        modelDir.mkdirs()
        val part = partFile()
        val existing = if (part.exists()) part.length() else 0L

        val conn = URL(baseUrl + PACKAGE_ENDPOINT).openConnection() as HttpURLConnection
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
                // Range 越界：说明 .part 已是完整文件（服务器已重建/换包）
                conn.disconnect()
                check(part.renameTo(tarFile())) { "无法将已完成下载的 part 改名" }
                return
            }
            else -> {
                conn.disconnect()
                throw IOException("下载模型包失败：HTTP $code")
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
        check(part.renameTo(tarFile())) { "无法将 part 改名为 $TAR_FILE" }
    }

    private fun parseTotalFromContentRange(header: String?): Long {
        // 形如 "bytes 0-123/456789"
        val h = header ?: return -1
        val slash = h.lastIndexOf('/')
        return if (slash > 0) h.substring(slash + 1).trim().toLongOrNull() ?: -1 else -1
    }

    /**
     * 解包 tar 并用包内 MANIFEST.json 校验各文件大小。
     */
    fun unpack(onProgress: (fileName: String, extractedBytes: Long) -> Unit = { _, _ -> }) {
        val tar = tarFile()
        check(tar.exists()) { "模型包 tar 不存在，请先下载" }
        val manifest = HashMap<String, Long>()
        val total = tar.length()
        tar.inputStream().buffered().use { input ->
            val header = ByteArray(512)
            var pos = 0L
            var sawZeroBlock = false
            while (true) {
                var n = 0
                while (n < 512) {
                    val r = input.read(header, n, 512 - n)
                    if (r < 0) break
                    n += r
                }
                if (n < 512) {
                    if (n == 0) break // 干净 EOF
                    throw IOException("tar 头截断")
                }
                val allZero = header.all { it == 0.toByte() }
                if (allZero) {
                    // 连续两个零块 = 归档结束；单个零块 = 块间填充
                    if (sawZeroBlock) break
                    sawZeroBlock = true
                    pos += 512
                    continue
                }
                sawZeroBlock = false

                val name = parseCString(header, 0, 100)
                val size = parseOctal(header, 124, 12)
                val type = header[156].toInt()
                val dataBlocks = (size + 511) / 512

                when (type) {
                    '0'.code, 0 -> {
                        val outName = name.substringAfterLast('/')
                        if (outName.isNotEmpty()) {
                            FileOutputStream(File(modelDir, outName)).use { out ->
                                copyN(input, size, out)
                                skipN(input, dataBlocks * 512 - size)
                            }
                        } else {
                            skipN(input, dataBlocks * 512)
                        }
                        if (outName == MANIFEST_FILE) {
                            manifest.putAll(readManifest(File(modelDir, outName)))
                        }
                    }
                    '5'.code -> {
                        skipN(input, dataBlocks * 512)
                    }
                    else -> {
                        // pax 扩展头 / 链接 / 其它类型：跳过数据
                        skipN(input, dataBlocks * 512)
                    }
                }
                pos += 512 + dataBlocks * 512
                onProgress(name, pos)
                if (pos >= total) break
            }
        }

        // MANIFEST 校验：包内声明的文件大小与解包结果一致
        val expected = manifest[ENCODER_FILE]
        if (expected != null) {
            val actual = File(modelDir, ENCODER_FILE).length()
            check(actual == expected) { "模型包校验失败：$ENCODER_FILE 实际 $actual != 期望 $expected" }
        }
        val expectedDec = manifest[DECODER_FILE]
        if (expectedDec != null) {
            val actualDec = File(modelDir, DECODER_FILE).length()
            check(actualDec == expectedDec) {
                "模型包校验失败：$DECODER_FILE 实际 $actualDec != 期望 $expectedDec"
            }
        }
        check(isModelReady()) { "解包后仍缺少 ONNX 模型文件" }
    }

    private fun readManifest(file: File): Map<String, Long> {
        val json = JSONObject(file.readText(Charsets.UTF_8))
        val files = json.getJSONObject("files")
        val map = HashMap<String, Long>()
        for (key in files.keys()) {
            map[key] = files.getJSONObject(key).getLong("size_bytes")
        }
        return map
    }

    /** 一步到位：缺模型则下载 + 解包 + 校验。返回是否就绪。 */
    fun ensureModel(
        onProgress: (phase: String, doneBytes: Long, totalBytes: Long) -> Unit = { _, _, _ -> },
    ): Boolean {
        if (isModelReady()) return true
        if (!isPackageDownloaded()) {
            download { d, t -> onProgress(PHASE_DOWNLOAD, d, t) }
        }
        val total = tarFile().length()
        unpack { _, done -> onProgress(PHASE_UNPACK, done, total) }
        return isModelReady()
    }

    private fun parseCString(buf: ByteArray, off: Int, len: Int): String {
        var end = off
        while (end < off + len && buf[end].toInt() != 0) end++
        return String(buf, off, end - off, Charsets.UTF_8)
    }

    private fun parseOctal(buf: ByteArray, off: Int, len: Int): Long {
        var s = 0
        var e = len
        // 跳过前导空格/NUL，去尾空格/NUL
        while (s < e && (buf[off + s] == ' '.code.toByte() || buf[off + s] == 0.toByte())) s++
        while (e > s && (buf[off + e - 1] == ' '.code.toByte() || buf[off + e - 1] == 0.toByte())) e--
        if (s == e) return 0
        var v = 0L
        for (i in s until e) {
            val c = buf[off + i].toInt() - '0'.code
            if (c !in 0..7) return 0
            v = v * 8 + c
        }
        return v
    }

    private fun copyN(input: InputStream, n: Long, out: FileOutputStream) {
        var remaining = n
        val buf = ByteArray(256 * 1024)
        while (remaining > 0) {
            val r = input.read(buf, 0, minOf(buf.size.toLong(), remaining).toInt())
            if (r < 0) throw IOException("tar 数据截断")
            out.write(buf, 0, r)
            remaining -= r
        }
    }

    private fun skipN(input: InputStream, n: Long) {
        var remaining = n
        val buf = ByteArray(64 * 1024)
        while (remaining > 0) {
            val r = input.read(buf, 0, minOf(buf.size.toLong(), remaining).toInt())
            if (r < 0) throw IOException("tar 数据截断")
            remaining -= r
        }
    }
}
