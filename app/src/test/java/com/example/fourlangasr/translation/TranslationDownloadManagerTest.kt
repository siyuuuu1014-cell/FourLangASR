package com.example.fourlangasr.translation

import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayOutputStream
import java.io.File

/**
 * TranslationDownloadManager 的 tar 解包 + MANIFEST 校验（纯 JVM）。
 */
class TranslationDownloadManagerTest {

    @Test
    fun unpack_extractsFilesAndVerifiesSizes() {
        val dir = File.createTempFile("tarTest", "").also { it.delete(); it.mkdirs() }
        try {
            val enc = ByteArray(1000) { 7 }
            val dec = ByteArray(2000) { 9 }
            val manifest = JSONObject()
                .put(
                    "files",
                    JSONObject()
                        .put(
                            "encoder_model.onnx",
                            JSONObject().put("size_bytes", enc.size.toLong()),
                        )
                        .put(
                            "decoder_model_merged.onnx",
                            JSONObject().put("size_bytes", dec.size.toLong()),
                        ),
                )
                .toString()
                .toByteArray()

            val tar = buildTar(
                mapOf(
                    "encoder_model.onnx" to enc,
                    "decoder_model_merged.onnx" to dec,
                    "MANIFEST.json" to manifest,
                ),
            )
            File(dir, TranslationDownloadManager.TAR_FILE).writeBytes(tar)

            val mgr = TranslationDownloadManager(baseUrl = "http://localhost:0", modelDir = dir)
            mgr.unpack()

            assertArrayEquals(enc, File(dir, "encoder_model.onnx").readBytes())
            assertArrayEquals(dec, File(dir, "decoder_model_merged.onnx").readBytes())
            assertTrue(mgr.isModelReady())
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun unpack_rejectsSizeMismatch() {
        val dir = File.createTempFile("tarTest", "").also { it.delete(); it.mkdirs() }
        try {
            val enc = ByteArray(100) { 1 }
            val manifest = JSONObject()
                .put(
                    "files",
                    JSONObject()
                        .put(
                            "encoder_model.onnx",
                            JSONObject().put("size_bytes", 999L), // 与实际不符
                        )
                        .put(
                            "decoder_model_merged.onnx",
                            JSONObject().put("size_bytes", 100L),
                        ),
                )
                .toString()
                .toByteArray()
            val tar = buildTar(
                mapOf(
                    "encoder_model.onnx" to enc,
                    "decoder_model_merged.onnx" to ByteArray(100),
                    "MANIFEST.json" to manifest,
                ),
            )
            File(dir, TranslationDownloadManager.TAR_FILE).writeBytes(tar)

            val mgr = TranslationDownloadManager(baseUrl = "http://localhost:0", modelDir = dir)
            var threw = false
            try {
                mgr.unpack()
            } catch (e: IllegalStateException) {
                threw = true
            }
            assertTrue("MANIFEST 大小不符应抛异常", threw)
        } finally {
            dir.deleteRecursively()
        }
    }

    /** 构造一个最小合法 tar（512 字节头 + 数据 + 填充 + 两个零块结尾）。 */
    private fun buildTar(files: Map<String, ByteArray>): ByteArray {
        val out = ByteArrayOutputStream()
        for ((name, data) in files) {
            val header = ByteArray(512)
            name.toByteArray(Charsets.UTF_8).copyInto(header, 0)
            val oct = data.size.toString(8)
            oct.toByteArray().copyInto(header, 124 + 12 - oct.length)
            header[156] = '0'.code.toByte()
            out.write(header)
            out.write(data)
            val pad = (512 - data.size % 512) % 512
            out.write(ByteArray(pad))
        }
        out.write(ByteArray(1024)) // 两个 512 零块 = 归档结束
        return out.toByteArray()
    }
}
