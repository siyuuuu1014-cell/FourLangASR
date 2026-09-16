package com.example.fourlangasr.translation.model

import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.nio.charset.StandardCharsets

/**
 * 最小 tar（ustar 兼容）解包器：模型包是服务器 `tar` 生成的平铺文件。
 * 支持普通文件 / 目录 / pax 扩展头跳过；路径取 basename 落盘。
 */
object TarExtractor {

    /**
     * 解包 [input] 到 [destDir]。返回 (解包字节数, MANIFEST.json 原文?)。
     */
    fun extract(input: InputStream, destDir: File, onProgress: (fileName: String, doneBytes: Long) -> Unit) {
        destDir.mkdirs()
        val header = ByteArray(512)
        var done = 0L
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
                        FileOutputStream(File(destDir, outName)).use { out ->
                            copyN(input, size, out)
                            skipN(input, dataBlocks * 512 - size)
                        }
                    } else {
                        skipN(input, dataBlocks * 512)
                    }
                }
                '5'.code -> skipN(input, dataBlocks * 512)
                else -> skipN(input, dataBlocks * 512) // pax 头 / 链接等
            }
            done += 512 + dataBlocks * 512
            onProgress(name, done)
        }
    }

    private fun parseCString(buf: ByteArray, off: Int, len: Int): String {
        var end = off
        while (end < off + len && buf[end].toInt() != 0) end++
        return String(buf, off, end - off, StandardCharsets.UTF_8)
    }

    private fun parseOctal(buf: ByteArray, off: Int, len: Int): Long {
        var s = 0
        var e = len
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
