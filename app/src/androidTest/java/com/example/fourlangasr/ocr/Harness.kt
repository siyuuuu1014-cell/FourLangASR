package com.example.fourlangasr.ocr

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.core.app.ApplicationProvider
import org.json.JSONArray
import org.json.JSONObject
import java.io.DataInputStream
import java.io.File
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * 阶段1 真机验证的公共工具：资产读取、PSS 采样、报告落盘、WAV 解析。
 *
 * 报告一律写到 app 外部私有目录 `files/ocr_phase1/` 下的同名 json，
 * 便于 `adb pull` 后离线核对，不依赖 logcat 抓取。
 */
object Harness {

    const val LOG_TAG = "OcrPhase1"

    /** 被测应用的 context：引擎、TranslationModules 等都以它为准。 */
    fun context(): Context = ApplicationProvider.getApplicationContext()

    /**
     * assets 必须走 instrumentation context：测试 APK 的 assets 不在被测应用的
     * AssetManager 搜索路径里，用 context() 读会得到 FileNotFoundException。
     */
    fun testContext(): Context =
        androidx.test.platform.app.InstrumentationRegistry.getInstrumentation().context

    fun assetStream(path: String): InputStream = testContext().assets.open(path)

    fun assetBytes(path: String): ByteArray = assetStream(path).use { it.readBytes() }

    fun assetJson(path: String): JSONObject =
        JSONObject(String(assetBytes(path), Charsets.UTF_8))

    /**
     * 解码资产图片，并把长边限制在 [maxLongSide] 以内。
     *
     * 手机原图（12MP ≈ 3969x5292）按 ARGB_8888 解码一次就是 ~84MB Java 堆，
     * 后面 bitmapToMat 还要再拷一份；实测这种图会让进程陷入 GC 风暴，
     * 所以限幅必须在解码时做（inSampleSize），不能等解码完再缩。
     */
    fun assetBitmap(path: String, maxLongSide: Int = 1600): Bitmap {
        val bytes = assetBytes(path)
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
        var sample = 1
        while (maxOf(bounds.outWidth, bounds.outHeight) / (sample * 2) >= maxLongSide) sample *= 2
        val opts = BitmapFactory.Options().apply { inSampleSize = sample }
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size, opts)
            ?: error("无法解码图片: $path")
    }

    /** float32 小端裸数据（PC 端 dump_det_prob_maps.py 的输出格式）。 */
    fun assetFloat32(path: String): FloatArray {
        val bytes = assetBytes(path)
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.nativeOrder())
        require(bytes.size % 4 == 0) { "$path 不是 float32 数组" }
        val out = FloatArray(bytes.size / 4)
        buf.asFloatBuffer().get(out)
        return out
    }

    /** 与 App 内指标同源：MemoryMeter 已经处理了 totalPss 的 KB->MB 换算。 */
    fun pssMb(): Float = com.example.fourlangasr.util.MemoryMeter.totalPssMb()

    /** 解析 16kHz 单声道 16-bit PCM WAV，与 OnDevicePipelineTest 的口径一致。 */
    fun loadWav(assetPath: String): FloatArray {
        assetStream(assetPath).use { raw ->
            val data = DataInputStream(raw.buffered())
            val header = ByteArray(12)
            data.readFully(header)
            require(String(header, 0, 4) == "RIFF" && String(header, 8, 4) == "WAVE") {
                "$assetPath 不是 RIFF/WAVE"
            }
            var pcm: ByteArray? = null
            var sampleRate = 0
            var channels = 0
            var bits = 0
            while (true) {
                val chunkHeader = ByteArray(8)
                try {
                    data.readFully(chunkHeader)
                } catch (e: Exception) {
                    break
                }
                val size = ByteBuffer.wrap(chunkHeader, 4, 4).order(ByteOrder.LITTLE_ENDIAN).int
                val body = ByteArray(size)
                data.readFully(body)
                when (String(chunkHeader, 0, 4)) {
                    "fmt " -> {
                        // fmt 块布局：0 音频格式 / 2 声道 / 4 采样率 / 8 字节率 / 12 块对齐 / 14 位深
                        val le = ByteOrder.LITTLE_ENDIAN
                        channels = ByteBuffer.wrap(body, 2, 2).order(le).short.toInt()
                        sampleRate = ByteBuffer.wrap(body, 4, 4).order(le).int
                        bits = ByteBuffer.wrap(body, 14, 2).order(le).short.toInt()
                    }
                    "data" -> pcm = body
                }
            }
            val pcmBytes = requireNotNull(pcm) { "$assetPath 缺少 data 块" }
            require(sampleRate == 16000) { "$assetPath 采样率 $sampleRate，管线要求 16000" }
            require(channels == 1) { "$assetPath 需要单声道，实际 $channels" }
            require(bits == 16) { "$assetPath 需要 16-bit，实际 $bits" }
            val samples = pcmBytes.size / 2
            val out = FloatArray(samples)
            val buf = ByteBuffer.wrap(pcmBytes).order(ByteOrder.LITTLE_ENDIAN)
            for (i in 0 until samples) {
                out[i] = buf.short.toInt() / 32768f
            }
            return out
        }
    }

    fun deviceInfo(): JSONObject {
        val activity = context()
        return JSONObject().apply {
            put("manufacturer", android.os.Build.MANUFACTURER)
            put("model", android.os.Build.MODEL)
            put("android", android.os.Build.VERSION.RELEASE)
            put("sdkInt", android.os.Build.VERSION.SDK_INT)
            put("abi", android.os.Build.SUPPORTED_ABIS.joinToString(","))
            put("ramTotalMb", (activity.getSystemService(Context.ACTIVITY_SERVICE)
                as android.app.ActivityManager).let { am ->
                val mi = android.app.ActivityManager.MemoryInfo()
                am.getMemoryInfo(mi)
                mi.totalMem / 1048576
            })
            put("cpuCount", Runtime.getRuntime().availableProcessors())
            put("javaMaxHeapMb", Runtime.getRuntime().maxMemory() / 1048576)
        }
    }

    fun stats(values: List<Double>): JSONObject {
        if (values.isEmpty()) return JSONObject()
        val sorted = values.sorted()
        fun pct(p: Double) = sorted[(sorted.size * p).toInt().coerceIn(0, sorted.size - 1)]
        return JSONObject().apply {
            put("n", sorted.size)
            put("mean", round(sorted.average()))
            put("min", round(sorted.first()))
            put("p50", round(pct(0.50)))
            put("p90", round(pct(0.90)))
            put("max", round(sorted.last()))
        }
    }

    fun round(v: Double): Double = Math.round(v * 1000.0) / 1000.0

    /**
     * 写报告并打一行标记，方便 `adb logcat -s System.out:I | grep OcrPhase1` 定位。
     *
     * Android 16 (OPPO PLW110) 上 adb shell 读不到 `/sdcard/Android/data/<pkg>`（scoped storage），
     * 且 connected test 跑完后包若被卸载，外部私有目录连同报告一起消失。所以这里除了落盘，
     * 还把整段 JSON 按块通过 android.util.Log 打出来，保证 `adb logcat` 一定能取到结果。
     */
    fun writeReport(name: String, payload: JSONObject): File {
        val dir = File(context().getExternalFilesDir(null), "ocr_phase1")
        var outFile = dir
        runCatching {
            dir.mkdirs()
            outFile = File(dir, name)
            outFile.writeText(payload.toString(1))
        }.onFailure {
            // 外部目录不可写时退回内部 filesDir（run-as 可读）。
            val internal = File(context().filesDir, "ocr_phase1").apply { mkdirs() }
            outFile = File(internal, name)
            runCatching { outFile.writeText(payload.toString(1)) }
        }
        println("[OcrPhase1] report=${outFile.absolutePath}")
        // 落盘用 pretty，logcat 转储用压缩单行（toString() 无缩进 → 内部无换行，分块可精确重接）。
        dumpToLogcat(name, payload.toString())
        return outFile
    }

    /** 分块打日志，规避 logcat 单条约 4000 字符上限；pull 端按 `##NAME##` 重新拼接。 */
    private fun dumpToLogcat(name: String, json: String) {
        val chunk = 3000
        var i = 0
        var part = 0
        while (i < json.length) {
            val end = minOf(i + chunk, json.length)
            android.util.Log.i(LOG_TAG, "REPORT[$name] part=$part ${json.substring(i, end)}")
            i = end
            part++
        }
        android.util.Log.i(LOG_TAG, "REPORT[$name] part=END")
    }

    fun jsonArray(items: List<JSONObject>): JSONArray {
        val a = JSONArray()
        items.forEach { a.put(it) }
        return a
    }
}
