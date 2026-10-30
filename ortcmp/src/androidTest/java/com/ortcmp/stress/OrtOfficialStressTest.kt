package com.ortcmp.stress

import android.os.Debug
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import java.io.File
import java.nio.FloatBuffer

/**
 * 方案(a) 纯净对照：只用官方 onnxruntime-android:1.27.0（未改名/未打补丁），
 * 不加载 sherpa/opencv/vosk/tflite/翻译。用与主 App 里 libort4jrt.so 完全相同的
 * 模型、大张量尺寸、SessionOptions(BASIC + intra-op=4)、调用顺序（1 det + 5 rec）
 * 连续跑 >=200 次，看官方原版是否也挂。
 *
 * 每轮打印迭代号 + 线程数 + FD + PSS + native + 单帧耗时，便于定位首次挂起的迭代。
 */
class OrtOfficialStressTest {

    private val tag = "OrtStress"
    private val iters: Int =
        InstrumentationRegistry.getArguments().getString("iters")?.toIntOrNull() ?: 200
    private val doSessionChurn: Boolean =
        InstrumentationRegistry.getArguments().getString("churn")?.toBoolean() ?: false

    private fun threadCount() = File("/proc/self/task").listFiles()?.size ?: -1
    private fun fdCount() = File("/proc/self/fd").listFiles()?.size ?: -1
    private fun pssMb(): Double {
        val mi = Debug.MemoryInfo()
        Debug.getMemoryInfo(mi)
        return mi.totalPss / 1024.0
    }
    private fun nativeMb() = Debug.getNativeHeapAllocatedSize() / 1048576.0

    private fun readAsset(path: String): ByteArray =
        InstrumentationRegistry.getInstrumentation().targetContext.assets.open(path).use { it.readBytes() }

    @Test
    fun officialRepeatInference() {
        val env = OrtEnvironment.getEnvironment()
        val opts = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
            setIntraOpNumThreads(4)
        }
        val detBytes = readAsset("ocr/models/det/inference.onnx")
        val recBytes = readAsset("ocr/models/rec/inference.onnx")
        Log.i(tag, "creating sessions (official ORT), det=${detBytes.size} rec=${recBytes.size}")
        val det = env.createSession(detBytes, opts)
        val rec = env.createSession(recBytes, opts)
        val detName = det.inputNames.iterator().next()
        val recName = rec.inputNames.iterator().next()
        Log.i(tag, "sessions created; detInput=$detName recInput=$recName")

        val dshape = longArrayOf(1, 3, 960, 1280)
        val din = FloatArray(1 * 3 * 960 * 1280) { (it % 251) / 255f }
        val rshape = longArrayOf(1, 3, 48, 800)
        val rin = FloatArray(1 * 3 * 48 * 800) { (it % 251) / 255f - 0.5f }

        val rows = JSONArray()
        var firstHangIter = -1
        try {
            for (i in 0 until iters) {
                val t0 = System.currentTimeMillis()
                runOne(env, det, detName, din, dshape)
                repeat(5) { runOne(env, rec, recName, rin, rshape) }
                val dt = System.currentTimeMillis() - t0
                val row = JSONObject().apply {
                    put("iter", i); put("opMs", dt)
                    put("threads", threadCount()); put("fd", fdCount())
                    put("pssMb", r1(pssMb())); put("nativeMb", r1(nativeMb()))
                }
                rows.put(row)
                if (i < 3 || i % 20 == 0 || i == iters - 1) Log.i(tag, "ROW $row")
                if (dt > 5000 && firstHangIter < 0) firstHangIter = i
            }
        } catch (t: Throwable) {
            Log.e(tag, "EXCEPTION after ${rows.length()} iters: ${t.javaClass.simpleName} ${t.message}")
        } finally {
            det.close(); rec.close(); opts.close()
        }
        writeReport("official_result.json", rows, firstHangIter)
        Log.i(tag, "DONE iters=${rows.length()} completed without hang (firstHangIter=$firstHangIter)")
    }

    private fun runOne(env: OrtEnvironment, session: OrtSession, inputName: String, data: FloatArray, shape: LongArray) {
        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape)
        try {
            session.run(mapOf(inputName to tensor)).use { result ->
                // 触发一次输出读取，与主 App 路径一致（拷贝 floatBuffer）
                val name = session.outputNames.iterator().next()
                result.get(name).ifPresent { v -> (v as? OnnxTensor)?.floatBuffer?.let { fb -> FloatArray(fb.remaining()).also { fb.duplicate().get(it) } } }
            }
        } finally {
            tensor.close()
        }
    }

    private fun writeReport(name: String, rows: JSONArray, firstHangIter: Int) {
        val dir = File(InstrumentationRegistry.getInstrumentation().targetContext.getExternalFilesDir(null), "ortcmp")
        dir.mkdirs()
        val payload = JSONObject().apply {
            put("build", "official-onnxruntime-android-1.27.0")
            put("itersRequested", iters)
            put("itersCompleted", rows.length())
            put("firstSlowIter", firstHangIter)
            put("rows", rows)
        }
        File(dir, name).writeText(payload.toString(1))
        Log.i(tag, "report=${File(dir, name).absolutePath}")
    }

    private fun r1(v: Double): Double = Math.round(v * 10.0) / 10.0
}
