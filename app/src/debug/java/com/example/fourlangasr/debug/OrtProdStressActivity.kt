package com.example.fourlangasr.debug

import android.app.Activity
import android.os.Build
import android.os.Bundle
import android.os.Debug
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.FloatBuffer
import java.security.MessageDigest

/**
 * 任务2：在**正式 App 的 debug 构建**里，用**实际生产运行库**（app/jniLibs 的
 * libonnxruntime4j_jni.so(补丁桥) + libort4jrt.so + classes.jar）做非 Instrumentation 压力测试。
 * 不加载 sherpa/opencv/vosk/tflite，只验证 OCR 使用的 ORT 运行时本身。
 *
 * 非 instrumentation（am start 触发，无 JDWP），避免上一阶段发现的 am instrument 假挂起。
 * 模型从 filesDir/models/ocr/models/... 读取（run-as 推入，sha256 与 androidTest 同源），
 * 没有则退回 assets。
 *
 * 触发：adb shell am start -n com.example.fourlangasr/.debug.OrtProdStressActivity --ei iters 200
 * 日志 tag = ProdOrtStress
 */
class OrtProdStressActivity : Activity() {
    private val tag = "ProdOrtStress"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val iters = intent?.getIntExtra("iters", 200) ?: 200
        Log.i(tag, "ENV device=${Build.MANUFACTURER}/${Build.MODEL} sdk=${Build.VERSION.SDK_INT} " +
            "abi=${Build.SUPPORTED_ABIS.joinToString(",")} isDebuggerConnected=${Debug.isDebuggerConnected()} " +
            "waitingForDebugger=${Debug.waitingForDebugger()} jdwp=absent(iters=$iters)")
        Thread({ try { run(iters) } catch (t: Throwable) { Log.e(tag, "EXCEPTION ${t.javaClass.simpleName}: ${t.message}") } },
            "ProdOrtRunner").start()
    }

    private fun threadCount() = File("/proc/self/task").listFiles()?.size ?: -1
    private fun fdCount() = File("/proc/self/fd").listFiles()?.size ?: -1
    private fun pssMb(): Double { val mi = Debug.MemoryInfo(); Debug.getMemoryInfo(mi); return mi.totalPss / 1024.0 }
    private fun nativeMb() = Debug.getNativeHeapAllocatedSize() / 1048576.0

    private fun readModel(rel: String): ByteArray {
        val f = File(filesDir, "models/$rel")
        return if (f.isFile) f.readBytes() else assets.open(rel).use { it.readBytes() }
    }
    private fun sha256(b: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(b).joinToString("") { "%02x".format(it) }

    /** 记录真正加载进来的 native 库（验证用的是哪个 ORT / 有没有误加载）。 */
    private fun logLoadedLibs() {
        val libs = runCatching {
            File("/proc/self/maps").readLines()
                .mapNotNull { it.substringAfterLast(' ', "").trim() }
                .filter { it.endsWith(".so") || it.contains("base.apk") }
                .distinct()
                .filter { it.contains("onnxruntime") || it.contains("ort4jrt") || it.contains("opencv") || it.contains("sherpa") || it.contains("vosk") || it.contains("tensorflow") }
        }.getOrDefault(emptyList())
        Log.i(tag, "LOADED_LIBS=${libs.distinct()}")
    }

    private fun run(iters: Int) {
        val detB = readModel("ocr/models/det/inference.onnx")
        val recB = readModel("ocr/models/rec/inference.onnx")
        Log.i(tag, "MODELS det.sha256=${sha256(detB)} rec.sha256=${sha256(recB)} detBytes=${detB.size} recBytes=${recB.size}")
        logLoadedLibs()
        val env = OrtEnvironment.getEnvironment()
        val opts = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
            setIntraOpNumThreads(4)
        }
        val det = env.createSession(detB, opts)
        val rec = env.createSession(recB, opts)
        opts.close()
        logLoadedLibs() // session 建好后再看一次（此时 ORT 库一定已映射）
        val detName = det.inputNames.iterator().next(); val detOut = det.outputNames.iterator().next()
        val recName = rec.inputNames.iterator().next(); val recOut = rec.outputNames.iterator().next()
        val din = FloatArray(1 * 3 * 960 * 1280) { (it % 251) / 255f }; val dshape = longArrayOf(1, 3, 960, 1280)
        val rin = FloatArray(1 * 3 * 48 * 800) { (it % 251) / 255f - 0.5f }; val rshape = longArrayOf(1, 3, 48, 800)
        Log.i(tag, "INPUT det=[1,3,960,1280] rec=[1,3,48,800] perRound=1det+5rec")
        try {
            for (i in 0 until iters) {
                Log.i(tag, "ROUND START iter=$i threads=${threadCount()} fd=${fdCount()}")
                val t0 = System.currentTimeMillis()
                one(env, det, detName, detOut, din, dshape)
                repeat(5) { one(env, rec, recName, recOut, rin, rshape) }
                val dt = System.currentTimeMillis() - t0
                Log.i(tag, "ROUND END iter=$i opMs=$dt threads=${threadCount()} fd=${fdCount()} pss=${pssMb().toInt()}MB native=${nativeMb().toInt()}MB")
            }
            Log.i(tag, "DONE completed=$iters")
        } finally { det.close(); rec.close() }
    }

    private fun one(env: OrtEnvironment, s: OrtSession, inName: String, outName: String, data: FloatArray, shape: LongArray) {
        val t = OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape)
        try {
            s.run(mapOf(inName to t)).use { r ->
                r.get(outName).ifPresent { v -> (v as? OnnxTensor)?.floatBuffer?.let { fb -> fb.duplicate().get(FloatArray(fb.remaining())) } }
            }
        } finally { t.close() }
    }
}
