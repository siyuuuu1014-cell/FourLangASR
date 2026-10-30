package com.ortcmp.stress

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

/**
 * 非 instrumentation 的压力运行器：用 `am start` 启动本 Activity 后，在后台线程跑与
 * OrtOfficialStressTest 完全相同的循环（官方 ORT、同模型、同大张量、1 det + 5 rec、BASIC）。
 *
 * 用途：区分"挂起是 ORT 运行时问题"还是"am instrument / JDWP 调试环境问题"。
 * instrumentation 会开 JDWP 并可能挂起线程；这里完全没有 JDWP。
 *
 * 触发：adb shell am start -n com.ortcmp.stress/.Placeholder --ei iters 200
 * 结果日志 tag = NonInstrStress（每轮 ROUND START/END）。
 */
class Placeholder : Activity() {
    private val tag = "NonInstrStress"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val iters = intent?.getIntExtra("iters", 200) ?: 200
        Log.i(tag, "ENV device=${Build.MANUFACTURER}/${Build.MODEL} sdk=${Build.VERSION.SDK_INT} iters=$iters jdwp=none")
        Thread({
            try { run(iters) } catch (t: Throwable) {
                Log.e(tag, "EXCEPTION ${t.javaClass.simpleName}: ${t.message}")
            }
        }, "NonInstrRunner").start()
    }

    private fun threadCount() = File("/proc/self/task").listFiles()?.size ?: -1
    private fun fdCount() = File("/proc/self/fd").listFiles()?.size ?: -1
    private fun pssMb(): Double { val mi = Debug.MemoryInfo(); Debug.getMemoryInfo(mi); return mi.totalPss / 1024.0 }
    private fun nativeMb() = Debug.getNativeHeapAllocatedSize() / 1048576.0
    private fun readAsset(p: String): ByteArray {
        // 优先从内部 filesDir 读（run-as 推入的模型，与 androidTest 同源同 sha256），
        // 没有再退回 assets。避免把 6MB 权重打进主 APK。
        val f = File(filesDir, "models/$p")
        if (f.isFile) return f.readBytes()
        return assets.open(p).use { it.readBytes() }
    }

    private fun run(iters: Int) {
        val env = OrtEnvironment.getEnvironment()
        val opts = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
            setIntraOpNumThreads(4)
        }
        val det = env.createSession(readAsset("ocr/models/det/inference.onnx"), opts)
        val rec = env.createSession(readAsset("ocr/models/rec/inference.onnx"), opts)
        opts.close()
        val detName = det.inputNames.iterator().next()
        val recName = rec.inputNames.iterator().next()
        val outDet = det.outputNames.iterator().next()
        val outRec = rec.outputNames.iterator().next()
        val din = FloatArray(1 * 3 * 960 * 1280) { (it % 251) / 255f }; val dshape = longArrayOf(1, 3, 960, 1280)
        val rin = FloatArray(1 * 3 * 48 * 800) { (it % 251) / 255f - 0.5f }; val rshape = longArrayOf(1, 3, 48, 800)
        try {
            for (i in 0 until iters) {
                Log.i(tag, "ROUND START iter=$i threads=${threadCount()} fd=${fdCount()}")
                val t0 = System.currentTimeMillis()
                one(env, det, detName, outDet, din, dshape)
                repeat(5) { one(env, rec, recName, outRec, rin, rshape) }
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
