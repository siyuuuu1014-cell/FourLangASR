package com.ortcmp.stress

import android.os.Build
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
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/**
 * 方案(a) 纯净对照：只用官方 onnxruntime-android:1.27.0（未改名/未打补丁），
 * 不加载 sherpa/opencv/vosk/tflite/翻译。
 *
 * 两个独立测试：
 *  - officialReuseSessionInference：Session 建一次、反复推理（对应主 App 的 N 测试）；
 *  - officialSessionChurn：每轮新建 det+rec Session、推理、关闭（隔离会话生命周期）。
 * 两者都用与主 App 完全相同的模型（Gradle 同步并校验 sha256）、大张量
 * det[1,3,960,1280] / rec[1,3,48,800]、SessionOptions(BASIC + intra-op=4)、每轮 1 det + 5 rec。
 *
 * 每轮打 START/END 日志；看门狗在某轮停滞超过 stallMs 时打 HANGDUMP（含已完成轮次、
 * 当前迭代、线程数、FD、PSS、native heap、设备信息）。进程冻结时以最后一条 START/END 判定挂起轮次。
 */
class OrtOfficialStressTest {

    private val tag = "OrtStress"
    private val args = InstrumentationRegistry.getArguments()
    private val iters: Int = args.getString("iters")?.toIntOrNull() ?: 200
    private val stallMs: Long = args.getString("stallMs")?.toLongOrNull() ?: 20_000L

    // test APK 的 assets 在 instrumentation context 下；targetContext 读不到。
    private fun readAsset(path: String): ByteArray =
        InstrumentationRegistry.getInstrumentation().context.assets.open(path).use { it.readBytes() }

    private fun threadCount() = File("/proc/self/task").listFiles()?.size ?: -1
    private fun fdCount() = File("/proc/self/fd").listFiles()?.size ?: -1
    private fun pssMb(): Double {
        val mi = Debug.MemoryInfo(); Debug.getMemoryInfo(mi); return mi.totalPss / 1024.0
    }
    private fun nativeMb() = Debug.getNativeHeapAllocatedSize() / 1048576.0

    @Volatile private var lastStartedIter = -1
    @Volatile private var lastCompletedIter = -1

    private fun deviceInfo(): JSONObject = JSONObject().apply {
        put("manufacturer", Build.MANUFACTURER); put("model", Build.MODEL)
        put("android", Build.VERSION.RELEASE); put("sdkInt", Build.VERSION.SDK_INT)
        put("abi", Build.SUPPORTED_ABIS.joinToString(","))
    }

    private fun logEnv(which: String) {
        Log.i(tag, "ENV $which device=${Build.MANUFACTURER}/${Build.MODEL} sdk=${Build.VERSION.SDK_INT} abi=${Build.SUPPORTED_ABIS.joinToString(",")} iters=$iters stallMs=$stallMs")
    }

    /** 看门狗：heartbeat 停滞超过 stallMs 就 dump 一次当前资源与进度（进程冻结时这是唯一线索）。 */
    private fun startWatchdog(which: String): AutoCloseable {
        val hb = AtomicLong(System.currentTimeMillis())
        val fired = AtomicBoolean(false)
        val stop = AtomicBoolean(false)
        val t = Thread {
            while (!stop.get()) {
                try { Thread.sleep(1_000) } catch (e: InterruptedException) { break }
                val idle = System.currentTimeMillis() - hb.get()
                if (idle > stallMs && fired.compareAndSet(false, true)) {
                    Log.e(tag, "HANGDUMP $which idleMs=$idle lastStartedIter=$lastStartedIter lastCompletedIter=$lastCompletedIter " +
                        "threads=${threadCount()} fd=${fdCount()} pss=${r1(pssMb())}MB native=${r1(nativeMb())}MB " +
                        "device=${Build.MANUFACTURER}/${Build.MODEL} sdk=${Build.VERSION.SDK_INT}")
                }
            }
        }
        t.isDaemon = true; t.name = "Watchdog-$which"; t.start()
        return AutoCloseable { stop.set(true); t.interrupt() }
    }

    @Test
    fun officialReuseSessionInference() {
        logEnv("reuse")
        val env = OrtEnvironment.getEnvironment()
        val opts = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
            setIntraOpNumThreads(4)
        }
        val det = try { env.createSession(readAsset(DET), opts) } finally { opts.close() }
        val recOpts = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
            setIntraOpNumThreads(4)
        }
        val rec = try { env.createSession(readAsset(REC), recOpts) } finally { recOpts.close() }
        val detName = det.inputNames.iterator().next()
        val recName = rec.inputNames.iterator().next()
        val din = FloatArray(1 * 3 * 960 * 1280) { (it % 251) / 255f }
        val dshape = longArrayOf(1, 3, 960, 1280)
        val rin = FloatArray(1 * 3 * 48 * 800) { (it % 251) / 255f - 0.5f }
        val rshape = longArrayOf(1, 3, 48, 800)

        val rows = JSONArray()
        val wd = startWatchdog("reuse")
        try {
            for (i in 0 until iters) {
                lastStartedIter = i
                Log.i(tag, "START reuse iter=$i threads=${threadCount()} fd=${fdCount()}")
                val t0 = System.currentTimeMillis()
                runOne(env, det, detName, din, dshape)
                repeat(5) { runOne(env, rec, recName, rin, rshape) }
                val dt = System.currentTimeMillis() - t0
                lastCompletedIter = i
                val row = JSONObject().apply {
                    put("iter", i); put("opMs", dt); put("threads", threadCount()); put("fd", fdCount())
                    put("pssMb", r1(pssMb())); put("nativeMb", r1(nativeMb()))
                }
                rows.put(row)
                Log.i(tag, "END reuse iter=$i opMs=$dt threads=${threadCount()} fd=${fdCount()} pss=${r1(pssMb())}MB native=${r1(nativeMb())}MB")
            }
        } finally {
            wd.close(); det.close(); rec.close()
        }
        writeReport("official_reuse_result.json", "reuse", rows)
    }

    @Test
    fun officialSessionChurn() {
        logEnv("churn")
        val env = OrtEnvironment.getEnvironment()
        val detBytes = readAsset(DET)
        val recBytes = readAsset(REC)
        val din = FloatArray(1 * 3 * 960 * 1280) { (it % 251) / 255f }
        val dshape = longArrayOf(1, 3, 960, 1280)
        val rin = FloatArray(1 * 3 * 48 * 800) { (it % 251) / 255f - 0.5f }
        val rshape = longArrayOf(1, 3, 48, 800)

        val rows = JSONArray()
        val wd = startWatchdog("churn")
        try {
            for (i in 0 until iters) {
                lastStartedIter = i
                Log.i(tag, "START churn iter=$i threads=${threadCount()} fd=${fdCount()}")
                val t0 = System.currentTimeMillis()
                val opts = OrtSession.SessionOptions().apply {
                    setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
                    setIntraOpNumThreads(4)
                }
                val det = try { env.createSession(detBytes, opts) } finally { opts.close() }
                val recOpts = OrtSession.SessionOptions().apply {
                    setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
                    setIntraOpNumThreads(4)
                }
                val rec = try { env.createSession(recBytes, recOpts) } finally { recOpts.close() }
                try {
                    val detName = det.inputNames.iterator().next()
                    val recName = rec.inputNames.iterator().next()
                    runOne(env, det, detName, din, dshape)
                    repeat(5) { runOne(env, rec, recName, rin, rshape) }
                } finally {
                    det.close(); rec.close()
                }
                val dt = System.currentTimeMillis() - t0
                lastCompletedIter = i
                val row = JSONObject().apply {
                    put("iter", i); put("opMs", dt); put("threads", threadCount()); put("fd", fdCount())
                    put("pssMb", r1(pssMb())); put("nativeMb", r1(nativeMb()))
                }
                rows.put(row)
                Log.i(tag, "END churn iter=$i opMs=$dt threads=${threadCount()} fd=${fdCount()} pss=${r1(pssMb())}MB native=${r1(nativeMb())}MB")
            }
        } finally {
            wd.close()
        }
        writeReport("official_churn_result.json", "churn", rows)
    }

    private fun runOne(env: OrtEnvironment, session: OrtSession, inputName: String, data: FloatArray, shape: LongArray) {
        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape)
        try {
            session.run(mapOf(inputName to tensor)).use { result ->
                val name = session.outputNames.iterator().next()
                result.get(name).ifPresent { v ->
                    (v as? OnnxTensor)?.floatBuffer?.let { fb -> fb.duplicate().get(FloatArray(fb.remaining())) }
                }
            }
        } finally {
            tensor.close()
        }
    }

    private fun writeReport(name: String, which: String, rows: JSONArray) {
        val dir = File(InstrumentationRegistry.getInstrumentation().context.getExternalFilesDir(null), "ortcmp")
        dir.mkdirs()
        val payload = JSONObject().apply {
            put("build", "official-onnxruntime-android-1.27.0")
            put("mode", which)
            put("device", deviceInfo())
            put("itersRequested", iters)
            put("itersCompleted", rows.length())
            put("rows", rows)
        }
        File(dir, name).writeText(payload.toString(1))
        Log.i(tag, "report=${File(dir, name).absolutePath} completed=${rows.length()}/$iters")
        Log.i(tag, "DONE $which completed=${rows.length()} lastCompletedIter=$lastCompletedIter")
    }

    private fun r1(v: Double): Double = Math.round(v * 10.0) / 10.0

    companion object {
        private const val DET = "ocr/models/det/inference.onnx"
        private const val REC = "ocr/models/rec/inference.onnx"
    }
}
