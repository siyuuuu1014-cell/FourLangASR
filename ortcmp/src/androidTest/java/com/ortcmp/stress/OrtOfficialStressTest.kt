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
 * 官方 ORT 纯净对照（方案 a + C1 配置对照）。只依赖官方 onnxruntime-android，不加载
 * sherpa/opencv/vosk/tflite/翻译。模板/输入/循环与主 App 完全一致：
 * det[1,3,960,1280] + 5×rec[1,3,48,800]，每轮 1 det + 5 rec，BASIC 优化。
 *
 * 通过 instrumentation 参数 -e config 选择**单一变量**配置（其余都是 baseline）：
 *   baseline / intra4(default) / intra2 / intra1 / nospin / noarena / nomempattern / seq
 * -e iters N（默认 200）、-e stallMs（看门狗，默认 15000）。
 *
 * 挂起定位：每轮打 ROUND START/END；每轮内部用 lastStep 记录到子步骤
 * （tensor.create / session.run / tensor.close / session.create / session.close），
 * 看门狗 HANGDUMP 会带 lastStep，可区分「run 内部 / tensor 生命周期 / session 生命周期 / 其他」。
 */
class OrtOfficialStressTest {

    private val tag = "OrtStress"
    private val args = InstrumentationRegistry.getArguments()
    private val iters: Int = args.getString("iters")?.toIntOrNull() ?: 200
    private val stallMs: Long = args.getString("stallMs")?.toLongOrNull() ?: 15_000L
    private val config: String = args.getString("config") ?: "baseline"
    private val reuseTensor: Boolean = args.getString("reuseTensor")?.toBoolean() ?: false

    private fun readAsset(path: String): ByteArray =
        InstrumentationRegistry.getInstrumentation().context.assets.open(path).use { it.readBytes() }

    private fun threadCount() = File("/proc/self/task").listFiles()?.size ?: -1
    private fun fdCount() = File("/proc/self/fd").listFiles()?.size ?: -1
    private fun pssMb(): Double { val mi = Debug.MemoryInfo(); Debug.getMemoryInfo(mi); return mi.totalPss / 1024.0 }
    private fun nativeMb() = Debug.getNativeHeapAllocatedSize() / 1048576.0

    @Volatile private var lastStartedIter = -1
    @Volatile private var lastCompletedIter = -1
    @Volatile private var lastStep = "none"

    private fun deviceInfo(): JSONObject = JSONObject().apply {
        put("manufacturer", Build.MANUFACTURER); put("model", Build.MODEL)
        put("android", Build.VERSION.RELEASE); put("sdkInt", Build.VERSION.SDK_INT)
        put("abi", Build.SUPPORTED_ABIS.joinToString(","))
    }

    /** 依据 config 构造 SessionOptions；只改一个变量，其它保持 baseline。返回给日志用的应用记录。 */
    private fun buildOptions(): Pair<OrtSession.SessionOptions, JSONObject> {
        val opts = OrtSession.SessionOptions()
        val applied = JSONObject()
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
        // baseline: intra-op=4, PARALLEL, spinning 默认开, arena 默认开, mempattern 默认开
        var intra = 4
        when (config) {
            "baseline", "intra4" -> {}
            "intra2" -> { intra = 2 }
            "intra1" -> { intra = 1 }
            "nospin" -> { opts.addConfigEntry("session.intra_op.allow_spinning", "0"); applied.put("allow_spinning", "0") }
            "noarena" -> { opts.setCPUArenaAllocator(false); applied.put("cpu_arena", false) }
            "nomempattern" -> { opts.setMemoryPatternOptimization(false); applied.put("mem_pattern", false) }
            "seq" -> { opts.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL); applied.put("exec_mode", "SEQUENTIAL") }
            else -> throw IllegalArgumentException("unknown config=$config")
        }
        opts.setIntraOpNumThreads(intra)
        applied.put("intra_op", intra)
        applied.put("config", config)
        return opts to applied
    }

    private fun startWatchdog(which: String): AutoCloseable {
        val hb = AtomicLong(System.currentTimeMillis())
        val fired = AtomicBoolean(false); val stop = AtomicBoolean(false)
        val t = Thread {
            while (!stop.get()) {
                try { Thread.sleep(1_000) } catch (e: InterruptedException) { break }
                val idle = System.currentTimeMillis() - hb.get()
                if (idle > stallMs && fired.compareAndSet(false, true)) {
                    Log.e(tag, "HANGDUMP $which config=$config idleMs=$idle lastStartedIter=$lastStartedIter " +
                        "lastCompletedIter=$lastCompletedIter lastStep='$lastStep' threads=${threadCount()} fd=${fdCount()} " +
                        "pss=${r1(pssMb())}MB native=${r1(nativeMb())}MB device=${Build.MANUFACTURER}/${Build.MODEL} sdk=${Build.VERSION.SDK_INT}")
                }
            }
        }
        t.isDaemon = true; t.name = "Watchdog-$which"; t.start()
        return AutoCloseable { stop.set(true); t.interrupt() }
    }

    @Test
    fun officialReuseSessionInference() {
        val (opts, applied) = buildOptions()
        Log.i(tag, "ENV reuse ortVersion=${BuildConfig.ORT_VERSION} config=$config applied=$applied entries=${opts.configEntries} " +
            "device=${Build.MANUFACTURER}/${Build.MODEL} sdk=${Build.VERSION.SDK_INT} iters=$iters stallMs=$stallMs")
        val env = OrtEnvironment.getEnvironment()
        lastStep = "session.create.det"; val det = env.createSession(readAsset(DET), opts)
        lastStep = "session.create.rec"; val rec = env.createSession(readAsset(REC), opts)
        opts.close()
        val detName = det.inputNames.iterator().next(); val recName = rec.inputNames.iterator().next()
        val din = FloatArray(1 * 3 * 960 * 1280) { (it % 251) / 255f }; val dshape = longArrayOf(1, 3, 960, 1280)
        val rin = FloatArray(1 * 3 * 48 * 800) { (it % 251) / 255f - 0.5f }; val rshape = longArrayOf(1, 3, 48, 800)
        val rows = JSONArray(); val wd = startWatchdog("reuse")
        // reuseTensor: 预建一次输入张量并复用，消除每轮 ~15MB×6 的原生分配/释放，
        // 用来判定挂起是否由"每轮张量创建/销毁的分配器抖动"引起（而非 ORT 计算本身）。
        val detT = if (reuseTensor) OnnxTensor.createTensor(env, FloatBuffer.wrap(din), dshape) else null
        val recT = if (reuseTensor) OnnxTensor.createTensor(env, FloatBuffer.wrap(rin), rshape) else null
        Log.i(tag, "reuseTensor=$reuseTensor")
        try {
            for (i in 0 until iters) {
                lastStartedIter = i
                Log.i(tag, "ROUND START reuse config=$config iter=$i threads=${threadCount()} fd=${fdCount()}")
                val t0 = System.currentTimeMillis()
                if (reuseTensor) { runOneReuse(env, det, detT!!, i, "det"); repeat(5) { k -> runOneReuse(env, rec, recT!!, i, "rec$k") } }
                else { runOne(env, det, detName, din, dshape, i, "det"); repeat(5) { k -> runOne(env, rec, recName, rin, rshape, i, "rec$k") } }
                val dt = System.currentTimeMillis() - t0
                lastCompletedIter = i; lastStep = "round.done"
                rows.put(JSONObject().apply {
                    put("iter", i); put("opMs", dt); put("threads", threadCount()); put("fd", fdCount())
                    put("pssMb", r1(pssMb())); put("nativeMb", r1(nativeMb()))
                })
                Log.i(tag, "ROUND END reuse config=$config iter=$i opMs=$dt threads=${threadCount()} fd=${fdCount()} pss=${r1(pssMb())}MB native=${r1(nativeMb())}MB")
            }
        } finally { wd.close(); detT?.close(); recT?.close(); det.close(); rec.close() }
        writeReport("official_reuse_${config}.json", "reuse", applied, rows)
    }

    @Test
    fun officialSessionChurn() {
        val (opts0, applied) = buildOptions(); opts0.close() // 仅用于确认配置合法
        Log.i(tag, "ENV churn ortVersion=${BuildConfig.ORT_VERSION} config=$config applied=$applied device=${Build.MANUFACTURER}/${Build.MODEL} sdk=${Build.VERSION.SDK_INT} iters=$iters")
        val env = OrtEnvironment.getEnvironment()
        val detBytes = readAsset(DET); val recBytes = readAsset(REC)
        val din = FloatArray(1 * 3 * 960 * 1280) { (it % 251) / 255f }; val dshape = longArrayOf(1, 3, 960, 1280)
        val rin = FloatArray(1 * 3 * 48 * 800) { (it % 251) / 255f - 0.5f }; val rshape = longArrayOf(1, 3, 48, 800)
        val rows = JSONArray(); val wd = startWatchdog("churn")
        try {
            for (i in 0 until iters) {
                lastStartedIter = i
                Log.i(tag, "ROUND START churn config=$config iter=$i threads=${threadCount()} fd=${fdCount()}")
                val t0 = System.currentTimeMillis()
                val (opts, _) = buildOptions()
                lastStep = "session.create.det"; val det = env.createSession(detBytes, opts)
                lastStep = "session.create.rec"; val rec = env.createSession(recBytes, opts); opts.close()
                try {
                    val detName = det.inputNames.iterator().next(); val recName = rec.inputNames.iterator().next()
                    runOne(env, det, detName, din, dshape, i, "det")
                    repeat(5) { k -> runOne(env, rec, recName, rin, rshape, i, "rec$k") }
                } finally {
                    lastStep = "session.close.det"; det.close()
                    lastStep = "session.close.rec"; rec.close()
                }
                val dt = System.currentTimeMillis() - t0
                lastCompletedIter = i; lastStep = "round.done"
                rows.put(JSONObject().apply { put("iter", i); put("opMs", dt); put("threads", threadCount()); put("fd", fdCount()); put("pssMb", r1(pssMb())); put("nativeMb", r1(nativeMb())) })
                Log.i(tag, "ROUND END churn config=$config iter=$i opMs=$dt threads=${threadCount()} fd=${fdCount()} native=${r1(nativeMb())}MB")
            }
        } finally { wd.close() }
        writeReport("official_churn_${config}.json", "churn", applied, rows)
    }

    /** 复用预建输入张量的推理（不 create/close 输入张量），仍读取输出，用于隔离分配抖动。 */
    private fun runOneReuse(env: OrtEnvironment, session: OrtSession, tensor: OnnxTensor, iter: Int, step: String) {
        lastStep = "iter=$iter $step session.run(reuse)"
        val name = session.outputNames.iterator().next()
        session.run(mapOf(session.inputNames.iterator().next() to tensor)).use { result ->
            lastStep = "iter=$iter $step result.read(reuse)"
            result.get(name).ifPresent { v -> (v as? OnnxTensor)?.floatBuffer?.let { fb -> fb.duplicate().get(FloatArray(fb.remaining())) } }
        }
    }

    private fun runOne(env: OrtEnvironment, session: OrtSession, inputName: String, data: FloatArray, shape: LongArray, iter: Int, step: String) {
        lastStep = "iter=$iter $step tensor.create"
        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape)
        try {
            lastStep = "iter=$iter $step session.run"
            session.run(mapOf(inputName to tensor)).use { result ->
                lastStep = "iter=$iter $step result.read"
                val name = session.outputNames.iterator().next()
                result.get(name).ifPresent { v -> (v as? OnnxTensor)?.floatBuffer?.let { fb -> fb.duplicate().get(FloatArray(fb.remaining())) } }
            }
        } finally {
            lastStep = "iter=$iter $step tensor.close"
            tensor.close()
        }
    }

    private fun writeReport(name: String, which: String, applied: JSONObject, rows: JSONArray) {
        val dir = File(InstrumentationRegistry.getInstrumentation().context.getExternalFilesDir(null), "ortcmp")
        dir.mkdirs()
        val payload = JSONObject().apply {
            put("build", "official-onnxruntime-android-${BuildConfig.ORT_VERSION}")
            put("mode", which); put("config", config); put("applied", applied)
            put("device", deviceInfo()); put("itersRequested", iters); put("itersCompleted", rows.length())
            put("rows", rows)
        }
        File(dir, name).writeText(payload.toString(1))
        Log.i(tag, "report=${File(dir, name).absolutePath} completed=${rows.length()}/$iters")
        Log.i(tag, "DONE $which config=$config completed=${rows.length()} lastCompletedIter=$lastCompletedIter")
    }

    private fun r1(v: Double): Double = Math.round(v * 10.0) / 10.0

    companion object { private const val DET = "ocr/models/det/inference.onnx"; private const val REC = "ocr/models/rec/inference.onnx" }
}
