package com.example.fourlangasr.ocr

import com.example.fourlangasr.ocr.paddle.EngineConfig
import com.example.fourlangasr.ocr.paddle.engine.ORTSessionManager
import com.example.fourlangasr.ocr.runtime.OcrModelCatalog
import com.example.fourlangasr.ocr.runtime.OcrRuntime
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import org.opencv.core.Core
import java.util.concurrent.atomic.AtomicLong

/**
 * 阶段2B-0：把"重复操作挂起"拆到最小面，逐面定位，且每面独立运行（用 am instrument 单方法跑，
 * 一个卡死不影响其它）。每组用 [HangProbe] 记录迭代号 / 线程数 / FD / 内存，
 * 配 [HangProbe.watchdog] 在某次操作停滞时抓全线程栈。
 *
 * 关键区分：
 *  - A/B/C 用**同一个** session 反复 run()，看是不是"每次推理"漏资源；
 *  - D 反复 create/close session，看是不是"会话生命周期"漏资源；
 *  - E 走引擎池，看生产复用模式下的真实表现。
 * 这样能把阶段 2A 里混在一起的"重建"和"连跑"两种挂起分开归因。
 */
class OcrHangIsolationTest {

    private val iters: Int =
        InstrumentationRegistry.getArguments().getString("iters")?.toIntOrNull() ?: 60

    private fun detInput(): Pair<FloatArray, LongArray> {
        val shape = longArrayOf(1, 3, 64, 64)
        return FloatArray(1 * 3 * 64 * 64) { (it % 251) / 255f } to shape
    }

    private fun recInput(): Pair<FloatArray, LongArray> {
        val shape = longArrayOf(1, 3, 48, 320)
        return FloatArray(1 * 3 * 48 * 320) { (it % 251) / 255f - 0.5f } to shape
    }

    /** A：只反复跑 Det 推理（同一 session）。 */
    @Test
    fun a_repeatDetInference() = repeatInfer("det")

    /** B：只反复跑 Rec 推理（同一 session）。 */
    @Test
    fun b_repeatRecInference() = repeatInfer("rec")

    private fun repeatInfer(which: String) {
        Harness.assertTrueOpenCv()
        val mgr = ORTSessionManager(Harness.testContext(), EngineConfig(4))
        mgr.loadModels(OcrModelCatalog.tinyDet.assetPath, OcrModelCatalog.tinyRec.assetPath)
        val (input, shape) = if (which == "det") detInput() else recInput()
        val hb = AtomicLong(System.currentTimeMillis())
        val wd = HangProbe.watchdog(hb, "A-$which", 6000)
        val rows = JSONArray()
        try {
            for (i in 0 until iters) {
                val t0 = System.currentTimeMillis()
                if (which == "det") mgr.runDetection(input, shape) else mgr.runRecognition(input, shape)
                hb.set(System.currentTimeMillis())
                val dt = System.currentTimeMillis() - t0
                if (true) {
                    val r = probeRow(which, i, dt)
                    rows.put(r)
                    Log(r.toString())
                }
            }
        } finally {
            wd.close(); mgr.release()
        }
        writeReport("h2b0_ab_${which}.json", which, rows)
    }

    /** C：完整 Det+Rec 管线，同一引擎反复 run()（阶段2A f3 的放大版）。 */
    @Test
    fun c_repeatFullPipeline() {
        Harness.assertTrueOpenCv()
        val rt = OcrRuntime(Harness.testContext())
        val engine = rt.engineFor(OcrModelCatalog.uzbek)
        val bmp = Harness.assetBitmap("ocr/images/print_00.png")
        val hb = AtomicLong(System.currentTimeMillis())
        val wd = HangProbe.watchdog(hb, "C-full", 6000)
        val rows = JSONArray()
        try {
            for (i in 0 until iters) {
                val t0 = System.currentTimeMillis()
                val res = engine.run(bmp)
                hb.set(System.currentTimeMillis())
                val dt = System.currentTimeMillis() - t0
                if (true) {
                    val r = probeRow("full", i, dt, "lines=${res.lineCount}")
                    rows.put(r); Log(r.toString())
                }
            }
        } finally {
            wd.close(); bmp.recycle(); rt.releaseAll()
        }
        writeReport("h2b0_c_full.json", "full", rows)
    }

    /**
     * I：完整管线，但把 ORT intra-op 线程数设成 1。
     * 若 C 的挂起是"ORT 线程池与 OpenCV 线程池交错竞争"，单线程 ORT 应显著改变（或消除）挂起。
     * 这是判定根因是否为线程池竞争的决定性对照。
     */
    @Test
    fun i_fullPipelineSingleThreadOrt() {
        Harness.assertTrueOpenCv()
        val engine = com.example.fourlangasr.ocr.paddle.engine.OCREngine(
            Harness.testContext(),
            com.example.fourlangasr.ocr.paddle.PaddleOCRConfig(),
            EngineConfig(1),
            OcrModelCatalog.tinyDet.assetPath,
            OcrModelCatalog.tinyRec.assetPath,
            OcrModelCatalog.tinyRec.configAssetPath,
        )
        val bmp = Harness.assetBitmap("ocr/images/print_00.png")
        val hb = AtomicLong(System.currentTimeMillis())
        val wd = HangProbe.watchdog(hb, "I-single", 6000)
        val rows = JSONArray()
        try {
            for (i in 0 until iters) {
                val t0 = System.currentTimeMillis()
                val res = engine.run(bmp)
                hb.set(System.currentTimeMillis())
                val r = probeRow("single-thread", i, System.currentTimeMillis() - t0, "lines=${res.lineCount}")
                rows.put(r); Log(r.toString())
            }
        } finally { wd.close(); bmp.recycle(); engine.release() }
        writeReport("h2b0_i_single_thread.json", "single-thread", rows)
    }

    /**
     * L：完整管线，但把 OpenCV 内部线程数设为 1（禁用 parallel_for 多线程）。
     * 若挂起是"OpenCV 线程池 × ORT 线程池交错竞争"，限制 OpenCV 线程应能规避。
     */
    @Test
    fun l_fullPipelineSingleThreadOpenCv() {
        Harness.assertTrueOpenCv()
        Core.setNumThreads(1)
        val rt = OcrRuntime(Harness.testContext())
        val engine = rt.engineFor(OcrModelCatalog.uzbek)
        val bmp = Harness.assetBitmap("ocr/images/print_00.png")
        val hb = AtomicLong(System.currentTimeMillis())
        val wd = HangProbe.watchdog(hb, "L-ocv1", 6000)
        val rows = JSONArray()
        try {
            for (i in 0 until iters) {
                val t0 = System.currentTimeMillis()
                val res = engine.run(bmp)
                hb.set(System.currentTimeMillis())
                val r = probeRow("opencv1", i, System.currentTimeMillis() - t0, "lines=${res.lineCount}")
                rows.put(r); Log(r.toString())
            }
        } finally { wd.close(); bmp.recycle(); rt.releaseAll(); Core.setNumThreads(0) }
        writeReport("h2b0_l_opencv1.json", "opencv1", rows)
    }

    /**
     * M：完整管线，ORT intra-op=1 且 OpenCV 线程=1 同时限制。
     * I 只限 ORT、L 只限 OpenCV 都仍挂；若两者同时限线程能跑完，
     * 说明挂起来自"两个原生线程池同时活跃"的交互。
     */
    @Test
    fun m_bothSingleThreaded() {
        Harness.assertTrueOpenCv()
        Core.setNumThreads(1)
        val engine = com.example.fourlangasr.ocr.paddle.engine.OCREngine(
            Harness.testContext(),
            com.example.fourlangasr.ocr.paddle.PaddleOCRConfig(),
            EngineConfig(1),
            OcrModelCatalog.tinyDet.assetPath,
            OcrModelCatalog.tinyRec.assetPath,
            OcrModelCatalog.tinyRec.configAssetPath,
        )
        val bmp = Harness.assetBitmap("ocr/images/print_00.png")
        val hb = AtomicLong(System.currentTimeMillis())
        val wd = HangProbe.watchdog(hb, "M-both1", 6000)
        val rows = JSONArray()
        try {
            for (i in 0 until iters) {
                val t0 = System.currentTimeMillis()
                val res = engine.run(bmp)
                hb.set(System.currentTimeMillis())
                val r = probeRow("both1", i, System.currentTimeMillis() - t0, "lines=${res.lineCount}")
                rows.put(r); Log(r.toString())
            }
        } finally { wd.close(); bmp.recycle(); engine.release(); Core.setNumThreads(0) }
        writeReport("h2b0_m_both_single.json", "both1", rows)
    }

    /**
     * N：只用 ORT，但喂**真实尺寸的大张量**（det 1×3×960×1280、rec 1×3×48×800），
     * 循环里完全不碰 OpenCV。用来把"大张量让 ORT intra-op 线程池真正并行"与
     * "OpenCV 交错调用"两个变量分开：
     *  - 若 N 挂 → 触发条件是 ORT 处理大张量时的线程池，与 OpenCV 无关；
     *  - 若 N 不挂 → 必须 OpenCV 与 ORT 交错才挂（回到 G 只跑小张量不挂的对照）。
     */
    @Test
    fun n_largeOrtTensorsNoOpenCv() {
        Harness.assertTrueOpenCv()
        val mgr = ORTSessionManager(Harness.testContext(), EngineConfig(4))
        mgr.loadModels(OcrModelCatalog.tinyDet.assetPath, OcrModelCatalog.tinyRec.assetPath)
        val dshape = longArrayOf(1, 3, 960, 1280)
        val din = FloatArray(1 * 3 * 960 * 1280) { (it % 251) / 255f }
        val rshape = longArrayOf(1, 3, 48, 800)
        val rin = FloatArray(1 * 3 * 48 * 800) { (it % 251) / 255f - 0.5f }
        val hb = AtomicLong(System.currentTimeMillis())
        val wd = HangProbe.watchdog(hb, "N-large", 6000)
        val rows = JSONArray()
        try {
            for (i in 0 until iters) {
                val t0 = System.currentTimeMillis()
                mgr.runDetection(din, dshape)
                repeat(5) { mgr.runRecognition(rin, rshape) }
                hb.set(System.currentTimeMillis())
                val r = probeRow("large-ort", i, System.currentTimeMillis() - t0)
                rows.put(r); Log(r.toString())
            }
        } finally { wd.close(); mgr.release() }
        writeReport("h2b0_n_large_ort.json", "large-ort", rows)
    }

    /** D：反复 create→run→close 会话（阶段2A f1 的放大版），隔离"会话生命周期"。 */
    @Test
    fun d_repeatSessionLifecycle() {
        Harness.assertTrueOpenCv()
        val (input, shape) = detInput()
        val hb = AtomicLong(System.currentTimeMillis())
        val wd = HangProbe.watchdog(hb, "D-lifecycle", 6000)
        val rows = JSONArray()
        for (i in 0 until iters) {
            val t0 = System.currentTimeMillis()
            val mgr = ORTSessionManager(Harness.testContext(), EngineConfig(4))
            mgr.loadModels(OcrModelCatalog.tinyDet.assetPath, OcrModelCatalog.tinyRec.assetPath)
            mgr.runDetection(input, shape)
            mgr.release()
            hb.set(System.currentTimeMillis())
            val dt = System.currentTimeMillis() - t0
            if (true) {
                val r = probeRow("lifecycle", i, dt)
                rows.put(r); Log(r.toString())
            }
        }
        wd.close()
        writeReport("h2b0_d_lifecycle.json", "lifecycle", rows)
    }

    /** E：引擎池连续识别不同图片（生产复用模式）。 */
    @Test
    fun e_poolAcrossImages() {
        Harness.assertTrueOpenCv()
        val rt = OcrRuntime(Harness.testContext())
        val assets = listOf(
            "ocr/images/print_00.png", "ocr/images/print_01.png", "ocr/images/print_02.png",
            "ocr/images/hand_00.jpg", "ocr/images/hand_01.jpg",
        )
        val hb = AtomicLong(System.currentTimeMillis())
        val wd = HangProbe.watchdog(hb, "E-pool", 6000)
        val rows = JSONArray()
        try {
            for (i in 0 until iters) {
                val path = assets[i % assets.size]
                val bmp = Harness.assetBitmap(path)
                val t0 = System.currentTimeMillis()
                rt.engineFor(OcrModelCatalog.uzbek).run(bmp)
                hb.set(System.currentTimeMillis())
                val dt = System.currentTimeMillis() - t0
                bmp.recycle()
                if (true) {
                    val r = probeRow("pool", i, dt, "img=${path.substringAfterLast('/')}")
                    rows.put(r); Log(r.toString())
                }
            }
        } finally {
            wd.close(); rt.releaseAll()
        }
        writeReport("h2b0_e_pool.json", "pool", rows)
    }

    /** G：交替 det+rec 推理，纯 ORT，不经任何 OpenCV。隔离"ORT 交替调用"这一面。 */
    @Test
    fun g_alternatingDetRecNoOpenCv() {
        Harness.assertTrueOpenCv()
        val mgr = ORTSessionManager(Harness.testContext(), EngineConfig(4))
        mgr.loadModels(OcrModelCatalog.tinyDet.assetPath, OcrModelCatalog.tinyRec.assetPath)
        val (din, dshape) = detInput()
        val (rin, rshape) = recInput()
        val hb = AtomicLong(System.currentTimeMillis())
        val wd = HangProbe.watchdog(hb, "G-alt", 6000)
        val rows = JSONArray()
        try {
            for (i in 0 until iters) {
                val t0 = System.currentTimeMillis()
                mgr.runDetection(din, dshape)   // 1 次 det
                repeat(5) { mgr.runRecognition(rin, rshape) } // 模拟一页 5 行
                hb.set(System.currentTimeMillis())
                val r = probeRow("alt-ort", i, System.currentTimeMillis() - t0)
                rows.put(r); Log(r.toString())
            }
        } finally { wd.close(); mgr.release() }
        writeReport("h2b0_g_alt_ort.json", "alt-ort", rows)
    }

    /** H：只跑 OpenCV 重活（解码→det 前处理→DB 后处理→裁剪），完全不碰 ORT 推理。隔离 OpenCV。 */
    @Test
    fun h_openCvPipelineNoOrt() {
        Harness.assertTrueOpenCv()
        val bmp = Harness.assetBitmap("ocr/images/print_00.png")
        // 先做一次真 det 拿到一张概率图，之后循环只复用这张概率图跑 OpenCV 侧。
        val mgr = ORTSessionManager(Harness.testContext(), EngineConfig(4))
        mgr.loadModels(OcrModelCatalog.tinyDet.assetPath, OcrModelCatalog.tinyRec.assetPath)
        val pcm = com.example.fourlangasr.ocr.paddle.preprocess.DetPreprocessor.preprocess(bmp, 64, "min", 1280, "BGR")
        val (pred, pshape) = mgr.runDetection(pcm.tensorData, pcm.shape)
        mgr.release()
        val hb = AtomicLong(System.currentTimeMillis())
        val wd = HangProbe.watchdog(hb, "H-opencv", 6000)
        val rows = JSONArray()
        try {
            for (i in 0 until iters) {
                val t0 = System.currentTimeMillis()
                // 纯 OpenCV：前处理(缩放/拆分/归一化) + DB(threshold/findContours/minAreaRect/unclip) + 裁剪
                val p = com.example.fourlangasr.ocr.paddle.preprocess.DetPreprocessor.preprocess(bmp, 64, "min", 1280, "BGR")
                val boxes = com.example.fourlangasr.ocr.paddle.postprocess.DBPostProcessor.process(
                    pred, pshape, 0.3f, 0.6f, 1.5f, 3000, false, "fast", "quad", p.originalH, p.originalW)
                val srcMat = com.example.fourlangasr.ocr.paddle.util.BitmapUtils.bitmapToBGRMat(bmp)
                for (b in boxes.take(8)) {
                    val crop = com.example.fourlangasr.ocr.paddle.postprocess.QuadTextCrop.crop(srcMat, b)
                    crop.release()
                }
                srcMat.release()
                hb.set(System.currentTimeMillis())
                val r = probeRow("opencv", i, System.currentTimeMillis() - t0, "boxes=${boxes.size}")
                rows.put(r); Log(r.toString())
            }
        } finally { wd.close(); bmp.recycle() }
        writeReport("h2b0_h_opencv.json", "opencv", rows)
    }

    private fun probeRow(which: String, i: Int, dt: Long, extra: String = ""): JSONObject =
        JSONObject().apply {
            put("which", which)
            put("iter", i)
            put("opMs", dt)
            put("threads", HangProbe.threadCount())
            put("fd", HangProbe.fdCount())
            put("javaMb", HangProbe.r(HangProbe.javaHeapUsedMb()))
            put("nativeMb", HangProbe.r(HangProbe.nativeHeapMb()))
            put("pssMb", HangProbe.r(HangProbe.pssMb()))
            put("thread", Thread.currentThread().name)
            if (extra.isNotEmpty()) put("extra", extra)
        }

    private fun writeReport(name: String, which: String, rows: JSONArray) {
        Harness.writeReport(name, JSONObject().apply {
            put("test", "h2b0-$which")
            put("device", Harness.deviceInfo())
            put("iters", iters)
            put("rows", rows)
        })
    }

    private fun Log(s: String) = android.util.Log.i(Harness.LOG_TAG, "PROBE $s")
}
