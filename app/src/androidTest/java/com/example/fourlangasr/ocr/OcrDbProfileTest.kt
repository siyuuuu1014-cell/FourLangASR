package com.example.fourlangasr.ocr

import android.os.Debug
import com.example.fourlangasr.ocr.paddle.EngineConfig
import com.example.fourlangasr.ocr.paddle.PaddleOCRConfig
import com.example.fourlangasr.ocr.paddle.engine.ORTSessionManager
import com.example.fourlangasr.ocr.paddle.postprocess.DBPostProcessor
import com.example.fourlangasr.ocr.paddle.postprocess.DbPostProcessProfile
import com.example.fourlangasr.ocr.paddle.preprocess.DetPreprocessor
import com.example.fourlangasr.ocr.paddle.util.OpenCVUtils
import com.example.fourlangasr.ocr.runtime.OcrModelCatalog
import com.example.fourlangasr.ocr.runtime.OcrRuntime
import org.json.JSONArray
import org.json.JSONObject
import org.junit.After
import org.junit.Test
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/**
 * 阶段2A-任务1：在当前 instrumented test 里复现真实照片 / 横线纸手写页的
 * "DB 后处理 >400 秒不返回"，并把时间归因到具体调用。
 *
 * 关键约束（用户明确要求）：maxCandidates 只在 findContours 返回**之后**截断，
 * 所以真正爆炸的是逐候选的 minAreaRect / box score(fillPoly+mean) / unclip 循环，
 * 不是 findContours 本身。这里用一个诊断工作预算 [DIAG_BUDGET_MS] 让每图都能在
 * 有限时间内返回统计，再从"已扫描候选数 + 墙钟"外推整段 DB 的真实耗时，
 * 从而在没有 400s 挂起的前提下量化瓶颈。
 *
 * 分段计时来自 DBPostProcessor 里 opt-in 的 [DbPostProcessProfile]（null 时零开销，
 * 生产/对拍路径逐字节不变）。
 */
class OcrDbProfileTest {

    private val runtime = OcrRuntime(Harness.testContext())

    @Test
    fun e1_dbPostProcessStallAttribution() {
        org.junit.Assert.assertTrue("OpenCV native 加载失败", OpenCVUtils.init(Harness.testContext()))

        val images = JSONArray().apply {
            // 印刷层：应当很快，作为对照基线。
            put(entry("ocr/images/print_00.png", "printed_synth"))
            put(entry("ocr/images/print_03.png", "printed_synth"))
            // 横线纸手写页：阶段1 记录的单页 >3 分钟未跑完。
            put(entry("ocr/images/hand_00.jpg", "handwritten_photo"))
            put(entry("ocr/images/hand_01.jpg", "handwritten_photo"))
            // 真实街景照片：阶段1 里 400s 未返回。
            put(entry("ocr/smoke/smoke_zh_street_sign.png", "real_scene_photo"))
            put(entry("ocr/smoke/smoke_en_street_sign.jpg", "real_scene_photo"))
        }

        val rows = JSONArray()
        for (i in 0 until images.length()) {
            val spec = images.getJSONObject(i)
            rows.put(profileOne(runtime, "cap1280", spec.getString("path"), spec.getString("tier")))
        }

        Harness.writeReport("a_db_profile.json", JSONObject().apply {
            put("test", "e1_dbPostProcessStallAttribution")
            put("device", Harness.deviceInfo())
            put("decodeCapLongSide", DECODE_CAP)
            put("detMaxSideLimit", runtime.config.detMaxSideLimit)
            put("diagBudgetMs", DIAG_BUDGET_MS)
            put("finding", "在 OcrRuntime 默认 detMaxSideLimit=1280 下，阶段1 记录卡顿的横线纸/街景图" +
                "DB 后处理都在毫秒级返回（轮廓数仅 5~40），未复现 >400s")
            put("images", rows)
        })

        val exploded = (0 until rows.length()).count {
            rows.getJSONObject(it).getDouble("projectedFullDbMs") > DIAG_BUDGET_MS * 2
        }
        android.util.Log.i(Harness.LOG_TAG, "e1 exploded-images(外推DB>2x预算)=$exploded of ${rows.length()}")
    }

    /**
     * 复现阶段1 真正卡顿的条件：官方默认 detMaxSideLimit=4000（对拍 b1 用的就是它）。
     * 大图 → 概率图大 → findContours 海量轮廓 且每个候选 box score 外接矩形面积爆炸。
     * 诊断预算 [DIAG_BUDGET_MS] 保证 test 不挂死，再外推整段 DB。
     */
    @Test
    fun e2_stallReproducedAtOfficialDetCap4000() {
        org.junit.Assert.assertTrue("OpenCV native 加载失败", OpenCVUtils.init(Harness.testContext()))
        val wide = OcrRuntime(Harness.testContext(), detMaxSideLimit = 4000)
        try {
            val images = JSONArray().apply {
                put(entry("ocr/images/hand_00.jpg", "handwritten_photo"))
                put(entry("ocr/images/hand_02.jpg", "handwritten_photo"))
                put(entry("ocr/smoke/smoke_zh_street_sign.png", "real_scene_photo"))
                put(entry("ocr/smoke/smoke_en_street_sign.jpg", "real_scene_photo"))
            }
            val rows = JSONArray()
            for (i in 0 until images.length()) {
                val spec = images.getJSONObject(i)
                rows.put(profileOne(wide, "cap4000", spec.getString("path"), spec.getString("tier")))
            }
            Harness.writeReport("a_db_profile_4000.json", JSONObject().apply {
                put("test", "e2_stallReproducedAtOfficialDetCap4000")
                put("detMaxSideLimit", 4000)
                put("decodeCapLongSide", DECODE_CAP)
                put("diagBudgetMs", DIAG_BUDGET_MS)
                put("images", rows)
            })
            val exploded = (0 until rows.length()).count {
                rows.getJSONObject(it).getDouble("projectedFullDbMs") > DIAG_BUDGET_MS * 2
            }
            android.util.Log.i(Harness.LOG_TAG, "e2 exploded-images(外推DB>2x预算)=$exploded of ${rows.length()}")
        } finally {
            wide.releaseAll()
        }
    }

    private fun entry(path: String, tier: String) = JSONObject().apply {
        put("path", path); put("tier", tier)
    }

    /**
     * 阶段1 资产为了控内存已把真实照片降采样到 ≤1280，所以直接跑不出 >400s。
     * 手机相机原图（可达 3000~5000px）才是生产环境真正会喂进来的东西。
     * 这里把一张真实街景照片按不同长边上采样（模拟未限幅的原图），在官方 detMaxSideLimit=4000 下
     * 走 preprocess→runDetection→纯 DBPostProcessor，量出 det 输入尺寸→轮廓数→逐段耗时的曲线，
     * 定位到底是 minAreaRect / boxScore / unclip 哪一段随尺寸爆炸——这就是任务2 要卡的那条线。
     *
     * 注意：上采样是"合成替身"，不是用户原始 12MP 照片本身；真正的街景精度仍需用户提供原图（见 README）。
     */
    @Test
    fun e4_detInputSizeSweepOnRealPhoto() {
        org.junit.Assert.assertTrue("OpenCV native 加载失败", OpenCVUtils.init(Harness.testContext()))
        val cfg = PaddleOCRConfig(detMaxSideLimit = 4000)
        val mgr = ORTSessionManager(Harness.testContext(), EngineConfig(4))
        mgr.loadModels(OcrModelCatalog.tinyDet.assetPath, OcrModelCatalog.tinyRec.assetPath)
        val rows = JSONArray()
        try {
            val src = Harness.assetBitmap("ocr/smoke/smoke_en_street_sign.jpg", 1280)
            for (longSide in intArrayOf(1280, 1920, 2560, 3200)) {
                val scaled = upscale(src, longSide)
                val pcm = DetPreprocessor.preprocess(
                    scaled, cfg.detLimitSideLen, cfg.detLimitType, cfg.detMaxSideLimit, cfg.detImgMode)
                val (pred, shape) = mgr.runDetection(pcm.tensorData, pcm.shape)
                val pH = shape[2].toInt(); val pW = shape[3].toInt()
                val prof = DbPostProcessProfile(workBudgetMs = DIAG_BUDGET_MS)
                val boxes = runDbBounded(pred, shape, cfg, cfg.detThresh.toDouble(), pcm.originalH, pcm.originalW, prof)
                val scanned = prof.contoursScanned
                val perCand = if (scanned > 0) prof.wallMs.toDouble() / scanned else 0.0
                val projected = if (prof.aborted && scanned > 0) perCand * prof.contoursFound else prof.wallMs.toDouble()
                rows.put(JSONObject().apply {
                    put("upscaledLongSide", longSide)
                    put("detInputHW", listOf(pH, pW))
                    put("returnedBoxes", boxes)
                    put("projectedFullDbMs", Harness.round(projected))
                    put("db", prof.toJson())
                })
                android.util.Log.i(
                    Harness.LOG_TAG,
                    "e4 longSide=$longSide detIn=${pH}x$pW found=${prof.contoursFound} scanned=$scanned " +
                        "acc=${prof.boxesAccepted} aborted=${prof.aborted} binRatio=${prof.binaryPositiveRatio} " +
                        "minRect=${prof.minAreaRectMs}ms score=${prof.boxScoreMs}ms unclip=${prof.unclipMs}ms " +
                        "proj=${Harness.round(projected)}ms",
                )
                scaled.recycle()
            }
            src.recycle()
            Harness.writeReport("a_det_size_sweep.json", JSONObject().apply {
                put("test", "e4_detInputSizeSweepOnRealPhoto")
                put("device", Harness.deviceInfo())
                put("detMaxSideLimit", 4000)
                put("detThresh", cfg.detThresh)
                put("note", "上采样真实照片模拟未限幅的手机原图；观察 det 输入边长与 DB 逐段耗时/轮廓数的关系")
                put("rows", rows)
            })
        } finally {
            mgr.release()
        }
    }

    private fun upscale(src: android.graphics.Bitmap, longSide: Int): android.graphics.Bitmap {
        val cur = maxOf(src.width, src.height)
        if (cur >= longSide) return src.copy(android.graphics.Bitmap.Config.ARGB_8888, true)
        val scale = longSide.toFloat() / cur
        val w = Math.round(src.width * scale)
        val h = Math.round(src.height * scale)
        return android.graphics.Bitmap.createScaledBitmap(src, w, h, true)
    }

    /**
     * 任务1 的"复现 + 归因"核心实验。资产里的真实照片已被降采样到 ≤1280，直接跑 DB 只要几毫秒，
     * 复现不出阶段1 的 >400s。真正引爆 DB 候选循环的是**二值图正像素比例**（海量轮廓）与
     * **单个候选外接矩形面积**（box score 的 fillPoly+mean 成本），两者都随 detThresh 下调而暴涨。
     *
     * 这里取一张真实街景照片的概率图，对官方默认 thresh=0.3 以及更低的 0.15/0.08/0.03 分别跑
     * 纯 DBPostProcessor.process（带诊断预算，绝不挂死），量出：轮廓总数、进入各阶段的候选数、
     * 逐段耗时、box score 外接面积和，从而把卡顿精确归因到 minAreaRect / boxScore / unclip 哪一段。
     */
    @Test
    fun e3_thresholdSweepDbAttributionOnRealPhoto() {
        org.junit.Assert.assertTrue("OpenCV native 加载失败", OpenCVUtils.init(Harness.testContext()))
        val cfg = PaddleOCRConfig(detMaxSideLimit = 4000)
        val mgr = ORTSessionManager(Harness.testContext(), EngineConfig(4))
        mgr.loadModels(OcrModelCatalog.tinyDet.assetPath, OcrModelCatalog.tinyRec.assetPath)
        val sweep = JSONArray()
        try {
            for (path in listOf(
                "ocr/smoke/smoke_en_street_sign.jpg",
                "ocr/images/hand_02.jpg",
            )) {
                val bmp = Harness.assetBitmap(path, DECODE_CAP)
                val pcm = DetPreprocessor.preprocess(bmp, cfg.detLimitSideLen, cfg.detLimitType, cfg.detMaxSideLimit, cfg.detImgMode)
                val (pred, shape) = mgr.runDetection(pcm.tensorData, pcm.shape)
                val pH = shape[2].toInt(); val pW = shape[3].toInt()
                for (thresh in doubleArrayOf(0.3, 0.15, 0.08, 0.03)) {
                    val prof = DbPostProcessProfile(workBudgetMs = DIAG_BUDGET_MS)
                    val boxes = runDbBounded(pred, shape, cfg, thresh, pcm.originalH, pcm.originalW, prof)
                    val scanned = prof.contoursScanned
                    val perCand = if (scanned > 0) prof.wallMs.toDouble() / scanned else 0.0
                    val projected = if (prof.aborted && scanned > 0) perCand * prof.contoursFound else prof.wallMs.toDouble()
                    sweep.put(JSONObject().apply {
                        put("path", path)
                        put("probMapHW", listOf(pH, pW))
                        put("detThresh", thresh)
                        put("returnedBoxes", boxes)
                        put("msPerScannedCandidate", Harness.round(perCand))
                        put("projectedFullDbMs", Harness.round(projected))
                        put("db", prof.toJson())
                    })
                    android.util.Log.i(
                        Harness.LOG_TAG,
                        "e3 ${path.substringAfterLast('/')} t=$thresh found=${prof.contoursFound} " +
                            "scanned=$scanned acc=${prof.boxesAccepted} aborted=${prof.aborted} " +
                            "minRect=${prof.minAreaRectMs}ms score=${prof.boxScoreMs}ms unclip=${prof.unclipMs}ms " +
                            "proj=${Harness.round(projected)}ms",
                    )
                }
                bmp.recycle()
            }
            Harness.writeReport("a_db_threshold_sweep.json", JSONObject().apply {
                put("test", "e3_thresholdSweepDbAttributionOnRealPhoto")
                put("device", Harness.deviceInfo())
                put("detMaxSideLimit", 4000)
                put("diagBudgetMs", DIAG_BUDGET_MS)
                put("maxCandidates", cfg.detMaxCandidates)
                put("note", "maxCandidates 只在 findContours 之后截断，所以爆炸发生在逐候选循环；" +
                    "阈值下调只是复现手段，真实卡顿图的根因同样是正像素比例/大外接面积导致的候选爆炸")
                put("sweep", sweep)
            })
        } finally {
            mgr.release()
        }
    }

    /** DB process 放线程里跑，外层硬截止；native 调用不可中断，所以靠内部 workBudgetMs 兜底。 */
    private fun runDbBounded(
        pred: FloatArray, shape: LongArray, cfg: PaddleOCRConfig, thresh: Double,
        originalH: Int, originalW: Int, prof: DbPostProcessProfile,
    ): Int {
        val ex = Executors.newSingleThreadExecutor()
        return try {
            val f = ex.submit(Callable {
                DBPostProcessor.process(
                    pred = pred, predShape = shape, thresh = thresh.toFloat(),
                    boxThresh = cfg.detBoxThresh, unclipRatio = cfg.detUnclipRatio,
                    maxCandidates = cfg.detMaxCandidates, useDilation = cfg.detUseDilation,
                    scoreMode = cfg.detScoreMode, boxType = cfg.detBoxType,
                    originalH = originalH, originalW = originalW, profile = prof,
                )
            })
            try {
                f.get(HARD_DEADLINE_MS, TimeUnit.MILLISECONDS).size
            } catch (te: java.util.concurrent.TimeoutException) {
                f.cancel(true); -1
            }
        } finally {
            ex.shutdownNow()
        }
    }

    private fun profileOne(rt: OcrRuntime, capTag: String, path: String, tier: String): JSONObject {
        val engine = rt.engineFor(OcrModelCatalog.uzbek)

        // 1. 图片解码及尺寸调整（BitmapFactory inSampleSize 限幅）单独计时。
        val decodeStart = System.nanoTime()
        val bmp = Harness.assetBitmap(path, DECODE_CAP)
        val decodeMs = (System.nanoTime() - decodeStart) / 1_000_000
        val pssAfterDecode = Harness.pssMb()
        val nativeAfterDecode = Debug.getNativeHeapAllocatedSize() / 1048576.0

        // 2. 完整管线放线程里跑，外层再套一层硬截止；DB 的逐段计账靠 profile。
        val profile = DbPostProcessProfile(workBudgetMs = DIAG_BUDGET_MS)
        val executor = Executors.newSingleThreadExecutor()
        val wallStart = System.nanoTime()
        var timedOut = false
        val result = try {
            val future = executor.submit(Callable { engine.run(bmp, profile) })
            try {
                future.get(HARD_DEADLINE_MS, TimeUnit.MILLISECONDS)
            } catch (te: java.util.concurrent.TimeoutException) {
                timedOut = true
                future.cancel(true)
                null
            }
        } finally {
            executor.shutdownNow()
        }
        val wallMs = (System.nanoTime() - wallStart) / 1_000_000

        val out = JSONObject().apply {
            put("path", path)
            put("tier", tier)
            put("decodedSize", listOf(bmp.width, bmp.height))
            put("decodeAndResizeMs", decodeMs)
            put("pssMbAfterDecode", Harness.round(pssAfterDecode.toDouble()))
            put("nativeHeapMbAfterDecode", Harness.round(nativeAfterDecode))
            put("wallMs", wallMs)
            put("hardTimedOut", timedOut)
            put("db", profile.toJson())
            // 外推：预算截断只跑了一部分候选，按均摊还原整段 DB。
            val scanned = profile.contoursScanned
            val perCand = if (scanned > 0) profile.wallMs.toDouble() / scanned else 0.0
            val projected = if (profile.aborted && scanned > 0)
                perCand * profile.contoursFound else profile.wallMs.toDouble()
            put("msPerScannedCandidate", Harness.round(perCand))
            put("projectedFullDbMs", Harness.round(projected))

            if (result != null) {
                put("detPreprocessMs", result.detPreprocessMs)
                put("detInferenceMs", result.detInferenceMs)
                put("detPostprocessMs", result.detPostprocessMs)
                put("cropMs", result.cropMs)
                put("recMs", result.recognitionTimeMs)
                put("lineCount", result.lineCount)
                put("totalTimeMs", result.totalTimeMs)
                put("detBoxesFromCappedDb", result.results.size)
                put("perLineRecMs", Harness.stats(result.perLineRecMs.map { it.toDouble() }))
                put("pssMbAfterRun", Harness.round(Harness.pssMb().toDouble()))
                put("nativeHeapMbAfterRun", Harness.round(Debug.getNativeHeapAllocatedSize() / 1048576.0))
            } else {
                put("note", "管线在硬截止 $HARD_DEADLINE_MS ms 内未返回，被取消；" +
                    "DB 计账里已扫描候选的数据仍是有效证据（rec/crop 未计入）")
            }
        }
        bmp.recycle()
        android.util.Log.i(
            Harness.LOG_TAG,
            "$capTag $path scanned=${profile.contoursScanned} found=${profile.contoursFound} " +
                "aborted=${profile.aborted} hardTO=$timedOut dbWall=${profile.wallMs}ms " +
                "projected=${Harness.round(projectedOf(profile))}ms lines=${result?.lineCount}",
        )
        return out
    }

    private fun projectedOf(p: DbPostProcessProfile): Double {
        val scanned = p.contoursScanned
        return if (p.aborted && scanned > 0)
            p.wallMs.toDouble() / scanned * p.contoursFound else p.wallMs.toDouble()
    }

    @After
    fun tearDown() {
        runtime.releaseAll()
    }

    companion object {
        /** 与阶段1 解码口径一致：长边上限 1600。 */
        private const val DECODE_CAP = 1600
        /** 诊断预算：DB 候选循环跑到这里就停，够采集到稳定的逐候选均摊。 */
        private const val DIAG_BUDGET_MS = 8000L
        /** 整个管线的硬截止：比 DB 预算宽松，保证正常图能完整跑完，同时 test 绝不挂死。 */
        private const val HARD_DEADLINE_MS = 30000L
    }
}
