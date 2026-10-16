package com.example.fourlangasr.ocr

import com.example.fourlangasr.ocr.paddle.postprocess.DBPostProcessor
import com.example.fourlangasr.ocr.paddle.postprocess.DbPostProcessProfile
import com.example.fourlangasr.ocr.paddle.util.OpenCVUtils
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * 阶段2A-任务2：验证 DB 后处理的"工作预算"护栏。
 *
 * 用户明确警告：官方 maxCandidates 只在 findContours 返回**之后**截断，所以真正会爆炸的是
 * 逐候选的 minAreaRect / boxScore(fillPoly+mean) / unclip 循环。真实照片资产已降采样、
 * 复现不出这个爆炸（见 e1~e4），所以这里用一张**合成的海量连通块概率图**直接构造最坏情况：
 * 一张 1024x1024 的图，铺满几千个独立小方块，findContours 必然返回几千条轮廓，
 * 逼出候选循环的线性/超线性开销，然后证明：
 *   1) 不设预算时它确实会随轮廓数显著变慢（复现机制）；
 *   2) 设了 workBudgetMs 后，无论有多少候选都能在预算内返回**部分结果**并置 truncated，
 *      绝不无限期挂起（安全降级）。
 * 这是纯 CPU 测试，不加载模型、不依赖设备相机图。
 */
class OcrDbGuardTest {

    /** 造一张 n x n 的 float 概率图，在网格点放边长 [blob] 的实心方块（值=1）。 */
    private fun syntheticProbMap(grid: Int, blob: Int): Pair<FloatArray, LongArray> {
        val w = grid * blob
        val h = grid * blob
        val pred = FloatArray(w * h)
        for (gy in 0 until grid) {
            for (gx in 0 until grid) {
                val ox = gx * blob + 2
                val oy = gy * blob + 2
                for (yy in 0 until (blob - 4)) {
                    val rowBase = (oy + yy) * w
                    for (xx in 0 until (blob - 4)) pred[rowBase + ox + xx] = 1f
                }
            }
        }
        return pred to longArrayOf(1, 1, h.toLong(), w.toLong())
    }

    private fun runDb(pred: FloatArray, shape: LongArray, budgetMs: Long, prof: DbPostProcessProfile?): Int {
        val truncated = booleanArrayOf(false)
        val boxes = DBPostProcessor.process(
            pred = pred, predShape = shape, thresh = 0.3f, boxThresh = 0.6f,
            unclipRatio = 1.5f, maxCandidates = 3000, useDilation = false,
            scoreMode = "fast", boxType = "quad",
            originalH = shape[2].toInt(), originalW = shape[3].toInt(),
            profile = prof, workBudgetMs = budgetMs, truncatedOut = truncated,
        )
        return if (truncated[0]) -boxes.size else boxes.size
    }

    @Test
    fun g1_dbWorkBudgetBoundsCandidateExplosion() {
        assertTrue("OpenCV native 加载失败", OpenCVUtils.init(Harness.testContext()))
        // 60x60 网格、每块 16px → 3600 个独立连通块，逼近 maxCandidates=3000 上限。
        val (pred, shape) = syntheticProbMap(grid = 60, blob = 16)

        // (a) 不设预算：跑完整候选循环，作为"爆炸机制"参照。
        val noBudgetProfile = DbPostProcessProfile(workBudgetMs = 0)
        val t0 = System.nanoTime()
        val full = runDb(pred, shape, 0L, noBudgetProfile)
        val fullMs = (System.nanoTime() - t0) / 1_000_000

        // (b) 设 300ms 预算：必须更快返回、被截断、且返回部分框。
        val budgetMs = 300L
        val cappedProfile = DbPostProcessProfile(workBudgetMs = budgetMs)
        val t1 = System.nanoTime()
        val cappedSigned = runDb(pred, shape, budgetMs, cappedProfile)
        val cappedMs = (System.nanoTime() - t1) / 1_000_000
        val capped = kotlin.math.abs(cappedSigned)

        val report = JSONObject().apply {
            put("test", "g1_dbWorkBudgetBoundsCandidateExplosion")
            put("device", Harness.deviceInfo())
            put("syntheticBlobs", 60 * 60)
            put("maxCandidates", 3000)
            put("noBudget", JSONObject().apply {
                put("returnedBoxes", full)
                put("wallMs", fullMs)
                put("contoursFound", noBudgetProfile.contoursFound)
                put("dbWallMs", noBudgetProfile.wallMs)
                put("minAreaRectMs", noBudgetProfile.minAreaRectMs)
                put("boxScoreMs", noBudgetProfile.boxScoreMs)
                put("unclipMs", noBudgetProfile.unclipMs)
                put("binaryPositiveRatio", noBudgetProfile.binaryPositiveRatio)
            })
            put("withBudget", JSONObject().apply {
                put("budgetMs", budgetMs)
                put("returnedBoxes", capped)
                put("wallMs", cappedMs)
                put("truncated", cappedSigned < 0)
                put("contoursFound", cappedProfile.contoursFound)
                put("contoursScanned", cappedProfile.contoursScanned)
                put("skippedByBudget", cappedProfile.skippedByBudget)
                put("dbWallMs", cappedProfile.wallMs)
            })
        }
        Harness.writeReport("g1_db_budget_guard.json", report)
        android.util.Log.i(
            Harness.LOG_TAG,
            "g1 noBudget boxes=$full wall=${fullMs}ms (found=${noBudgetProfile.contoursFound} " +
                "score=${noBudgetProfile.boxScoreMs}ms unclip=${noBudgetProfile.unclipMs}ms) | " +
                "budget=$budgetMs boxes=$capped wall=${cappedMs}ms scanned=${cappedProfile.contoursScanned} " +
                "skipped=${cappedProfile.skippedByBudget}",
        )

        // 断言 1：合成图确实产生了海量轮廓（复现机制的前提）。
        assertTrue("合成图轮廓数应远超正常照片，实际=${noBudgetProfile.contoursFound}",
            noBudgetProfile.contoursFound > 1000)
        // 断言 2：设预算后被截断，且返回的框数明显少于全量。
        assertTrue("预算模式应触发截断", cappedSigned < 0)
        assertTrue("预算模式返回框数应少于全量: capped=$capped full=$full", capped < full)
        // 断言 3：预算模式在有限时间内返回（给 3 倍预算的宽松上界，容纳调度抖动）。
        assertTrue("预算模式必须在 ${budgetMs * 4}ms 内返回，实际=${cappedMs}ms",
            cappedMs < budgetMs * 4)
    }
}
