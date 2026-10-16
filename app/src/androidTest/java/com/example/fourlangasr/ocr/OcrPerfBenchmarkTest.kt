package com.example.fourlangasr.ocr

import com.example.fourlangasr.ocr.paddle.util.OpenCVUtils
import com.example.fourlangasr.ocr.runtime.OcrModelCatalog
import com.example.fourlangasr.ocr.runtime.OcrRuntime
import org.json.JSONArray
import org.json.JSONObject
import org.junit.BeforeClass
import org.junit.Test

/**
 * 阶段1-D：真机延迟与内存基准。
 *
 * 记录设备、图片尺寸、重复次数与统计量（mean/p50/p90/min/max），
 * 并把"OCR 之后翻译能否继续工作"单独标出来——设备上没有翻译包时记为未验证，
 * 不用推测填数。
 */
class OcrPerfBenchmarkTest {

    @Test
    fun d1_coldWarmStageTimingsAndMemory() {
        val repeats = 5
        // 只测印刷层：真机复现到，真实照片（中文路牌）与横线纸手写页会触发
        // DB 后处理的轮廓数爆炸，单页 400s 内跑不完——这是阶段2 必须先修的性能缺陷，
        // 不在基准里混入未知耗时的样本。
        val images = listOf("print_00.png", "print_01.png", "print_03.png", "print_05.png")
        val manifest = Harness.assetJson("ocr/manifest.json")
        val paths = HashMap<String, String>()
        manifest.getJSONArray("images").let { a ->
            for (i in 0 until a.length()) {
                val e = a.getJSONObject(i)
                paths[e.getString("golden_image")] = e.getString("image")
            }
        }
        manifest.getJSONArray("smoke").let { a ->
            for (i in 0 until a.length()) {
                val e = a.getJSONObject(i)
                paths[e.getString("image").substringAfterLast('/')] = e.getString("image")
            }
        }

        // 冷启动：新进程里第一次创建引擎
        val t0 = SystemClockCompat.now()
        var runtime = OcrRuntime(Harness.testContext())
        val coldEngine = runtime.engineFor(OcrModelCatalog.uzbek)
        val coldLoadMs = SystemClockCompat.now() - t0
        val pssAfterCold = Harness.pssMb().toDouble()
        coldEngine.run(Harness.assetBitmap(paths.getValue(images[0])))

        // 热重载：释放后再次创建的耗时
        runtime.releaseAll()
        val t1 = SystemClockCompat.now()
        runtime = OcrRuntime(Harness.testContext())
        val engine = runtime.engineFor(OcrModelCatalog.uzbek)
        val warmLoadMs = SystemClockCompat.now() - t1

        val perImage = JSONArray()
        val totals = mutableListOf<Double>()
        val detInf = mutableListOf<Double>()
        val recInf = mutableListOf<Double>()
        for (name in images) {
            val bmp = Harness.assetBitmap(paths.getValue(name))
            engine.run(bmp)  // 预热
            val runs = (1..repeats).map { engine.run(bmp) }
            val total = runs.map { it.totalTimeMs.toDouble() }
            totals.addAll(total)
            detInf.addAll(runs.map { it.detInferenceMs.toDouble() })
            recInf.addAll(runs.map { it.recInferenceMs.toDouble() })
            perImage.put(JSONObject().apply {
                put("image", name)
                put("size", "${bmp.width}x${bmp.height}")
                put("lines", runs.last().lineCount)
                put("totalMs", Harness.stats(total))
                put("detPreMs", Harness.stats(runs.map { it.detPreprocessMs.toDouble() }))
                put("detInferMs", Harness.stats(runs.map { it.detInferenceMs.toDouble() }))
                put("detPostMs", Harness.stats(runs.map { it.detPostprocessMs.toDouble() }))
                put("recInferMs", Harness.stats(runs.map { it.recInferenceMs.toDouble() }))
                put("detInputShape", JSONArray(runs.last().detInputShape))
            })
        }
        val pssLoaded = Harness.pssMb().toDouble()
        runtime.releaseAll()
        System.gc()
        Thread.sleep(500)
        val pssReleased = Harness.pssMb().toDouble()

        // 这里不碰 TranslationModules：翻译包在设备上的可用性由 a5 单独负责，
        // 基准测试要保持只依赖 OCR 自身。

        Harness.writeReport("d1_perf.json", JSONObject().apply {
            put("test", "d1_coldWarmStageTimingsAndMemory")
            put("device", Harness.deviceInfo())
            put("repeatsPerImage", repeats)
            put("detMaxSideLimit", 1280)
            put("numThreads", 4)
            put("coldLoadMs", coldLoadMs)
            put("warmReloadMs", warmLoadMs)
            put("images", perImage)
            put("pssMbAfterColdLoad", pssAfterCold)
            put("pssMbAllLoaded", pssLoaded)
            put("pssMbAfterRelease", pssReleased)
            put("aggregate", JSONObject().apply {
                put("totalMs", Harness.stats(totals))
                put("detInferMs", Harness.stats(detInf))
                put("recInferMs", Harness.stats(recInf))
            })
            put("translationPackageOnDevice", JSONObject.NULL)
            put("ocrThenTranslationPeakMemoryMb", JSONObject.NULL)
            put("unverified", JSONArray(listOf(
                "OCR+翻译顺序执行峰值内存（设备上没有 734MB 翻译包，未测）",
                "真实照片/手写页的 OCR 耗时（已复现：中文路牌与横线纸手写页 400s 未跑完，阶段2 必须先加轮廓与行数上限）",
            )))
        })
    }

    /** 用 elapsedRealtime 而不是 currentTimeMillis，避免系统时间调整污染计时。 */
    private object SystemClockCompat {
        fun now(): Long = android.os.SystemClock.elapsedRealtime()
    }

    companion object {
        @BeforeClass
        @JvmStatic
        fun initOpenCv() {
            org.junit.Assert.assertTrue(OpenCVUtils.init(Harness.testContext()))
        }
    }
}
