package com.example.fourlangasr.ocr

import com.example.fourlangasr.ocr.runtime.OcrModelCatalog
import com.example.fourlangasr.ocr.runtime.OcrRuntime
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.atomic.AtomicReference

/**
 * 迁移评审（任务三）：① 真实 OCREngine 端到端（解码→det→DB→裁剪→rec→CTC→顺序→释放）在真实照片上
 * 记录精度相关量与性能/内存/降级；② 验证 OcrRuntime.releaseAll() 与并发 OCREngine.run() 的生命周期竞争。
 *
 * 全部是**短**运行（几张图 / 一次 race），< JDWP 长循环阈值，可安全在 am instrument 下执行。
 * 长循环（200 轮）由正式 App 的 debug 入口非 instrument 验证（OrtProdStressActivity）。
 */
class OcrMigrationReviewTest {

    @Test
    fun e2e_realPhotos() {
        Harness.assertTrueOpenCv()
        val rt = OcrRuntime(Harness.testContext())
        val engine = rt.engineFor(OcrModelCatalog.uzbek)
        val images = listOf(
            "ocr/images/print_00.png" to "printed",
            "ocr/images/print_02.png" to "printed",
            "ocr/smoke/smoke_zh_street_sign.png" to "real_photo_street",
            "ocr/smoke/smoke_en_street_sign.jpg" to "real_photo_street",
            "ocr/images/hand_00.jpg" to "handwritten_photo",
        )
        val rows = JSONArray()
        val totals = mutableListOf<Double>()
        try {
            for ((path, tier) in images) {
                val bmp = Harness.assetBitmap(path)
                val t0 = System.currentTimeMillis()
                val r = engine.run(bmp)
                val wall = System.currentTimeMillis() - t0
                bmp.recycle()
                totals.add(wall.toDouble())
                rows.put(JSONObject().apply {
                    put("image", path.substringAfterLast('/')); put("tier", tier)
                    put("lines", r.lineCount); put("detectedBoxes", r.detectedBoxCount); put("degraded", r.degraded)
                    put("wallMs", wall); put("detInferMs", r.detInferenceMs); put("dbPostMs", r.detPostprocessMs)
                    put("cropMs", r.cropMs); put("recMs", r.recognitionTimeMs); put("totalMs", r.totalTimeMs)
                    put("pssMb", Harness.round(Harness.pssMb().toDouble()))
                    put("sampleText", r.results.take(3).joinToString(" | ") { it.text })
                })
                android.util.Log.i(Harness.LOG_TAG, "e2e $path lines=${r.lineCount} degraded=${r.degraded} wall=${wall}ms")
            }
        } finally { rt.releaseAll() }
        Harness.writeReport("migration_e2e.json", JSONObject().apply {
            put("test", "e2e_realPhotos"); put("device", Harness.deviceInfo())
            put("note", "smoke 为真实街景照片；hand 为横线纸手写页；印刷为合成印刷体")
            put("rows", rows); put("wallMsStats", Harness.stats(totals))
        })
    }

    @Test
    fun releaseDuringRunDoesNotCrash() {
        Harness.assertTrueOpenCv()
        val rt = OcrRuntime(Harness.testContext())
        val engine = rt.engineFor(OcrModelCatalog.uzbek)
        val bmp = Harness.assetBitmap("ocr/images/print_00.png")
        val err = AtomicReference<Throwable?>(null)
        val done = java.util.concurrent.atomic.AtomicBoolean(false)
        val t0 = System.currentTimeMillis()
        val t = Thread({ try { engine.run(bmp); done.set(true) } catch (e: Throwable) { err.set(e) } }, "ocr-run")
        t.start()
        Thread.sleep(30) // 让 run 处于推理进行中
        val relStart = System.currentTimeMillis()
        rt.releaseAll()   // 应等待在跑的 run 结束（写锁）再关 session
        val relMs = System.currentTimeMillis() - relStart
        t.join(30_000)
        bmp.recycle()
        android.util.Log.i(Harness.LOG_TAG, "releaseRace done=${done.get()} err=${err.get()?.javaClass?.simpleName} releaseWaitedMs=$relMs alive=${t.isAlive} totalMs=${System.currentTimeMillis()-t0}")
        Harness.writeReport("migration_release_race.json", JSONObject().apply {
            put("test", "releaseDuringRunDoesNotCrash"); put("device", Harness.deviceInfo())
            put("runCompleted", done.get()); put("error", err.get()?.toString() ?: JSONObject.NULL)
            put("releaseWaitedMs", relMs); put("runThreadAlive", t.isAlive)
        })
        assertTrue("run 线程不应仍存活", !t.isAlive)
        assertNull("release 与 run 竞争不应抛异常", err.get())
    }
}
