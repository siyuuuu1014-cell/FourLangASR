package com.example.fourlangasr.ocr

import com.example.fourlangasr.asr.ChineseSherpaEngine
import com.example.fourlangasr.asr.RussianVoskEngine
import com.example.fourlangasr.lid.LidEngine
import com.example.fourlangasr.ocr.paddle.model.OCRResult
import com.example.fourlangasr.ocr.paddle.util.OpenCVUtils
import com.example.fourlangasr.ocr.runtime.OcrLanguageProfile
import com.example.fourlangasr.ocr.runtime.OcrModelCatalog
import com.example.fourlangasr.ocr.runtime.OcrRuntime
import com.example.fourlangasr.translation.TranslationModules
import org.json.JSONArray
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assume
import org.junit.BeforeClass
import org.junit.Test

/**
 * 阶段1-A：真机 native 共存与 OCR 推理验证。
 *
 * 要回答的问题很具体：官方 ppocr-sdk 后处理依赖 OpenCV，而这个 App 已经同时背着
 * sherpa 自带的 libonnxruntime.so、我们改名补丁过的 libort4jrt.so + 桥、Vosk/JNA、LiteRT。
 * 任何一处 dlopen/符号冲突都会让 B 路线作废，所以先测这个再谈别的。
 */
class OcrNativeCompatTest {

    // OCREngine 只用传入的 context 读 assets，所以这里必须是 instrumentation context
    private val runtime = OcrRuntime(Harness.testContext())

    @After
    fun tearDown() {
        runtime.releaseAll()
    }

    private fun recognize(profile: OcrLanguageProfile, asset: String): List<OCRResult> =
        runtime.engineFor(profile).run(Harness.assetBitmap(asset)).results

    /** 1. OpenCV 的 fat so 能否与 App 现有 native 库一起加载，并且真的能干活。 */
    @Test
    fun a1_openCvLoadsAndOperatesInThisProcess() {
        val mat = org.opencv.core.Mat.zeros(64, 64, org.opencv.core.CvType.CV_8UC1)
        try {
            org.opencv.imgproc.Imgproc.rectangle(
                mat,
                org.opencv.core.Point(10.0, 10.0),
                org.opencv.core.Point(50.0, 40.0),
                org.opencv.core.Scalar(255.0),
                -1,
            )
            val contours = mutableListOf<org.opencv.core.MatOfPoint>()
            org.opencv.imgproc.Imgproc.findContours(
                mat, contours, org.opencv.core.Mat(),
                org.opencv.imgproc.Imgproc.RETR_LIST,
                org.opencv.imgproc.Imgproc.CHAIN_APPROX_SIMPLE,
            )
            val probe = org.opencv.core.MatOfPoint2f(*contours[0].toArray())
            val rect = org.opencv.imgproc.Imgproc.minAreaRect(probe)
            probe.release()
            Harness.writeReport("a1_opencv.json", JSONObject().apply {
                put("test", "a1_openCvLoadsAndOperatesInThisProcess")
                put("device", Harness.deviceInfo())
                put("contourCount", contours.size)
                put("rectWidth", Harness.round(rect.size.width.toDouble()))
                put("rectHeight", Harness.round(rect.size.height.toDouble()))
                put("pssMbAfterOpenCv", Harness.pssMb().toDouble())
            })
            assertTrue("findContours 找不到轮廓，OpenCV 运行时无效", contours.isNotEmpty())
            // minAreaRect 不保证宽高顺序：实测返回 30x40（与绘制的 40x30 互为转置）
            val longSide = maxOf(rect.size.width, rect.size.height)
            val shortSide = minOf(rect.size.width, rect.size.height)
            assertTrue("minAreaRect 尺寸不合理: ${rect.size}", longSide >= 40.0 && shortSide >= 30.0)
        } finally {
            mat.release()
        }
    }

    /** 2. 打过补丁的 ORT 桥能否加载 PP-OCRv6 tiny det+rec 并推理出文本。 */
    @Test
    fun a2_ocrModelsLoadAndInferThroughThePatchedOrtBridge() {
        val engine = runtime.engineFor(OcrModelCatalog.uzbek)
        val res = engine.run(Harness.assetBitmap("ocr/images/print_00.png"))
        val text = res.results.joinToString(" ") { it.text }
        Harness.writeReport("a2_ort_ocr.json", JSONObject().apply {
            put("test", "a2_ocrModelsLoadAndInferThroughThePatchedOrtBridge")
            put("device", Harness.deviceInfo())
            put("coldLoadMs", engine.coldLoadTimeMs)
            put("detInputShape", JSONArray(res.detInputShape))
            put("detPreprocessMs", res.detPreprocessMs)
            put("detInferenceMs", res.detInferenceMs)
            put("detPostprocessMs", res.detPostprocessMs)
            put("recInferenceMs", res.recInferenceMs)
            put("lineCount", res.lineCount)
            put("totalMs", res.totalTimeMs)
            put("pssMbAfterLoad", Harness.pssMb().toDouble())
            put("firstLine", res.results.firstOrNull()?.text ?: "")
        })
        assertTrue("检测/识别没有输出任何行", res.lineCount >= 1)
        assertTrue("识别文本为空", text.isNotBlank())
        assertEquals("zh/en/uz 应共用同一份权重", 1, runtime.liveEngineCount)
    }

    /** 3. zh / en / uz 必须命中同一个引擎实例，并共用它做冒烟识别。 */
    @Test
    fun a3_threeLanguagesShareOneEngineAndSmokeRecognize() {
        val uz = runtime.engineFor(OcrModelCatalog.uzbek)
        val zh = runtime.engineFor(OcrModelCatalog.chinese)
        val en = runtime.engineFor(OcrModelCatalog.english)
        assertTrue("不同语言拿到了不同引擎实例（会重复占内存）", uz === zh)
        assertTrue("不同语言拿到了不同引擎实例（会重复占内存）", zh === en)
        assertEquals("不该为每种语言各加载一份 tiny 模型", 1, runtime.liveEngineCount)

        val smoke = Harness.assetJson("ocr/manifest.json").getJSONArray("smoke")
        val perLang = JSONObject()
        val misses = mutableListOf<String>()
        for (i in 0 until smoke.length()) {
            val e = smoke.getJSONObject(i)
            val lang = e.getString("lang")
            val expectedArr = e.getJSONArray("expect_any")
            val expected = (0 until expectedArr.length()).map { expectedArr.getString(it) }
            val results = recognize(OcrModelCatalog.forCode(lang)!!, e.getString("image"))
            val text = results.joinToString(" ") { it.text }
            val hits = expected.filter { text.contains(it, ignoreCase = true) }
            perLang.put(lang, JSONObject().apply {
                put("image", e.getString("image"))
                put("origin", e.getString("origin"))
                put("license", e.getString("license"))
                put("size", "${e.getInt("w")}x${e.getInt("h")}")
                put("lines", results.size)
                put("meanConf", Harness.round(
                    if (results.isEmpty()) 0.0 else results.map { it.confidence.toDouble() }.average(),
                ))
                put("expected", JSONArray(expected))
                put("expectedHits", JSONArray(hits))
                put("text", text.take(220))
            })
            if (hits.isEmpty()) misses.add(lang)
        }
        Harness.writeReport("a3_multilang_smoke.json", JSONObject().apply {
            put("test", "a3_threeLanguagesShareOneEngineAndSmokeRecognize")
            put("device", Harness.deviceInfo())
            put("liveEngineCount", runtime.liveEngineCount)
            put("engineCacheKeys", JSONArray(runtime.engineIds()))
            put("results", perLang)
            put("missedLanguages", JSONArray(misses))
        })
        // 报告先落盘再断言，失败时也能看到真机到底识别出了什么
        assertTrue("这些语言的期望词一个都没命中: $misses", misses.isEmpty())
    }

    /** 4a. OCR 引擎存活时，sherpa-onnx 中文 ASR 仍能加载并识别。 */
    @Test
    fun a4a_sherpaZhWithOcrAlive() {
        val report = speechStep("a4a_sherpa_zh.json")
        val samples = Harness.loadWav("audio/zh_1.wav")
        val sherpa = ChineseSherpaEngine(Harness.context().assets)
        val load = sherpa.load()
        report.put("sherpaLoadMs", load.loadMs)
        report.put("sherpaPssAfterLoadMb", Harness.pssMb().toDouble())
        val text = sherpa.transcribe(samples, 16000).text
        report.put("text", text.take(80))
        Harness.writeReport("a4a_sherpa_zh.json", report)
        sherpa.release()
        assertTrue("sherpa 中文 ASR 在 OCR 加载后返回空文本", text.isNotBlank())
    }

    /** 4b. Vosk 俄语 ASR（自带另一份 ORT + JNA）在 OCR 存活时仍能工作。 */
    @Test
    fun a4b_voskRuWithOcrAlive() {
        val report = speechStep("a4b_vosk_ru.json")
        val vosk = RussianVoskEngine(Harness.context())
        val load = vosk.load()
        report.put("voskLoadMs", load.loadMs)
        report.put("voskPssAfterLoadMb", Harness.pssMb().toDouble())
        val text = vosk.transcribe(Harness.loadWav("audio/ru_1.wav"), 16000).text
        report.put("text", text.take(80))
        Harness.writeReport("a4b_vosk_ru.json", report)
        vosk.release()
        assertTrue("Vosk 俄语 ASR 在 OCR 加载后返回空文本", text.isNotBlank())
    }

    /** 4c. LiteRT（TFLite）LID 与 OpenCV / 两个 ORT 同进程共存。 */
    @Test
    fun a4c_lidWithOcrAlive() {
        val report = speechStep("a4c_lid.json")
        val lid = LidEngine(Harness.context())
        val t0 = android.os.SystemClock.elapsedRealtime()
        lid.load()
        report.put("lidLoadMs", android.os.SystemClock.elapsedRealtime() - t0)
        report.put("lidPssAfterLoadMb", Harness.pssMb().toDouble())
        val res = lid.identify(Harness.loadWav("audio/uz_1.wav"), 16000)
        report.put("lidCode", res.languageCode)
        Harness.writeReport("a4c_lid.json", report)
        lid.release()
        assertNotNull("LID 无输出", res.languageCode)
    }

    private fun speechStep(name: String): JSONObject {
        runtime.engineFor(OcrModelCatalog.uzbek)
        return JSONObject().apply {
            put("test", name)
            put("device", Harness.deviceInfo())
            put("ocrEngineAlive", runtime.liveEngineCount)
            put("pssMbBeforeSpeechLoad", Harness.pssMb().toDouble())
        }
    }

    /** 5. OCR 之后翻译引擎能否加载并推理（设备无 734MB 包时按未验证处理）。 */
    @Test
    fun a5_translationEngineStillUsableAfterOcr() {
        runtime.engineFor(OcrModelCatalog.uzbek)
        val modules = TranslationModules.getInstance(Harness.context())
        val manifest = modules.sync.latestManifest
        Assume.assumeTrue(
            "设备上没有已安装的翻译模型包（需先跑一次清单同步），本项未验证",
            manifest != null && modules.packages.isModelReady(manifest.generalist().modelId),
        )
        val result = modules.service.translate(
            "Assalomu alaykum, ish qanday ketmoqda?",
            com.example.fourlangasr.model.AppLanguage.UZ,
            com.example.fourlangasr.model.AppLanguage.ZH,
        )
        Harness.writeReport("a5_translation_after_ocr.json", JSONObject().apply {
            put("test", "a5_translationEngineStillUsableAfterOcr")
            put("status", "executed")
            put("engine", result.modelLabel ?: "server")
            put("output", result.outputText.take(80))
            put("pssMb", Harness.pssMb().toDouble())
        })
        assertTrue(result.outputText.isNotBlank())
    }

    /** 6. 反复创建/释放 OCR 引擎后内存是否回收（泄漏检查）。 */
    @Test
    fun a6_repeatedLoadReleaseReclaimsMemory() {
        val bmp = Harness.assetBitmap("ocr/images/print_01.png")
        val series = JSONArray()
        var baseline = 0
        repeat(5) { i ->
            val r = OcrRuntime(Harness.testContext())
            val before = Harness.pssMb().toInt()
            if (i == 0) baseline = before
            val engine = r.engineFor(OcrModelCatalog.uzbek)
            val loaded = Harness.pssMb().toInt()
            engine.run(bmp)
            r.releaseAll()
            System.gc()
            Thread.sleep(400)
            series.put(JSONObject().apply {
                put("cycle", i)
                put("pssBeforeLoadMb", before)
                put("pssAfterLoadMb", loaded)
                put("loadDeltaMb", loaded - before)
                put("pssAfterReleaseMb", Harness.pssMb().toDouble())
            })
        }
        val finalPss = Harness.pssMb().toInt()
        Harness.writeReport("a6_load_release_memory.json", JSONObject().apply {
            put("test", "a6_repeatedLoadReleaseReclaimsMemory")
            put("device", Harness.deviceInfo())
            put("detMaxSideLimit", 1280)
            put("cycles", series)
            put("baselineMb", baseline)
            put("finalMb", finalPss)
            put("growthMb", finalPss - baseline)
        })
        assertTrue(
            "5 轮加载/释放后 PSS 增长 ${finalPss - baseline}MB 超过 80MB 阈值：$series",
            finalPss - baseline < 80,
        )
    }

    companion object {
        @BeforeClass
        @JvmStatic
        fun loadOpenCvOnce() {
            // 官方 SDK 要求在创建引擎之前手动加载 OpenCV native。
            assertTrue("OpenCV native 加载失败", OpenCVUtils.init(Harness.context()))
        }
    }
}
