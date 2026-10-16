package com.example.fourlangasr.ocr.runtime

import android.content.Context
import com.example.fourlangasr.ocr.paddle.EngineConfig
import com.example.fourlangasr.ocr.paddle.PaddleOCRConfig
import com.example.fourlangasr.ocr.paddle.engine.OCREngine
import com.example.fourlangasr.ocr.text.LatinCjkTextNormalizer
import com.example.fourlangasr.ocr.text.TextNormalizer
import com.example.fourlangasr.ocr.text.UzbekTextNormalizer

/**
 * 语言 profile：只描述"用哪套权重 + 用哪套文本规则"，不含任何推理逻辑。
 *
 * 关键前提（阶段0 核实）：PP-OCRv6 Tiny 的 rec 模型字典有 6904 条，
 * 一个模型就覆盖简中、繁中、英 + 46 种拉丁语系共 49 种语言。
 * 所以 zh / en / uz **必须共用同一份 det+rec 权重**，
 * 语言之间只是规范化规则不同；分别加载三份相同模型是纯浪费
 * （这个 App 已经背着 ASR 与 ~1GB 翻译引擎）。
 */
data class OcrDetectionModel(
    val id: String,
    val assetPath: String,
)

data class OcrRecognitionModel(
    val id: String,
    val assetPath: String,
    /** 包内 inference.yml，提供 CTC 解码用的 character_dict。 */
    val configAssetPath: String,
)

data class OcrLanguageProfile(
    val code: String,
    val detection: OcrDetectionModel,
    val recognition: OcrRecognitionModel,
    val normalizer: TextNormalizer,
) {
    /** 引擎复用键：只有真正不同的权重才各自占一份内存。 */
    val cacheKey: String get() = "${detection.id}|${recognition.id}"
}

object OcrModelCatalog {

    const val ASSET_ROOT = "ocr"

    /** 共享基础模型：tiny det + tiny rec。 */
    val tinyDet = OcrDetectionModel("ppocrv6_tiny_det", "$ASSET_ROOT/models/det/inference.onnx")
    val tinyRec = OcrRecognitionModel(
        "ppocrv6_tiny_rec",
        "$ASSET_ROOT/models/rec/inference.onnx",
        "$ASSET_ROOT/models/rec/inference.yml",
    )

    /** 第一阶段验收语言。 */
    val uzbek = OcrLanguageProfile("uz", tinyDet, tinyRec, UzbekTextNormalizer)

    /** 同权重、不同文本规则。 */
    val chinese = OcrLanguageProfile("zh", tinyDet, tinyRec, LatinCjkTextNormalizer)
    val english = OcrLanguageProfile("en", tinyDet, tinyRec, LatinCjkTextNormalizer)

    /**
     * 俄语预留位：PP-OCRv6 的 50 语言里**没有西里尔**（阶段0 核实），
     * 需要换 `cyrillic_PP-OCRv5_mobile_rec_onnx`（7.68MB，apache-2.0）。
     * 权重本阶段不下载、不实测，只把切换点留在类型系统里：
     * 新增一个 profile 指向不同的 recognition 即可，推理代码不用改。
     */
    val cyrillicRecV5 = OcrRecognitionModel(
        "cyrillic_ppocrv5_mobile_rec",
        "$ASSET_ROOT/models/rec_cyrillic/inference.onnx",
        "$ASSET_ROOT/models/rec_cyrillic/inference.yml",
    )

    val available: List<OcrLanguageProfile> = listOf(uzbek, chinese, english)

    fun forCode(code: String): OcrLanguageProfile? =
        available.firstOrNull { it.code == code.lowercase() }
}

/**
 * 引擎池：按 [OcrLanguageProfile.cacheKey] 复用 [OCREngine]。
 *
 * zh / en / uz 的 cacheKey 相同 → 全进程只有一份 det+rec session。
 * 这里刻意用 OCREngine 而不是 PaddleOCR 门面：后者的 create() 是 suspend，
 * 无法在锁内安全地做"只创建一次"的双重检查；OCREngine 构造即加载，语义更直白。
 */
class OcrRuntime(
    private val context: Context,
    private val numThreads: Int = 4,
    /**
     * 官方默认 detMaxSideLimit=4000，一张 4000px 照片的输入张量就要 ~144MB，
     * 手机相机原图很容易触发；这里默认限到 1280（阶段0 报告里的内存风险项）。
     */
    private val detMaxSideLimit: Int = 1280,
) {
    private val engines = HashMap<String, OCREngine>()

    val config: PaddleOCRConfig = PaddleOCRConfig(detMaxSideLimit = detMaxSideLimit)

    val liveEngineCount: Int
        @Synchronized get() = engines.size

    fun engineIds(): List<String> = synchronized(this) { engines.keys.toList() }

    @Synchronized
    fun engineFor(profile: OcrLanguageProfile): OCREngine {
        engines[profile.cacheKey]?.let { return it }
        val created = OCREngine(
            context,
            config,
            EngineConfig(numThreads),
            detModelAsset = profile.detection.assetPath,
            recModelAsset = profile.recognition.assetPath,
            recConfigAsset = profile.recognition.configAssetPath,
        )
        engines[profile.cacheKey] = created
        return created
    }

    @Synchronized
    fun releaseAll() {
        engines.values.forEach { runCatching { it.release() } }
        engines.clear()
    }
}
