package com.example.fourlangasr.pipeline

import android.content.res.AssetManager
import com.example.fourlangasr.asr.EnglishSherpaEngine
import com.example.fourlangasr.model.LoadResult

/**
 * EXP5/EXP6 的 code-switch 单词回填。
 *
 * 中文为主语言的句子经过流式 CTC 后，夹在句中的英文词会落成
 * <unk>。本类以这些 <unk> 的时间戳为中心裁出音频片段，交给
 * EN（GigaSpeech）验证识别，再把识别文本回填到中文输出里。
 *
 * PC 基准（experiments/codeswitch_unk_exp5）：
 *   - 纯中文 15 条零 <unk>，误报 0
 *   - 30 条中英夹杂句约 60% 能回填出正确英文词，另有约 20%
 *     回填近似词（POST / FAMILY / MOBILE 级别，仍有提示意义）
 *
 * 生命周期：EN 模型（约 73 MB）按需懒加载；
 * 释放交给 MainActivity 的释放按钮 / onDestroy。
 */
class CodeSwitchRecovery(
    assetManager: AssetManager,
) {

    private val enEngine =
        EnglishSherpaEngine(
            assetManager
        )

    private var enLoaded =
        false

    /**
     * 是否已识别过任意 <unk>（外部据此决定要不要调 recover）。
     */
    fun isTriggered(
        unkTimestampsSec: FloatArray,
    ): Boolean {

        return unkTimestampsSec.isNotEmpty()
    }

    /**
     * 加载 EN 验证模型（幂等，首次调用后复用）。
     */
    fun ensureLoaded(): LoadResult? {

        if (enLoaded) {
            return null
        }

        val result =
            enEngine.load()

        enLoaded =
            true

        return result
    }

    /**
     * 对一条 ZH 识别结果做回填。
     *
     * @param zhText        ZH CTC 输出文本（含字面 "<unk>"）
     * @param unkTimestampsSec 每个 <unk> token 的起始时间（秒）
     */
    fun recover(
        samples: FloatArray,
        sampleRate: Int,
        zhText: String,
        unkTimestampsSec: FloatArray,
    ): RecoveryResult {

        if (
            unkTimestampsSec.isEmpty() ||
            zhText.isEmpty()
        ) {

            return RecoveryResult(
                recoveredText = zhText,
                regionCount = 0,
                enTexts = emptyList(),
                enLoadMs = null,
            )
        }

        val enLoadMs =
            ensureLoaded()
                ?.loadMs

        val durationSec =
            samples.size /
                    sampleRate.toFloat()

        /*
         * 1. 每个 <unk> 展开成 [ts - pad, ts + pad]，
         *    相邻（gap <= MERGE_GAP_SEC）区域合并。
         */
        val regions =
            mergeRegions(
                unkTimesSec =
                    unkTimestampsSec,
                padSec =
                    REGION_PAD_SEC,
                maxGapSec =
                    MERGE_GAP_SEC,
                durationSec =
                    durationSec,
            )

        if (
            regions.isEmpty()
        ) {

            return RecoveryResult(
                recoveredText =
                    stripUnk(
                        zhText
                    ),
                regionCount = 0,
                enTexts = emptyList(),
                enLoadMs = enLoadMs,
            )
        }

        /*
         * 2. 每个 region 用 EN 模型识别。
         *    region[i] 覆盖 unkTimesSec 的区间，
         *    enTexts[i] 为空表示该 region 没识别出英文实词。
         */
        val enTexts =
            regions.map {
                (startSec, endSec) ->

                val enText =
                    verifyRegion(
                        samples = samples,
                        sampleRate = sampleRate,
                        startSec = startSec,
                        endSec = endSec,
                    )

                enText
            }

        /*
         * 3. 回填：zhText 中第 k 个 "<unk>" 属于 region r。
         *    该 region 的第一个 <unk> 替换为 EN 文本，
         *    其余 <unk>（同 region 内的相邻 unk）删除；
         *    region 没有有效 EN 文本时也删除该标记。
         */
        val recovered =
            rebuildText(
                zhText = zhText,
                unkTimestampsSec =
                    unkTimestampsSec,
                regions = regions,
                enTexts = enTexts,
            )

        return RecoveryResult(
            recoveredText =
                recovered,
            regionCount =
                regions.size,
            enTexts =
                enTexts.filter {
                    it.isNotEmpty()
                },
            enLoadMs =
                enLoadMs,
        )
    }

    fun release() {

        enEngine.release()
        enLoaded =
            false
    }

    /**
     * 用 EN 模型验证一个音频区域，返回清理后的英文文本。
     *
     * 中文语境里的孤词对 EN 模型很难一次认准。与 PC EXP5 最终配置
     * 对齐：区域较长时把"整窗 + 左半窗 + 右半窗"各识别一次，
     * 取实词最多的结果。三个都只有停用词/空则返回 ""。
     *
     * 注意：EN GigaSpeech 对纯中文语音也会幻觉英文实词（EXP6：15/15），
     * 因此本类的精度不依赖 EN 文本内容，而依赖调用方只在本区域
     * 确实存在 <unk> 触发时才调用（EXP6 结论）。
     */
    private fun verifyRegion(
        samples: FloatArray,
        sampleRate: Int,
        startSec: Float,
        endSec: Float,
    ): String {

        val startSample =
            (startSec *
                    sampleRate).toInt()
                .coerceAtLeast(0)

        val endSample =
            (endSec *
                    sampleRate).toInt()
                .coerceAtMost(
                    samples.size
                )

        if (
            endSample -
            startSample <
            MIN_REGION_SAMPLES
        ) {

            return ""
        }

        /*
         * 候选窗口：整窗；区域 >= 0.5 s 时再加左/右半窗。
         */
        val candidates =
            mutableListOf<
                    IntRange
                    >()

        candidates.add(
            startSample until
                    endSample
        )

        if (
            endSample -
            startSample >=
            HALF_WINDOW_MIN_SAMPLES
        ) {

            val mid =
                (startSample +
                        endSample) /
                        2

            candidates.add(
                startSample until
                        mid
            )

            candidates.add(
                mid until
                        endSample
            )
        }

        /*
         * 每个候选独立识别 + 停用词过滤。
         */
        val cleanedTexts =
            candidates.mapNotNull {
                range ->

                val cleaned =
                    decodeRegion(
                        samples = samples,
                        sampleRate =
                            sampleRate,
                        startSample =
                            range.first,
                        endSample =
                            range.last +
                                    1,
                    )

                cleaned.takeIf {
                    it.isNotEmpty()
                }
            }

        /*
         * 取实词最多的候选；平手取更长文本。
         */
        return cleanedTexts
            .maxWithOrNull(
                compareBy(
                    {
                        contentWordCount(
                            it
                        )
                    },
                    {
                        it.length
                    },
                )
            )
            ?: ""
    }

    private fun decodeRegion(
        samples: FloatArray,
        sampleRate: Int,
        startSample: Int,
        endSample: Int,
    ): String {

        if (
            endSample -
            startSample <
            MIN_REGION_SAMPLES
        ) {

            return ""
        }

        val regionSamples =
            samples.copyOfRange(
                startSample,
                endSample,
            )

        val result =
            enEngine.transcribe(
                samples =
                    regionSamples,
                sampleRate =
                    sampleRate,
            )

        val cleaned =
            result.text
                .trim()
                .lowercase()

        if (
            !containsContentWord(
                cleaned
            )
        ) {

            return ""
        }

        return cleaned
    }

    /**
     * EN 输出是否含英文实词（相对停用词表）。
     */
    private fun containsContentWord(
        text: String,
    ): Boolean {

        return contentTokens(
            text
        ).isNotEmpty()
    }

    /**
     * 提取文本中的英文实词（去掉停用词与单字母）。
     */
    private fun contentTokens(
        text: String,
    ): List<String> {

        return text
            .split(
                Regex(
                    "[^a-z']+"
                )
            )
            .filter { token ->

                token.length >=
                        2 &&
                        token !in
                        STOP_WORDS
            }
    }

    private fun contentWordCount(
        text: String,
    ): Int {

        return contentTokens(
            text
        ).size
    }

    /**
     * 把合并后的时间区域按升序拼成 (start, end) 对。
     */
    private fun mergeRegions(
        unkTimesSec: FloatArray,
        padSec: Float,
        maxGapSec: Float,
        durationSec: Float,
    ): List<FloatArray> {

        val spans =
            unkTimesSec
                .map { ts ->

                    floatArrayOf(
                        (ts -
                                padSec)
                            .coerceAtLeast(0f),
                        (ts +
                                padSec)
                            .coerceAtMost(
                                durationSec
                            ),
                    )
                }
                .sortedBy {
                    it[0]
                }

        val merged =
            mutableListOf<FloatArray>()

        for (
        span
        in spans
        ) {

            val last =
                merged.lastOrNull()

            if (
                last != null &&
                span[0] <=
                last[1] +
                        maxGapSec
            ) {

                last[1] =
                    maxOf(
                        last[1],
                        span[1],
                    )

            } else {

                merged.add(
                    floatArrayOf(
                        span[0],
                        span[1],
                    )
                )
            }
        }

        return merged
    }

    /**
     * 把 zhText 里出现的 "<unk>" 标记替换成对应 EN 文本
     * （同 region 的后续 <unk> 与空文本一律删除）。
     */
    private fun rebuildText(
        zhText: String,
        unkTimestampsSec: FloatArray,
        regions: List<FloatArray>,
        enTexts: List<String>,
    ): String {

        /*
         * 每个 <unk> 属于哪个 region：
         * unk 落在 [start, end] 内的第一个 region。
         */
        val regionIndexOfUnk =
            unkTimestampsSec.map { ts ->

                regions.indexOfFirst {
                    region ->

                    ts >=
                            region[0] &&
                            ts <=
                            region[1]
                }
            }

        /*
         * region 内的第几个 <unk>。
         */
        val unkIndexInRegion =
            IntArray(
                unkTimestampsSec.size
            )

        val regionUnkCount =
            IntArray(
                regions.size
            )

        regionIndexOfUnk.forEachIndexed {
            i,
            regionIndex ->

            if (
                regionIndex >= 0
            ) {

                unkIndexInRegion[i] =
                    regionUnkCount[
                        regionIndex
                    ]

                regionUnkCount[
                    regionIndex
                ] +=
                    1
            }
        }

        /*
         * 在原文本上先定位所有 "<unk>" 的位置，
         * 再从后往前替换，保证位置不因前面替换而偏移。
         */
        val markerPositions =
            mutableListOf<Int>()

        var searchFrom =
            0

        while (true) {

            val pos =
                zhText.indexOf(
                    UNK_MARKER,
                    searchFrom,
                )

            if (
                pos < 0
            ) {
                break
            }

            markerPositions.add(
                pos
            )

            searchFrom =
                pos +
                        UNK_MARKER.length
        }

        val builder =
            StringBuilder(
                zhText
            )

        for (
        unkIndex
        in markerPositions.indices.reversed()
        ) {

            val pos =
                markerPositions[
                    unkIndex
                ]

            val regionIndex =
                regionIndexOfUnk
                    .getOrNull(
                        unkIndex
                    )
                    ?: -1

            val isFirstInRegion =
                regionIndex >= 0 &&
                        unkIndexInRegion[
                            unkIndex
                        ] == 0

            val enText =
                if (
                    regionIndex in
                    enTexts.indices
                ) {

                    enTexts[
                        regionIndex
                    ]

                } else {

                    ""
                }

            val replacement =
                if (
                    isFirstInRegion &&
                    enText.isNotEmpty()
                ) {

                    enText

                } else {

                    ""
                }

            builder.replace(
                pos,
                pos +
                        UNK_MARKER.length,
                replacement,
            )
        }

        return builder
            .toString()
            .replace(
                Regex(
                    "\\s+"
                ),
                " ",
            )
            .trim()
    }

    private fun stripUnk(
        text: String,
    ): String {

        return text
            .replace(
                UNK_MARKER,
                "",
            )
            .replace(
                Regex(
                    "\\s+"
                ),
                " ",
            )
            .trim()
    }

    companion object {

        const val UNK_MARKER =
            "<unk>"

        /*
         * PC 校准（codeswitch_unk_exp5 diag_backfill_strategy）：
         * pad 越大回填质量越高 —— 0.6 s 严格词级命中 48% vs
         * 0.45 s 的 36%（检测命中率两者持平，但回填文本完整性
         * 0.6 明显更好，EN 需要更多上下文才能输出完整词）。
         */
        const val REGION_PAD_SEC =
            0.60f

        const val MERGE_GAP_SEC =
            0.2f

        const val MIN_REGION_SAMPLES =
            1600 // 0.1 s @ 16 kHz

        /*
         * 区域 >= 0.5 s 时启用多候选（整窗 + 左右半窗）识别。
         */
        const val HALF_WINDOW_MIN_SAMPLES =
            8000 // 0.5 s @ 16 kHz

        /*
         * EN GigaSpeech 对中文/静音段的常见幻觉功能词。
         * 全为小写比较。
         */
        private val STOP_WORDS =
            setOf(
                "a", "an", "and", "are", "as", "at", "be", "but",
                "by", "can", "did", "do", "does", "for", "from",
                "had", "has", "have", "he", "her", "his", "i",
                "if", "in", "into", "is", "it", "its", "me", "my",
                "no", "not", "of", "on", "or", "our", "out", "over",
                "so", "some", "that", "the", "their", "them", "then",
                "there", "they", "this", "to", "up", "um", "uh",
                "yeah", "ah", "we", "were", "what", "when", "which",
                "who", "will", "with", "you", "your", "s", "t",
                "ve", "re", "gonna", "got",
            )
    }
}

/**
 * 一条中文句子的 code-switch 回填结果。
 */
data class RecoveryResult(
    val recoveredText: String,
    val regionCount: Int,
    val enTexts: List<String>,

    /*
     * 首次懒加载 EN 验证模型的耗时（毫秒）。
     * 已加载过时为 null，可用作真机基准指标。
     */
    val enLoadMs: Long? = null,
)
