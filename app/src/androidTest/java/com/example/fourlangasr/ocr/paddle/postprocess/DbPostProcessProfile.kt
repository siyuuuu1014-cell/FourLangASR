package com.example.fourlangasr.ocr.paddle.postprocess

import org.json.JSONObject

/**
 * DB 后处理的分段计账器，只用于阶段2A 的性能定位，不参与任何结果计算。
 *
 * 之所以要单独一个对象：卡顿发生在候选框循环内部，整段 postprocessMs 一个数字
 * 看不出是 findContours、minAreaRect 还是 box score 吃掉的时间。这里把每个候选
 * 都要走的路径分别累加，并记录轮廓规模分布，才能把"400 秒不返回"归因到具体调用。
 */
class DbPostProcessProfile(
    /**
     * 诊断用的工作预算（毫秒）。超出的候选被跳过并置 [aborted]，
     * 让一张会卡死的图也能在有限时间内给出统计，而不是让 instrumented test 挂住。
     * <=0 表示不限幅，即生产行为。
     */
    val workBudgetMs: Long = 0L,
) {
    var thresholdMs = 0L
    var findContoursMs = 0L
    var contourToPointMs = 0L
    var minAreaRectMs = 0L
    var boxScoreMs = 0L
    var unclipMs = 0L
    var expandedRectMs = 0L
    var scaleMs = 0L

    /** 二值图上非零像素比例：干净文本页约 1%，纹理照片/横线纸可以高一个数量级。 */
    var binaryPositiveRatio: Double = 0.0
    var maskWidth: Int = 0
    var maskHeight: Int = 0

    var contoursFound: Int = 0
    var contoursScanned: Int = 0
    var passedMinSizeBeforeUnclip: Int = 0
    var passedBoxScore: Int = 0
    var boxesAccepted: Int = 0
    var skippedByBudget: Int = 0

    var contourPointsTotal: Long = 0
    var contourPointsMax: Int = 0
    var contourPointsSumOfScanned: Long = 0

    /** box score 里 fillPoly + mean 的工作量：候选外接矩形面积之和（像素）。 */
    var scorePixelWork: Long = 0
    var scorePixelWorkMax: Long = 0

    var aborted: Boolean = false
    var wallMs: Long = 0L
        private set

    /** 生产路径（无预算）下每次调用都计时，用于回归对拍；预算模式只跑到超时为止。 */
    fun onThreshold(ms: Long) { thresholdMs += ms }
    fun onFindContours(ms: Long) { findContoursMs += ms }
    fun onToPoint(ms: Long) { contourToPointMs += ms }
    fun onMinAreaRect(ms: Long) { minAreaRectMs += ms }
    fun onBoxScore(ms: Long) { boxScoreMs += ms }
    fun onUnclip(ms: Long) { unclipMs += ms }
    fun onExpandedRect(ms: Long) { expandedRectMs += ms }
    fun onScale(ms: Long) { scaleMs += ms }

    fun onWall(ms: Long) { wallMs = ms }

    fun incPassedMinSize() { passedMinSizeBeforeUnclip++ }
    fun incPassedBoxScore() { passedBoxScore++ }
    fun incBoxesAccepted() { boxesAccepted++ }
    fun addScannedPoints(n: Int) { contourPointsSumOfScanned += n }

    /** 记录一次 box score 的外接矩形面积工作量（fillPoly+mean 的实际开销代理）。 */
    fun noteScoreWork(points: List<org.opencv.core.Point>, h: Int, w: Int) {
        if (points.isEmpty()) return
        val xmin = points.minOf { kotlin.math.floor(it.x).toInt() }.coerceIn(0, w - 1)
        val xmax = points.maxOf { kotlin.math.ceil(it.x).toInt() }.coerceIn(0, w - 1)
        val ymin = points.minOf { kotlin.math.floor(it.y).toInt() }.coerceIn(0, h - 1)
        val ymax = points.maxOf { kotlin.math.ceil(it.y).toInt() }.coerceIn(0, h - 1)
        val area = (ymax - ymin + 1).toLong() * (xmax - xmin + 1)
        scorePixelWork += area
        if (area > scorePixelWorkMax) scorePixelWorkMax = area
    }

    fun toJson(): JSONObject = JSONObject().apply {
        put("thresholdMs", thresholdMs)
        put("findContoursMs", findContoursMs)
        put("contourToPointMs", contourToPointMs)
        put("minAreaRectMs", minAreaRectMs)
        put("boxScoreMs", boxScoreMs)
        put("unclipMs", unclipMs)
        put("expandedRectMs", expandedRectMs)
        put("scaleMs", scaleMs)
        put("wallMs", wallMs)
        put("workBudgetMs", workBudgetMs)
        put("aborted", aborted)
        put("maskSize", listOf(maskWidth, maskHeight))
        put("binaryPositiveRatio", r6(binaryPositiveRatio))
        put("contoursFound", contoursFound)
        put("contoursScanned", contoursScanned)
        put("passedMinSizeBeforeUnclip", passedMinSizeBeforeUnclip)
        put("passedBoxScore", passedBoxScore)
        put("boxesAccepted", boxesAccepted)
        put("skippedByBudget", skippedByBudget)
        put("contourPointsTotal", contourPointsTotal)
        put("contourPointsMax", contourPointsMax)
        put("contourPointsMeanScanned",
            if (contoursScanned == 0) 0.0 else r6(contourPointsSumOfScanned.toDouble() / contoursScanned))
        put("boxScorePixelWork", scorePixelWork)
        put("boxScorePixelWorkMax", scorePixelWorkMax)
    }

    private fun r6(v: Double): Double = Math.round(v * 1e6) / 1e6
}
