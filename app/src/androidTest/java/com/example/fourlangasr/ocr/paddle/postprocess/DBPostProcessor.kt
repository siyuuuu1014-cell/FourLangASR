// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.example.fourlangasr.ocr.paddle.postprocess

import android.graphics.PointF
import com.example.fourlangasr.ocr.paddle.model.OCRBox
import com.example.fourlangasr.ocr.paddle.util.MathUtils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc

object DBPostProcessor {
    private const val MIN_SIZE_BEFORE_UNCLIP = 3f
    private const val MIN_SIZE_AFTER_UNCLIP = 5f

    /**
     * 本地修改（阶段2A）：
     * 1) 新增可选 [profile] 探针，默认 null → null 时代码路径与官方逐行一致，对拍不受影响；
     * 2) 新增生产可用的 [workBudgetMs]：>0 时给逐候选循环设硬预算，超时立即返回**已识别到的部分框**
     *    （安全降级），保证极复杂照片也不会无限期挂住。0 = 不限幅，等价官方行为。
     * 两者都只影响"处理多少候选"，不改动任何几何/阈值判断，正常图片结果逐字节不变。
     */
    fun process(
        pred: FloatArray,
        predShape: LongArray,
        thresh: Float,
        boxThresh: Float,
        unclipRatio: Float,
        maxCandidates: Int,
        useDilation: Boolean,
        scoreMode: String,
        boxType: String,
        originalH: Int,
        originalW: Int,
        profile: DbPostProcessProfile? = null,
        workBudgetMs: Long = 0L,
        truncatedOut: BooleanArray? = null,
    ): List<OCRBox> {
        require(boxType == "quad") { "Only DBPostProcess box_type=quad is supported" }

        val pH = predShape[2].toInt()
        val pW = predShape[3].toInt()
        val scaleX = originalW.toDouble() / pW
        val scaleY = originalH.toDouble() / pH
        val normalizedScoreMode = scoreMode.lowercase()

        profile?.let { it.maskWidth = pW; it.maskHeight = pH }

        val rawProb = Mat(pH, pW, CvType.CV_32FC1)
        val mask = Mat(pH, pW, CvType.CV_8UC1)
        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        var contourMask = mask

        val wallStart = System.nanoTime()
        val effectiveBudgetMs = if (workBudgetMs > 0) workBudgetMs else profile?.workBudgetMs ?: 0L
        return try {
            rawProb.put(0, 0, pred)
            val threshMat = Mat()
            val t0 = System.nanoTime()
            try {
                Imgproc.threshold(rawProb, threshMat, thresh.toDouble(), 255.0, Imgproc.THRESH_BINARY)
                threshMat.convertTo(mask, CvType.CV_8UC1)
            } finally {
                threshMat.release()
            }
            profile?.onThreshold((System.nanoTime() - t0) / 1_000_000)

            // 二值图正像素比例：定位"海量轮廓"用，countNonZero 只走原生一遍。
            profile?.let {
                val nz = Core.countNonZero(mask).toDouble()
                it.binaryPositiveRatio = nz / (pW.toDouble() * pH)
            }

            contourMask = if (useDilation) {
                val kernel = Mat.ones(2, 2, CvType.CV_8UC1)
                try {
                    Mat().also { Imgproc.dilate(mask, it, kernel) }
                } finally {
                    kernel.release()
                }
            } else {
                mask
            }

            val fcStart = System.nanoTime()
            Imgproc.findContours(contourMask, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)
            profile?.onFindContours((System.nanoTime() - fcStart) / 1_000_000)

            val boxes = mutableListOf<OCRBox>()
            val numContours = minOf(contours.size, maxCandidates)
            profile?.let {
                it.contoursFound = contours.size
                for (c in contours) {
                    val n = c.total().toInt()
                    it.contourPointsTotal += n
                    if (n > it.contourPointsMax) it.contourPointsMax = n
                }
            }

            for (index in 0 until numContours) {
                // 生产/诊断硬预算：超时则停止处理剩余候选，返回已识别到的部分框（安全降级）。
                if (effectiveBudgetMs > 0 &&
                    (System.nanoTime() - wallStart) / 1_000_000 > effectiveBudgetMs
                ) {
                    profile?.aborted = true
                    profile?.skippedByBudget = numContours - index
                    truncatedOut?.set(0, true)
                    break
                }

                val contour = contours[index]
                val toPointStart = System.nanoTime()
                val contourArray = contour.toArray()
                val contour2f = MatOfPoint2f(*contourArray)
                profile?.onToPoint((System.nanoTime() - toPointStart) / 1_000_000)
                profile?.addScannedPoints(contourArray.size)

                val rectStart = System.nanoTime()
                val rect = try {
                    Imgproc.minAreaRect(contour2f)
                } finally {
                    contour2f.release()
                }
                profile?.onMinAreaRect((System.nanoTime() - rectStart) / 1_000_000)
                val sside = minOf(rect.size.width, rect.size.height)
                if (sside < MIN_SIZE_BEFORE_UNCLIP) continue
                profile?.incPassedMinSize()

                val rectPoints = Array(4) { Point() }
                rect.points(rectPoints)
                val orderedPoints = QuadGeometry.orderMinAreaRectPoints(rectPoints)
                val scoreStart = System.nanoTime()
                val score = if (normalizedScoreMode == "slow") {
                    computeBoxScore(rawProb, contour.toList())
                } else {
                    computeBoxScore(rawProb, orderedPoints)
                }
                if (profile != null) {
                    profile.onBoxScore((System.nanoTime() - scoreStart) / 1_000_000)
                    profile.noteScoreWork(orderedPoints, pH, pW)
                }
                if (score < boxThresh) continue
                profile?.incPassedBoxScore()

                val unclipStart = System.nanoTime()
                val expandedPts = PolygonUnclip.unclip(orderedPoints, unclipRatio)
                profile?.onUnclip((System.nanoTime() - unclipStart) / 1_000_000)
                val expRectStart = System.nanoTime()
                val expandedRect = MatOfPoint2f().let { expanded2f ->
                    try {
                        expanded2f.fromList(expandedPts)
                        Imgproc.minAreaRect(expanded2f)
                    } finally {
                        expanded2f.release()
                    }
                }
                profile?.onExpandedRect((System.nanoTime() - expRectStart) / 1_000_000)
                val sside2 = minOf(expandedRect.size.width, expandedRect.size.height)
                if (sside2 < MIN_SIZE_AFTER_UNCLIP) continue

                val expandedBox = Array(4) { Point() }
                expandedRect.points(expandedBox)
                val ePts = QuadGeometry.orderMinAreaRectPoints(expandedBox)
                val scaleStart = System.nanoTime()
                val scaled = listOf(
                    scalePoint(ePts[0], scaleX, scaleY, originalW, originalH),
                    scalePoint(ePts[1], scaleX, scaleY, originalW, originalH),
                    scalePoint(ePts[2], scaleX, scaleY, originalW, originalH),
                    scalePoint(ePts[3], scaleX, scaleY, originalW, originalH),
                )
                profile?.onScale((System.nanoTime() - scaleStart) / 1_000_000)

                val boxW = kotlin.math.hypot(
                    scaled[1].x - scaled[0].x,
                    scaled[1].y - scaled[0].y,
                )
                val boxH = kotlin.math.hypot(
                    scaled[3].x - scaled[0].x,
                    scaled[3].y - scaled[0].y,
                )
                if (boxW <= 3 || boxH <= 3) continue

                boxes.add(OCRBox(points = scaled))
                profile?.incBoxesAccepted()
            }
            profile?.contoursScanned =
                if (profile?.aborted == true) numContours - profile.skippedByBudget else numContours
            profile?.onWall((System.nanoTime() - wallStart) / 1_000_000)
            boxes
        } finally {
            hierarchy.release()
            if (contourMask !== mask) {
                contourMask.release()
            }
            contours.forEach { it.release() }
            mask.release()
            rawProb.release()
        }
    }

    private fun scalePoint(
        point: Point,
        scaleX: Double,
        scaleY: Double,
        originalW: Int,
        originalH: Int,
    ): PointF {
        return PointF(
            MathUtils.roundHalfToEven(point.x * scaleX).coerceIn(0, originalW).toFloat(),
            MathUtils.roundHalfToEven(point.y * scaleY).coerceIn(0, originalH).toFloat(),
        )
    }

    private fun computeBoxScore(probMap: Mat, points: List<Point>): Float {
        if (points.isEmpty()) return 0f

        val h = probMap.rows()
        val w = probMap.cols()
        val xmin = points.minOf { kotlin.math.floor(it.x).toInt() }.coerceIn(0, w - 1)
        val xmax = points.maxOf { kotlin.math.ceil(it.x).toInt() }.coerceIn(0, w - 1)
        val ymin = points.minOf { kotlin.math.floor(it.y).toInt() }.coerceIn(0, h - 1)
        val ymax = points.maxOf { kotlin.math.ceil(it.y).toInt() }.coerceIn(0, h - 1)

        val mask = Mat(ymax - ymin + 1, xmax - xmin + 1, CvType.CV_8UC1, Scalar(0.0))
        val pts = MatOfPoint().apply {
            fromList(
                points.map { Point((it.x - xmin).toInt().toDouble(), (it.y - ymin).toInt().toDouble()) }
            )
        }
        Imgproc.fillPoly(mask, mutableListOf(pts), Scalar(1.0))

        val roi = probMap.submat(ymin, ymax + 1, xmin, xmax + 1)
        val mean = Core.mean(roi, mask)
        roi.release()
        mask.release()
        pts.release()
        return mean.`val`[0].toFloat()
    }
}
