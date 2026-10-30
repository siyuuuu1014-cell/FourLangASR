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

package com.example.fourlangasr.ocr.paddle.engine

import android.content.Context
import android.graphics.Bitmap
import com.example.fourlangasr.ocr.paddle.EngineConfig
import com.example.fourlangasr.ocr.paddle.PaddleOCRConfig
import com.example.fourlangasr.ocr.paddle.model.ModelConfig
import com.example.fourlangasr.ocr.paddle.model.OCRError
import com.example.fourlangasr.ocr.paddle.model.OCRResult
import com.example.fourlangasr.ocr.paddle.postprocess.BoxSorter
import com.example.fourlangasr.ocr.paddle.postprocess.DbPostProcessProfile
import com.example.fourlangasr.ocr.paddle.postprocess.QuadTextCrop
import com.example.fourlangasr.ocr.paddle.util.BitmapUtils

class OCREngine(
    context: Context,
    private val config: PaddleOCRConfig,
    engineConfig: EngineConfig,
    detModelAsset: String = "models/det/inference.onnx",
    recModelAsset: String = "models/rec/inference.onnx",
    recConfigAsset: String = "models/rec/inference.yml",
) {
    private val ortManager = ORTSessionManager(context, engineConfig)
    // 生命周期护栏：run() 持读锁、release() 持写锁，保证 release 不会在推理进行中关闭 Session
    // （OcrRuntime.releaseAll 与并发 run 的竞争）。读锁允许多个并发 run。
    private val lifecycle = java.util.concurrent.locks.ReentrantReadWriteLock()
    private val detectionEngine: DetectionEngine
    private val recognitionEngine: RecognitionEngine
    val coldLoadTimeMs: Long get() = ortManager.coldLoadTimeMs

    init {
        val configured = try {
            ortManager.loadModels(detModelAsset, recModelAsset)
            val recModelConfig = ModelConfig.parse(context, recConfigAsset)
            recModelConfig
        } catch (t: Throwable) {
            ortManager.release()
            throw t
        }
        detectionEngine = DetectionEngine(ortManager, config)
        recognitionEngine = RecognitionEngine(ortManager, configured.characterList)
    }

    fun run(bitmap: Bitmap, profile: DbPostProcessProfile? = null): OCREngineResult {
        val rl = lifecycle.readLock(); rl.lock()
        try {
            val srcMat = BitmapUtils.bitmapToBGRMat(bitmap)
            return try {
                runWithOwnedMat(srcMat, profile)
            } finally {
                srcMat.release()
            }
        } finally {
            rl.unlock()
        }
    }

    fun run(imageBytes: ByteArray, profile: DbPostProcessProfile? = null): OCREngineResult {
        val rl = lifecycle.readLock(); rl.lock()
        try {
            val srcMat = BitmapUtils.imdecodeBGR(imageBytes)
            if (srcMat.empty()) {
                srcMat.release()
                throw OCRError.InvalidImage()
            }
            return try {
                runWithOwnedMat(srcMat, profile)
            } finally {
                srcMat.release()
            }
        } finally {
            rl.unlock()
        }
    }

    private fun runWithOwnedMat(srcMat: org.opencv.core.Mat, profile: DbPostProcessProfile?): OCREngineResult {
        return run(srcMat, profile)
    }

    private fun run(srcMat: org.opencv.core.Mat, profile: DbPostProcessProfile?): OCREngineResult {
        val totalStart = System.currentTimeMillis()
        val detResult = detectionEngine.detect(srcMat, profile)
        val boxes = detResult.boxes

        if (boxes.isEmpty()) {
            val elapsed = System.currentTimeMillis() - totalStart
            return OCREngineResult(
                results = emptyList(),
                detectionTimeMs = detResult.timeMs,
                recognitionTimeMs = 0,
                totalTimeMs = elapsed,
                lineCount = 0,
                detPreprocessMs = detResult.preprocessMs,
                detInferenceMs = detResult.inferenceMs,
                detPostprocessMs = detResult.postprocessMs,
                detInputShape = detResult.inputShape,
                coldLoadTimeMs = ortManager.coldLoadTimeMs,
            )
        }

        // 2. 阶段2A 安全降级：识别是逐行线性耗时（实测 ~22-30ms/行），框数才是总时长主因。
        //    极端多框图只按面积保留最可能是正文的前 maxRecBoxes 框，其余丢弃并置降级标志，
        //    正常图（几十框以内）完全不受影响，结果与官方逐字节一致。
        val cap = config.maxRecBoxes
        val cappedByCount = cap > 0 && boxes.size > cap
        val keptBoxes = if (cappedByCount) boxes.sortedByDescending { quadArea(it) }.take(cap) else boxes
        val degraded = detResult.truncated || cappedByCount

        // 3. Sort boxes
        val sortedBoxes = BoxSorter.sortInReadingOrder(keptBoxes)

        // 3. Crop and recognize text regions
        var totalRecPreMs = 0L
        var totalRecInfMs = 0L
        var totalRecPostMs = 0L
        var totalRecMs = 0L
        var totalCropMs = 0L
        val allResults = mutableListOf<OCRResult>()
        val recInputShapes = mutableListOf<List<Int>>()
        val perLineRecMs = mutableListOf<Long>()
        val batchSize = config.recBatchSize.coerceAtLeast(1)

        var i = 0
        while (i < sortedBoxes.size) {
            val batchCrops = mutableListOf<org.opencv.core.Mat>()
            val batchBoxIndices = mutableListOf<Int>()
            var next = i
            while (next < sortedBoxes.size && batchCrops.size < batchSize) {
                val cropStart = System.currentTimeMillis()
                val crop = QuadTextCrop.crop(srcMat, sortedBoxes[next])
                totalCropMs += System.currentTimeMillis() - cropStart
                if (crop.rows() > 0 && crop.cols() > 0) {
                    batchCrops.add(crop)
                    batchBoxIndices.add(next)
                } else {
                    crop.release()
                }
                next++
            }

            try {
                if (batchCrops.isNotEmpty()) {
                    val batchResult = recognitionEngine.recognize(batchCrops)
                    totalRecPreMs += batchResult.preprocessMs
                    totalRecInfMs += batchResult.inferenceMs
                    totalRecPostMs += batchResult.postprocessMs
                    totalRecMs += batchResult.timeMs
                    recInputShapes.add(batchResult.inputShape)
                    if (batchSize == 1) {
                        perLineRecMs.add(batchResult.timeMs)
                    }

                    for (j in batchResult.texts.indices) {
                        val boxIdx = batchBoxIndices[j]
                        val (text, confidence) = batchResult.texts[j]
                        if (confidence >= config.recScoreThresh) {
                            allResults.add(
                                OCRResult(
                                    box = sortedBoxes[boxIdx],
                                    text = text,
                                    confidence = confidence,
                                )
                            )
                        }
                    }
                }
            } finally {
                batchCrops.forEach { it.release() }
            }
            i = next
        }

        val totalElapsed = System.currentTimeMillis() - totalStart
        val pipelineOverhead = totalElapsed - detResult.timeMs - totalRecMs

        return OCREngineResult(
            results = allResults,
            detectionTimeMs = detResult.timeMs,
            recognitionTimeMs = totalRecMs,
            totalTimeMs = totalElapsed,
            lineCount = allResults.size,
            detPreprocessMs = detResult.preprocessMs,
            detInferenceMs = detResult.inferenceMs,
            detPostprocessMs = detResult.postprocessMs,
            recPreprocessMs = totalRecPreMs,
            recInferenceMs = totalRecInfMs,
            recPostprocessMs = totalRecPostMs,
            cropMs = totalCropMs,
            pipelineOverheadMs = pipelineOverhead,
            coldLoadTimeMs = ortManager.coldLoadTimeMs,
            detInputShape = detResult.inputShape,
            recInputShapes = recInputShapes,
            perLineRecMs = perLineRecMs,
            degraded = degraded,
            detectedBoxCount = boxes.size,
        )
    }

    /** 四角框鞋带公式面积，用于降级时按"最可能是正文"排序保留。 */
    private fun quadArea(box: com.example.fourlangasr.ocr.paddle.model.OCRBox): Double {
        val p = box.points
        var s = 0.0
        for (i in p.indices) {
            val a = p[i]; val b = p[(i + 1) % p.size]
            s += a.x.toDouble() * b.y - b.x.toDouble() * a.y
        }
        return kotlin.math.abs(s) / 2.0
    }

    fun release() {
        // 写锁：等待所有在跑的 run() 结束再关 Session，避免 use-after-close。
        val wl = lifecycle.writeLock(); wl.lock()
        try {
            ortManager.release()
        } finally {
            wl.unlock()
        }
    }
}
