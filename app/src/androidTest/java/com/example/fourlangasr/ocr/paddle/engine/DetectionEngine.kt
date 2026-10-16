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

import android.graphics.Bitmap
import com.example.fourlangasr.ocr.paddle.PaddleOCRConfig
import com.example.fourlangasr.ocr.paddle.model.OCRBox
import com.example.fourlangasr.ocr.paddle.postprocess.DBPostProcessor
import com.example.fourlangasr.ocr.paddle.postprocess.DbPostProcessProfile
import com.example.fourlangasr.ocr.paddle.preprocess.DetPreprocessResult
import com.example.fourlangasr.ocr.paddle.preprocess.DetPreprocessor
import org.opencv.core.Mat

class DetectionEngine(
    private val ortManager: ORTSessionManager,
    private val config: PaddleOCRConfig,
) {
    data class DetectionResult(
        val boxes: List<OCRBox>,
        val preprocessMs: Long,
        val inferenceMs: Long,
        val postprocessMs: Long,
        val timeMs: Long,
        val inputShape: List<Int>,
        /** DB 后处理是否因工作预算被截断（安全降级信号）。 */
        val truncated: Boolean = false,
    )

    fun detect(bitmap: Bitmap, profile: DbPostProcessProfile? = null): DetectionResult {
        return detect(profile) {
            DetPreprocessor.preprocess(
                bitmap,
                config.detLimitSideLen,
                config.detLimitType,
                config.detMaxSideLimit,
                config.detImgMode,
            )
        }
    }

    fun detect(src: Mat, profile: DbPostProcessProfile? = null): DetectionResult {
        return detect(profile) {
            DetPreprocessor.preprocess(
                src,
                config.detLimitSideLen,
                config.detLimitType,
                config.detMaxSideLimit,
                config.detImgMode,
            )
        }
    }

    private fun detect(
        profile: DbPostProcessProfile?,
        preprocessor: () -> DetPreprocessResult,
    ): DetectionResult {
        val preStart = System.currentTimeMillis()
        val preResult = preprocessor()
        val preprocessMs = System.currentTimeMillis() - preStart

        val infStart = System.currentTimeMillis()
        val (outputData, outputShape) = ortManager.runDetection(preResult.tensorData, preResult.shape)
        val inferenceMs = System.currentTimeMillis() - infStart

        val postStart = System.currentTimeMillis()
        val truncated = booleanArrayOf(false)
        val boxes = DBPostProcessor.process(
            pred = outputData,
            predShape = outputShape,
            thresh = config.detThresh,
            boxThresh = config.detBoxThresh,
            unclipRatio = config.detUnclipRatio,
            maxCandidates = config.detMaxCandidates,
            useDilation = config.detUseDilation,
            scoreMode = config.detScoreMode,
            boxType = config.detBoxType,
            originalH = preResult.originalH,
            originalW = preResult.originalW,
            profile = profile,
            workBudgetMs = config.detPostBudgetMs,
            truncatedOut = truncated,
        )
        val postprocessMs = System.currentTimeMillis() - postStart

        return DetectionResult(
            boxes = boxes,
            preprocessMs = preprocessMs,
            inferenceMs = inferenceMs,
            postprocessMs = postprocessMs,
            timeMs = preprocessMs + inferenceMs + postprocessMs,
            inputShape = listOf(1, 3, preResult.shape[2].toInt(), preResult.shape[3].toInt()),
            truncated = truncated[0],
        )
    }
}
