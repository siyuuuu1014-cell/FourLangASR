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

import com.example.fourlangasr.ocr.paddle.model.OCRResult

data class OCREngineResult(
    val results: List<OCRResult>,
    val detectionTimeMs: Long,
    val recognitionTimeMs: Long,
    val totalTimeMs: Long,
    val lineCount: Int,
    // Detailed per-stage timing
    val detPreprocessMs: Long = 0,
    val detInferenceMs: Long = 0,
    val detPostprocessMs: Long = 0,
    val recPreprocessMs: Long = 0,
    val recInferenceMs: Long = 0,
    val recPostprocessMs: Long = 0,
    // 文字区域透视裁剪耗时（与 rec 推理分开计，阶段2A 定位卡顿用）
    val cropMs: Long = 0,
    val pipelineOverheadMs: Long = 0,
    val coldLoadTimeMs: Long = 0,
    // Input tensor shapes
    val detInputShape: List<Int> = emptyList(),
    val recInputShapes: List<List<Int>> = emptyList(),
    // Per-line timing (only populated when recBatchSize == 1)
    val perLineRecMs: List<Long> = emptyList(),
    // 阶段2A 安全降级信号：true 表示这张图触发了 DB 预算截断或识别框数上限，
    // 只返回了部分结果，调用方可据此提示"请手动裁剪后重试"。正常图恒为 false。
    val degraded: Boolean = false,
    // DB 检出（截断/上限前）的原始框数，用于报告降级幅度。
    val detectedBoxCount: Int = 0,
)
