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

package com.example.fourlangasr.ocr.paddle

data class PaddleOCRConfig(
    val detImgMode: String = "BGR",
    val detLimitSideLen: Int = 64,
    val detLimitType: String = "min",
    val detMaxSideLimit: Int = 4000,
    val detThresh: Float = 0.3f,
    val detBoxThresh: Float = 0.6f,
    val detUnclipRatio: Float = 1.5f,
    val detMaxCandidates: Int = 3000,
    val detUseDilation: Boolean = false,
    val detScoreMode: String = "fast",
    val detBoxType: String = "quad",
    val recScoreThresh: Float = 0.0f,
    val recBatchSize: Int = 1,
    // ---- 阶段2A 本地新增：生产可用的"绝不无限期挂起"护栏，默认全开但只影响极端图 ----
    /**
     * DB 逐候选后处理的硬预算（毫秒）。0 = 不限幅（官方行为）。
     * 真实照片实测 DB 只要 ≤25ms，这条只在病态海量轮廓时兜底，超时返回已识别的部分框。
     */
    val detPostBudgetMs: Long = 2500,
    /**
     * 送入识别的最大文本框数。0 = 不限。实测每行 rec ~22-30ms，框数才是总耗时的线性主因，
     * 这里按面积保留最可能是正文的前 N 框，超出部分丢弃并置降级标志，避免"几百行→几十秒"。
     */
    val maxRecBoxes: Int = 120,
)
