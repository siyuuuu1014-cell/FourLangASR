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

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import com.example.fourlangasr.ocr.paddle.EngineConfig
import com.example.fourlangasr.ocr.paddle.model.OCRError
import java.nio.FloatBuffer

class ORTSessionManager(
    private val context: Context,
    private val config: EngineConfig,
) {
    private var env: OrtEnvironment? = null
    private var detSession: OrtSession? = null
    private var recSession: OrtSession? = null
    private var detInputName: String = "x"
    private var recInputName: String = "x"
    var coldLoadTimeMs: Long = 0
        private set

    companion object {
        /** 阶段2B-0 诊断开关（默认关，保持对拍基线）：true 时关闭 CPU 内存 arena（省 ~200MB native）。 */
        const val DISABLE_CPU_MEM_ARENA = false
        /** 阶段2B-0 诊断开关：true 会抛错（单例 env 不支持），仅作记录，勿开。 */
        const val DISABLE_PER_SESSION_THREADS = false
    }

    fun loadModels(detAssetPath: String, recAssetPath: String) {
        val loadStart = System.currentTimeMillis()
        env = OrtEnvironment.getEnvironment()
        val opts = OrtSession.SessionOptions().apply {
            // 相对上游的唯一改动：ALL_OPT -> BASIC_OPT。
            // 阶段0 实测（experiments/ocr/ort_compat_check.py）：PP-OCRv6 五个模型在
            // ORT_ENABLE_BASIC 下，ORT 1.27 与 1.31 的输出逐字节一致；EXTENDED/ALL 会出现
            // 内核选择差异（small_rec max|Δ|=2.2e-6）。固定 BASIC 才能让 PC golden 对拍有意义，
            // 也与 KitTranslationEngine.makeSession 的既有约束一致。
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
            setIntraOpNumThreads(config.numThreads)
            // 阶段2B-0 诊断结论（见 README 2B-0）：挂起来自 libort4jrt.so(ORT 1.27) 内部线程池
            // 在反复大张量推理后的 futex 等待；关 arena 能把 native 常驻从 ~232MB 降到 ~25MB，
            // 但不解决挂起；disablePerSessionThreads() 因走的是单例 env 会直接抛错，不可用。
            // 两个开关默认关闭，保持与阶段1/2A 一致的对拍基线；需要内存对照时再单独打开 arena 开关。
            if (DISABLE_CPU_MEM_ARENA) setCPUArenaAllocator(false)
            if (DISABLE_PER_SESSION_THREADS) disablePerSessionThreads()
        }
        try {
            val ortEnv = env ?: throw OCRError.ModelLoadFailed("OCR", Exception("Environment not initialized"))
            val detBytes = readModelAsset(detAssetPath)
            val recBytes = readModelAsset(recAssetPath)
            try {
                detSession = ortEnv.createSession(detBytes, opts)
            } catch (t: Throwable) {
                throw OCRError.ModelLoadFailed("detection", t)
            }
            try {
                recSession = ortEnv.createSession(recBytes, opts)
            } catch (t: Throwable) {
                detSession?.close()
                detSession = null
                throw OCRError.ModelLoadFailed("recognition", t)
            }

            detInputName = try {
                detSession!!.inputNames.iterator().next()
            } catch (t: Throwable) {
                throw OCRError.ModelLoadFailed("detection", t)
            }
            recInputName = try {
                recSession!!.inputNames.iterator().next()
            } catch (t: Throwable) {
                throw OCRError.ModelLoadFailed("recognition", t)
            }
            coldLoadTimeMs = System.currentTimeMillis() - loadStart
        } finally {
            opts.close()
        }
    }

    fun runDetection(input: FloatArray, shape: LongArray): Pair<FloatArray, LongArray> {
        val session = detSession
            ?: throw OCRError.ModelLoadFailed("detection", Exception("Session not initialized"))
        val ortEnv = env
            ?: throw OCRError.ModelLoadFailed("detection", Exception("Environment not initialized"))
        return runSession(ortEnv, session, detInputName, input, shape, "detection")
    }

    fun runRecognition(input: FloatArray, shape: LongArray): Pair<FloatArray, LongArray> {
        val session = recSession
            ?: throw OCRError.ModelLoadFailed("recognition", Exception("Session not initialized"))
        val ortEnv = env
            ?: throw OCRError.ModelLoadFailed("recognition", Exception("Environment not initialized"))
        return runSession(ortEnv, session, recInputName, input, shape, "recognition")
    }

    fun release() {
        try {
            detSession?.close()
        } finally {
            detSession = null
            try {
                recSession?.close()
            } finally {
                recSession = null
                env = null
            }
        }
    }

    private fun readModelAsset(assetPath: String): ByteArray {
        return try {
            context.assets.open(assetPath).use { it.readBytes() }
        } catch (t: Throwable) {
            throw OCRError.ModelNotFound(assetPath, t)
        }
    }

    private fun runSession(
        ortEnv: OrtEnvironment,
        session: OrtSession,
        inputName: String,
        input: FloatArray,
        shape: LongArray,
        modelName: String,
    ): Pair<FloatArray, LongArray> {
        val tensor = try {
            OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(input), shape)
        } catch (t: Throwable) {
            throw OCRError.InferenceFailed(modelName, t)
        }
        val result = try {
            try {
                session.run(mapOf(inputName to tensor))
            } catch (t: Throwable) {
                throw OCRError.InferenceFailed(modelName, t)
            }
        } finally {
            tensor.close()
        }

        return try {
            try {
                val outputName = session.outputNames.iterator().next()
                val ortValue = result.get(outputName)
                    .orElseThrow { Exception("No output tensor found") }
                val outputTensor = ortValue as? OnnxTensor
                    ?: throw Exception("Output is not an ONNX tensor")
                Pair(copyFloatBuffer(outputTensor.floatBuffer), outputTensor.info.shape)
            } catch (t: Throwable) {
                throw OCRError.InferenceFailed(modelName, t)
            }
        } finally {
            result.close()
        }
    }

    private fun copyFloatBuffer(buffer: FloatBuffer): FloatArray {
        val duplicate = buffer.duplicate()
        duplicate.rewind()
        val output = FloatArray(duplicate.remaining())
        duplicate.get(output)
        return output
    }
}
