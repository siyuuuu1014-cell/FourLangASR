package com.example.fourlangasr.asr

import android.content.res.AssetManager
import com.example.fourlangasr.model.AppLanguage
import com.example.fourlangasr.model.AsrResult
import com.example.fourlangasr.model.LoadResult
import com.example.fourlangasr.util.MemoryMeter
import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.OnlineModelConfig
import com.k2fsa.sherpa.onnx.OnlineRecognizer
import com.k2fsa.sherpa.onnx.OnlineRecognizerConfig
import com.k2fsa.sherpa.onnx.OnlineZipformer2CtcModelConfig

class ChineseSherpaEngine(
    private val assetManager: AssetManager,
) : AsrEngine {

    override val language = AppLanguage.ZH

    private var recognizer: OnlineRecognizer? = null

    override fun load(): LoadResult {
        release()

        val before = MemoryMeter.totalPssMb()
        val start = System.nanoTime()

        val modelConfig = OnlineModelConfig(
            zipformer2Ctc = OnlineZipformer2CtcModelConfig(
                model = "models/zh/model.int8.onnx",
            ),
            tokens = "models/zh/tokens.txt",
            numThreads = 2,
            debug = false,
            provider = "cpu",
        )

        val config = OnlineRecognizerConfig(
            featConfig = FeatureConfig(
                sampleRate = 16000,
                featureDim = 80,
                dither = 0.0f,
            ),
            modelConfig = modelConfig,
            enableEndpoint = false,
            decodingMethod = "greedy_search",
        )

        recognizer = OnlineRecognizer(
            assetManager = assetManager,
            config = config,
        )

        val loadMs = (System.nanoTime() - start) / 1_000_000
        val after = MemoryMeter.totalPssMb()

        return LoadResult(loadMs, before, after)
    }

    override fun transcribe(
        samples: FloatArray,
        sampleRate: Int,
    ): AsrResult {
        val r = checkNotNull(recognizer) { "Chinese model is not loaded" }

        val pssBefore = MemoryMeter.totalPssMb()
        val duration = samples.size / sampleRate.toFloat()

        val stream = r.createStream()

        val start = System.nanoTime()

        stream.acceptWaveform(samples, sampleRate)

        // Keep the same flush behavior used in our PC EXP4-A benchmark.
        stream.acceptWaveform(
            FloatArray((0.66f * sampleRate).toInt()),
            sampleRate,
        )
        stream.inputFinished()

        while (r.isReady(stream)) {
            r.decode(stream)
        }

        val result = r.getResult(stream)
        val text = result.text.trim()
        val elapsedMs = (System.nanoTime() - start) / 1_000_000

        /*
         * EXP5/EXP6: token 级时间戳作为 code-switch 触发信号。
         * timestamps 单位为秒（与 sherpa-onnx 1.13.7 PC 版一致）。
         */
        val tokens = result.tokens
        val timestamps = result.timestamps
        val unkTimes: FloatArray =
            if (
                timestamps.size ==
                tokens.size
            ) {

                tokens.indices
                    .filter { i ->
                        tokens[i] ==
                                "<unk>"
                    }
                    .map { i ->
                        timestamps[i]
                    }
                    .toFloatArray()

            } else {

                FloatArray(0)
            }

        stream.release()

        val pssAfter = MemoryMeter.totalPssMb()
        val elapsedSec = elapsedMs / 1000.0f
        val rtf = if (duration > 0f) elapsedSec / duration else 0f

        return AsrResult(
            text = text,
            audioDurationSec = duration,
            inferenceMs = elapsedMs,
            rtf = rtf,
            pssBeforeMb = pssBefore,
            pssAfterMb = pssAfter,
            unkTimestampsSec = unkTimes,
        )
    }

    override fun release() {
        recognizer?.release()
        recognizer = null
    }
}
