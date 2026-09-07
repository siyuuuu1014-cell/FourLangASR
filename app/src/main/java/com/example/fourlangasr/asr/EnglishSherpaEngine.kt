package com.example.fourlangasr.asr

import android.content.res.AssetManager
import com.example.fourlangasr.model.AppLanguage
import com.example.fourlangasr.model.AsrResult
import com.example.fourlangasr.model.LoadResult
import com.example.fourlangasr.util.MemoryMeter
import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig
import com.k2fsa.sherpa.onnx.OfflineTransducerModelConfig

class EnglishSherpaEngine(
    private val assetManager: AssetManager,
) : AsrEngine {

    override val language = AppLanguage.EN

    private var recognizer: OfflineRecognizer? = null

    override fun load(): LoadResult {
        release()

        val before = MemoryMeter.totalPssMb()
        val start = System.nanoTime()

        val modelConfig = OfflineModelConfig(
            transducer = OfflineTransducerModelConfig(
                encoder = "models/en/encoder.int8.onnx",
                decoder = "models/en/decoder.onnx",
                joiner = "models/en/joiner.int8.onnx",
            ),
            tokens = "models/en/tokens.txt",
            numThreads = 2,
            debug = false,
            provider = "cpu",
            modelType = "transducer",
        )

        val config = OfflineRecognizerConfig(
            featConfig = FeatureConfig(
                sampleRate = 16000,
                featureDim = 80,
                dither = 0.0f,
            ),
            modelConfig = modelConfig,
            decodingMethod = "greedy_search",
        )

        recognizer = OfflineRecognizer(
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
        val r = checkNotNull(recognizer) { "English model is not loaded" }

        val pssBefore = MemoryMeter.totalPssMb()
        val duration = samples.size / sampleRate.toFloat()

        val stream = r.createStream()
        stream.acceptWaveform(samples, sampleRate)

        val start = System.nanoTime()
        r.decode(stream)
        val elapsedMs = (System.nanoTime() - start) / 1_000_000

        val text = r.getResult(stream).text.trim()
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
        )
    }

    override fun release() {
        recognizer?.release()
        recognizer = null
    }
}
