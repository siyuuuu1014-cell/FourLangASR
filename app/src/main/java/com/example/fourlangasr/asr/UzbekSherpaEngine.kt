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
import com.k2fsa.sherpa.onnx.OnlineTransducerModelConfig

class UzbekSherpaEngine(
    private val assetManager: AssetManager,
) : AsrEngine {

    override val language = AppLanguage.UZ

    private var recognizer: OnlineRecognizer? = null

    override fun load(): LoadResult {
        release()

        val before = MemoryMeter.totalPssMb()
        val start = System.nanoTime()

        val modelConfig = OnlineModelConfig(
            transducer = OnlineTransducerModelConfig(
                encoder = "models/uz/encoder.onnx",
                decoder = "models/uz/decoder.onnx",
                joiner = "models/uz/joiner.onnx",
            ),
            tokens = "models/uz/tokens.txt",
            numThreads = 2,
            debug = false,
            provider = "cpu",
            modelType = "zipformer2",
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
        val r = checkNotNull(recognizer) { "Uzbek model is not loaded" }

        val pssBefore = MemoryMeter.totalPssMb()
        val duration = samples.size / sampleRate.toFloat()

        val stream = r.createStream()
        val start = System.nanoTime()

        stream.acceptWaveform(samples, sampleRate)
        stream.acceptWaveform(
            FloatArray((0.8f * sampleRate).toInt()),
            sampleRate,
        )
        stream.inputFinished()

        while (r.isReady(stream)) {
            r.decode(stream)
        }

        val text = r.getResult(stream).text.trim()
        val elapsedMs = (System.nanoTime() - start) / 1_000_000
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
