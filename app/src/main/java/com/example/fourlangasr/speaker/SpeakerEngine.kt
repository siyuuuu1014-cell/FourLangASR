package com.example.fourlangasr.speaker

import android.content.res.AssetManager
import com.k2fsa.sherpa.onnx.SpeakerEmbeddingExtractor
import com.k2fsa.sherpa.onnx.SpeakerEmbeddingExtractorConfig

internal class SpeakerEngine(
    private val assetManager: AssetManager,
) {

    private var extractor:
            SpeakerEmbeddingExtractor? =
        null

    @Synchronized
    fun load() {

        if (extractor != null) {
            return
        }

        extractor =
            SpeakerEmbeddingExtractor(
                assetManager = assetManager,
                config =
                    SpeakerEmbeddingExtractorConfig(
                        model =
                            MODEL_ASSET,
                        numThreads = 1,
                        debug = false,
                        provider = "cpu",
                    ),
            )
    }

    @Synchronized
    fun extract(
        samples: FloatArray,
        sampleRate: Int,
    ): FloatArray {

        require(
            sampleRate == 16000
        ) {
            "Speaker model expects 16000 Hz, " +
                    "got $sampleRate Hz."
        }

        require(
            samples.isNotEmpty()
        ) {
            "Speaker samples are empty."
        }

        load()

        val currentExtractor =
            extractor
                ?: error(
                    "Speaker extractor is not loaded."
                )

        val stream =
            currentExtractor
                .createStream()

        try {

            stream.acceptWaveform(
                samples,
                sampleRate,
            )

            stream.inputFinished()

            if (
                !currentExtractor
                    .isReady(stream)
            ) {

                error(
                    "Speaker extractor is not ready. " +
                            "The utterance may be too short."
                )
            }

            return SpeakerProfile
                .normalize(
                    currentExtractor
                        .compute(stream)
                )

        } finally {

            stream.release()
        }
    }

    @Synchronized
    fun release() {

        extractor
            ?.release()

        extractor =
            null
    }

    companion object {

        /*
         * Put this file directly under:
         *
         * app/src/main/assets/speaker_embedding.onnx
         */
        const val MODEL_ASSET =
            "speaker_embedding.onnx"
    }
}
