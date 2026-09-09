package com.example.fourlangasr.lid

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Collections
import kotlin.math.exp

class LidEngine(
    context: Context,
) {

    private val appContext =
        context.applicationContext

    private val environment =
        OrtEnvironment.getEnvironment()

    private var sessionOptions:
            OrtSession.SessionOptions? =
        null

    private var session:
            OrtSession? =
        null

    private var inputName:
            String? =
        null

    @Synchronized
    fun load() {

        if (session != null) {
            return
        }

        val modelFile =
            ensureModelFile()

        val options =
            OrtSession.SessionOptions().apply {

                setIntraOpNumThreads(1)
                setInterOpNumThreads(1)
            }

        try {

            val newSession =
                environment.createSession(
                    modelFile.absolutePath,
                    options,
                )

            require(
                "logits" in
                        newSession.outputNames
            ) {
                "VoxLingua model has no 'logits' output. " +
                        "Outputs=${newSession.outputNames}"
            }

            val newInputName =
                newSession.inputNames
                    .firstOrNull()
                    ?: error(
                        "VoxLingua model has no input."
                    )

            sessionOptions =
                options

            session =
                newSession

            inputName =
                newInputName

        } catch (
            e: Throwable
        ) {

            try {
                options.close()
            } catch (_: Throwable) {
            }

            throw e
        }
    }

    @Synchronized
    fun identify(
        samples: FloatArray,
        sampleRate: Int,
    ): LidResult {

        require(
            sampleRate ==
                    SAMPLE_RATE
        ) {
            "LID expects 16000 Hz, " +
                    "got $sampleRate Hz."
        }

        require(
            samples.isNotEmpty()
        ) {
            "LID samples are empty."
        }

        load()

        val currentSession =
            session
                ?: error(
                    "LID session is not loaded."
                )

        val currentInputName =
            inputName
                ?: error(
                    "LID input name is unavailable."
                )

        val buffer =
            ByteBuffer
                .allocateDirect(
                    samples.size *
                            Float.SIZE_BYTES
                )
                .order(
                    ByteOrder.nativeOrder()
                )
                .asFloatBuffer()

        buffer.put(
            samples
        )

        buffer.flip()

        val inputTensor =
            OnnxTensor.createTensor(
                environment,
                buffer,
                longArrayOf(
                    1L,
                    samples.size.toLong(),
                ),
            )

        val startNs =
            System.nanoTime()

        try {

            currentSession.run(
                Collections.singletonMap(
                    currentInputName,
                    inputTensor,
                ),
                setOf(
                    OUTPUT_LOGITS
                ),
            ).use { outputs ->

                val output =
                    outputs
                        .get(
                            OUTPUT_LOGITS
                        )
                        .orElseThrow {
                            IllegalStateException(
                                "Missing logits output."
                            )
                        }

                val logits =
                    extractLogits(
                        output.value
                    )

                require(
                    logits.size >=
                            NUM_CLASSES
                ) {
                    "Expected at least $NUM_CLASSES logits, " +
                            "got ${logits.size}."
                }

                val elapsedMs =
                    (
                        System.nanoTime() -
                                startNs
                        ) /
                            1_000_000L

                return buildFourWayResult(
                    logits = logits,
                    inferenceMs = elapsedMs,
                )
            }

        } finally {

            inputTensor.close()
        }
    }

    @Synchronized
    fun release() {

        try {

            session
                ?.close()

        } finally {

            session =
                null

            inputName =
                null

            try {

                sessionOptions
                    ?.close()

            } finally {

                sessionOptions =
                    null
            }
        }
    }

    private fun extractLogits(
        value: Any?,
    ): FloatArray {

        return when (
            value
        ) {

            is FloatArray ->
                value

            is Array<*> -> {

                val first =
                    value.firstOrNull()

                when (
                    first
                ) {

                    is FloatArray ->
                        first

                    else ->
                        error(
                            "Unexpected logits array type: " +
                                    "${first?.javaClass?.name}"
                        )
                }
            }

            else ->
                error(
                    "Unexpected logits type: " +
                            "${value?.javaClass?.name}"
                )
        }
    }

    private fun buildFourWayResult(
        logits: FloatArray,
        inferenceMs: Long,
    ): LidResult {

        val candidates =
            TARGETS.map {
                target ->

                target to
                        logits[
                            target.index
                        ]
            }

        val fourWayProbabilities =
            softmax(
                candidates.map {
                    it.second
                }.toFloatArray()
            )

        val ranking =
            candidates.indices
                .map {
                    i ->

                    TargetScore(
                        code =
                            candidates[i]
                                .first
                                .code,
                        probability =
                            fourWayProbabilities[i],
                    )
                }
                .sortedByDescending {
                    it.probability
                }

        val best =
            ranking[0]

        val second =
            ranking[1]

        val fullProbabilities =
            softmax(
                logits
            )

        var targetMass =
            0.0f

        for (
            target
            in TARGETS
        ) {

            targetMass +=
                fullProbabilities[
                    target.index
                ]
        }

        return LidResult(
            languageCode =
                best.code,
            confidence =
                best.probability,
            secondLanguageCode =
                second.code,
            secondConfidence =
                second.probability,
            margin =
                best.probability -
                        second.probability,
            targetMass =
                targetMass,
            inferenceMs =
                inferenceMs,
            skipped =
                false,
            reason =
                "VOXLINGUA_4WAY",
        )
    }

    private fun softmax(
        values: FloatArray,
    ): FloatArray {

        val maxValue =
            values.maxOrNull()
                ?: return FloatArray(0)

        val exps =
            DoubleArray(
                values.size
            )

        var sum =
            0.0

        for (
            i
            in values.indices
        ) {

            val e =
                exp(
                    (
                        values[i] -
                                maxValue
                        ).toDouble()
                )

            exps[i] =
                e

            sum +=
                e
        }

        return FloatArray(
            values.size
        ) {
            i ->

            (
                exps[i] /
                        sum
                ).toFloat()
        }
    }

    private fun ensureModelFile():
            File {

        val modelDir =
            File(
                appContext.filesDir,
                "lid",
            )

        if (
            !modelDir.exists()
        ) {

            check(
                modelDir.mkdirs()
            ) {
                "Unable to create: " +
                        modelDir.absolutePath
            }
        }

        val outputFile =
            File(
                modelDir,
                MODEL_FILE_NAME,
            )

        val assetLength =
            appContext
                .assets
                .openFd(
                    MODEL_ASSET
                )
                .use {
                    it.length
                }

        if (
            outputFile.isFile &&
            outputFile.length() ==
            assetLength
        ) {

            return outputFile
        }

        val tempFile =
            File(
                modelDir,
                "$MODEL_FILE_NAME.tmp",
            )

        if (
            tempFile.exists()
        ) {

            tempFile.delete()
        }

        appContext
            .assets
            .open(
                MODEL_ASSET
            )
            .use {
                input ->

                FileOutputStream(
                    tempFile
                ).use {
                    output ->

                    input.copyTo(
                        output,
                        bufferSize =
                            1024 * 1024,
                    )
                }
            }

        check(
            tempFile.length() ==
                    assetLength
        ) {
            "LID model copy size mismatch: " +
                    "${tempFile.length()} vs " +
                    assetLength
        }

        if (
            outputFile.exists()
        ) {

            check(
                outputFile.delete()
            ) {
                "Unable to replace old LID model."
            }
        }

        check(
            tempFile.renameTo(
                outputFile
            )
        ) {
            "Unable to move LID model into place."
        }

        return outputFile
    }

    private data class Target(
        val code: String,
        val index: Int,
    )

    private data class TargetScore(
        val code: String,
        val probability: Float,
    )

    companion object {

        const val SAMPLE_RATE =
            16000

        /*
         * Very short utterances are less reliable for LID.
         * v0.4 keeps the currently loaded ASR for <1 s audio.
         */
        const val MIN_AUDIO_SEC =
            1.0f

        const val MODEL_ASSET =
            "voxlingua107.onnx"

        private const val MODEL_FILE_NAME =
            "voxlingua107.onnx"

        private const val OUTPUT_LOGITS =
            "logits"

        private const val NUM_CLASSES =
            107

        /*
         * From the model's lang_map.json:
         *
         * en -> 20
         * ru -> 77
         * uz -> 101
         * zh -> 106
         */
        private val TARGETS =
            listOf(
                Target(
                    code = "zh",
                    index = 106,
                ),
                Target(
                    code = "en",
                    index = 20,
                ),
                Target(
                    code = "ru",
                    index = 77,
                ),
                Target(
                    code = "uz",
                    index = 101,
                ),
            )
    }
}
