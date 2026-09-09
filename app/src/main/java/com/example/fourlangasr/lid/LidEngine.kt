package com.example.fourlangasr.lid

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp

class LidEngine(
    context: Context,
) {

    private val appContext =
        context.applicationContext

    private var interpreter:
            Interpreter? =
        null

    /*
     * Keep a strong reference to the mmap backing the Interpreter.
     */
    private var modelBuffer:
            MappedByteBuffer? =
        null

    @Synchronized
    fun load() {

        if (interpreter != null) {
            return
        }

        val modelFile =
            ensureModelFile()

        val mapped =
            FileInputStream(
                modelFile
            ).channel.use {
                channel ->

                channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    0L,
                    channel.size(),
                )
            }

        val options =
            Interpreter.Options().apply {

                setNumThreads(
                    1
                )
            }

        val newInterpreter =
            Interpreter(
                mapped,
                options,
            )

        try {

            /*
             * EXP4 was converted as a true static 5-second model.
             * Fail early if a different model was accidentally packaged.
             */
            val inputShape =
                newInterpreter
                    .getInputTensor(
                        0
                    )
                    .shape()

            require(
                inputShape.contentEquals(
                    intArrayOf(
                        1,
                        FIXED_SAMPLES,
                    )
                )
            ) {
                "Unexpected LID input shape: " +
                        inputShape.contentToString() +
                        ", expected [1, $FIXED_SAMPLES]"
            }

            val outputShape =
                newInterpreter
                    .getOutputTensor(
                        0
                    )
                    .shape()

            val outputElements =
                outputShape.fold(
                    1
                ) {
                    acc,
                    value ->

                    acc * value
                }

            require(
                outputElements ==
                        NUM_CLASSES
            ) {
                "Unexpected LID output shape: " +
                        outputShape.contentToString() +
                        ", expected $NUM_CLASSES logits."
            }

            modelBuffer =
                mapped

            interpreter =
                newInterpreter

        } catch (
            e: Throwable
        ) {

            try {
                newInterpreter.close()
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
            "LID expects $SAMPLE_RATE Hz, " +
                    "got $sampleRate Hz."
        }

        require(
            samples.isNotEmpty()
        ) {
            "LID samples are empty."
        }

        load()

        val currentInterpreter =
            interpreter
                ?: error(
                    "LID Interpreter is not loaded."
                )

        val prepared =
            makeFixedWindow(
                samples
            )

        /*
         * Model input shape is [1, 80000].
         * Model output shape is [1, 107].
         */
        val input =
            arrayOf(
                prepared.samples
            )

        val output =
            arrayOf(
                FloatArray(
                    NUM_CLASSES
                )
            )

        val startNs =
            System.nanoTime()

        currentInterpreter.run(
            input,
            output,
        )

        val inferenceMs =
            (
                System.nanoTime() -
                        startNs
                ) /
                    1_000_000L

        return buildFourWayResult(
            logits =
                output[0],
            inferenceMs =
                inferenceMs,
            windowMode =
                prepared.mode,
        )
    }

    @Synchronized
    fun release() {

        try {

            interpreter
                ?.close()

        } finally {

            interpreter =
                null

            modelBuffer =
                null
        }
    }

    private fun makeFixedWindow(
        samples: FloatArray,
    ): PreparedAudio {

        if (
            samples.size ==
            FIXED_SAMPLES
        ) {

            return PreparedAudio(
                samples =
                    samples.copyOf(),
                mode =
                    "exact",
            )
        }

        if (
            samples.size >
            FIXED_SAMPLES
        ) {

            val start =
                (
                    samples.size -
                            FIXED_SAMPLES
                    ) /
                        2

            return PreparedAudio(
                samples =
                    samples.copyOfRange(
                        start,
                        start +
                                FIXED_SAMPLES,
                    ),
                mode =
                    "center_crop",
            )
        }

        /*
         * Match the successful PC EXP4 benchmark:
         * short utterances are symmetrically zero-padded to 5 s.
         */
        val result =
            FloatArray(
                FIXED_SAMPLES
            )

        val totalPadding =
            FIXED_SAMPLES -
                    samples.size

        val leftPadding =
            totalPadding /
                    2

        samples.copyInto(
            destination =
                result,
            destinationOffset =
                leftPadding,
        )

        return PreparedAudio(
            samples =
                result,
            mode =
                "zero_pad",
        )
    }

    private fun buildFourWayResult(
        logits: FloatArray,
        inferenceMs: Long,
        windowMode: String,
    ): LidResult {

        require(
            logits.size ==
                    NUM_CLASSES
        ) {
            "Expected $NUM_CLASSES LID logits, " +
                    "got ${logits.size}."
        }

        val fourWayLogits =
            FloatArray(
                LidTarget.entries.size
            ) {
                i ->

                logits[
                    LidTarget.entries[i].classIndex
                ]
            }

        val fourWayProbabilities =
            softmax(
                fourWayLogits
            )

        val ranking =
            LidTarget.entries.indices
                .map {
                    i ->

                    TargetScore(
                        code =
                            LidTarget.entries[i].code,
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
            in LidTarget.entries
        ) {

            targetMass +=
                fullProbabilities[
                    target.classIndex
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
            windowMode =
                windowMode,
            skipped =
                false,
            reason =
                "VOXLINGUA_TFLITE_4WAY",
        )
    }

    private fun softmax(
        values: FloatArray,
    ): FloatArray {

        val maxValue =
            values.maxOrNull()
                ?: return FloatArray(
                    0
                )

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
                "Unable to create LID model directory: " +
                        modelDir.absolutePath
            }
        }

        val outputFile =
            File(
                modelDir,
                MODEL_CACHE_FILE,
            )

        if (
            outputFile.isFile &&
            outputFile.length() >
            MIN_EXPECTED_MODEL_BYTES
        ) {

            return outputFile
        }

        val tempFile =
            File(
                modelDir,
                "$MODEL_CACHE_FILE.tmp",
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
                inputStream ->

                FileOutputStream(
                    tempFile
                ).use {
                    outputStream ->

                    inputStream.copyTo(
                        outputStream,
                        bufferSize =
                            1024 *
                                    1024,
                    )
                }
            }

        check(
            tempFile.length() >
                    MIN_EXPECTED_MODEL_BYTES
        ) {
            "Copied LID model is unexpectedly small: " +
                    "${tempFile.length()} bytes."
        }

        if (
            outputFile.exists()
        ) {

            check(
                outputFile.delete()
            ) {
                "Unable to replace old cached LID model."
            }
        }

        check(
            tempFile.renameTo(
                outputFile
            )
        ) {
            "Unable to move cached LID model into place."
        }

        return outputFile
    }

    private data class PreparedAudio(
        val samples: FloatArray,
        val mode: String,
    )

    private data class TargetScore(
        val code: String,
        val probability: Float,
    )

    companion object {

        const val SAMPLE_RATE =
            16000

        const val MIN_AUDIO_SEC =
            1.0f

        const val FIXED_SECONDS =
            5.0f

        const val FIXED_SAMPLES =
            80000

        const val MODEL_ASSET =
            "voxlingua107_float32.tflite"

        private const val MODEL_CACHE_FILE =
            "voxlingua107_float32_v1.tflite"

        private const val MIN_EXPECTED_MODEL_BYTES =
            80_000_000L

        /*
         * VoxLingua107 输出 107 类 logits。
         * 四个目标语种的索引见 lid/LidTarget.kt。
         */
        private const val NUM_CLASSES =
            107
    }
}
