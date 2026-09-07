package com.example.fourlangasr.asr

import android.content.Context
import com.example.fourlangasr.model.AppLanguage
import com.example.fourlangasr.model.AsrResult
import com.example.fourlangasr.model.LoadResult
import com.example.fourlangasr.util.AssetUtils
import com.example.fourlangasr.util.MemoryMeter
import org.json.JSONObject
import org.vosk.Model
import org.vosk.Recognizer
import java.io.File
import kotlin.math.roundToInt

class RussianVoskEngine(
    private val context: Context,
) : AsrEngine {

    override val language = AppLanguage.RU

    private var model: Model? = null

    override fun load(): LoadResult {
        release()

        val before = MemoryMeter.totalPssMb()
        val start = System.nanoTime()

        val modelDir = File(
            context.filesDir,
            "asr_models/ru/vosk-model-small-ru-0.22",
        )

        AssetUtils.copyAssetFolderIfNeeded(
            context = context,
            assetPath = "models/ru/vosk-model-small-ru-0.22",
            destinationDir = modelDir,
            requiredMarker = "am/final.mdl",
        )

        model = Model(modelDir.absolutePath)

        val loadMs = (System.nanoTime() - start) / 1_000_000
        val after = MemoryMeter.totalPssMb()

        return LoadResult(loadMs, before, after)
    }

    override fun transcribe(
        samples: FloatArray,
        sampleRate: Int,
    ): AsrResult {
        val loadedModel = checkNotNull(model) { "Russian Vosk model is not loaded" }

        val pssBefore = MemoryMeter.totalPssMb()
        val duration = samples.size / sampleRate.toFloat()

        val pcm = ShortArray(samples.size) { i ->
            (samples[i].coerceIn(-1.0f, 1.0f) * 32767.0f)
                .roundToInt()
                .toShort()
        }

        val start = System.nanoTime()

        val recognizer = Recognizer(
            loadedModel,
            sampleRate.toFloat(),
        )

        // Feed in ~200 ms chunks, matching Vosk's online-friendly use.
        val chunkSize = (sampleRate * 0.2f).toInt()
        var offset = 0

        while (offset < pcm.size) {
            val count = minOf(chunkSize, pcm.size - offset)
            val chunk = pcm.copyOfRange(offset, offset + count)
            recognizer.acceptWaveForm(chunk, chunk.size)
            offset += count
        }

        val json = recognizer.finalResult
        val text = JSONObject(json).optString("text", "").trim()

        recognizer.close()

        val elapsedMs = (System.nanoTime() - start) / 1_000_000
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
        model?.close()
        model = null
    }
}
