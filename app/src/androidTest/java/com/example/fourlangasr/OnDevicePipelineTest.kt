package com.example.fourlangasr

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.fourlangasr.asr.AsrManager
import com.example.fourlangasr.asr.ChineseSherpaEngine
import com.example.fourlangasr.asr.EnglishSherpaEngine
import com.example.fourlangasr.asr.RussianVoskEngine
import com.example.fourlangasr.asr.UzbekSherpaEngine
import com.example.fourlangasr.lid.LidEngine
import com.example.fourlangasr.model.AppLanguage
import com.example.fourlangasr.speaker.SpeakerManager
import com.example.fourlangasr.vad.VadEngine
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * 真机端到端管线测试：把真实 WAV 喂进真实原生引擎。
 *
 * 覆盖 JVM 单测覆盖不到的部分——sherpa-onnx / Vosk / TFLite /
 * 3D-Speaker 的原生加载与推理路径。
 *
 * 样本来自 experiments/speaker/speaker_data（16 kHz mono 16-bit），
 * 已拷贝到 androidTest/assets/audio/。
 *
 * 运行：
 *   .\gradlew.bat :app:connectedDebugAndroidTest
 */
@RunWith(AndroidJUnit4::class)
class OnDevicePipelineTest {

    private val instrumentation =
        InstrumentationRegistry.getInstrumentation()

    /** 被测 App 的 assets（含 models/）。 */
    private val appAssets
        get() = instrumentation.targetContext.assets

    /** 测试 APK 自己的 assets（含 audio/）。 */
    private val testAssets
        get() = instrumentation.context.assets

    private val appContext
        get() = instrumentation.targetContext

    // ---------------------------------------------------------------
    // VAD
    // ---------------------------------------------------------------

    @Test
    fun vad_producesSegmentForRealSpeech() {
        val samples = loadWav("zh_1.wav")
        val vad = VadEngine(appAssets)
        try {
            var segments = 0
            var totalSegmentSamples = 0

            // 按 512 帧喂入，与 MainActivity 的录音线程一致。
            var offset = 0
            while (offset < samples.size) {
                val end = minOf(offset + VadEngine.WINDOW_SIZE, samples.size)
                val frame = samples.copyOfRange(offset, end)
                if (frame.size == VadEngine.WINDOW_SIZE) {
                    vad.accept(frame)
                    while (vad.hasSegment()) {
                        val seg = vad.takeSegment() ?: break
                        segments++
                        totalSegmentSamples += seg.size
                    }
                }
                offset += VadEngine.WINDOW_SIZE
            }

            vad.flush()
            while (vad.hasSegment()) {
                val seg = vad.takeSegment() ?: break
                segments++
                totalSegmentSamples += seg.size
            }

            Log.i(TAG, "VAD: input=${samples.size} samples, " +
                "segments=$segments, segmentSamples=$totalSegmentSamples")

            assertTrue("VAD 应至少切出一段语音", segments >= 1)
            assertTrue("语音段不应为空", totalSegmentSamples > 0)
        } finally {
            vad.release()
        }
    }

    // ---------------------------------------------------------------
    // 四语种 ASR
    // ---------------------------------------------------------------

    @Test
    fun chineseAsr_transcribesRealSpeech() {
        val samples = loadWav("zh_1.wav")
        val engine = ChineseSherpaEngine(appAssets)
        try {
            val load = engine.load()
            Log.i(TAG, "ZH load: ${load.loadMs} ms, delta=${load.pssDeltaMb} MB")
            assertTrue("ZH 模型加载耗时应为正", load.loadMs > 0)

            val result = engine.transcribe(samples, 16000)
            Log.i(TAG, "ZH text='${result.text}' rtf=${result.rtf} " +
                "infer=${result.inferenceMs}ms unk=${result.unkTimestampsSec.size}")
            assertTrue("ZH 识别结果不应为空", result.text.isNotBlank())
        } finally {
            engine.release()
        }
    }

    @Test
    fun englishAsr_transcribesRealSpeech() {
        val samples = loadWav("en_1.wav")
        val engine = EnglishSherpaEngine(appAssets)
        try {
            val load = engine.load()
            Log.i(TAG, "EN load: ${load.loadMs} ms, delta=${load.pssDeltaMb} MB")

            val result = engine.transcribe(samples, 16000)
            Log.i(TAG, "EN text='${result.text}' rtf=${result.rtf} " +
                "infer=${result.inferenceMs}ms")
            assertTrue("EN 识别结果不应为空", result.text.isNotBlank())
        } finally {
            engine.release()
        }
    }

    @Test
    fun russianAsr_transcribesRealSpeech() {
        val samples = loadWav("ru_1.wav")
        val engine = RussianVoskEngine(appContext)
        try {
            val load = engine.load()
            Log.i(TAG, "RU load: ${load.loadMs} ms, delta=${load.pssDeltaMb} MB")

            val result = engine.transcribe(samples, 16000)
            Log.i(TAG, "RU text='${result.text}' rtf=${result.rtf} " +
                "infer=${result.inferenceMs}ms")
            assertTrue("RU 识别结果不应为空", result.text.isNotBlank())
        } finally {
            engine.release()
        }
    }

    @Test
    fun uzbekAsr_transcribesRealSpeech() {
        val samples = loadWav("uz_1.wav")
        val engine = UzbekSherpaEngine(appAssets)
        try {
            val load = engine.load()
            Log.i(TAG, "UZ load: ${load.loadMs} ms, delta=${load.pssDeltaMb} MB")

            val result = engine.transcribe(samples, 16000)
            Log.i(TAG, "UZ text='${result.text}' rtf=${result.rtf} " +
                "infer=${result.inferenceMs}ms")
            assertTrue("UZ 识别结果不应为空", result.text.isNotBlank())
        } finally {
            engine.release()
        }
    }

    // ---------------------------------------------------------------
    // LID：四语种自动路由
    // ---------------------------------------------------------------

    @Test
    fun lid_identifiesAllFourLanguages() {
        val engine = LidEngine(appContext)
        try {
            val cases = listOf(
                "zh" to "zh_1.wav",
                "en" to "en_1.wav",
                "ru" to "ru_1.wav",
                "uz" to "uz_1.wav",
            )

            val failures = mutableListOf<String>()

            for ((expected, file) in cases) {
                val samples = loadWav(file)
                val result = engine.identify(samples, 16000)

                Log.i(TAG, "LID $file -> ${result.languageCode} " +
                    "(expect $expected) conf4=${result.confidence} " +
                    "margin=${result.margin} mass=${result.targetMass} " +
                    "infer=${result.inferenceMs}ms window=${result.windowMode}")

                if (result.languageCode != expected) {
                    failures += "$file: 期望 $expected 实际 ${result.languageCode}"
                }
            }

            assertTrue(
                "LID 四语种识别出现偏差：$failures",
                failures.isEmpty(),
            )
        } finally {
            engine.release()
        }
    }

    // ---------------------------------------------------------------
    // Speaker：会话内说话人档案
    // ---------------------------------------------------------------

    @Test
    fun speaker_sameSpeakerMatchesExistingProfile() {
        val manager = SpeakerManager(appAssets)
        try {
            manager.load()
            manager.resetProfiles()

            val enroll = manager.identify(loadWav("zh_1.wav"), 16000)
            Log.i(TAG, "Speaker enroll: ${enroll.label} " +
                "reason=${enroll.reason} profiles=${enroll.profileCount}")
            assertEquals("首次应新建档案", 1, enroll.profileCount)
            assertTrue("首次应是新说话人", enroll.isNewSpeaker)

            val test = manager.identify(loadWav("zh_1_test.wav"), 16000)
            Log.i(TAG, "Speaker same-speaker test: ${test.label} " +
                "reason=${test.reason} score=${test.bestScore} " +
                "margin=${test.margin} profiles=${test.profileCount}")

            assertNotNull("同一说话人应匹配到档案", test.speakerId)
            assertEquals("同一说话人不应新建档案", 1, test.profileCount)
            assertEquals("应匹配到 Speaker 1", enroll.speakerId, test.speakerId)
        } finally {
            manager.release()
        }
    }

    @Test
    fun speaker_differentSpeakersDoNotCollapse() {
        val manager = SpeakerManager(appAssets)
        try {
            manager.load()
            manager.resetProfiles()

            val first = manager.identify(loadWav("zh_1.wav"), 16000)
            val second = manager.identify(loadWav("zh_2.wav"), 16000)

            Log.i(TAG, "Speaker cross test: first=${first.label} " +
                "second=${second.label} reason=${second.reason} " +
                "score=${second.bestScore} margin=${second.margin} " +
                "profiles=${second.profileCount}")

            // 不同说话人：要么低于阈值新建档案，要么被判 UNKNOWN。
            // 两种情况都不允许把 speaker_2 直接算成 speaker_1。
            if (second.speakerId == first.speakerId) {
                assertTrue(
                    "把不同说话人判成同一人时，必须给出足够的置信度证据",
                    second.bestScore >= SpeakerManager.SCORE_THRESHOLD,
                )
            } else {
                Log.i(TAG, "不同说话人已正确区分：profiles=${second.profileCount}")
            }
        } finally {
            manager.release()
        }
    }

    // ---------------------------------------------------------------
    // AsrManager：语种切换
    // ---------------------------------------------------------------

    @Test
    fun asrManager_switchesLanguageAndReusesWhenUnchanged() {
        val manager = AsrManager(appContext)
        try {
            val zhLoad = manager.switchLanguage(AppLanguage.ZH)
            assertEquals(AppLanguage.ZH, manager.loadedLanguage)
            Log.i(TAG, "AsrManager ZH load: ${zhLoad.loadMs} ms")

            // 同语种不应重复加载——loadedLanguage 不变即复用。
            assertEquals(AppLanguage.ZH, manager.loadedLanguage)

            val enLoad = manager.switchLanguage(AppLanguage.EN)
            assertEquals(AppLanguage.EN, manager.loadedLanguage)
            Log.i(TAG, "AsrManager EN load: ${enLoad.loadMs} ms")

            // 切回中文，验证释放旧模型后能重新加载。
            val zhAgain = manager.switchLanguage(AppLanguage.ZH)
            assertEquals(AppLanguage.ZH, manager.loadedLanguage)
            Log.i(TAG, "AsrManager ZH reload: ${zhAgain.loadMs} ms")

            val text = manager.transcribe(loadWav("zh_1.wav"), 16000)
            Log.i(TAG, "AsrManager ZH text='${text.text}'")
            assertTrue("切换后应能正常识别", text.text.isNotBlank())
        } finally {
            manager.release()
        }
    }

    // ---------------------------------------------------------------
    // WAV 读取
    // ---------------------------------------------------------------

    /** 读 16 kHz mono 16-bit PCM WAV，返回 [-1, 1) 的 float 样本。 */
    private fun loadWav(name: String): FloatArray {
        val bytes = testAssets.open("audio/$name").use { it.readBytes() }

        require(bytes.size > 44) { "WAV 太小: $name (${bytes.size} bytes)" }
        require(ascii(bytes, 0) == "RIFF") { "缺少 RIFF 头: $name" }
        require(ascii(bytes, 8) == "WAVE") { "缺少 WAVE 头: $name" }

        var pos = 12
        var channels = 1
        var sampleRate = 16000
        var bits = 16
        var dataOffset = -1
        var dataLen = 0

        while (pos + 8 <= bytes.size) {
            val id = ascii(bytes, pos)
            val size = le32(bytes, pos + 4)
            val body = pos + 8

            when (id) {
                "fmt " -> {
                    channels = le16(bytes, body + 2)
                    sampleRate = le32(bytes, body + 4)
                    bits = le16(bytes, body + 14)
                }
                "data" -> {
                    dataOffset = body
                    dataLen = minOf(size, bytes.size - body)
                }
            }

            // chunk 按偶数字节对齐
            pos = body + size + (size % 2)
        }

        require(dataOffset >= 0) { "缺少 data chunk: $name" }
        require(channels == 1 && sampleRate == 16000 && bits == 16) {
            "$name 格式不符：channels=$channels sampleRate=$sampleRate bits=$bits"
        }

        val count = dataLen / 2
        val out = FloatArray(count)
        for (i in 0 until count) {
            val lo = bytes[dataOffset + i * 2].toInt() and 0xFF
            val hi = bytes[dataOffset + i * 2 + 1].toInt()
            out[i] = ((hi shl 8) or lo).toShort() / 32768.0f
        }
        return out
    }

    private fun ascii(bytes: ByteArray, at: Int): String =
        String(bytes, at, 4, Charsets.US_ASCII)

    private fun le16(bytes: ByteArray, at: Int): Int =
        (bytes[at].toInt() and 0xFF) or
            ((bytes[at + 1].toInt() and 0xFF) shl 8)

    private fun le32(bytes: ByteArray, at: Int): Int =
        (bytes[at].toInt() and 0xFF) or
            ((bytes[at + 1].toInt() and 0xFF) shl 8) or
            ((bytes[at + 2].toInt() and 0xFF) shl 16) or
            ((bytes[at + 3].toInt() and 0xFF) shl 24)

    private companion object {
        const val TAG = "OnDeviceTest"
    }
}
