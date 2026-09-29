package com.example.fourlangasr.pipeline

import com.example.fourlangasr.asr.AsrEngine
import com.example.fourlangasr.model.AppLanguage
import com.example.fourlangasr.model.AsrResult
import com.example.fourlangasr.model.LoadResult
import kotlinx.coroutines.CancellationException
import org.junit.Assert.*
import org.junit.Test
import java.io.IOException
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException

class CodeSwitchRecoveryTest {
    private val samples = FloatArray(48_000)

    @Test fun overlappingWindowsKeepBothOriginalWordGroups() {
        val engine = FakeEnglishEngine { call -> if (call <= 3) "review" else "plan" }
        val result = CodeSwitchRecovery(engine).recover(
            samples, 16000, "先<unk>一下今天的<unk>", floatArrayOf(1f, 1.5f),
        )
        assertEquals("先review一下今天的plan", result.recoveredText)
        assertEquals(2, result.regionCount)
        assertEquals(listOf("review", "plan"), result.enTexts)
    }

    @Test fun adjacentTokensStillFormOneWordGroup() {
        val result = CodeSwitchRecovery(FakeEnglishEngine { "smart phone" }).recover(
            samples, 16000, "我的<unk><unk>很好用", floatArrayOf(1f, 1.2f),
        )
        assertEquals("我的smart phone很好用", result.recoveredText)
        assertEquals(1, result.regionCount)
    }

    @Test fun groupsAtAudioEdgesKeepTheirOwnText() {
        val result = CodeSwitchRecovery(
            FakeEnglishEngine { call -> if (call <= 3) "review" else "plan" },
        ).recover(samples, 16000, "<unk>然后<unk>", floatArrayOf(0.1f, 2.9f))
        assertEquals("review然后plan", result.recoveredText)
        assertEquals(2, result.regionCount)
    }

    @Test fun plainChineseDoesNotLoadEnglishModel() {
        val engine = FakeEnglishEngine()
        val result = CodeSwitchRecovery(engine).recover(samples, 16000, "你好", floatArrayOf())
        assertEquals("你好", result.recoveredText)
        assertEquals(0, engine.loads)
        assertEquals(0, engine.decodes)
    }

    @Test fun emptyEnglishCandidateOnlyRemovesItsOwnMarker() {
        val result = CodeSwitchRecovery(
            FakeEnglishEngine { call -> if (call <= 3) "" else "plan" },
        ).recover(samples, 16000, "先<unk>然后<unk>", floatArrayOf(1f, 1.5f))
        assertEquals("先然后plan", result.recoveredText)
    }

    @Test fun loadFailureKeepsChineseAndAllowsRetry() {
        val engine = FakeEnglishEngine().apply { loadError = IOException("missing model") }
        val recovery = CodeSwitchRecovery(engine)
        val failed = recovery.recover(samples, 16000, "我的<unk>", floatArrayOf(1f))
        // 回填失败要保留中文，但不能把 <unk> 占位符留给用户（它会被 TTS 念出来）
        assertEquals("我的", failed.recoveredText)
        assertTrue(failed.error!!.contains("missing model"))
        assertEquals(1, engine.releases)
        engine.loadError = null
        val retried = recovery.recover(samples, 16000, "我的<unk>", floatArrayOf(1f))
        assertEquals("我的review", retried.recoveredText)
        assertEquals(2, engine.loads)
        assertNull(retried.error)
    }

    @Test fun decodeFailureKeepsOriginalChinese() {
        val engine = FakeEnglishEngine { throw IllegalStateException("decode failed") }
        val result = CodeSwitchRecovery(engine).recover(
            samples, 16000, "我的<unk>", floatArrayOf(1f),
        )
        // 兜底保留中文正文，但 <unk> 占位符必须剥掉：它会进转录并被 TTS 念出来
        assertEquals("我的", result.recoveredText)
        assertTrue(result.error!!.contains("decode failed"))
    }

    @Test(expected = CancellationException::class)
    fun cancellationIsNotConvertedIntoAResult() {
        CodeSwitchRecovery(FakeEnglishEngine { throw CancellationException("stopped") })
            .recover(samples, 16000, "我的<unk>", floatArrayOf(1f))
    }

    @Test fun releaseWaitsForAnInFlightRecovery() {
        val enteredDecode = CountDownLatch(1)
        val finishDecode = CountDownLatch(1)
        val releaseRequested = CountDownLatch(1)
        val engine = FakeEnglishEngine {
            enteredDecode.countDown()
            check(finishDecode.await(5, TimeUnit.SECONDS))
            "review"
        }
        val recovery = CodeSwitchRecovery(engine)
        val executor = Executors.newFixedThreadPool(2)
        try {
            val recognition = executor.submit<RecoveryResult> {
                recovery.recover(samples, 16000, "我的<unk>", floatArrayOf(1f))
            }
            assertTrue(enteredDecode.await(5, TimeUnit.SECONDS))
            val release = executor.submit {
                releaseRequested.countDown()
                recovery.release()
            }
            assertTrue(releaseRequested.await(5, TimeUnit.SECONDS))
            try {
                release.get(150, TimeUnit.MILLISECONDS)
                fail("Model was released while recovery was using it")
            } catch (_: TimeoutException) {
                // release must wait for the whole recovery, including all candidate windows.
            }
            assertEquals(0, engine.releases)
            finishDecode.countDown()
            assertEquals("我的review", recognition.get(5, TimeUnit.SECONDS).recoveredText)
            release.get(5, TimeUnit.SECONDS)
            assertEquals(1, engine.releases)
        } finally {
            finishDecode.countDown()
            executor.shutdownNow()
        }
    }

    @Test fun releaseStopsAutoReloadUntilNextSession() {
        val engine = FakeEnglishEngine { "review" }
        val recovery = CodeSwitchRecovery(engine)
        recovery.recover(samples, 16000, "我的<unk>", floatArrayOf(1f))
        assertEquals(1, engine.loads)

        recovery.release()

        // 排空窗口里剩下的句子不得把已释放的模型偷偷加载回来
        val afterRelease = recovery.recover(samples, 16000, "我的<unk>", floatArrayOf(1f))
        assertEquals(1, engine.loads)
        assertEquals("我的", afterRelease.recoveredText)

        recovery.beginSession()
        recovery.recover(samples, 16000, "我的<unk>", floatArrayOf(1f))
        assertEquals(2, engine.loads)
    }

    private class FakeEnglishEngine(
        val output: (Int) -> String = { "review" },
    ) : AsrEngine {
        override val language = AppLanguage.EN
        var loads = 0
        var decodes = 0
        @Volatile var releases = 0
        var loadError: Exception? = null

        override fun load(): LoadResult {
            loads++
            loadError?.let { throw it }
            return LoadResult(12, 0f, 0f)
        }

        override fun transcribe(samples: FloatArray, sampleRate: Int): AsrResult {
            decodes++
            return AsrResult(output(decodes), samples.size / sampleRate.toFloat(), 1, 0f, 0f, 0f)
        }

        override fun release() { releases++ }
    }
}
