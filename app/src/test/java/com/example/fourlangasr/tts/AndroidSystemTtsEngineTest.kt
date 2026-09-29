package com.example.fourlangasr.tts

import android.content.Context
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import com.example.fourlangasr.model.AppLanguage
import org.junit.Assert.*
import org.junit.Test
import org.mockito.ArgumentMatchers.*
import org.mockito.Mockito.*
import java.util.Locale

class AndroidSystemTtsEngineTest {
    @Test fun stopAlsoStopsTheFallbackThatIsActuallySpeaking() {
        val fixture = Fixture(useFallback = true)
        fixture.initialize()
        assertTrue(fixture.subject.speak("hello", AppLanguage.EN))
        fixture.subject.stop()
        verify(fixture.primary).stop()
        verify(fixture.fallback).stop()
        assertFalse(fixture.subject.isSpeaking)
        // A delayed start from the stopped fallback must not close the microphone again.
        fixture.progress.getValue(fixture.fallback).onStart(fixture.utteranceIds.last())
        assertFalse(fixture.subject.isSpeaking)
    }

    @Test fun oldUtteranceCallbacksDoNotClearNewPlayback() {
        val fixture = Fixture()
        fixture.initialize()
        fixture.subject.speak("first", AppLanguage.EN)
        val oldId = fixture.utteranceIds.last()
        fixture.subject.speak("second", AppLanguage.EN)
        val newId = fixture.utteranceIds.last()
        val progress = fixture.progress.getValue(fixture.primary)
        progress.onStart(newId)
        progress.onStop(oldId, true)
        assertTrue(fixture.subject.isSpeaking)
        progress.onDone(oldId)
        progress.onError(oldId, 1)
        assertTrue(fixture.subject.isSpeaking)
        assertEquals(0, fixture.listener.done)
        assertEquals(0, fixture.listener.errors)
        progress.onDone(newId)
        assertFalse(fixture.subject.isSpeaking)
        assertEquals(1, fixture.listener.done)
    }

    @Test fun completionInsideSpeakSubmissionDoesNotLeavePlaybackStuck() {
        val fixture = Fixture()
        fixture.initialize()
        doAnswer { invocation ->
            fixture.progress.getValue(fixture.primary).onDone(invocation.getArgument(3))
            TextToSpeech.SUCCESS
        }.`when`(fixture.primary).speak(anyString(), anyInt(), isNull<Bundle>(), anyString())
        assertTrue(fixture.subject.speak("hello", AppLanguage.EN))
        assertFalse(fixture.subject.isSpeaking)
        assertEquals(1, fixture.listener.done)
    }

    @Test fun failedSubmissionReopensTheMicrophone() {
        val fixture = Fixture()
        fixture.initialize()
        doReturn(TextToSpeech.ERROR).`when`(fixture.primary)
            .speak(anyString(), anyInt(), isNull<Bundle>(), anyString())
        assertFalse(fixture.subject.speak("hello", AppLanguage.EN))
        assertFalse(fixture.subject.isSpeaking)
        assertEquals(1, fixture.listener.errors)
    }

    @Test fun lateInitializationCannotReviveReleasedEngine() {
        val fixture = Fixture()
        fixture.subject.release()
        fixture.initialize()
        assertFalse(fixture.subject.isReady)
        assertEquals(0, fixture.listener.ready)
        assertFalse(fixture.subject.speak("hello", AppLanguage.EN))
    }

    @Test fun submissionExceptionDoesNotLeaveTheMicrophoneClosed() {
        val fixture = Fixture()
        fixture.initialize()
        doThrow(IllegalStateException("engine disconnected")).`when`(fixture.primary)
            .speak(anyString(), anyInt(), isNull<Bundle>(), anyString())
        assertFalse(fixture.subject.speak("hello", AppLanguage.EN))
        assertFalse(fixture.subject.isSpeaking)
        assertEquals(1, fixture.listener.errors)
    }

    private class Fixture(useFallback: Boolean = false) {
        val primary = mock(TextToSpeech::class.java)
        val fallback = mock(TextToSpeech::class.java)
        val listener = RecordingListener()
        val progress = mutableMapOf<TextToSpeech, UtteranceProgressListener>()
        val utteranceIds = mutableListOf<String>()
        val initCallbacks = mutableMapOf<String, TextToSpeech.OnInitListener>()
        val subject: AndroidSystemTtsEngine

        init {
            val context = mock(Context::class.java)
            `when`(context.applicationContext).thenReturn(context)
            `when`(primary.defaultEngine).thenReturn("primary")
            `when`(primary.setLanguage(any(Locale::class.java))).thenReturn(
                if (useFallback) TextToSpeech.LANG_NOT_SUPPORTED else TextToSpeech.LANG_AVAILABLE,
            )
            `when`(fallback.setLanguage(any(Locale::class.java)))
                .thenReturn(TextToSpeech.LANG_AVAILABLE)
            `when`(primary.engines).thenReturn(
                if (useFallback) listOf(TextToSpeech.EngineInfo().apply { name = "fallback" })
                else emptyList(),
            )
            for (engine in listOf(primary, fallback)) {
                doAnswer { invocation ->
                    progress[engine] = invocation.getArgument(0)
                    TextToSpeech.SUCCESS
                }.`when`(engine).setOnUtteranceProgressListener(any(UtteranceProgressListener::class.java))
                doAnswer { invocation ->
                    utteranceIds += invocation.getArgument<String>(3)
                    TextToSpeech.SUCCESS
                }.`when`(engine).speak(anyString(), anyInt(), isNull<Bundle>(), anyString())
            }
            subject = AndroidSystemTtsEngine(context, listener) { _, callback, packageName ->
                initCallbacks[packageName ?: "primary"] = callback
                if (packageName == null) primary else fallback
            }
        }

        fun initialize() {
            initCallbacks.getValue("primary").onInit(TextToSpeech.SUCCESS)
            initCallbacks["fallback"]?.onInit(TextToSpeech.SUCCESS)
        }
    }

    private class RecordingListener : TtsEngine.Listener {
        var done = 0
        var errors = 0
        var ready = 0
        override fun onSpeakStart() = Unit
        override fun onSpeakDone() { done++ }
        override fun onSpeakError(reason: String) { errors++ }
        override fun onEngineReady(availability: Map<String, Boolean>) { ready++ }
    }
}
