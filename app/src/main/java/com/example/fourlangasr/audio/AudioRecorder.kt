package com.example.fourlangasr.audio

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Process
import androidx.core.content.ContextCompat
import java.util.concurrent.atomic.AtomicBoolean

class AudioRecorder(
    private val context: Context,
    val sampleRate: Int = 16000,
) {
    private val recording = AtomicBoolean(false)

    private var audioRecord: AudioRecord? = null
    private var recordThread: Thread? = null

    /*
     * 原来的整段录音模式仍然保留。
     */
    private val chunks = mutableListOf<ShortArray>()
    private var totalSamples = 0

    fun isRecording(): Boolean = recording.get()

    /**
     * ============================================================
     * 传统模式：
     *
     * start()
     * ↓
     * 整段录音
     * ↓
     * stopAndGetFloatSamples()
     * ============================================================
     */
    fun start() {

        check(!recording.get()) {
            "Already recording"
        }

        checkPermission()

        val recorder = createAudioRecord()

        synchronized(chunks) {
            chunks.clear()
            totalSamples = 0
        }

        audioRecord = recorder
        recording.set(true)

        recorder.startRecording()

        recordThread = Thread {

            Process.setThreadPriority(
                Process.THREAD_PRIORITY_AUDIO
            )

            val bufferSizeSamples =
                maxOf(
                    recorder.bufferSizeInFrames,
                    2048
                )

            val shortBuffer =
                ShortArray(bufferSizeSamples)

            while (recording.get()) {

                val count =
                    recorder.read(
                        shortBuffer,
                        0,
                        shortBuffer.size,
                        AudioRecord.READ_BLOCKING,
                    )

                if (count > 0) {

                    val copy =
                        shortBuffer.copyOf(count)

                    synchronized(chunks) {
                        chunks.add(copy)
                        totalSamples += count
                    }
                }
            }

        }.apply {

            name = "FourLang-AudioRecord"
            start()
        }
    }

    /**
     * ============================================================
     * VAD 连续模式
     *
     * AudioRecord 返回的数据长度不一定正好是 512，
     * 因此这里自动重新整理成固定 frameSize。
     *
     * 默认：
     * 16000 Hz
     * 512 samples
     * = 32 ms
     * ============================================================
     */
    fun startStreaming(
        frameSize: Int = 512,
        onFrame: (FloatArray) -> Unit,
    ) {

        check(!recording.get()) {
            "Already recording"
        }

        check(frameSize > 0) {
            "frameSize must be > 0"
        }

        checkPermission()

        val recorder = createAudioRecord()

        audioRecord = recorder
        recording.set(true)

        recorder.startRecording()

        recordThread = Thread {

            Process.setThreadPriority(
                Process.THREAD_PRIORITY_AUDIO
            )

            val readBuffer =
                ShortArray(2048)

            val frameBuffer =
                ShortArray(frameSize)

            var frameOffset = 0

            while (recording.get()) {

                val count =
                    recorder.read(
                        readBuffer,
                        0,
                        readBuffer.size,
                        AudioRecord.READ_BLOCKING,
                    )

                if (count <= 0) {
                    continue
                }

                var readOffset = 0

                while (
                    readOffset < count &&
                    recording.get()
                ) {

                    val remainingInput =
                        count - readOffset

                    val remainingFrame =
                        frameSize - frameOffset

                    val copyCount =
                        minOf(
                            remainingInput,
                            remainingFrame
                        )

                    System.arraycopy(
                        readBuffer,
                        readOffset,
                        frameBuffer,
                        frameOffset,
                        copyCount,
                    )

                    readOffset += copyCount
                    frameOffset += copyCount

                    /*
                     * 凑够一个完整 VAD window。
                     */
                    if (frameOffset == frameSize) {

                        val floatFrame =
                            FloatArray(frameSize)

                        for (i in 0 until frameSize) {

                            floatFrame[i] =
                                frameBuffer[i] /
                                        32768.0f
                        }

                        /*
                         * 交给 VAD。
                         */
                        onFrame(floatFrame)

                        frameOffset = 0
                    }
                }
            }

        }.apply {

            name = "FourLang-AudioStream"
            start()
        }
    }

    /**
     * 停止连续 VAD 录音。
     */
    fun stopStreaming() {

        if (!recording.getAndSet(false)) {
            return
        }

        stopAudioRecord()
    }

    /**
     * 停止传统整段录音。
     */
    fun stopAndGetFloatSamples(): FloatArray {

        if (!recording.getAndSet(false)) {
            return FloatArray(0)
        }

        stopAudioRecord()

        val result: FloatArray

        synchronized(chunks) {

            result =
                FloatArray(totalSamples)

            var offset = 0

            for (chunk in chunks) {

                for (sample in chunk) {

                    result[offset++] =
                        sample / 32768.0f
                }
            }

            chunks.clear()
            totalSamples = 0
        }

        return result
    }

    /**
     * Activity / App 销毁时调用。
     */
    fun cancel() {

        if (recording.getAndSet(false)) {

            stopAudioRecord()

        } else {

            try {
                audioRecord?.release()
            } catch (_: Throwable) {
            }

            audioRecord = null
        }

        synchronized(chunks) {
            chunks.clear()
            totalSamples = 0
        }
    }

    private fun checkPermission() {

        check(
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.RECORD_AUDIO,
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            "RECORD_AUDIO permission not granted"
        }
    }

    private fun createAudioRecord(): AudioRecord {

        val minBufferBytes =
            AudioRecord.getMinBufferSize(
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
            )

        check(minBufferBytes > 0) {
            "AudioRecord.getMinBufferSize failed: $minBufferBytes"
        }

        val bufferBytes =
            maxOf(
                minBufferBytes * 2,
                4096
            )

        val recorder =
            AudioRecord(
                MediaRecorder.AudioSource.VOICE_RECOGNITION,
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufferBytes,
            )

        if (
            recorder.state !=
            AudioRecord.STATE_INITIALIZED
        ) {

            recorder.release()

            error(
                "AudioRecord initialization failed"
            )
        }

        return recorder
    }

    private fun stopAudioRecord() {

        try {
            audioRecord?.stop()
        } catch (_: Throwable) {
        }

        /*
         * stop() 后，阻塞中的 read() 会返回。
         */
        try {

            recordThread?.join(1500)

        } catch (_: InterruptedException) {

            Thread.currentThread().interrupt()
        }

        recordThread = null

        try {
            audioRecord?.release()
        } catch (_: Throwable) {
        }

        audioRecord = null
    }
}