package com.example.fourlangasr.asr

import android.content.Context
import com.example.fourlangasr.model.AppLanguage
import com.example.fourlangasr.model.AsrResult
import com.example.fourlangasr.model.LoadResult

class AsrManager(
    private val context: Context,
) {

    private var currentEngine: AsrEngine? = null

    val loadedLanguage: AppLanguage?
        get() =
            synchronized(this) {
                currentEngine?.language
            }

    /**
     * 切换语言。
     *
     * 自动：
     * 释放旧模型
     * ↓
     * 创建新模型
     * ↓
     * 加载新模型
     */
    @Synchronized
    fun switchLanguage(
        language: AppLanguage
    ): LoadResult {

        release()

        val engine: AsrEngine =
            when (language) {

                AppLanguage.ZH ->
                    ChineseSherpaEngine(
                        context.assets
                    )

                AppLanguage.EN ->
                    EnglishSherpaEngine(
                        context.assets
                    )

                AppLanguage.RU ->
                    RussianVoskEngine(
                        context
                    )

                AppLanguage.UZ ->
                    UzbekSherpaEngine(
                        context.assets
                    )
            }

        return try {

            val result =
                engine.load()

            currentEngine =
                engine

            result

        } catch (e: Throwable) {

            try {
                engine.release()
            } catch (_: Throwable) {
            }

            throw e
        }
    }

    /**
     * 统一识别入口。
     *
     * ZH / EN / RU / UZ
     * 都通过这个方法调用。
     */
    @Synchronized
    fun transcribe(
        samples: FloatArray,
        sampleRate: Int = 16000,
    ): AsrResult {

        val engine =
            checkNotNull(
                currentEngine
            ) {
                "No ASR model is loaded"
            }

        return engine.transcribe(
            samples,
            sampleRate,
        )
    }

    /**
     * 释放当前语言模型。
     */
    @Synchronized
    fun release() {

        try {

            currentEngine?.release()

        } finally {

            currentEngine = null
        }

        /*
         * 只是 GC hint。
         */
        System.gc()
    }
}