package com.example.fourlangasr.translation

import android.content.Context

/**
 * 性能配置（低内存档位）：
 * [releaseAsrDuringTranslation] = true 时，翻译前先释放 ASR 模型、
 * 翻译完成后自动重载（以切换延迟换内存安全，面向低内存设备）。
 * 默认关闭 —— 16GB 级设备无需开启。
 */
object PerformanceConfig {

    const val DEFAULT_RELEASE_ASR_DURING_TRANSLATION = false

    private const val PREFS_NAME = "performance_config"
    private const val KEY_RELEASE_ASR = "release_asr_during_translation"

    fun releaseAsrDuringTranslation(context: Context): Boolean =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getBoolean(KEY_RELEASE_ASR, DEFAULT_RELEASE_ASR_DURING_TRANSLATION)

    fun setReleaseAsrDuringTranslation(context: Context, enabled: Boolean) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putBoolean(KEY_RELEASE_ASR, enabled)
            .apply()
    }
}
