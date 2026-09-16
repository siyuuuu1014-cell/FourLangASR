package com.example.fourlangasr.translation

import android.content.Context

/**
 * 服务器地址配置（全 App 唯一配置点）。
 *
 * 优先级：用户在专项模型页设置的地址（SharedPreferences 持久化）
 * > [DEFAULT_BASE_URL]。所有下游（manifest 客户端、模型包下载、服务器翻译）
 * 通过 provider 动态读取，修改后无需重建任何模块。
 *
 * 当前默认值 = 开发机 SSH 隧道 + adb reverse 路径：
 *   PC 上保持 ssh -p 28893 -L 8000:localhost:8000 root@region-46.seetacloud.com
 *   USB 连接后执行 adb reverse tcp:8000 tcp:8000
 *   AutoDL 自定义服务映射恢复后改回 http://region-46.seetacloud.com:<远程端口>
 */
object ServerConfig {

    const val DEFAULT_BASE_URL = "http://127.0.0.1:8000"

    private const val PREFS_NAME = "server_config"
    private const val KEY_BASE_URL = "base_url"

    @Volatile
    private var cached: String? = null

    fun baseUrl(context: Context): String {
        cached?.let { return it }
        val stored = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getString(KEY_BASE_URL, null)
        cached = stored
        return (stored ?: DEFAULT_BASE_URL).trimEnd('/')
    }

    fun setBaseUrl(context: Context, url: String) {
        val normalized = url.trim().trimEnd('/')
        require(normalized.startsWith("http://") || normalized.startsWith("https://")) {
            "地址必须以 http:// 或 https:// 开头"
        }
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_BASE_URL, normalized)
            .apply()
        cached = normalized
    }

    fun resetToDefault(context: Context) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .remove(KEY_BASE_URL)
            .apply()
        cached = null
    }
}
