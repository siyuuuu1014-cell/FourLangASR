package com.example.fourlangasr.translation.engine

import com.example.fourlangasr.model.AppLanguage
import com.example.fourlangasr.translation.TranslationEngine
import com.example.fourlangasr.translation.TranslationResult
import org.json.JSONObject
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

/**
 * 服务器翻译引擎：六组 SMaLL-100 pair 模型经 FastAPI 提供
 * （POST {baseUrl}/translate，契约见训练项目 service/schemas.py）。
 *
 * 必须在后台线程调用（同步阻塞网络请求，与 TranslationEngine 契约一致）。
 */
class ServerTranslationEngine(
    private val baseUrl: String = DEFAULT_BASE_URL,
    private val connectTimeoutMs: Int = 15_000,
    private val readTimeoutMs: Int = 60_000,
) : TranslationEngine {

    companion object {
        /*
         * 四语 API 服务器基址（全 App 唯一配置点，TranslationModules 引用）。
         *
         * 当前 = 开发机 SSH 隧道 + adb reverse 路径：
         *   PC 上保持 ssh -p 28893 -L 8000:localhost:8000 root@region-46.seetacloud.com
         *   USB 连接后执行 adb reverse tcp:8000 tcp:8000
         *   手机访问自身 127.0.0.1:8000 即到服务器。
         * AutoDL 自定义服务映射恢复后改回：
         *   http://region-46.seetacloud.com:<远程端口>
         */
        const val DEFAULT_BASE_URL = "http://127.0.0.1:8000"

        /** TranslationResult.modelLabel 中服务器路径的标识。 */
        const val MODEL_LABEL_SERVER = "server"
    }

    override fun translate(
        text: String,
        source: AppLanguage,
        target: AppLanguage,
    ): TranslationResult {
        val body = JSONObject()
            .put("source_lang", source.code)
            .put("target_lang", target.code)
            .put("text", text)
            .toString()

        val conn = URL("$baseUrl/translate").openConnection() as HttpURLConnection
        try {
            conn.requestMethod = "POST"
            conn.connectTimeout = connectTimeoutMs
            conn.readTimeout = readTimeoutMs
            conn.doOutput = true
            conn.setRequestProperty("Content-Type", "application/json; charset=utf-8")
            conn.outputStream.use { it.write(body.toByteArray(Charsets.UTF_8)) }

            val code = conn.responseCode
            val stream = if (code in 200..299) conn.inputStream else conn.errorStream
            val resp = stream?.bufferedReader(Charsets.UTF_8)?.use { it.readText() } ?: ""
            if (code !in 200..299) {
                throw IOException("translate HTTP $code: ${resp.take(200)}")
            }
            val translation = JSONObject(resp).getString("translation")
            return TranslationResult(
                sourceText = text,
                outputText = translation,
                sourceLanguage = source,
                targetLanguage = target,
                speechLanguage = target,
                translated = true,
                modelLabel = MODEL_LABEL_SERVER,
            )
        } finally {
            conn.disconnect()
        }
    }
}
