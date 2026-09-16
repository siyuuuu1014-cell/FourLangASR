package com.example.fourlangasr.translation

import com.example.fourlangasr.model.AppLanguage
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
        /** 四语 API 服务器（用户配置，可用构造函数覆盖）。 */
        const val DEFAULT_BASE_URL = "http://region-46.seetacloud.com:8000"
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
            )
        } finally {
            conn.disconnect()
        }
    }
}
