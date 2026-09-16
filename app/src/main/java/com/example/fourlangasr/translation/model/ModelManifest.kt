package com.example.fourlangasr.translation.model

import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

/**
 * 服务器模型清单（GET /models/manifest）的数据模型与解析。
 *
 * 每个模型条目：
 * - [modelId]：唯一 id（如 "m2m100_fourlang"、"zh_uz"）
 * - [family]：模型家族（"m2m100" = 四语总模型；"small100" = 六组专项）
 * - [version] / [sourceSha256]：远端版本指纹（模型权重 SHA），
 *   与本地已装版本对比，不一致即需要重新下载
 * - [directions]：该模型覆盖的方向（如 ["zh-uz", "uz-zh"]）
 * - [downloadUrl]：相对下载路径（如 "/download/zh_uz"），
 *   必须与 [baseUrl] 拼接使用，代码不写死任何模型 id
 */
data class ManifestModel(
    val modelId: String,
    val family: String,
    val version: String,
    val sourceSha256: String,
    val sizeBytes: Long,
    val directions: List<String>,
    val downloadUrl: String,
) {
    fun covers(srcLang: String, tgtLang: String): Boolean =
        directions.contains("$srcLang-$tgtLang")
}

data class ModelsManifest(
    val models: List<ManifestModel>,
) {
    /** 总模型 = 覆盖方向最多的模型（四语 12 方向 vs 专项 2 方向，按清单语义区分）。 */
    fun generalist(): ManifestModel =
        models.maxByOrNull { it.directions.size }
            ?: error("manifest 为空")

    fun specialists(): List<ManifestModel> =
        models.filter { it.directions.size < generalist().directions.size }

    fun byId(modelId: String): ManifestModel? =
        models.firstOrNull { it.modelId == modelId }
}

object ModelManifestParser {

    fun parse(jsonText: String): ModelsManifest {
        val root = JSONObject(jsonText)
        val arr = root.getJSONArray("models")
        val models = ArrayList<ManifestModel>(arr.length())
        for (i in 0 until arr.length()) {
            val m = arr.getJSONObject(i)
            val dirs = ArrayList<String>()
            val dirArr = m.getJSONArray("directions")
            for (j in 0 until dirArr.length()) {
                dirs.add(dirArr.getString(j))
            }
            models.add(
                ManifestModel(
                    modelId = m.getString("model_id"),
                    family = m.optString("family", ""),
                    version = m.optString("version", ""),
                    sourceSha256 = m.optString("source_sha256", ""),
                    sizeBytes = m.optLong("size_bytes", 0L),
                    directions = dirs,
                    downloadUrl = m.getString("download_url"),
                ),
            )
        }
        return ModelsManifest(models)
    }
}

/**
 * manifest 客户端：GET {baseUrl}/models/manifest。
 * 必须在后台线程调用（同步阻塞网络请求）。
 */
class ModelManifestClient(
    private val baseUrl: String,
    private val connectTimeoutMs: Int = 15_000,
    private val readTimeoutMs: Int = 30_000,
) {

    fun fetchManifest(): ModelsManifest {
        val conn = URL("$baseUrl/models/manifest").openConnection() as HttpURLConnection
        try {
            conn.connectTimeout = connectTimeoutMs
            conn.readTimeout = readTimeoutMs
            val code = conn.responseCode
            val stream = if (code in 200..299) conn.inputStream else conn.errorStream
            val body = stream?.bufferedReader(Charsets.UTF_8)?.use { it.readText() } ?: ""
            if (code !in 200..299) {
                throw IOException("fetch manifest HTTP $code: ${body.take(200)}")
            }
            return ModelManifestParser.parse(body)
        } finally {
            conn.disconnect()
        }
    }
}
