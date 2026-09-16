package com.example.fourlangasr.translation.model

import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * 本地已安装模型的状态登记（版本指纹 + 启用开关），JSON 持久化。
 *
 * 语义：
 * - installed = 包已下载并解包、SHA 与下载时的远端 [sourceSha256] 一致；
 * - [enabled] = 用户在"专项模型"页开启的模型（某方向已启用专项模型时，
 *   翻译优先走专项，否则走总模型）；
 * - 远端 manifest 的 sha 与本地记录不一致 → 视为过期，需要重新下载。
 *
 * 线程安全：synchronized + 原子写（临时文件 + rename）。
 */
class ModelRegistry(
    private val registryFile: File,
) {

    data class InstalledModel(
        val modelId: String,
        val version: String,
        val sourceSha256: String,
        val enabled: Boolean,
    )

    @Volatile
    private var cache: Map<String, InstalledModel> = readFromDisk()

    private fun readFromDisk(): Map<String, InstalledModel> {
        if (!registryFile.exists()) return emptyMap()
        return try {
            val arr = JSONArray(registryFile.readText(Charsets.UTF_8))
            val map = HashMap<String, InstalledModel>()
            for (i in 0 until arr.length()) {
                val o = arr.getJSONObject(i)
                val id = o.getString("model_id")
                map[id] = InstalledModel(
                    modelId = id,
                    version = o.optString("version", ""),
                    sourceSha256 = o.optString("source_sha256", ""),
                    enabled = o.optBoolean("enabled", false),
                )
            }
            map
        } catch (e: Exception) {
            // 登记文件损坏时从空状态重建，不阻断 App
            emptyMap()
        }
    }

    private fun writeToDisk(map: Map<String, InstalledModel>) {
        val arr = JSONArray()
        for (m in map.values.sortedBy { it.modelId }) {
            arr.put(
                JSONObject()
                    .put("model_id", m.modelId)
                    .put("version", m.version)
                    .put("source_sha256", m.sourceSha256)
                    .put("enabled", m.enabled),
            )
        }
        val tmp = File(registryFile.parentFile, registryFile.name + ".tmp")
        registryFile.parentFile?.mkdirs()
        tmp.writeText(arr.toString(), Charsets.UTF_8)
        if (!tmp.renameTo(registryFile)) {
            tmp.copyTo(registryFile, overwrite = true)
            tmp.delete()
        }
    }

    fun installed(): List<InstalledModel> = synchronized(this) { cache.values.toList() }

    fun get(modelId: String): InstalledModel? = synchronized(this) { cache[modelId] }

    fun markInstalled(modelId: String, version: String, sourceSha256: String) =
        synchronized(this) {
            val prev = cache[modelId]
            cache = cache + (
                modelId to InstalledModel(
                    modelId = modelId,
                    version = version,
                    sourceSha256 = sourceSha256,
                    enabled = prev?.enabled ?: false,
                )
                )
            writeToDisk(cache)
        }

    fun setEnabled(modelId: String, enabled: Boolean) = synchronized(this) {
        val prev = cache[modelId] ?: return@synchronized
        cache = cache + (modelId to prev.copy(enabled = enabled))
        writeToDisk(cache)
    }

    fun remove(modelId: String) = synchronized(this) {
        cache = cache - modelId
        writeToDisk(cache)
    }
}
