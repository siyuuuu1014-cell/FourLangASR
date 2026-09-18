package com.example.fourlangasr.translation.model

import org.json.JSONObject

/**
 * 方向机制（编码/解码契约，由家族决定）：
 * - [M2M100_STYLE]：encoder = [src_lang] + X + [eos]；decoder 首步 = [eos, tgt_lang]（forced bos）；
 * - [SMALL100_STYLE]：encoder = [tgt_lang] + X + [eos]（目标语言前缀，训练项目
 *   set_lang_special_tokens 实测契约）；decoder 首步 = [eos]（no forced_bos）。
 */
enum class DirectionMechanism {
    M2M100_STYLE,
    SMALL100_STYLE,
}

/**
 * 模型运行时配置：从包内 MANIFEST.json（+ config.json / generation_config.json）解析。
 *
 * 专项包（SMaLL-100）与四语包（m2m100）的字段布局不同，本解析器兼容两者：
 * - languages 支持平铺 {"en": 128022} 与嵌套 {"en": {"token_id": 128022}}；
 * - 结构参数优先 MANIFEST，缺失时回退 config.json；
 * - 方向机制按 family / MANIFEST.direction_mechanism 判定。
 * 代码不写死任何模型参数。
 */
data class ModelRuntimeConfig(
    val modelId: String,
    val family: String,
    val vocabSize: Int,
    val encoderLayers: Int,
    val decoderLayers: Int,
    val attentionHeads: Int,
    val headDim: Int,
    /** 语言 code → 语言 token id（如 en → 128022）。 */
    val languages: Map<String, Long>,
    val bosId: Long,
    val eosId: Long,
    val padId: Long,
    val maxNewTokens: Int,
    val directionMechanism: DirectionMechanism,
)

object ModelRuntimeConfigParser {

    fun parse(
        packageManifestJson: String,
        configJson: String?,
        generationConfigJson: String?,
        modelId: String,
        family: String,
    ): ModelRuntimeConfig {
        val m = JSONObject(packageManifestJson)
        val cfg = configJson?.let { runCatching { JSONObject(it) }.getOrNull() }
        val gen = generationConfigJson?.let { runCatching { JSONObject(it) }.getOrNull() }

        // ---- 语言 token：兼容平铺与嵌套两种格式 ----
        val languages = LinkedHashMap<String, Long>()
        val langObj = m.optJSONObject("languages")
        if (langObj != null) {
            for (code in langObj.keys()) {
                val v = langObj.get(code)
                val id = when (v) {
                    is Number -> v.toLong()
                    is JSONObject -> v.getLong("token_id")
                    else -> null
                }
                if (id != null) languages[code] = id
            }
        }
        check(languages.isNotEmpty()) { "$modelId: MANIFEST 缺少 languages 映射" }

        // ---- 结构参数：MANIFEST 优先，config.json 回退 ----
        val vocabSize = m.optInt("vocab_size", -1)
            .let { if (it > 0) it else cfg?.optInt("vocab_size", -1) ?: -1 }
        check(vocabSize > 0) { "$modelId: 缺少 vocab_size（MANIFEST/config.json 均无）" }

        val encoderLayers = m.optInt("encoder_layers", -1)
            .let { if (it > 0) it else cfg?.optInt("encoder_layers", -1) ?: -1 }
            .let { if (it > 0) it else cfg?.optInt("num_hidden_layers", -1) ?: 12 }
        val decoderLayers = m.optInt("decoder_layers", -1)
            .let { if (it > 0) it else cfg?.optInt("decoder_layers", -1) ?: -1 }
            .let { if (it > 0) it else cfg?.optInt("num_hidden_layers", -1) ?: 12 }

        val attentionHeads = m.optInt("attention_heads", -1)
            .let { if (it > 0) it else cfg?.optInt("decoder_attention_heads", -1) ?: -1 }
            .let { if (it > 0) it else cfg?.optInt("encoder_attention_heads", -1) ?: -1 }
            .let { if (it > 0) it else cfg?.optInt("num_attention_heads", -1) ?: 16 }

        val dModel = m.optInt("d_model", -1)
            .let { if (it > 0) it else cfg?.optInt("d_model", -1) ?: -1 }
        val headDim = m.optInt("head_dim", -1)
            .let { if (it > 0) it else (if (dModel > 0 && attentionHeads > 0) dModel / attentionHeads else 64) }

        // ---- 特殊 token：MANIFEST.special_tokens 优先，config.json 回退 ----
        val specials = m.optJSONObject("special_tokens")
        val bos = specials?.optLong("bos_token_id", -1)?.takeIf { it >= 0 }
            ?: cfg?.optLong("bos_token_id", 0) ?: 0
        val eos = specials?.optLong("eos_token_id", -1)?.takeIf { it >= 0 }
            ?: cfg?.optLong("eos_token_id", 2) ?: 2
        val pad = specials?.optLong("pad_token_id", -1)?.takeIf { it >= 0 }
            ?: cfg?.optLong("pad_token_id", 1) ?: 1

        // ---- 生成长度：MANIFEST.generation 优先，generation_config.json 回退 ----
        val generation = m.optJSONObject("generation")
        val maxNew = generation?.optInt("max_new_tokens", -1)?.takeIf { it > 0 }
            ?: gen?.optInt("max_length", -1)?.takeIf { it > 0 }
            ?: 200

        // ---- 方向机制 ----
        val mechanismText = m.optString("direction_mechanism", "")
        val mechanism = if (
            family == "small100" ||
            mechanismText.contains("no forced_bos", ignoreCase = true) ||
            mechanismText.contains("tgt_lang prefix", ignoreCase = true)
        ) {
            DirectionMechanism.SMALL100_STYLE
        } else {
            DirectionMechanism.M2M100_STYLE
        }

        return ModelRuntimeConfig(
            modelId = modelId,
            family = family,
            vocabSize = vocabSize,
            encoderLayers = encoderLayers,
            decoderLayers = decoderLayers,
            attentionHeads = attentionHeads,
            headDim = headDim,
            languages = languages,
            bosId = bos,
            eosId = eos,
            padId = pad,
            maxNewTokens = maxNew,
            directionMechanism = mechanism,
        )
    }
}
