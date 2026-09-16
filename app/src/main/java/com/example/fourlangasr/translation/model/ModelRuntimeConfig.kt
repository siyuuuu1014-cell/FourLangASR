package com.example.fourlangasr.translation.model

import org.json.JSONObject

/**
 * 模型运行时配置：从包内 MANIFEST.json 解析（模型结构参数 + 语言 token + 特殊 token）。
 *
 * 引擎解码循环所需的全部模型相关常量都来自这里 —— 代码不写死任何
 * 词表大小/层数/语言 token（四语包与六组专项包各自携带自己的值）。
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
)

object ModelRuntimeConfigParser {

    fun parse(jsonText: String, modelId: String, family: String): ModelRuntimeConfig {
        val m = JSONObject(jsonText)

        val vocabSize = m.optInt("vocab_size", -1)
        check(vocabSize > 0) { "$modelId: MANIFEST 缺少 vocab_size" }

        val languages = LinkedHashMap<String, Long>()
        val langObj = m.optJSONObject("languages")
        if (langObj != null) {
            for (code in langObj.keys()) {
                languages[code] = langObj.getLong(code)
            }
        }
        check(languages.isNotEmpty()) { "$modelId: MANIFEST 缺少 languages 映射" }

        val specials = m.optJSONObject("special_tokens")
        val bos = specials?.optLong("bos_token_id", 0L) ?: 0L
        val eos = specials?.optLong("eos_token_id", 2L) ?: 2L
        val pad = specials?.optLong("pad_token_id", 1L) ?: 1L

        val generation = m.optJSONObject("generation")
        val maxNew = generation?.optInt("max_new_tokens", 200) ?: 200

        return ModelRuntimeConfig(
            modelId = modelId,
            family = family,
            vocabSize = vocabSize,
            encoderLayers = m.optInt("encoder_layers", 12),
            decoderLayers = m.optInt("decoder_layers", 12),
            attentionHeads = m.optInt("attention_heads", 16),
            headDim = m.optInt("head_dim", 64),
            languages = languages,
            bosId = bos,
            eosId = eos,
            padId = pad,
            maxNewTokens = maxNew,
        )
    }
}
