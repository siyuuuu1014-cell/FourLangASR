package com.example.fourlangasr.translation.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayOutputStream
import java.io.File

class ModelModelTests {

    private val sampleManifest = """
        {"models": [
          {"model_id": "m2m100_fourlang", "family": "m2m100", "version": "1.0.0",
           "source_sha256": "abc123", "size_bytes": 769590133,
           "directions": ["en-ru","en-uz","en-zh","ru-en","ru-uz","ru-zh","uz-en","uz-ru","uz-zh","zh-en","zh-ru","zh-uz"],
           "download_url": "/download/m2m100_fourlang"},
          {"model_id": "zh_uz", "family": "small100", "version": "1.0.0",
           "source_sha256": "def456", "size_bytes": 615080387,
           "directions": ["uz-zh","zh-uz"],
           "download_url": "/download/zh_uz"}
        ]}
    """.trimIndent()

    @Test
    fun manifest_parsesAndRoutesCorrectly() {
        val manifest = ModelManifestParser.parse(sampleManifest)
        assertEquals(2, manifest.models.size)
        assertEquals("m2m100_fourlang", manifest.generalist().modelId)
        assertEquals(1, manifest.specialists().size)
        assertEquals("zh_uz", manifest.specialists()[0].modelId)
        assertTrue(manifest.byId("zh_uz")!!.covers("zh", "uz"))
        assertFalse(manifest.byId("zh_uz")!!.covers("en", "zh"))
    }

    @Test
    fun registry_persistsAndRoundsTrips() {
        val dir = File.createTempFile("reg", "").also { it.delete(); it.mkdirs() }
        try {
            val file = File(dir, "model_registry.json")
            val reg = ModelRegistry(file)
            assertNull(reg.get("zh_uz"))
            reg.markInstalled("zh_uz", "1.0.0", "def456")
            reg.setEnabled("zh_uz", true)
            // 重新加载（模拟重启）
            val reg2 = ModelRegistry(file)
            val m = reg2.get("zh_uz")!!
            assertEquals("1.0.0", m.version)
            assertEquals("def456", m.sourceSha256)
            assertTrue(m.enabled)
            reg2.remove("zh_uz")
            assertNull(ModelRegistry(file).get("zh_uz"))
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun runtimeConfig_parsesPackageManifest() {
        val json = """
            {"name":"x","vocab_size":128112,"d_model":1024,
             "encoder_layers":12,"decoder_layers":12,"attention_heads":16,"head_dim":64,
             "languages":{"en":128022,"zh":128102,"uz":128096,"ru":128077},
             "special_tokens":{"bos_token_id":0,"eos_token_id":2,"pad_token_id":1},
             "generation":{"max_new_tokens":256,"max_source_length":1024}}
        """.trimIndent()
        val cfg = ModelRuntimeConfigParser.parse(json, null, null, "m2m100_fourlang", "m2m100")
        assertEquals(128112, cfg.vocabSize)
        assertEquals(12, cfg.encoderLayers)
        assertEquals(16, cfg.attentionHeads)
        assertEquals(64, cfg.headDim)
        assertEquals(128102L, cfg.languages["zh"])
        assertEquals(2L, cfg.eosId)
        assertEquals(256, cfg.maxNewTokens)
        // 编码器截断上限必须被读出来，否则超长句整段喂入会直接推理失败
        assertEquals(1024, cfg.maxSourceTokens)
        assertEquals(DirectionMechanism.M2M100_STYLE, cfg.directionMechanism)
    }

    /** 真实 zh_uz 专项包结构（嵌套 languages + config.json 结构参数 + SMALL100 契约）。 */
    @Test
    fun runtimeConfig_parsesRealSmall100Package() {
        val manifestJson = """
            {"name":"shared","version":"1.0.0","family":"small100",
             "languages":{
               "en":{"token_id":128022},"zh":{"token_id":128102},
               "uz":{"token_id":128096},"ru":{"token_id":128077}},
             "special_tokens":{"bos_token_id":0,"eos_token_id":2,"pad_token_id":1},
             "direction_mechanism":"tgt_lang prefix (target), no forced_bos",
             "files":{}}
        """.trimIndent()
        val configJson = """
            {"architectures":["M2M100ForConditionalGeneration"],
             "vocab_size":128112,"d_model":1024,
             "encoder_layers":12,"decoder_layers":3,
             "encoder_attention_heads":16,"decoder_attention_heads":16,
             "eos_token_id":2,"pad_token_id":1,"bos_token_id":0}
        """.trimIndent()
        val generationJson = """
            {"decoder_start_token_id":2,"max_length":256,"eos_token_id":2}
        """.trimIndent()
        val cfg = ModelRuntimeConfigParser.parse(
            manifestJson, configJson, generationJson, "zh_uz", "small100",
        )
        assertEquals(128112, cfg.vocabSize)
        assertEquals(12, cfg.encoderLayers)
        assertEquals(3, cfg.decoderLayers)
        assertEquals(16, cfg.attentionHeads)
        assertEquals(64, cfg.headDim)
        assertEquals(128096L, cfg.languages["uz"])
        assertEquals(256, cfg.maxNewTokens)
        assertEquals(DirectionMechanism.SMALL100_STYLE, cfg.directionMechanism)
    }

    @Test
    fun tarExtractor_unpacksAndSkipsDirectories() {
        val dir = File.createTempFile("tar", "").also { it.delete(); it.mkdirs() }
        try {
            val enc = ByteArray(1000) { 7 }
            val dec = ByteArray(2000) { 9 }
            val tar = buildTar(
                mapOf(
                    "m2m100_fourlang/encoder_model.onnx" to enc,
                    "m2m100_fourlang/decoder_model_merged.onnx" to dec,
                ),
            )
            TarExtractor.extract(tar.inputStream(), dir) { _, _ -> }
            assertTrue(File(dir, "encoder_model.onnx").readBytes().contentEquals(enc))
            assertTrue(File(dir, "decoder_model_merged.onnx").readBytes().contentEquals(dec))
        } finally {
            dir.deleteRecursively()
        }
    }

    private fun buildTar(files: Map<String, ByteArray>): ByteArray {
        val out = ByteArrayOutputStream()
        for ((name, data) in files) {
            val header = ByteArray(512)
            name.toByteArray(Charsets.UTF_8).copyInto(header, 0)
            val oct = data.size.toString(8)
            oct.toByteArray().copyInto(header, 124 + 12 - oct.length)
            header[156] = '0'.code.toByte()
            out.write(header)
            out.write(data)
            val pad = (512 - data.size % 512) % 512
            out.write(ByteArray(pad))
        }
        out.write(ByteArray(1024))
        return out.toByteArray()
    }
}
