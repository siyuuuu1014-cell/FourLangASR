package com.example.fourlangasr.translation.engine

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import android.os.SystemClock
import android.util.Log
import com.example.fourlangasr.BuildConfig
import com.example.fourlangasr.model.AppLanguage
import com.example.fourlangasr.translation.TranslationResult
import com.example.fourlangasr.translation.model.DirectionMechanism
import com.example.fourlangasr.translation.model.ModelRuntimeConfig
import com.example.fourlangasr.translation.tokenizer.M2m100Tokenizer
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * 本机离线翻译引擎：int8 ONNX encoder-decoder 模型（merged decoder + KV cache）。
 * 适用于四语 m2m100 与六组 SMaLL-100 包（同一 optimum seq2seq-lm-with-past 导出管线）。
 *
 * 解码循环移植自 experiments/translation/run_m2m100_reference.py
 * （PC 端 6 方向已验证）。全部模型相关常量（词表大小/层数/头数/语言 token/
 * 特殊 token）来自包内 MANIFEST.json 解析出的 [ModelRuntimeConfig]，
 * 本类不写死任何模型参数。关键契约：
 * - 图优化必须 BASIC（EXTENDED 有 TransposeDQWeightsForMatMulNBits bug）；
 * - 首步 input_ids=[eos, tgt_lang]，use_cache_branch=false，
 *   past 喂 48 个 [1,heads,0,head_dim] 零张量；
 * - 后续步 use_cache_branch=true；encoder KV 自首步冻结
 *   （cache 分支的 encoder 输出是占位张量，不能喂回）；
 * - greedy 解码。
 *
 * 实例内存约 1GB（四语模型），必须受控使用：由 TranslationService
 * 做单实例缓存。translate 加锁串行执行。
 */
class KitTranslationEngine(
    private val config: ModelRuntimeConfig,
    private val tokenizer: M2m100Tokenizer,
    private val encoderFile: File,
    private val decoderFile: File,
    private val numThreads: Int = 4,
) : com.example.fourlangasr.translation.TranslationEngine {

    private class NativeState {
        val env = OrtEnvironment.getEnvironment()
        lateinit var enc: OrtSession
        lateinit var dec: OrtSession
        lateinit var pastInputNames: List<String>
        lateinit var pastInputIsDecoder: BooleanArray
    }

    private var native: NativeState? = null

    private fun makeSession(env: OrtEnvironment, path: File): OrtSession {
        val opts = OrtSession.SessionOptions()
        // 必须 BASIC：ORT EXTENDED 的 TransposeDQWeightsForMatMulNBits
        // 对"子图内 DQ 引用主图 int8 权重"有 bug（包内 README 已实测）
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
        opts.setIntraOpNumThreads(numThreads)
        return env.createSession(path.absolutePath, opts)
    }

    private fun ensureLoaded(): NativeState {
        native?.let { return it }
        synchronized(this) {
            native?.let { return it }
            val t0 = SystemClock.elapsedRealtime()
            val st = NativeState()
            check(encoderFile.isFile && decoderFile.isFile) {
                "本地模型未就绪：缺 ${encoderFile.name}/${decoderFile.name}"
            }
            val decSession = makeSession(st.env, decoderFile)
            val names = decSession.inputInfo.keys.filter { it.startsWith("past_key_values.") }
            val isDecoder = BooleanArray(names.size)
            var decCount = 0
            var encCount = 0
            for (i in names.indices) {
                // past_key_values.{layer}.{decoder|encoder}.{key|value}
                val parts = names[i].split(".")
                when {
                    parts.size == 4 && parts[2] == "decoder" -> {
                        isDecoder[i] = true
                        decCount++
                    }
                    parts.size == 4 && parts[2] == "encoder" -> encCount++
                }
            }
            val expected = config.decoderLayers * 2
            check(decCount == expected && encCount == expected) {
                "past 张量数与 MANIFEST 不符：dec=$decCount enc=$encCount 期望各 $expected"
            }
            st.dec = decSession
            st.enc = makeSession(st.env, encoderFile)
            st.pastInputNames = names
            st.pastInputIsDecoder = isDecoder
            native = st
            Log.d(
                "KitTranslation",
                "${config.modelId}: sessions loaded in ${SystemClock.elapsedRealtime() - t0}ms (threads=$numThreads)",
            )
            return st
        }
    }

    private fun int64Tensor(env: OrtEnvironment, values: LongArray): OnnxTensor {
        // 模型输入声明为 [batch, seq] 二维（encoder 的 input_ids/attention_mask、
        // decoder 的 input_ids 均如此），batch 恒为 1。
        val buf = ByteBuffer.allocateDirect(values.size * 8)
            .order(ByteOrder.nativeOrder()).asLongBuffer()
        buf.put(values)
        buf.rewind()
        return OnnxTensor.createTensor(env, buf, longArrayOf(1, values.size.toLong()))
    }

    private fun boolTensor(env: OrtEnvironment, value: Boolean): OnnxTensor {
        val buf = ByteBuffer.allocateDirect(1)
        buf.put(if (value) 1 else 0)
        buf.rewind()
        return OnnxTensor.createTensor(env, buf, longArrayOf(1), OnnxJavaType.BOOL)
    }

    /** 拷贝一个输出张量为自有的输入张量（结果 Result 关闭后依然有效）。 */
    private fun copyTensor(env: OrtEnvironment, t: OnnxTensor): OnnxTensor {
        val shape = t.info.shape
        val src = t.floatBuffer
        val size = src.remaining()
        val arr = FloatArray(size)
        src.get(arr)
        val dst = ByteBuffer.allocateDirect(size * 4)
            .order(ByteOrder.nativeOrder()).asFloatBuffer()
        dst.put(arr)
        dst.rewind()
        return OnnxTensor.createTensor(env, dst, shape)
    }

    private fun argmaxLast(
        fb: FloatBuffer,
        seqLen: Int,
        vocab: Int,
        banned: IntArray,
    ): Int {
        val start = (seqLen - 1) * vocab
        var best = 0
        var bestV = Float.NEGATIVE_INFINITY
        for (i in 0 until vocab) {
            if (banned.binarySearch(i) >= 0) continue
            val v = fb.get(start + i)
            if (v > bestV) {
                bestV = v
                best = i
            }
        }
        return best
    }

    /**
     * 禁止生成的 token：pad / bos / 四个语言标记。
     * PC 端 beam 参考实现同样屏蔽 pad+bos；语言标记漏进正文会让译文带
     * `__en__` 这类字面量，而 Kotlin 端 detok 与 sentencepiece 逐字节对拍过，
     * 不能在那里加过滤（会破坏对拍契约），所以只在采样处屏蔽。
     */
    private val bannedTokenIds: IntArray by lazy {
        val ids = sortedSetOf(config.padId.toInt(), config.bosId.toInt())
        config.languages.values.forEach { ids.add(it.toInt()) }
        ids.toIntArray()
    }

    /**
     * 尾部是否已进入循环复读（连续 3 个完整周期，周期 1..3 token）。
     * greedy 解码没有任何停止保护，实测会把同一个词一路刷到 maxNewTokens。
     */
    private fun tailRepeats(gen: List<Long>): Boolean {
        val n = gen.size
        for (period in 1..3) {
            val need = period * 3
            if (n < need) continue
            var same = true
            for (i in (n - need) until n) {
                if (gen[i] != gen[i - period]) {
                    same = false
                    break
                }
            }
            if (same) return true
        }
        return false
    }

    /**
     * 一次解码步。present 张量在 Result 关闭前拷贝为自有张量。
     * 返回 (nextToken, (presentDecoder, presentEncoder))。
     */
    private fun decoderStep(
        st: NativeState,
        inputIds: LongArray,
        encMask: OnnxTensor,
        encHidden: OnnxTensor,
        pastDec: List<OnnxTensor>,
        pastEnc: List<OnnxTensor>,
        useCache: Boolean,
    ): Pair<Int, Pair<List<OnnxTensor>, List<OnnxTensor>>> {
        val feeds = HashMap<String, OnnxTensor>(60)
        val idTensor = int64Tensor(st.env, inputIds)
        val boolTensor = boolTensor(st.env, useCache)
        feeds["encoder_attention_mask"] = encMask
        feeds["input_ids"] = idTensor
        feeds["encoder_hidden_states"] = encHidden
        feeds["use_cache_branch"] = boolTensor
        var di = 0
        var ei = 0
        for (i in st.pastInputNames.indices) {
            feeds[st.pastInputNames[i]] =
                if (st.pastInputIsDecoder[i]) pastDec[di++] else pastEnc[ei++]
        }
        val result = st.dec.run(feeds)
        try {
            val logits = result.get("logits").get() as OnnxTensor
            val seqLen = logits.info.shape[1].toInt()
            val next = argmaxLast(logits.floatBuffer, seqLen, config.vocabSize, bannedTokenIds)
            val nPairs = config.decoderLayers * 2
            val presentDec = ArrayList<OnnxTensor>(nPairs)
            val presentEnc = ArrayList<OnnxTensor>(nPairs)
            for (i in st.pastInputNames.indices) {
                val outName = "present." + st.pastInputNames[i].removePrefix("past_key_values.")
                val t = copyTensor(st.env, result.get(outName).get() as OnnxTensor)
                if (st.pastInputIsDecoder[i]) presentDec.add(t) else presentEnc.add(t)
            }
            return next to (presentDec to presentEnc)
        } finally {
            result.close()
            idTensor.close()
            boolTensor.close()
        }
    }

    override fun translate(
        text: String,
        source: AppLanguage,
        target: AppLanguage,
    ): TranslationResult {
        val srcCode = source.code
        val tgtCode = target.code
        synchronized(this) {
            val tLoad0 = SystemClock.elapsedRealtime()
            val st = ensureLoaded()
            val tLoadMs = SystemClock.elapsedRealtime() - tLoad0

            // 张量所有权集中在这里回收：异常路径同样必须关闭，
            // 否则一次解码失败就把 48 组 cache 张量留在 native 里。
            var encHidden: OnnxTensor? = null
            var maskTensor: OnnxTensor? = null
            var emptyPast: List<OnnxTensor> = emptyList()
            var pastDec: List<OnnxTensor> = emptyList()
            var frozenEnc: List<OnnxTensor> = emptyList()
            try {
                // ---- 编码（一次） ----
                // 方向机制（家族契约）：
                //   M2M100：  [src_lang] + X + [eos]
                //   SMALL100：[tgt_lang] + X + [eos]（目标语言前缀，无 forced_bos）
                var bodyIds = tokenizer.piecesToModelIds(tokenizer.encodePieces(text))
                // 位置编码上限是硬约束（语言前缀 + eos 各占一位）：
                // 超长句整段喂入会直接推理失败或质量塌陷，只能截正文。
                val budget = config.maxSourceTokens - 2
                val truncated = budget in 0 until bodyIds.size
                if (truncated) bodyIds = bodyIds.copyOf(budget)

                val srcIds = when (config.directionMechanism) {
                    DirectionMechanism.M2M100_STYLE ->
                        longArrayOf(tokenizer.langToken(srcCode)) + bodyIds + config.eosId
                    DirectionMechanism.SMALL100_STYLE ->
                        longArrayOf(tokenizer.langToken(tgtCode)) + bodyIds + config.eosId
                }
                val srcIdsTensor = int64Tensor(st.env, srcIds)
                val mask = LongArray(srcIds.size) { 1L }
                val maskT = int64Tensor(st.env, mask)
                maskTensor = maskT
                val encFeeds = HashMap<String, OnnxTensor>(2)
                encFeeds["input_ids"] = srcIdsTensor
                encFeeds["attention_mask"] = maskT
                val encResult = st.enc.run(encFeeds)
                val hidden: OnnxTensor
                try {
                    hidden = copyTensor(st.env, encResult.get("last_hidden_state").get() as OnnxTensor)
                    encHidden = hidden
                } finally {
                    encResult.close()
                    srcIdsTensor.close()
                }

                // ---- 首步解码：input=[eos, tgt_lang]，空 past ----
                val nPast = config.decoderLayers * 4
                val emptyShape = longArrayOf(
                    1,
                    config.attentionHeads.toLong(),
                    0,
                    config.headDim.toLong(),
                )
                val emptyBuf = ByteBuffer.allocateDirect(0)
                    .order(ByteOrder.nativeOrder()).asFloatBuffer()
                emptyPast = List(nPast) { OnnxTensor.createTensor(st.env, emptyBuf, emptyShape) }

                // 首步 decoder 输入：
                //   M2M100：  [eos, tgt_lang]（forced bos）
                //   SMALL100：[eos]（no forced_bos）
                val firstInput = when (config.directionMechanism) {
                    DirectionMechanism.M2M100_STYLE ->
                        longArrayOf(config.eosId, tokenizer.langToken(tgtCode))
                    DirectionMechanism.SMALL100_STYLE ->
                        longArrayOf(config.eosId)
                }
                val (first, present0) = decoderStep(
                    st,
                    firstInput,
                    maskT,
                    hidden,
                    emptyPast,
                    emptyPast,
                    useCache = false,
                )

                // 冻结 encoder KV：cache 分支的 encoder 输出是占位张量，不能喂回
                frozenEnc = present0.second

                var next = first
                val gen = ArrayList<Long>()
                var decPast = present0.first
                pastDec = decPast
                val eos = config.eosId.toInt()
                var stoppedByRepetition = false
                while (next != eos && gen.size < config.maxNewTokens) {
                    if (tailRepeats(gen)) {
                        stoppedByRepetition = true
                        break
                    }
                    gen.add(next.toLong())
                    val (nxt, present) = decoderStep(
                        st,
                        longArrayOf(next.toLong()),
                        maskT,
                        hidden,
                        decPast,
                        frozenEnc,
                        useCache = true,
                    )
                    decPast.forEach { it.close() }
                    decPast = present.first
                    pastDec = decPast
                    next = nxt
                }

                val output = tokenizer.detok(gen.toLongArray())
                val totalMs = SystemClock.elapsedRealtime() - tLoad0
                if (BuildConfig.DEBUG) {
                    // 只有 debug 构建记录原文/译文内容，release 包不落到 logcat。
                    Log.d(
                        "KitTranslation",
                        "${config.modelId}: load=${tLoadMs}ms decode=${totalMs - tLoadMs}ms " +
                                "tokens=${gen.size} $srcCode->$tgtCode truncated=$truncated " +
                                "loopcut=$stoppedByRepetition " +
                                "in='${text.take(60)}' out='${output.take(60)}'",
                    )
                }
                if (truncated) {
                    Log.w(
                        "KitTranslation",
                        "${config.modelId}: 源句超过 ${config.maxSourceTokens} token，已截断后段",
                    )
                }
                return TranslationResult(
                    sourceText = text,
                    outputText = output,
                    sourceLanguage = source,
                    targetLanguage = target,
                    speechLanguage = target,
                    translated = true,
                    modelLabel = config.modelId,
                )
            } finally {
                pastDec.forEach { runCatching { it.close() } }
                frozenEnc.forEach { runCatching { it.close() } }
                emptyPast.forEach { runCatching { it.close() } }
                runCatching { encHidden?.close() }
                runCatching { maskTensor?.close() }
            }
        }
    }

    /** 释放 ORT session（引擎缓存替换时调用）。 */
    fun close() {
        synchronized(this) {
            val st = native ?: return
            try {
                st.enc.close()
            } catch (_: OrtException) {
            }
            try {
                st.dec.close()
            } catch (_: OrtException) {
            }
            native = null
        }
    }
}
