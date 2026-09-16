package com.example.fourlangasr.translation

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import com.example.fourlangasr.model.AppLanguage
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * 本机离线翻译引擎：m2m100 四语 int8 ONNX 模型（2 文件 merged decoder）。
 *
 * 解码循环完整移植 experiments/translation/run_m2m100_reference.py
 * （PC 端 6 方向已验证正确）。关键契约：
 * - 图优化必须 BASIC（EXTENDED 有 TransposeDQWeightsForMatMulNBits bug）；
 * - 首步 input_ids=[eos(2), tgt_lang]，use_cache_branch=false，
 *   past 喂 48 个 [1,16,0,64] 零张量；
 * - 后续步 use_cache_branch=true，input_ids=[上一 token]；
 *   encoder KV 自首步起冻结（cache 分支输出的 encoder 占位张量
 *   [0,16,1,64] 必须丢弃，喂回会 MatMul 广播报错）；
 * - greedy 解码（beam 在手机端延迟不可接受，v1 不做）。
 *
 * 内存约 1GB/实例（两个 session 的 int8 权重 + 激活），
 * 必须单实例：translate 加锁串行执行。
 * ORT 原生库在首次 translate 时才加载（构造不触碰 native）。
 */
class KitTranslationEngine(
    private val tokenizer: M2m100Tokenizer,
    private val modelDir: File,
    private val maxNewTokens: Int = 200,
    private val numThreads: Int = 4,
) : TranslationEngine {

    companion object {
        private const val VOCAB = 128112
        private const val N_LAYERS = 12
        private const val N_HEADS = 16
        private const val HEAD_DIM = 64
        private const val ENCODER_FILE = "encoder_model.onnx"
        private const val DECODER_FILE = "decoder_model_merged.onnx"
    }

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
            val st = NativeState()
            val encFile = File(modelDir, ENCODER_FILE)
            val decFile = File(modelDir, DECODER_FILE)
            check(encFile.exists() && decFile.exists()) {
                "本地翻译模型未就绪：需要先下载模型包（缺 $ENCODER_FILE/$DECODER_FILE）"
            }
            val decSession = makeSession(st.env, decFile)
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
            check(decCount == N_LAYERS * 2 && encCount == N_LAYERS * 2) {
                "decoder 输入 past 张量数不符：dec=$decCount enc=$encCount"
            }
            st.dec = decSession
            st.enc = makeSession(st.env, encFile)
            st.pastInputNames = names
            st.pastInputIsDecoder = isDecoder
            native = st
            return st
        }
    }

    private fun int64Tensor(env: OrtEnvironment, values: LongArray): OnnxTensor {
        val buf = ByteBuffer.allocateDirect(values.size * 8)
            .order(ByteOrder.nativeOrder()).asLongBuffer()
        buf.put(values)
        buf.rewind()
        return OnnxTensor.createTensor(env, buf, longArrayOf(values.size.toLong()))
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

    private fun argmaxLast(fb: FloatBuffer, seqLen: Int): Int {
        val start = (seqLen - 1) * VOCAB
        var best = 0
        var bestV = Float.NEGATIVE_INFINITY
        for (i in 0 until VOCAB) {
            val v = fb.get(start + i)
            if (v > bestV) {
                bestV = v
                best = i
            }
        }
        return best
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
            val next = argmaxLast(logits.floatBuffer, seqLen)
            val presentDec = ArrayList<OnnxTensor>(N_LAYERS * 2)
            val presentEnc = ArrayList<OnnxTensor>(N_LAYERS * 2)
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
        val tgtLang = target.code
        synchronized(this) {
            val st = ensureLoaded()

            // ---- 编码（一次） ----
            val srcIds = tokenizer.encodeSourceIds(text, source.code)
            val srcIdsTensor = int64Tensor(st.env, srcIds)
            val mask = LongArray(srcIds.size) { 1L }
            val maskTensor = int64Tensor(st.env, mask)
            val encFeeds = HashMap<String, OnnxTensor>(2)
            encFeeds["input_ids"] = srcIdsTensor
            encFeeds["attention_mask"] = maskTensor
            val encResult = st.enc.run(encFeeds)
            val encHidden: OnnxTensor
            try {
                encHidden = copyTensor(st.env, encResult.get("last_hidden_state").get() as OnnxTensor)
            } finally {
                encResult.close()
                srcIdsTensor.close()
            }

            // ---- 首步解码：input=[eos, tgt_lang]，空 past ----
            val emptyPast = ArrayList<OnnxTensor>(N_LAYERS * 4)
            val emptyShape = longArrayOf(1, N_HEADS.toLong(), 0, HEAD_DIM.toLong())
            val emptyBuf = ByteBuffer.allocateDirect(0)
                .order(ByteOrder.nativeOrder()).asFloatBuffer()
            repeat(N_LAYERS * 4) {
                emptyPast.add(OnnxTensor.createTensor(st.env, emptyBuf, emptyShape))
            }

            try {
                val (first, present0) = decoderStep(
                    st,
                    longArrayOf(M2m100Tokenizer.EOS_ID, M2m100Tokenizer.LANG_TOKENS[tgtLang]!!),
                    maskTensor,
                    encHidden,
                    emptyPast,
                    emptyPast,
                    useCache = false,
                )
                emptyPast.forEach { it.close() }

                // 冻结 encoder KV：cache 分支的 encoder 输出是 [0,16,1,64] 占位，不能喂回
                val frozenEnc = present0.second

                var next = first
                val gen = ArrayList<Long>()
                var pastDec = present0.first
                val eos = M2m100Tokenizer.EOS_ID.toInt()
                while (next != eos && gen.size < maxNewTokens) {
                    gen.add(next.toLong())
                    val (nxt, present) = decoderStep(
                        st,
                        longArrayOf(next.toLong()),
                        maskTensor,
                        encHidden,
                        pastDec,
                        frozenEnc,
                        useCache = true,
                    )
                    pastDec.forEach { it.close() }
                    pastDec = present.first
                    next = nxt
                }
                pastDec.forEach { it.close() }
                frozenEnc.forEach { it.close() }
                encHidden.close()
                maskTensor.close()

                val output = tokenizer.detok(gen.toLongArray())
                return TranslationResult(
                    sourceText = text,
                    outputText = output,
                    sourceLanguage = source,
                    targetLanguage = target,
                    speechLanguage = target,
                    translated = true,
                )
            } catch (e: Throwable) {
                encHidden.close()
                maskTensor.close()
                throw e
            }
        }
    }

    /** 释放 ORT session（进程退出/引擎替换时调用）。 */
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
