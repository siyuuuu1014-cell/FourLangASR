package com.example.fourlangasr.ocr

import androidx.test.core.app.ApplicationProvider
import com.example.fourlangasr.model.AppLanguage
import com.example.fourlangasr.translation.TranslationModules
import android.os.Debug
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Assume
import org.junit.Test

/**
 * 阶段2B-0：翻译引擎资源泄漏回归。
 *
 * 本轮修了 KitTranslationEngine 三处：解码循环每步释放不再使用的 present.second（原每步泄漏
 * decoderLayers*2 个原生张量）、SessionOptions 关闭、ensureLoaded 异常路径关闭已建 session。
 * 这里用端到端 translate() 反复调用，观察 native 堆是否随调用次数持续增长，作为回归护栏。
 *
 * 注意：本机若没有本地翻译包（~734MB），无法验证，**如实 Assume 跳过**，不当作通过。
 * 该用例在有本地包的设备上会自动执行。
 */
class KitTranslationLeakRegressionTest {

    private fun nativeMb() = Debug.getNativeHeapAllocatedSize() / 1048576.0

    @Test
    fun repeatTranslateDoesNotGrowNativeHeap() {
        val ctx = ApplicationProvider.getApplicationContext<android.content.Context>()
        val modules = TranslationModules.getInstance(ctx)
        val src = AppLanguage.EN
        val tgt = AppLanguage.ZH
        val label = modules.service.engineLabelFor(src, tgt)
        Assume.assumeFalse(
            "本机无可用本地翻译包（将走服务器），无法验证 native 泄漏，跳过。label=$label",
            label.startsWith("服务器"),
        )

        // 预热 3 次，排除首次加载分配。
        repeat(3) { modules.service.translate("hello", src, tgt) }
        val n0 = nativeMb()

        val n = 30
        var nonEmpty = 0
        val texts = mutableListOf<String>()
        repeat(n) {
            val r = modules.service.translate("this is a regression sentence for leak check", src, tgt)
            if (r.outputText.isNotBlank()) nonEmpty++
            texts.add(r.outputText.take(40))
        }
        val n1 = nativeMb()

        Harness.writeReport("kit_translation_leak.json", JSONObject().apply {
            put("test", "repeatTranslateDoesNotGrowNativeHeap")
            put("device", Harness.deviceInfo())
            put("engineLabel", label)
            put("iterations", n)
            put("nonEmpty", nonEmpty)
            put("nativeMbBefore", Math.round(n0 * 10.0) / 10.0)
            put("nativeMbAfter", Math.round(n1 * 10.0) / 10.0)
            put("nativeGrowthMb", Math.round((n1 - n0) * 10.0) / 10.0)
            put("sampleOutputs", org.json.JSONArray(texts.take(5)))
        })
        android.util.Log.i(
            Harness.LOG_TAG,
            "kitleak label=$label native ${Math.round(n0*10)/10.0}->${Math.round(n1*10)/10.0}MB nonEmpty=$nonEmpty/$n",
        )
        assertTrue("译文应非空", nonEmpty == n)
        // 30 次翻译 native 增长应远小于单次解码泄漏累加（原 bug 每 token 步泄漏 48 张量）。
        assertTrue("native 堆增长过大: ${n1 - n0}MB", (n1 - n0) < 40.0)
    }
}
