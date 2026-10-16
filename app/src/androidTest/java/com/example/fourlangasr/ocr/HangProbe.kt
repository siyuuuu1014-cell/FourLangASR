package com.example.fourlangasr.ocr

import android.os.Debug
import android.util.Log
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/**
 * 阶段2B-0 挂起取证探针：把"重复操作到底卡在哪儿"变成可测量的量。
 *
 * 用户要求：不要假设死锁或句柄泄漏，用调用栈 + 资源计数证明。这里统计
 *  - 线程数：/proc/self/task（含未 attach 到 JVM 的原生线程，ORT 工作线程就在这里）
 *  - 文件描述符数：/proc/self/fd（会话/模型 mmap 的 fd 泄漏会体现在这）
 *  - Java 堆 / Native 堆 / PSS
 * 并在某次操作超过 [staleMs] 没推进时，用 Debug.dumpThreadsToLogcat 抓一次全线程栈
 * （ART 会同时 dump Java 与 native 栈），再打一条 HANGDUMP 标记，供 adb 侧 timeout 后离线分析。
 *
 * 探针只读不写业务对象，本身不改变被测路径的行为。
 */
object HangProbe {

    private val reading = AtomicBoolean(false)

    fun threadCount(): Int = countEntries("/proc/self/task")

    fun fdCount(): Int = countEntries("/proc/self/fd")

    private fun countEntries(path: String): Int =
        try {
            File(path).listFiles()?.size ?: -1
        } catch (t: Throwable) {
            -1
        }

    fun javaHeapUsedMb(): Double =
        (Runtime.getRuntime().let { it.totalMemory() - it.freeMemory() }) / 1048576.0

    fun nativeHeapMb(): Double = Debug.getNativeHeapAllocatedSize() / 1048576.0

    fun pssMb(): Double {
        val mi = Debug.MemoryInfo()
        Debug.getMemoryInfo(mi)
        return mi.totalPss / 1024.0
    }

    fun snapshot(label: String, iter: Int, extra: String = "") {
        Log.i(
            Harness.LOG_TAG,
            "PROBE $label iter=$iter threads=${threadCount()} fd=${fdCount()} " +
                "java=${r1(javaHeapUsedMb())}MB native=${r1(nativeHeapMb())}MB pss=${r1(pssMb())}MB $extra",
        )
    }

    /**
     * 启一个守护看门狗：只要 [heartbeat] 在 [staleMs] 内有变化就认为在推进；
     * 停滞则 dump 一次全线程栈并打 HANGDUMP 标记（只 dump 一次，避免刷屏）。
     * 返回一个可 close 的句柄，测试结束时关掉。
     */
    fun watchdog(heartbeat: AtomicLong, label: String, staleMs: Long = 20_000): AutoCloseable {
        val dumped = AtomicBoolean(false)
        val stop = AtomicBoolean(false)
        val t = Thread {
            while (!stop.get()) {
                try {
                    Thread.sleep(1000)
                } catch (e: InterruptedException) {
                    break
                }
                val idle = System.currentTimeMillis() - heartbeat.get()
                if (idle > staleMs && !dumped.getAndSet(true)) {
                    Log.e(
                        Harness.LOG_TAG,
                        "HANGDUMP $label stalledMs=$idle threads=${threadCount()} fd=${fdCount()} " +
                            "java=${r1(javaHeapUsedMb())}MB native=${r1(nativeHeapMb())}MB pss=${r1(pssMb())}MB",
                    )
                    // 先试 @hide 的 Debug.dumpThreadsToLogcat（反射调用），它能连原生栈一起 dump 到 logcat；
                    // 失败再退回 Thread.getAllStackTraces（只 Java 栈，且可能因卡在不可 suspend 的
                    // native 线程而阻塞，所以放在后面并单独放线程里）。
                    if (!dumpViaReflection(label)) dumpJavaStacks(label)
                    Log.e(Harness.LOG_TAG, "HANGDUMP $label end")
                }
            }
        }
        t.isDaemon = true
        t.name = "HangWatchdog-$label"
        t.start()
        return AutoCloseable { stop.set(true); t.interrupt() }
    }

    fun r(v: Double): Double = Math.round(v * 10.0) / 10.0

    private fun r1(v: Double): Double = r(v)

    /** 反射调用 @hide 的 Debug.dumpThreadsToLogcat：ART 会把含 native 帧的全线程栈写进 logcat。 */
    private fun dumpViaReflection(label: String): Boolean = try {
        val m = Debug::class.java.getDeclaredMethod("dumpThreadsToLogcat", String::class.java)
        m.isAccessible = true
        m.invoke(null, "ANRDUMP<$label> ")
        true
    } catch (t: Throwable) {
        Log.e(Harness.LOG_TAG, "dumpThreadsToLogcat reflection failed: ${t.javaClass.simpleName} ${t.message}")
        false
    }

    /** 抓所有 Java 线程栈；卡在 ORT/OpenCV 的线程会停在 native 方法（栈顶显示 ...Native Method）。 */
    private fun dumpJavaStacks(label: String) {
        val stacks = Thread.getAllStackTraces()
        Log.e(Harness.LOG_TAG, "JAVASTACKS<$label> threadCount=${stacks.size}")
        for ((t, st) in stacks) {
            val top = st.take(6).joinToString(" <- ") {
                "${it.className}.${it.methodName}:${it.lineNumber}"
            }
            Log.e(Harness.LOG_TAG, "JSTACK<$label> [${t.name}] state=${t.state} $top")
        }
    }
}
