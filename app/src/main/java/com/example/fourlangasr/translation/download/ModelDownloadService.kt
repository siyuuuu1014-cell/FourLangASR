package com.example.fourlangasr.translation.download

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.IBinder
import androidx.core.app.NotificationCompat
import com.example.fourlangasr.R
import com.example.fourlangasr.translation.TranslationModules
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

/**
 * 模型下载前台服务：App 退后台/被杀（START_STICKY 重启）后继续下载。
 * 下载队列与进度由 [ModelDownloadCoordinator]（进程内单例）共享，
 * 通知栏实时显示进度；.part 断点续传保证中断不白下。
 */
class ModelDownloadService : Service() {

    companion object {
        const val ACTION_ENQUEUE = "com.example.fourlangasr.translation.download.ENQUEUE"
        const val EXTRA_MODEL_ID = "model_id"

        private const val CHANNEL_ID = "model_download"
        private const val NOTIFICATION_ID = 1
    }

    private val modules by lazy { TranslationModules.getInstance(this) }
    private val coordinator by lazy { modules.downloadCoordinator }
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var processingJob: Job? = null
    private var lastNotifyMs = 0L

    override fun onCreate() {
        super.onCreate()
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(
            NotificationChannel(
                CHANNEL_ID,
                "翻译模型下载",
                NotificationManager.IMPORTANCE_LOW,
            ),
        )
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_ENQUEUE -> {
                val modelId = intent.getStringExtra(EXTRA_MODEL_ID)
                if (modelId != null) {
                    coordinator.enqueue(listOf(modelId))
                }
                startForeground(NOTIFICATION_ID, buildNotification("准备下载…"))
                startProcessing()
            }
            else -> {
                // 进程重启（START_STICKY）：恢复前台状态；队列在内存中已空则直接停
                startForeground(NOTIFICATION_ID, buildNotification("准备下载…"))
                startProcessing()
            }
        }
        return START_STICKY
    }

    private fun startProcessing() {
        if (processingJob?.isActive == true) return
        processingJob = scope.launch {
            while (true) {
                val modelId = coordinator.nextPending() ?: break
                processOne(modelId)
            }
            // 队列清空：保留完成通知，服务退出
            notify(buildNotification("模型已就绪", progress = null))
            stopForeground(STOP_FOREGROUND_DETACH)
            stopSelf()
        }
    }

    private suspend fun processOne(modelId: String) {
        val model = resolveModel(modelId)
        if (model == null) {
            coordinator.fail(modelId, "清单中找不到该模型")
            notify(buildNotification("$modelId 失败：清单中无此模型", progress = null))
            return
        }
        try {
            // 版本更新：本地文件与新清单 SHA/版本不一致时，先清旧包再重新下载
            //（ensureModel 对"文件已就绪"会短路，不清包会导致旧文件配新版本号）
            val installed = modules.registry.get(modelId)
            val stale = installed != null &&
                (installed.sourceSha256 != model.sourceSha256 || installed.version != model.version)
            if (stale) {
                modules.packages.removePackage(modelId)
            }
            modules.packages.ensureModel(model) { phase, done, total ->
                coordinator.updateProgress(modelId, phase, done, total)
                notify(
                    buildNotification(
                        text = if (phase == "unpack") "解包中…" else "下载中",
                        progress = if (total > 0) (done * 100 / total).toInt().coerceIn(0, 100) else null,
                    ),
                    force = false,
                )
            }
            modules.registry.markInstalled(modelId, model.version, model.sourceSha256)
            val isSpecialist = modules.sync.latestManifest
                ?.specialists()?.any { it.modelId == modelId } == true
            if (isSpecialist) {
                modules.registry.setEnabled(modelId, true)
            }
            coordinator.complete(modelId)
        } catch (e: Exception) {
            coordinator.fail(modelId, e.javaClass.simpleName + ": " + e.message)
            notify(buildNotification("$modelId 下载失败，可重试", progress = null))
        }
    }

    private suspend fun resolveModel(modelId: String) =
        modules.sync.latestManifest?.byId(modelId)
            ?: runCatching { modules.manifestClient.fetchManifest() }
                .getOrNull()?.byId(modelId)

    private fun buildNotification(
        text: String,
        progress: Int? = null,
    ): android.app.Notification {
        val state = coordinator.state.value
        val current = state.queue.firstOrNull()
        val builder = NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setContentTitle(
                if (current != null) "下载模型 $current" else "翻译模型",
            )
            .setContentText(text)
            .setOngoing(progress != null)
        if (progress != null) {
            builder.setProgress(100, progress, false)
        }
        return builder.build()
    }

    private fun notify(notification: android.app.Notification, force: Boolean = true) {
        val now = System.currentTimeMillis()
        if (!force && now - lastNotifyMs < 1000) return
        lastNotifyMs = now
        val nm = getSystemService(NotificationManager::class.java)
        nm.notify(NOTIFICATION_ID, notification)
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        scope.cancel()
        super.onDestroy()
    }
}
