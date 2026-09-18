package com.example.fourlangasr.translation.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.widget.Button
import android.widget.ListView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.example.fourlangasr.R
import com.example.fourlangasr.translation.PerformanceConfig
import com.example.fourlangasr.translation.ServerConfig
import com.example.fourlangasr.translation.TranslationModules
import com.example.fourlangasr.translation.download.ModelDownloadCoordinator
import com.example.fourlangasr.translation.download.ModelDownloadService
import com.example.fourlangasr.translation.download.ModelDownloadState
import com.example.fourlangasr.translation.download.ModelDownloadStatus
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * 专项模型页：六个语言对模型的按需下载 / 更新 / 启用 / 删除。
 *
 * 下载统一交给 [ModelDownloadService]（前台服务）执行 —— App 退后台
 * 下载不中断、通知栏实时进度；本页订阅 [ModelDownloadCoordinator]
 * 的状态流渲染列表（进程内单例，页面重建/重进后状态一致）。
 */
class SpecialistModelsActivity : AppCompatActivity() {

    private lateinit var modules: TranslationModules
    private lateinit var adapter: SpecialistModelAdapter
    private lateinit var statusText: TextView

    private var downloadState: ModelDownloadState = ModelDownloadState(emptyList(), emptyMap())

    private val notificationPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (!granted) {
                Toast.makeText(
                    this,
                    "未授予通知权限：下载仍会在前台继续，但通知栏看不到进度",
                    Toast.LENGTH_LONG,
                ).show()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_specialist_models)
        setSupportActionBar(findViewById(R.id.specialistToolbar))
        supportActionBar?.setDisplayHomeAsUpEnabled(true)

        modules = TranslationModules.getInstance(this)
        statusText = findViewById(R.id.specialistStatus)

        val serverUrlText = findViewById<TextView>(R.id.specialistServerUrl)
        updateServerUrlText(serverUrlText)
        serverUrlText.setOnClickListener { showServerUrlDialog(serverUrlText) }

        val memoryModeText = findViewById<TextView>(R.id.specialistMemoryMode)
        updateMemoryModeText(memoryModeText)
        memoryModeText.setOnClickListener {
            val current = PerformanceConfig.releaseAsrDuringTranslation(this)
            PerformanceConfig.setReleaseAsrDuringTranslation(this, !current)
            updateMemoryModeText(memoryModeText)
            Toast.makeText(
                this,
                if (!current) {
                    "已开启：翻译时释放 ASR 模型（省约 1GB 内存，识别有短暂切换延迟）"
                } else {
                    "已关闭低内存档位"
                },
                Toast.LENGTH_LONG,
            ).show()
        }

        adapter = SpecialistModelAdapter(
            layoutInflater = layoutInflater,
            onDownload = { row -> downloadModel(row) },
            onToggleEnabled = { row -> toggleEnabled(row) },
            onDelete = { row -> deleteModel(row) },
        )
        findViewById<ListView>(R.id.specialistList).adapter = adapter

        findViewById<Button>(R.id.specialistRefresh).setOnClickListener {
            refreshManifest()
        }

        // 订阅下载协调器状态（下载中/排队中/失败实时反映到列表）
        lifecycleScope.launch {
            modules.downloadCoordinator.state.collectLatest { state ->
                downloadState = state
                rebuildRows()
            }
        }

        refreshManifest()
    }

    private fun downloadModel(row: SpecialistRow) {
        // Android 13+ 需要通知权限才能显示下载进度通知
        if (
            Build.VERSION.SDK_INT >= 33 &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) !=
            PackageManager.PERMISSION_GRANTED
        ) {
            notificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
        }
        modules.downloadCoordinator.enqueue(listOf(row.model.modelId))
        ContextCompat.startForegroundService(
            this,
            Intent(this, ModelDownloadService::class.java).apply {
                action = ModelDownloadService.ACTION_ENQUEUE
                putExtra(ModelDownloadService.EXTRA_MODEL_ID, row.model.modelId)
            },
        )
        rebuildRows()
    }

    private fun updateServerUrlText(view: TextView) {
        view.text = "服务器地址：${ServerConfig.baseUrl(this)}（点击修改）"
    }

    private fun updateMemoryModeText(view: TextView) {
        val enabled = PerformanceConfig.releaseAsrDuringTranslation(this)
        view.text = "低内存档位（翻译时释放 ASR 模型，约省 1GB）：${if (enabled) "开" else "关"}（点击切换）"
    }

    private fun showServerUrlDialog(view: TextView) {
        val input = android.widget.EditText(this)
        input.setText(ServerConfig.baseUrl(this))
        input.setSingleLine(true)
        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("服务器地址")
            .setMessage("修改后立即生效（清单/下载/服务器翻译都会走新地址）")
            .setView(input)
            .setPositiveButton("保存") { _, _ ->
                try {
                    ServerConfig.setBaseUrl(this, input.text.toString())
                    updateServerUrlText(view)
                    Toast.makeText(this, "已保存，正在刷新清单…", Toast.LENGTH_SHORT).show()
                    refreshManifest()
                } catch (e: Exception) {
                    Toast.makeText(this, "地址无效：${e.message}", Toast.LENGTH_LONG).show()
                }
            }
            .setNegativeButton("取消", null)
            .show()
    }

    private fun refreshManifest() {
        statusText.text = "正在拉取模型清单…"
        lifecycleScope.launch {
            try {
                withContext(Dispatchers.IO) { modules.sync.refreshManifest() }
                statusText.text = "清单已更新（${modules.sync.latestManifest?.models?.size ?: 0} 个模型）"
                rebuildRows()
            } catch (e: Exception) {
                statusText.text = "清单拉取失败（服务器不可达？）：${e.javaClass.simpleName}"
                rebuildRows()
            }
        }
    }

    private fun rebuildRows() {
        val manifest = modules.sync.latestManifest ?: run {
            adapter.submit(emptyList())
            return
        }
        val generalistId = manifest.generalist().modelId
        val rows = manifest.models.map { model ->
            val installed = modules.registry.get(model.modelId)
            val ready = modules.packages.isModelReady(model.modelId)
            val stale = installed != null &&
                (installed.sourceSha256 != model.sourceSha256 || installed.version != model.version)
            val status = downloadState.statuses[model.modelId]
            val progress = when (status) {
                is ModelDownloadStatus.Running ->
                    status.progress.phase to (status.progress.doneBytes to status.progress.totalBytes)
                ModelDownloadStatus.Pending -> "queue" to (0L to -1L)
                else -> null
            }
            SpecialistRow(
                model = model,
                isGeneralist = model.modelId == generalistId,
                installed = installed != null,
                ready = ready,
                stale = stale,
                enabled = installed?.enabled == true,
                progress = progress,
            )
        }
        adapter.submit(rows)
    }

    private fun toggleEnabled(row: SpecialistRow) {
        modules.registry.setEnabled(row.model.modelId, !row.enabled)
        rebuildRows()
    }

    private fun deleteModel(row: SpecialistRow) {
        lifecycleScope.launch {
            withContext(Dispatchers.IO) {
                modules.packages.removePackage(row.model.modelId)
                modules.registry.remove(row.model.modelId)
            }
            Toast.makeText(this@SpecialistModelsActivity, "${row.model.modelId} 已删除", Toast.LENGTH_SHORT).show()
            rebuildRows()
        }
    }
}
