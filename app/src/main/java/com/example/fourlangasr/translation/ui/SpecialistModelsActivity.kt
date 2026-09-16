package com.example.fourlangasr.translation.ui

import android.os.Bundle
import android.widget.Button
import android.widget.ListView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.fourlangasr.R
import com.example.fourlangasr.translation.ServerConfig
import com.example.fourlangasr.translation.TranslationModules
import com.example.fourlangasr.translation.model.ManifestModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * 专项模型页：六个语言对模型的按需下载 / 更新 / 启用 / 删除。
 *
 * 数据流：远端 /models/manifest（清单）+ 本地 ModelRegistry（版本/启用状态）
 * + ModelPackageManager（包就绪/下载进度）→ 组装为 SpecialistRow 列表。
 * 所有网络/解包操作在 IO 线程，UI 更新回主线程。
 */
class SpecialistModelsActivity : AppCompatActivity() {

    private lateinit var modules: TranslationModules
    private lateinit var adapter: SpecialistModelAdapter
    private lateinit var statusText: TextView

    private val progressMap = HashMap<String, Pair<String, Pair<Long, Long>>>()

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

        adapter = SpecialistModelAdapter(
            layoutInflater = layoutInflater,
            onDownload = { row -> downloadModel(row.model) },
            onToggleEnabled = { row -> toggleEnabled(row) },
            onDelete = { row -> deleteModel(row) },
        )
        findViewById<ListView>(R.id.specialistList).adapter = adapter

        findViewById<Button>(R.id.specialistRefresh).setOnClickListener {
            refreshManifest()
        }

        refreshManifest()
    }

    private fun updateServerUrlText(view: TextView) {
        view.text = "服务器地址：${ServerConfig.baseUrl(this)}（点击修改）"
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
            SpecialistRow(
                model = model,
                isGeneralist = model.modelId == generalistId,
                installed = installed != null,
                ready = ready,
                stale = stale,
                enabled = installed?.enabled == true,
                progress = progressMap[model.modelId],
            )
        }
        adapter.submit(rows)
    }

    private fun downloadModel(model: ManifestModel) {
        progressMap[model.modelId] = ("download" to (0L to model.sizeBytes))
        rebuildRows()
        lifecycleScope.launch {
            try {
                withContext(Dispatchers.IO) {
                    modules.packages.ensureModel(model) { phase, done, total ->
                        progressMap[model.modelId] = (phase to (done to total))
                        runOnUiThread { adapter.update(rowFor(model.modelId) ?: return@runOnUiThread) }
                    }
                }
                withContext(Dispatchers.IO) {
                    val isSpecialist = modules.sync.latestManifest
                        ?.specialists()?.any { it.modelId == model.modelId } == true
                    modules.registry.markInstalled(
                        model.modelId,
                        model.version,
                        model.sourceSha256,
                    )
                    if (isSpecialist) modules.registry.setEnabled(model.modelId, true)
                }
                Toast.makeText(this@SpecialistModelsActivity, "${model.modelId} 已就绪", Toast.LENGTH_SHORT).show()
            } catch (e: Exception) {
                Toast.makeText(
                    this@SpecialistModelsActivity,
                    "${model.modelId} 下载失败：${e.javaClass.simpleName}",
                    Toast.LENGTH_LONG,
                ).show()
            } finally {
                progressMap.remove(model.modelId)
                rebuildRows()
            }
        }
    }

    private fun rowFor(modelId: String): SpecialistRow? {
        val manifest = modules.sync.latestManifest ?: return null
        val model = manifest.byId(modelId) ?: return null
        val installed = modules.registry.get(modelId)
        val generalistId = manifest.generalist().modelId
        return SpecialistRow(
            model = model,
            isGeneralist = modelId == generalistId,
            installed = installed != null,
            ready = modules.packages.isModelReady(modelId),
            stale = installed != null &&
                (installed.sourceSha256 != model.sourceSha256 || installed.version != model.version),
            enabled = installed?.enabled == true,
            progress = progressMap[modelId],
        )
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
