package com.example.fourlangasr.translation.ui

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.BaseAdapter
import android.widget.Button
import android.widget.TextView
import com.example.fourlangasr.R

/**
 * 专项模型列表适配器：只做渲染，行为回调交给 [SpecialistModelsActivity]。
 * 行结构见 res/layout/item_specialist_model.xml。
 */
class SpecialistModelAdapter(
    private val layoutInflater: LayoutInflater,
    private val onDownload: (SpecialistRow) -> Unit,
    private val onToggleEnabled: (SpecialistRow) -> Unit,
    private val onDelete: (SpecialistRow) -> Unit,
) : BaseAdapter() {

    private val rows = ArrayList<SpecialistRow>()

    fun submit(newRows: List<SpecialistRow>) {
        rows.clear()
        rows.addAll(newRows)
        notifyDataSetChanged()
    }

    fun update(row: SpecialistRow) {
        val idx = rows.indexOfFirst { it.model.modelId == row.model.modelId }
        if (idx >= 0) {
            rows[idx] = row
            notifyDataSetChanged()
        }
    }

    override fun getCount(): Int = rows.size

    override fun getItem(position: Int): SpecialistRow = rows[position]

    override fun getItemId(position: Int): Long = position.toLong()

    override fun getView(position: Int, convertView: View?, parent: ViewGroup): View {
        val view = convertView ?: layoutInflater.inflate(R.layout.item_specialist_model, parent, false)
        val row = rows[position]

        val title = view.findViewById<TextView>(R.id.specialistTitle)
        val subtitle = view.findViewById<TextView>(R.id.specialistSubtitle)
        val btnPrimary = view.findViewById<Button>(R.id.specialistBtnPrimary)
        val btnSecondary = view.findViewById<Button>(R.id.specialistBtnSecondary)

        val familyLabel = if (row.isGeneralist) "总模型" else "专项模型"
        title.text = "${row.model.modelId}（$familyLabel · ${SpecialistRow.formatBytes(row.model.sizeBytes)}）"

        val lines = ArrayList<String>()
        lines.add("方向：" + row.model.directions.joinToString(", "))
        val (phase, p) = row.progress ?: ("" to (0L to -1L))
        val status = when {
            row.progress != null -> {
                val pct = if (p.second > 0) "${p.first * 100 / p.second}%" else ""
                if (phase == "unpack") "解包中…" else "下载中 $pct"
            }
            row.ready && row.stale -> "已就绪（有更新）"
            row.ready -> "已就绪"
            row.installed -> "包已下载，未解包"
            else -> "未下载"
        }
        lines.add("状态：$status")
        subtitle.text = lines.joinToString("\n")

        // 主按钮：下载 / 更新 / 启用切换
        when {
            row.progress != null -> {
                btnPrimary.isEnabled = false
                btnPrimary.text = "处理中…"
            }
            !row.ready && !row.installed -> {
                btnPrimary.isEnabled = true
                btnPrimary.text = "下载"
                btnPrimary.setOnClickListener { onDownload(row) }
            }
            row.ready && row.stale -> {
                btnPrimary.isEnabled = true
                btnPrimary.text = "更新"
                btnPrimary.setOnClickListener { onDownload(row) }
            }
            row.isGeneralist -> {
                btnPrimary.isEnabled = false
                btnPrimary.text = "主力已启用"
            }
            else -> {
                btnPrimary.isEnabled = true
                btnPrimary.text = if (row.enabled) "已启用（点击停用）" else "启用"
                btnPrimary.setOnClickListener { onToggleEnabled(row) }
            }
        }

        // 次按钮：删除（已下载时可用）
        if (row.ready || row.installed) {
            btnSecondary.visibility = View.VISIBLE
            btnSecondary.isEnabled = row.progress == null
            btnSecondary.text = "删除"
            btnSecondary.setOnClickListener { onDelete(row) }
        } else {
            btnSecondary.visibility = View.GONE
        }

        return view
    }
}
