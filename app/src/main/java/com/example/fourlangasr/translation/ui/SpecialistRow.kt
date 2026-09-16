package com.example.fourlangasr.translation.ui

import com.example.fourlangasr.translation.model.ManifestModel

/**
 * 专项模型页一行的展示状态（由 Activity 组装，Adapter 只负责渲染）。
 */
data class SpecialistRow(
    val model: ManifestModel,
    val isGeneralist: Boolean,
    val installed: Boolean,
    val ready: Boolean,
    val stale: Boolean,
    val enabled: Boolean,
    /** 非空 = 正在下载/解包：(phase, doneBytes, totalBytes)，total 未知为 -1。 */
    val progress: Pair<String, Pair<Long, Long>>?,
) {
    companion object {
        fun formatBytes(bytes: Long): String {
            if (bytes <= 0) return "?"
            return "%.0fMB".format(bytes / 1024.0 / 1024.0)
        }
    }
}
