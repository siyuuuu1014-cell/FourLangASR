package com.example.fourlangasr.translation.download

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

/**
 * 模型下载协调器（进程内单例，由 TranslationModules 持有）：
 * 维护下载队列与各模型状态，ModelDownloadService 写、UI 订阅读。
 * 状态在进程内存中（进程被杀后队列清空，但 .part 断点续传保证不白下）。
 */
data class ModelDownloadProgress(
    val phase: String,
    val doneBytes: Long,
    val totalBytes: Long,
)

sealed interface ModelDownloadStatus {
    object Pending : ModelDownloadStatus
    data class Running(val progress: ModelDownloadProgress) : ModelDownloadStatus
    object Complete : ModelDownloadStatus
    data class Failed(val message: String) : ModelDownloadStatus
}

data class ModelDownloadState(
    val queue: List<String>,
    val statuses: Map<String, ModelDownloadStatus>,
)

class ModelDownloadCoordinator {

    private val _state = MutableStateFlow(
        ModelDownloadState(queue = emptyList(), statuses = emptyMap()),
    )
    val state: StateFlow<ModelDownloadState> = _state

    @Synchronized
    fun enqueue(modelIds: List<String>) {
        val s = _state.value
        val newQueue = (s.queue + modelIds).distinct()
        val newStatuses = s.statuses + modelIds.map { it to ModelDownloadStatus.Pending }
        _state.value = s.copy(queue = newQueue, statuses = newStatuses)
    }

    /** 取队列头（不弹出）；无任务返回 null。 */
    @Synchronized
    fun nextPending(): String? = _state.value.queue.firstOrNull()

    @Synchronized
    fun updateProgress(modelId: String, phase: String, doneBytes: Long, totalBytes: Long) {
        val s = _state.value
        _state.value = s.copy(
            statuses = s.statuses + (
                modelId to ModelDownloadStatus.Running(
                    ModelDownloadProgress(phase, doneBytes, totalBytes),
                )
                ),
        )
    }

    @Synchronized
    fun complete(modelId: String) {
        val s = _state.value
        _state.value = s.copy(
            queue = s.queue - modelId,
            statuses = s.statuses + (modelId to ModelDownloadStatus.Complete),
        )
    }

    @Synchronized
    fun fail(modelId: String, message: String) {
        val s = _state.value
        _state.value = s.copy(
            queue = s.queue - modelId,
            statuses = s.statuses + (modelId to ModelDownloadStatus.Failed(message)),
        )
    }
}
