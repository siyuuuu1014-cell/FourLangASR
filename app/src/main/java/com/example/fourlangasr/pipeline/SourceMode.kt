package com.example.fourlangasr.pipeline

/**
 * v0.5 源语言模式。
 *
 * AUTO   -> 每句跑 VoxLingua LID，自动路由 zh/en/ru/uz。
 * MANUAL -> 完全跳过 LID，锁定用户手选语言。
 */
enum class SourceMode {
    AUTO,
    MANUAL,
}
