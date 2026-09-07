package com.example.fourlangasr.util

import android.os.Debug

object MemoryMeter {
    fun totalPssMb(): Float {
        val info = Debug.MemoryInfo()
        Debug.getMemoryInfo(info)
        return info.totalPss / 1024.0f
    }
}
