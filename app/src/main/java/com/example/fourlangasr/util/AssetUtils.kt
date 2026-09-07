package com.example.fourlangasr.util

import android.content.Context
import java.io.File
import java.io.FileOutputStream

object AssetUtils {

    fun copyAssetFolderIfNeeded(
        context: Context,
        assetPath: String,
        destinationDir: File,
        requiredMarker: String,
    ) {
        val marker = File(destinationDir, requiredMarker)
        if (marker.exists()) {
            return
        }

        if (destinationDir.exists()) {
            destinationDir.deleteRecursively()
        }
        destinationDir.mkdirs()

        copyRecursive(
            context = context,
            assetPath = assetPath,
            destination = destinationDir,
        )

        require(marker.exists()) {
            "Asset copy finished but marker is missing: ${marker.absolutePath}"
        }
    }

    private fun copyRecursive(
        context: Context,
        assetPath: String,
        destination: File,
    ) {
        val children = context.assets.list(assetPath) ?: emptyArray()

        if (children.isEmpty()) {
            destination.parentFile?.mkdirs()
            context.assets.open(assetPath).use { input ->
                FileOutputStream(destination).use { output ->
                    input.copyTo(output, bufferSize = 1024 * 1024)
                }
            }
            return
        }

        destination.mkdirs()

        for (child in children) {
            val childAsset = if (assetPath.isBlank()) child else "$assetPath/$child"
            val childDest = File(destination, child)
            copyRecursive(context, childAsset, childDest)
        }
    }
}
