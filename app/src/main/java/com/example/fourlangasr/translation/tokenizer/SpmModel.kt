package com.example.fourlangasr.translation.tokenizer

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets

/**
 * sentencepiece ModelProto 最小解析（protobuf wire format）。
 * 只取分词器需要的部分：
 * - pieces（piece 字符串 / score / type，按 spm id 顺序）
 * - normalizer_spec.precompiled_charsmap（Darts double-array + 替换串池）
 */
internal object SpmModel {

    class Parsed(
        val pieces: List<String>,
        val scores: FloatArray,
        val charsmapUnits: IntArray,
        val charsmapNormalized: ByteArray,
    )

    private class Field(
        val kind: Int, // 0 varint, 1 fixed64, 2 bytes, 5 fixed32
        val intVal: Long,
        val floatBits: Int,
        val bytesOff: Int,
        val bytesLen: Int,
    )

    private fun parseFields(
        buf: ByteArray,
        start: Int = 0,
        end: Int = buf.size,
    ): Map<Int, MutableList<Field>> {
        val out = HashMap<Int, MutableList<Field>>()
        var i = start
        while (i < end) {
            var tag = 0
            var shift = 0
            while (true) {
                val b = buf[i].toInt() and 0xFF
                i++
                tag = tag or ((b and 0x7F) shl shift)
                if (b and 0x80 == 0) break
                shift += 7
            }
            val fieldNum = tag ushr 3
            when (tag and 7) {
                0 -> {
                    var v = 0L
                    var s = 0
                    while (true) {
                        val b = buf[i].toInt() and 0xFF
                        i++
                        v = v or ((b and 0x7F).toLong() shl s)
                        if (b and 0x80 == 0) break
                        s += 7
                    }
                    out.getOrPut(fieldNum) { ArrayList() }
                        .add(Field(0, v, 0, -1, -1))
                }
                1 -> {
                    i += 8
                }
                2 -> {
                    var len = 0
                    var s = 0
                    while (true) {
                        val b = buf[i].toInt() and 0xFF
                        i++
                        len = len or ((b and 0x7F) shl s)
                        if (b and 0x80 == 0) break
                        s += 7
                    }
                    out.getOrPut(fieldNum) { ArrayList() }
                        .add(Field(2, 0, 0, i, len))
                    i += len
                }
                5 -> {
                    val bits = ByteBuffer.wrap(buf, i, 4)
                        .order(ByteOrder.LITTLE_ENDIAN).int
                    i += 4
                    out.getOrPut(fieldNum) { ArrayList() }
                        .add(Field(5, 0, bits, -1, -1))
                }
                else -> error("unsupported wire type ${tag and 7}")
            }
        }
        return out
    }

    fun parse(proto: ByteArray): Parsed {
        val top = parseFields(proto)

        val pieceEntries = top[1] ?: emptyList()
        val pieces = ArrayList<String>(pieceEntries.size)
        val scores = FloatArray(pieceEntries.size)
        for ((idx, entry) in pieceEntries.withIndex()) {
            if (entry.kind != 2) continue
            val fields = parseFields(proto, entry.bytesOff, entry.bytesOff + entry.bytesLen)
            val pieceField = fields[1]?.firstOrNull()
            val piece = if (pieceField != null) {
                String(proto, pieceField.bytesOff, pieceField.bytesLen, StandardCharsets.UTF_8)
            } else {
                ""
            }
            pieces.add(piece)
            val scoreField = fields[2]?.firstOrNull()
            if (scoreField != null && scoreField.kind == 5) {
                scores[idx] = java.lang.Float.intBitsToFloat(scoreField.floatBits)
            }
        }

        var units = IntArray(0)
        var normalized = ByteArray(0)
        val normSpec = top[3]?.firstOrNull()
        if (normSpec != null && normSpec.kind == 2) {
            val normFields = parseFields(proto, normSpec.bytesOff, normSpec.bytesOff + normSpec.bytesLen)
            val charsmap = normFields[2]?.firstOrNull()
            if (charsmap != null && charsmap.kind == 2) {
                // blob = <u32 units_len><units><normalized>
                val unitsLen = ByteBuffer.wrap(
                    proto, charsmap.bytesOff, 4,
                ).order(ByteOrder.LITTLE_ENDIAN).int
                units = IntArray(unitsLen / 4)
                ByteBuffer.wrap(proto, charsmap.bytesOff + 4, unitsLen)
                    .order(ByteOrder.LITTLE_ENDIAN)
                    .asIntBuffer()
                    .get(units)
                normalized = proto.copyOfRange(
                    charsmap.bytesOff + 4 + unitsLen,
                    charsmap.bytesOff + charsmap.bytesLen,
                )
            }
        }

        return Parsed(pieces, scores, units, normalized)
    }
}
