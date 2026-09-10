package com.example.fourlangasr.tts

import java.util.Locale

/**
 * 乌兹别克语 拉丁字母 -> 西里尔字母 转写（规则映射）。
 *
 * 背景：MMS 乌兹别克 TTS 模型只接受西里尔字母输入，
 * 而 App 的 uz ASR 输出拉丁字母。官方乌兹别克拉丁字母表与
 * 西里尔字母表一一对应（含 sh/ch/ng/gʻ/oʻ 双字母），
 * 纯规则转写即可，无需词典或第三方依赖。
 *
 * 已知边界（可接受）：
 * - ASR 文本通常全小写，转写按小写处理；输入中的非字母字符
 *   （空格、标点、数字）原样保留；
 * - 字母组合按最长匹配优先（先双字母后单字母）。
 */
object UzbekScript {

    /*
     * 双字母（含 ʻ/' 组合）在前，单字母在后。
     */
    private val DIGRAPHS =
        linkedMapOf(
            "sh" to "ш",
            "ch" to "ч",
            "ng" to "нг",
            "gʻ" to "ғ",
            "g'" to "ғ",
            "oʻ" to "ў",
            "o'" to "ў",
            "ye" to "е",
            "yo" to "ё",
            "ya" to "я",
            "yu" to "ю",
            "ts" to "ц",
        )

    private val SINGLE =
        mapOf(
            'a' to 'а',
            'b' to 'б',
            'd' to 'д',
            'e' to 'е',
            'f' to 'ф',
            'g' to 'г',
            'h' to 'ҳ',
            'i' to 'и',
            'j' to 'ж',
            'k' to 'к',
            'l' to 'л',
            'm' to 'м',
            'n' to 'н',
            'o' to 'о',
            'p' to 'п',
            'q' to 'қ',
            'r' to 'р',
            's' to 'с',
            't' to 'т',
            'u' to 'у',
            'v' to 'в',
            'x' to 'х',
            'y' to 'й',
            'z' to 'з',
            'c' to 'ц',
            'w' to 'в',
        )

    fun latinToCyrillic(
        text: String,
    ): String {

        val lower =
            text.lowercase(
                Locale.ROOT
            )

        val builder =
            StringBuilder(
                lower.length
            )

        var i =
            0

        while (
            i <
            lower.length
        ) {

            var matched =
                false

            for (
                (latin, cyrillic)
                in DIGRAPHS
            ) {

                if (
                    lower.startsWith(
                        latin,
                        i,
                    )
                ) {

                    builder.append(
                        cyrillic
                    )

                    i +=
                        latin.length

                    matched =
                        true

                    break
                }
            }

            if (
                matched
            ) {

                continue
            }

            val ch =
                lower[i]

            builder.append(
                SINGLE[ch]
                    ?: ch
            )

            i += 1
        }

        return builder.toString()
    }
}
