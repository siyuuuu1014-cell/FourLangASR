package com.example.fourlangasr.model

enum class AppLanguage(
    val code: String,
    val displayName: String,
) {
    ZH("zh", "中文 / Chinese"),
    EN("en", "English"),
    RU("ru", "Русский / Russian"),
    UZ("uz", "O'zbek / Uzbek");

    override fun toString(): String = displayName
}
