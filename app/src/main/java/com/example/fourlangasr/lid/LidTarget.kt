package com.example.fourlangasr.lid

/**
 * VoxLingua107 的 107 类输出中，四个目标语种所在的类别索引。
 *
 * 索引顺序来自模型自带的 labels 列表（TFLite 转换时固定），
 * 并经 PC 基准 lid_exp1-3 验证：four-way 79/80 = 98.75%，
 * TFLite 与 ONNX 预测一致率 100%（见
 * experiments/lid_exp1-3 与 experiments/lid_exp4_tflite 的 summary CSV）。
 *
 * 换用任何新版 VoxLingua 模型前必须重新核对这些索引，
 * 否则会静默错位（例如把中文识别成英语）。
 */
enum class LidTarget(
    val code: String,
    val classIndex: Int,
) {
    ZH("zh", 106),
    EN("en", 20),
    RU("ru", 77),
    UZ("uz", 101),
}
