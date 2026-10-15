"""阶段0-e：乌兹别克撇号全链路实验（OCR → 规范化 → SentencePiece → 翻译模型）。

复用同目录上级的 PC 参考解码器 experiments/translation/run_m2m100_reference.py
（它加载的就是 App 用的同一个 int8 ONNX 包，ORT_ENABLE_BASIC + greedy，
与 KitTranslationEngine 的解码契约一致），只测"撇号写法差异对译文的影响"。

    python experiments/ocr/apostrophe_chain_test.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "translation"))

import run_m2m100_reference as ref  # noqa: E402
import sentencepiece as spm  # noqa: E402

SENT = {
    "A_官方ʻ": "Oʻzbekiston iqtisodiyoti sanoat va xizmat koʻrsatish tarmoqlariga tayanadi.",
    "B_排版‘": "O‘zbekiston iqtisodiyoti sanoat va xizmat ko‘rsatish tarmoqlariga tayanadi.",
    "C_ASCII'": "O'zbekiston iqtisodiyoti sanoat va xizmat ko'rsatish tarmoqlariga tayanadi.",
    "D_丢撇号": "Ozbekiston iqtisodiyoti sanoat va xizmat korsatish tarmoqlariga tayanadi.",
}


def main():
    enc, dec = ref.make_sessions(4)
    sp = spm.SentencePieceProcessor(model_file=str(ref.SPM))

    print("== 1) 分词差异（App 的 Kotlin 分词器与这份 .model 已逐条对拍） ==")
    for k, t in SENT.items():
        pieces = sp.encode(t, out_type=str)
        print(f"  {k:9s} pieces={len(pieces):3d} unk={pieces.count('<unk>')}  {pieces[:8]}…")

    for tgt in ("zh", "en"):
        print(f"\n== 2) uz→{tgt} greedy 译文（同一句，只有撇号写法不同） ==")
        outs = {}
        for k, t in SENT.items():
            ids, ms = ref.greedy(sp, enc, dec, "uz", tgt, t)
            outs[k] = ref.detok(sp, ids)
            print(f"  {k:9s} {ms*1000:6.0f}ms → {outs[k]}")
        base = outs["A_官方ʻ"]
        print("  与官方写法是否完全相同:",
              {k: (v.strip() == base.strip()) for k, v in outs.items()})


if __name__ == "__main__":
    main()
