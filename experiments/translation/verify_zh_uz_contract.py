"""zh_uz SMaLL-100 解码契约对拍：在 PC 上按不同契约跑同一批句子，看哪种输出正确。

契约候选：
  A) M2M100 式：encoder=[src_lang]+X+[eos]，decoder 首步=[eos,tgt_lang]
  B) SMALL100 式：encoder=[tgt_lang]+X+[eos]，decoder 首步=[eos]
  C) 混合：     encoder=[tgt_lang]+X+[eos]，decoder 首步=[eos,tgt_lang]
"""
import time
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import sentencepiece as spm

PKG = Path(__file__).resolve().parent / "zh_uz_pkg"
ENCODER = PKG / "encoder_model.onnx"
DECODER = PKG / "decoder_model_merged.onnx"
SPM = PKG / "sentencepiece.bpe.model"
VOCAB_JSON = PKG / "vocab.json"

LANG = {"en": 128022, "zh": 128102, "uz": 128096, "ru": 128077}
EOS, PAD, BOS = 2, 1, 0

_vocab = json.load(open(VOCAB_JSON, encoding="utf-8"))
_piece_to_id = _vocab
_id_to_piece = {v: k for k, v in _vocab.items()}

# 结构参数（config.json）
N_DECODER_LAYERS = 3
N_HEADS = 16
HEAD_DIM = 64
VOCAB = 128112

PAST_NAMES = []
PRESENT_NAMES = []
for layer in range(N_DECODER_LAYERS):
    for kv in ("key", "value"):
        for cross in ("decoder", "encoder"):
            PAST_NAMES.append(f"past_key_values.{layer}.{cross}.{kv}")
            PRESENT_NAMES.append(f"present.{layer}.{cross}.{kv}")


def make_sessions():
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    opts.intra_op_num_threads = 4
    enc = ort.InferenceSession(str(ENCODER), sess_options=opts, providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(str(DECODER), sess_options=opts, providers=["CPUExecutionProvider"])
    return enc, dec


def encode(sp, text, prefix_lang):
    pieces = sp.encode(text, out_type=str)
    ids = [_piece_to_id.get(p, 3) for p in pieces]
    return np.array([[LANG[prefix_lang]] + ids + [EOS]], dtype=np.int64)


def decoder_step(dec, input_ids, enc_mask, enc_hidden, past, use_cache):
    feeds = {
        "encoder_attention_mask": enc_mask,
        "input_ids": input_ids,
        "encoder_hidden_states": enc_hidden,
        "use_cache_branch": np.array([use_cache], dtype=np.bool_),
    }
    for name in PAST_NAMES:
        feeds[name] = past[name]
    out = dec.run(None, feeds)
    names = [o.name for o in dec.get_outputs()]
    logits = out[names.index("logits")]
    present = {PAST_NAMES[i]: out[names.index(PRESENT_NAMES[i])] for i in range(len(PAST_NAMES))}
    return logits, present


def split_past(past):
    dec = {k: v for k, v in past.items() if ".encoder." not in k}
    enc = {k: v for k, v in past.items() if ".encoder." in k}
    return dec, enc


def run(enc, dec, sp, src, tgt, text, encoder_prefix_lang, first_step_tokens):
    input_ids = encode(sp, text, encoder_prefix_lang)
    mask = np.ones(input_ids.shape, dtype=np.int64)
    hidden = enc.run(None, {"input_ids": input_ids, "attention_mask": mask})[0]

    empty = {n: np.zeros((1, N_HEADS, 0, HEAD_DIM), dtype=np.float32) for n in PAST_NAMES}
    logits, past = decoder_step(dec, np.array([first_step_tokens], dtype=np.int64), mask, hidden, empty, False)
    past_dec, past_enc = split_past(past)
    next_id = int(np.argmax(logits[0, -1]))
    gen = [next_id]
    while next_id != EOS and len(gen) < 128:
        logits, past = decoder_step(dec, np.array([[next_id]], dtype=np.int64), mask, hidden,
                                    {**past_dec, **past_enc}, True)
        past_dec, _ = split_past(past)
        next_id = int(np.argmax(logits[0, -1]))
        if next_id != EOS:
            gen.append(next_id)
    pieces = [_id_to_piece.get(i, "<unk>") for i in gen]
    return sp.decode_pieces(pieces).replace("▁", " ").strip()


def main():
    print("loading sessions...")
    enc, dec = make_sessions()
    sp = spm.SentencePieceProcessor(model_file=str(SPM))
    print("ok")

    cases = [
        ("uz", "zh", "Bugun ob-havo qanday?"),
        ("uz", "zh", "Salom, tanishganimdan xursandman."),
        ("uz", "zh", "Bu telefon qancha turadi?"),
        ("zh", "uz", "今天天气怎么样？"),
        ("zh", "uz", "你好，很高兴认识你。"),
    ]

    variants = {
        "A m2m100式 (src前缀 + [eos,tgt]首步)": ("src", [EOS, "tgt"]),
        "B small100式 (tgt前缀 + [eos]首步)": ("tgt", [EOS]),
        "C 混合 (tgt前缀 + [eos,tgt]首步)": ("tgt", [EOS, "tgt"]),
    }

    for name, (prefix_mode, first_mode) in variants.items():
        print(f"\n===== {name} =====")
        for src, tgt, text in cases:
            prefix_lang = src if prefix_mode == "src" else tgt
            first = [EOS]
            if len(first_mode) == 2:
                first = [EOS, LANG[tgt]]
            t0 = time.perf_counter()
            out = run(enc, dec, sp, src, tgt, text, prefix_lang, first)
            ms = (time.perf_counter() - t0) * 1000
            print(f"  [{src}->{tgt}] {text!r}")
            print(f"       -> {out!r}  ({ms:.0f}ms)")


if __name__ == "__main__":
    main()


def beam_run(enc, dec, sp, src, tgt, text, num_beams):
    input_ids = encode(sp, text, tgt)
    mask = np.ones(input_ids.shape, dtype=np.int64)
    hidden = enc.run(None, {"input_ids": input_ids, "attention_mask": mask})[0]

    empty = {n: np.zeros((1, N_HEADS, 0, HEAD_DIM), dtype=np.float32) for n in PAST_NAMES}
    logits, past = decoder_step(dec, np.array([[EOS]], dtype=np.int64), mask, hidden, empty, False)
    _, past_enc = split_past(past)
    dec0, _ = split_past(past)
    step = logits[0, -1].astype(np.float64)
    step[PAD] = -1e9
    step[BOS] = -1e9
    top = np.argsort(step)[::-1][:num_beams]
    beams = [{"ids": [EOS, int(t)], "score": float(step[int(t)]), "past": dec0} for t in top]
    done = []
    for _ in range(128):
        candidates = []
        for b in beams:
            if b["ids"][-1] == EOS:
                done.append(b)
                continue
            logits, past = decoder_step(dec, np.array([[b["ids"][-1]]], dtype=np.int64), mask, hidden,
                                        {**b["past"], **past_enc}, True)
            nd, _ = split_past(past)
            step = logits[0, -1].astype(np.float64)
            step[PAD] = -1e9
            step[BOS] = -1e9
            for t in np.argsort(step)[::-1][:num_beams]:
                candidates.append({"ids": b["ids"] + [int(t)],
                                   "score": b["score"] + float(step[int(t)]),
                                   "past": nd})
        if not candidates:
            break
        candidates.sort(key=lambda c: -c["score"])
        beams, done = [], sorted(done, key=lambda c: -c["score"])[:num_beams]
        for c in candidates:
            if c["ids"][-1] == EOS:
                done.append(c)
            else:
                beams.append(c)
            if len(beams) + len(done) >= num_beams:
                break
        if not beams:
            break
    best = sorted(done + beams, key=lambda c: -c["score"])[0]
    pieces = [_id_to_piece.get(i, "<unk>") for i in best["ids"][2:]]
    return sp.decode_pieces(pieces).replace("▁", " ").strip()


if __name__ == "__main__" and "--beam" in __import__("sys").argv:
    print("loading sessions...")
    enc, dec = make_sessions()
    sp = spm.SentencePieceProcessor(model_file=str(SPM))
    cases = [
        ("uz", "zh", "Bugun ob-havo qanday?"),
        ("uz", "zh", "Salom, tanishganimdan xursandman."),
        ("uz", "zh", "Bu telefon qancha turadi?"),
        ("uz", "zh", "Yig'ilish ertaga ertalab soat to'qqizda boshlanadi."),
        ("zh", "uz", "今天天气怎么样？"),
        ("zh", "uz", "你好，很高兴认识你。"),
        ("zh", "uz", "会议将在明天上午九点开始。"),
    ]
    for nb in (3, 5):
        print(f"\n===== beam{nb} =====")
        for src, tgt, text in cases:
            t0 = __import__("time").perf_counter()
            out = beam_run(enc, dec, sp, src, tgt, text, nb)
            ms = (__import__("time").perf_counter() - t0) * 1000
            print(f"  [{src}->{tgt}] {text!r}")
            print(f"       -> {out!r}  ({ms:.0f}ms)")
