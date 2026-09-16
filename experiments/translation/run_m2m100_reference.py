"""m2m100 四语 ONNX 参考解码器（PC 验证）。

作用：
  1. 验证 int8 包的加载契约（ORT_ENABLE_BASIC 必须）
  2. 验证 merged decoder 的 If 分支解码循环（首步/后续步 + KV cache）
  3. 实测 greedy / beam 的 RTF 与译文质量（App 端照此转写）

输入输出契约（由包内模型图导出）：
  encoder: input_ids[1,L] i64, attention_mask[1,L] i64 -> last_hidden_state[1,L,1024]
  decoder_merged:
    入: encoder_attention_mask, input_ids[1,s], encoder_hidden_states,
        past_key_values.{0..11}.{decoder,encoder}.{key,value} (48 个 f32),
        use_cache_branch[1] bool (False=首步, True=后续步)
    出: logits[1,s,128112] + present.{0..11}... (48 个 f32)

语言 token: en=128022 zh=128102 uz=128096 ru=128077
特殊 token: bos=0 eos=2 pad=1
"""

import time
import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import sentencepiece as spm

BASE = Path(__file__).resolve().parent
PKG = BASE.parents[1] / "translation_pkg"

ENCODER = PKG / "encoder_model.onnx"
DECODER = PKG / "decoder_model_merged.onnx"
SPM = PKG / "sentencepiece.bpe.model"

LANG = {"en": 128022, "zh": 128102, "uz": 128096, "ru": 128077}

# M2M100 关键怪癖：模型词表 id 与 sentencepiece id 不一致，
# 需经 vocab.json 做 piece -> model_id 重映射。
_VOCAB_FWD = None
_VOCAB_REV = None


def piece_to_id(piece: str) -> int:
    global _VOCAB_FWD
    if _VOCAB_FWD is None:
        import json
        _VOCAB_FWD = json.load(open(PKG / "vocab.json", encoding="utf-8"))
    return _VOCAB_FWD.get(piece, 3)


def id_to_piece(mid: int) -> str:
    global _VOCAB_REV
    if _VOCAB_REV is None:
        import json
        fwd = json.load(open(PKG / "vocab.json", encoding="utf-8"))
        _VOCAB_REV = {v: k for k, v in fwd.items()}
    return _VOCAB_REV.get(mid, "<unk>")
EOS, PAD, BOS = 2, 1, 0
N_LAYERS = 12
N_HEADS = 16
HEAD_DIM = 64
VOCAB = 128112
MAX_NEW_TOKENS = 200

PAST_NAMES = []
PRESENT_NAMES = []
for layer in range(N_LAYERS):
    for kv in ("key", "value"):
        for cross in ("decoder", "encoder"):
            PAST_NAMES.append(f"past_key_values.{layer}.{cross}.{kv}")
            PRESENT_NAMES.append(f"present.{layer}.{cross}.{kv}")


def make_sessions(threads: int):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    opts.intra_op_num_threads = threads
    enc = ort.InferenceSession(str(ENCODER), sess_options=opts, providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(str(DECODER), sess_options=opts, providers=["CPUExecutionProvider"])
    return enc, dec


def encode_source(sp, text: str, src: str):
    """HF M2M100Tokenizer 约定：
    [src_lang_token] + spm_ids + [eos(2)]
    （源序列末尾必须带 </s>，缺失时编码表示错位、译文乱码）
    """
    pieces = sp.encode(text, out_type=str)
    ids = [piece_to_id(p) for p in pieces]
    input_ids = np.array([[LANG[src]] + ids + [EOS]], dtype=np.int64)
    mask = np.ones(input_ids.shape, dtype=np.int64)
    return input_ids, mask


def empty_past(batch=1):
    return {
        name: np.zeros((batch, N_HEADS, 0, HEAD_DIM), dtype=np.float32)
        for name in PAST_NAMES
    }


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
    # present.* -> past_key_values.*（同名索引顺序一致）
    present = {
        PAST_NAMES[i]: out[names.index(PRESENT_NAMES[i])]
        for i in range(len(PAST_NAMES))
    }
    return logits, present


def split_past(past):
    """拆成 decoder / encoder 两组缓存。"""
    dec = {}
    enc = {}
    for name, value in past.items():
        if ".encoder." in name:
            enc[name] = value
        else:
            dec[name] = value
    return dec, enc


def join_past(dec, enc):
    merged = {}
    merged.update(dec)
    merged.update(enc)
    return merged


def greedy(sp, enc, dec, src, tgt, text):
    input_ids, mask = encode_source(sp, text, src)
    hidden = enc.run(None, {"input_ids": input_ids, "attention_mask": mask})[0]

    t0 = time.perf_counter()
    dec_input = np.array([[EOS, LANG[tgt]]], dtype=np.int64)
    logits, past = decoder_step(dec, dec_input, mask, hidden, empty_past(), use_cache=False)

    # encoder KV 自首步起冻结；cache 分支输出的 encoder 占位张量
    # （[0,16,1,64]）不能喂回，否则 MatMul 广播报错。
    past_dec, past_enc = split_past(past)

    next_id = int(np.argmax(logits[0, -1]))
    gen = [next_id]
    while next_id != EOS and len(gen) < MAX_NEW_TOKENS:
        dec_input = np.array([[next_id]], dtype=np.int64)
        logits, past = decoder_step(
            dec,
            dec_input,
            mask,
            hidden,
            join_past(past_dec, past_enc),
            use_cache=True,
        )
        past_dec, _ = split_past(past)
        next_id = int(np.argmax(logits[0, -1]))
        if next_id != EOS:
            gen.append(next_id)
    elapsed = time.perf_counter() - t0
    return gen, elapsed


def beam_search(sp, enc, dec, src, tgt, text, num_beams=5):
    input_ids, mask = encode_source(sp, text, src)
    hidden = enc.run(None, {"input_ids": input_ids, "attention_mask": mask})[0]

    t0 = time.perf_counter()
    # 首步只跑一次（beam 共享）；encoder KV 冻结
    dec_input = np.array([[EOS, LANG[tgt]]], dtype=np.int64)
    logits, past0 = decoder_step(dec, dec_input, mask, hidden, empty_past(), use_cache=False)
    _, enc_frozen = split_past(past0)
    step_logits = logits[0, -1].astype(np.float64)
    # 阻止生成特殊 token
    for bad in (PAD, BOS):
        step_logits[bad] = -1e9
    top = np.argsort(step_logits)[::-1][:num_beams]
    dec0, _ = split_past(past0)
    beams = [
        {
            "ids": [EOS, LANG[tgt], int(t)],
            "score": float(step_logits[int(t)]),
            "past_dec": dec0,
        }
        for t in top
    ]
    done = []
    for _ in range(MAX_NEW_TOKENS):
        candidates = []
        for b in beams:
            if b["ids"][-1] == EOS:
                done.append(b)
                continue
            dec_input = np.array([[b["ids"][-1]]], dtype=np.int64)
            logits, present = decoder_step(
                dec, dec_input, mask, hidden,
                join_past(b["past_dec"], enc_frozen), use_cache=True)
            step = logits[0, -1].astype(np.float64)
            for bad in (PAD, BOS):
                step[bad] = -1e9
            topk = np.argsort(step)[::-1][:num_beams]
            for t in topk:
                new_dec, _ = split_past(present)
                candidates.append(
                    {
                        "ids": b["ids"] + [int(t)],
                        "score": b["score"] + float(step[int(t)]),
                        "past_dec": new_dec,
                    }
                )
        if not candidates:
            break
        candidates.sort(key=lambda c: -c["score"])
        beams = []
        for c in candidates:
            if c["ids"][-1] == EOS:
                done.append(c)
            else:
                beams.append(c)
            if len(beams) + len(done) >= num_beams:
                break
        done = sorted(done, key=lambda c: -c["score"])[:num_beams]
        if len(beams) == 0:
            break
    all_seqs = sorted(done + beams, key=lambda c: -c["score"])
    best = all_seqs[0]
    elapsed = time.perf_counter() - t0
    # ids = [EOS, lang, tok...]; 去掉前两个
    return best["ids"][2:], elapsed


def detok(sp, ids):
    pieces = [id_to_piece(i) for i in ids]
    return sp.decode_pieces(pieces).replace("▁", " ").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--greedy", action="store_true")
    args = ap.parse_args()

    print("loading sessions (ORT_ENABLE_BASIC)...")
    enc, dec = make_sessions(args.threads)
    sp = spm.SentencePieceProcessor(model_file=str(SPM))
    print("loaded ok")

    cases = [
        ("zh", "en", "今天天气怎么样？"),
        ("zh", "ru", "今天天气怎么样？"),
        ("zh", "uz", "今天天气怎么样？"),
        ("en", "zh", "What is the weather like today?"),
        ("ru", "zh", "Какая сегодня погода?"),
        ("uz", "zh", "Bugun ob-havo qanday?"),
    ]

    for src, tgt, text in cases:
        if args.greedy:
            ids, elapsed = greedy(sp, enc, dec, src, tgt, text)
            mode = "greedy"
        else:
            ids, elapsed = beam_search(sp, enc, dec, src, tgt, text, args.beam)
            mode = f"beam{args.beam}"
        out = detok(sp, ids)
        print(f"[{src}->{tgt}] {mode} {elapsed*1000:7.0f}ms  {text}")
        print(f"         -> {out}")


if __name__ == "__main__":
    main()
