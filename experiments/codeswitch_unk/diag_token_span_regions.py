"""EXP5.1 probe: token-span regions vs fixed-pad regions for backfill.

Hypothesis: the <unk> token's real audio span is [ts_unk, ts_next_token]
(the CTC decoder emits one timestamp per token start), so cropping by the
neighbouring Chinese token boundary excludes trailing Chinese syllables
that the fixed ±pad window drags in (e.g. "relax 一下" -> EN hears
"relaxy sir").

Compares, on the 30 positive ASCEND clips at the same EN verification
pipeline (whole + half windows, max-content pick, content-word gate):
  A) fixed pad 0.45 s  (current Android / EXP5 config; from existing CSVs)
  B) token-span region (computed here)

Judgments identical to diag_backfill_strategy.py: strict (token match)
and relax (len>=4 substring).
"""
import csv
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import sherpa_onnx

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]
EXP4_DIR = PROJECT_ROOT / "experiments" / "codeswitch_ascend"
MODEL_DIR = Path(r"D:\dev\projects\fourlang_translation\asr_exp1\models")
ZH_MODEL_DIR = MODEL_DIR / "sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01"
EN_MODEL_DIR = MODEL_DIR / "sherpa-onnx-zipformer-gigaspeech-2023-12-12"
TAIL_PAD_SEC = 0.66
UNK = "<unk>"
PAD_SEC = 0.45
MAX_GAP_SEC = 0.2
MIN_REGION_SAMPLES = 1600
HALF_WINDOW_SAMPLES = 8000
STOP = set(
    "a an and are as at be but by can did do does for from had has have he her his i if in into is it its "
    "like me my no not of on or our out over so some that the their them then there they this to um uh yeah "
    "ah up we were what when which who will with you your s t ve re gonna got one two".split()
)


def load_manifest():
    with open(EXP4_DIR / "ascend_natural_manifest.csv", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_audio(path):
    samples, sr = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim == 2:
        samples = samples.mean(axis=1).astype(np.float32)
    return np.ascontiguousarray(samples, dtype=np.float32)


def build_zh():
    return sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
        tokens=str(ZH_MODEL_DIR / "tokens.txt"),
        model=str(ZH_MODEL_DIR / "model.int8.onnx"),
        num_threads=2, sample_rate=16000, feature_dim=80,
        decoding_method="greedy_search", enable_endpoint_detection=False,
        provider="cpu", debug=False,
    )


def build_en():
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(EN_MODEL_DIR / "encoder-epoch-30-avg-1.int8.onnx"),
        decoder=str(EN_MODEL_DIR / "decoder-epoch-30-avg-1.int8.onnx"),
        joiner=str(EN_MODEL_DIR / "joiner-epoch-30-avg-1.int8.onnx"),
        tokens=str(EN_MODEL_DIR / "tokens.txt"),
        num_threads=2, sample_rate=16000, feature_dim=80,
        decoding_method="greedy_search", provider="cpu", debug=False,
    )


def zh_pass(rec, samples):
    s = rec.create_stream()
    s.accept_waveform(16000, samples)
    s.accept_waveform(16000, np.zeros(int(TAIL_PAD_SEC * 16000), dtype=np.float32))
    s.input_finished()
    while rec.is_ready(s):
        rec.decode_stream(s)
    res = rec.get_result_all(s)
    return list(res.tokens), list(res.timestamps)


def token_span_regions(tokens, times):
    """Each maximal run of <unk> tokens -> [ts of run start, ts after run]."""
    regions = []
    i = 0
    n = len(tokens)
    while i < n:
        if tokens[i] != UNK:
            i += 1
            continue
        start = times[i]
        j = i
        while j < n and tokens[j] == UNK:
            j += 1
        end = times[j] if j < n else times[i] + 0.35  # tail unk: small pad
        regions.append((start, min(end, times[-1] + 0.35)))
        i = j
    return regions


def pad_regions(unk_times, duration):
    spans = sorted((max(0.0, t - PAD_SEC), min(duration, t + PAD_SEC)) for t in unk_times)
    merged = []
    for st, en in spans:
        if merged and st <= merged[-1][1] + MAX_GAP_SEC:
            merged[-1][1] = max(merged[-1][1], en)
        else:
            merged.append([st, en])
    return merged


def en_verify(rec, samples, start_sec, end_sec):
    start = max(0, int(start_sec * 16000))
    end = min(samples.size, int(end_sec * 16000))
    if end - start < MIN_REGION_SAMPLES:
        return []
    cands = [(start, end)]
    if end - start >= HALF_WINDOW_SAMPLES:
        mid = (start + end) // 2
        cands = [(start, end), (start, mid), (mid, end)]
    out = []
    for cs, ce in cands:
        if ce - cs < MIN_REGION_SAMPLES:
            continue
        st = rec.create_stream()
        st.accept_waveform(16000, samples[cs:ce])
        rec.decode_streams([st])
        t = st.result.text.strip().lower()
        if t:
            out.append(t)
    return out


def content_words(text):
    return {t for t in re.findall(r"[a-z']+", text) if len(t) >= 2 and t not in STOP}


def pick(cands_texts):
    cands = [(t, content_words(t)) for t in cands_texts]
    cands = [c for c in cands if c[1]]
    if not cands:
        return ""
    return max(cands, key=lambda c: (len(c[1]), len(c[0])))[0]


def gt_words(field):
    return [w for w in re.split(r"[^A-Za-z']+", (field or "").lower()) if w]


def judge(chosen, gt, strict):
    if not chosen:
        return False
    toks = set(re.findall(r"[a-z']+", chosen))
    multi = [w for w in gt if len(w) > 1]
    single = [w for w in gt if len(w) == 1]
    if strict:
        return bool((set(multi) | set(single)) & toks)
    for w in multi:
        if w in toks:
            return True
        if len(w) >= 4:
            for t in toks:
                if w in t:
                    return True
    return bool(set(single) & toks)


def main():
    zh = build_zh()
    en = build_en()
    manifest = [r for r in load_manifest() if r["category"] == "positive_zh_en"]
    print(f"positive clips: {len(manifest)}")

    stats = {"pad": {"strict": 0, "relax": 0}, "span": {"strict": 0, "relax": 0}}
    examples = []
    for row in manifest:
        audio = load_audio(EXP4_DIR / row["file"])
        duration = audio.size / 16000.0
        gt = gt_words(row.get("english_words", ""))
        tokens, times = zh_pass(zh, audio)
        unk_times = [t for tok, t in zip(tokens, times) if tok == UNK]
        if not unk_times:
            continue

        # A) pad regions
        pa = en_verify(en, audio, *pad_regions(unk_times, duration)[0]) if pad_regions(unk_times, duration) else []
        # B) token-span regions
        spans = token_span_regions(tokens, times)
        texts_b = []
        for st, en2 in spans:
            texts_b.extend(en_verify(en, audio, st, en2))

        for name, texts in (("pad", pa), ("span", texts_b)):
            chosen = pick(texts)
            s = judge(chosen, gt, True)
            r = judge(chosen, gt, False)
            stats[name]["strict"] += s
            stats[name]["relax"] += r
            if name == "span":
                examples.append((row["id"], gt, chosen, "STRICT" if s else ("RELAX" if r else "miss")))

    print(f"{'region scheme':10s} {'clip strict':>12s} {'clip relax':>11s}")
    for name, v in stats.items():
        print(f"{name:10s} {v['strict']}/25 {v['relax']}/25")
    print("\nspan examples:")
    for e in examples:
        print(f"  id {e[0]} gt={e[1]} -> {e[2]!r} [{e[3]}]")


if __name__ == "__main__":
    main()
