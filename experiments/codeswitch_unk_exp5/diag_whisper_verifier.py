"""EXP5.2 probe: Whisper-tiny as code-switch verifier vs GigaSpeech.

Same ZH <unk> trigger pipeline (run-based regions, pad 0.6, content-word
gate, edge-filler trim) with the verifier swapped to sherpa-onnx Whisper
tiny (language=en). Reports strict/relax clip hits on the 30 ASCEND
positives plus per-region inference cost, against the GigaSpeech baseline
(12/25 strict, 18/25 relax, ~50 ms/region).

Whisper decodes fixed 30 s windows, so per-region cost is ~30 s of model
compute regardless of region length - the number that decides whether it
can ever run on a phone.
"""
import csv
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import sherpa_onnx

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]
EXP4_DIR = PROJECT_ROOT / "experiments" / "codeswitch_natural_exp4"
MODEL_DIR = PROJECT_ROOT / "experiments" / "lid_exp1-3" / "models"
ZH_MODEL_DIR = Path(r"D:\dev\projects\fourlang_translation\asr_exp1\models") / "sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01"
WHISPER_DIR = MODEL_DIR / "sherpa-onnx-whisper-tiny"
TAIL_PAD_SEC = 0.66
UNK = "<unk>"
PAD_SEC = 0.6
RUN_GAP_SEC = 0.35
MIN_REGION_SAMPLES = 1600
EDGE = set("a an and is are was were the yeah um uh oh well sir that it's your you know so just like of to in on".split())
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


def build_whisper():
    return sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=str(WHISPER_DIR / "tiny-encoder.int8.onnx"),
        decoder=str(WHISPER_DIR / "tiny-decoder.int8.onnx"),
        tokens=str(WHISPER_DIR / "tiny-tokens.txt"),
        language="en",
        task="transcribe",
        tail_paddings=0,
        num_threads=4,
        provider="cpu",
        debug=False,
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


def run_regions(unk_times, duration):
    if not unk_times:
        return []
    runs = [[unk_times[0]]]
    for t in unk_times[1:]:
        if t - runs[-1][-1] <= RUN_GAP_SEC:
            runs[-1].append(t)
        else:
            runs.append([t])
    return [(max(0.0, r[0] - PAD_SEC), min(duration, r[-1] + PAD_SEC)) for r in runs]


def trim(text):
    t = text.strip()
    changed = True
    while changed:
        changed = False
        toks = t.split(" ")
        if len(toks) >= 2 and toks[0] in EDGE:
            t = " ".join(toks[1:])
            changed = True
        elif len(toks) >= 2 and toks[-1] in EDGE:
            t = " ".join(toks[:-1])
            changed = True
    return t


def content_words(text):
    return {w for w in re.findall(r"[a-z']+", text) if len(w) >= 2 and w not in STOP}


def whisper_verify(rec, samples, start_sec, end_sec):
    start = max(0, int(start_sec * 16000))
    end = min(samples.size, int(end_sec * 16000))
    if end - start < MIN_REGION_SAMPLES:
        return []
    t0 = time.perf_counter()
    s = rec.create_stream()
    s.accept_waveform(16000, samples[start:end])
    rec.decode_streams([s])
    text = s.result.text.strip().lower()
    cost = time.perf_counter() - t0
    cleaned = trim(text)
    return cleaned, cost


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
    whisper = build_whisper()
    manifest = [r for r in load_manifest() if r["category"] == "positive_zh_en"]
    print(f"whisper verifier: {WHISPER_DIR}")

    st = {"strict": 0, "relax": 0, "total": 0, "regions": 0}
    cost_sum = 0.0
    examples = []
    for row in manifest:
        audio = load_audio(EXP4_DIR / row["file"])
        dur = audio.size / 16000.0
        gt = gt_words(row.get("english_words", ""))
        tokens, times = zh_pass(zh, audio)
        unk = [t for tok, t in zip(tokens, times) if tok == UNK]
        if not unk:
            continue
        st["total"] += 1
        chosen_region_texts = []
        for s, e in run_regions(unk, dur):
            cleaned, cost = whisper_verify(whisper, audio, s, e)
            cost_sum += cost
            st["regions"] += 1
            if content_words(cleaned):
                chosen_region_texts.append(cleaned)
        chosen = chosen_region_texts[0] if chosen_region_texts else ""
        if len(chosen_region_texts) > 1:
            chosen = max(chosen_region_texts, key=lambda t: (len(content_words(t)), len(t)))
        hs = judge(chosen, gt, True)
        hr = judge(chosen, gt, False)
        st["strict"] += hs
        st["relax"] += hr
        examples.append((row["id"], gt, chosen, "STRICT" if hs else ("RELAX" if hr else "miss")))

    print(f"whisper-tiny : clip strict {st['strict']}/{st['total']}  relax {st['relax']}/{st['total']}")
    print(f"giggaspeech  : clip strict 12/25  relax 18/25   (baseline)")
    print(f"regions={st['regions']}  mean whisper cost {cost_sum/max(st['regions'],1):.2f} s/region")
    print("\nexamples (chosen backfill):")
    for e in examples:
        print(f"  id {e[0]} gt={e[1]} -> {e[2]!r} [{e[3]}]")


if __name__ == "__main__":
    main()
