"""EXP6 diagnostic: dump per-token confidence for the known swallow cases.

Verifies the assumption that English words folded into Chinese syllables
(e.g. topic -> "他配", baseline -> "斯兰") show low token confidence in the
ZH CTC output, unlike ordinary high-confidence Chinese tokens.

Usage: python diag_token_confidence.py <manifest_id> [more ids...]
"""

import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import sherpa_onnx

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]
EXP4_DIR = PROJECT_ROOT / "experiments" / "codeswitch_ascend"
MODEL_DIR = Path(r"D:\dev\projects\fourlang_translation\asr_exp1\models")
ZH_MODEL_DIR = MODEL_DIR / "sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01"
TAIL_PAD_SEC = 0.66
UNK = "<unk>"

# ids with swallowed words, plus a couple of clean Chinese controls.
DEFAULT_IDS = ["00023", "00158", "00011", "00004", "00006"]


def load_manifest():
    with open(EXP4_DIR / "ascend_natural_manifest.csv", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_audio(path: Path):
    samples, sr = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim == 2:
        samples = samples.mean(axis=1).astype(np.float32)
    return np.ascontiguousarray(samples, dtype=np.float32)


def make_recognizer():
    return sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
        tokens=str(ZH_MODEL_DIR / "tokens.txt"),
        model=str(ZH_MODEL_DIR / "model.int8.onnx"),
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        enable_endpoint_detection=False,
        provider="cpu",
        debug=False,
    )


def dump(recognizer, samples, reference):
    stream = recognizer.create_stream()
    stream.accept_waveform(16000, samples)
    stream.accept_waveform(16000, np.zeros(int(TAIL_PAD_SEC * 16000), dtype=np.float32))
    stream.input_finished()
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)

    result = recognizer.get_result_all(stream)
    tokens = list(result.tokens)
    times = list(result.timestamps)
    ys = list(result.ys_probs)

    print(f"ref : {reference}")
    print(f"text: {result.text.strip()}")

    if not (len(tokens) == len(ys)):
        print(f"  !! len mismatch tokens={len(tokens)} ys={len(ys)}")
        return
    if len(times) != len(tokens):
        print(f"  !! len mismatch tokens={len(tokens)} times={len(times)} (keep tokens)")
        times = [float("nan")] * len(tokens)

    for tok, ts, lp in zip(tokens, times, ys):
        prob = float(np.exp(lp))
        mark = "  <-- LOW" if prob < 0.9 else ""
        print(f"  t={ts:6.2f}s p={prob:6.3f} {tok}{mark}")


def main():
    ids = sys.argv[1:] or DEFAULT_IDS
    manifest = {row["id"]: row for row in load_manifest()}
    recognizer = make_recognizer()

    for clip_id in ids:
        row = manifest.get(clip_id)
        if not row:
            print(f"id {clip_id} not in manifest")
            continue
        audio = load_audio(EXP4_DIR / row["file"])
        print("=" * 70)
        print(f"id {clip_id} ({row['category']}) {row['file']}")
        dump(recognizer, audio, row.get("transcription", ""))


if __name__ == "__main__":
    main()
