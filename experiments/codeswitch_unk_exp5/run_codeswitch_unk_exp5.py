"""
Code-switch UNK-Trigger EXP5.

Route: main-language ZH CTC ASR over the whole utterance
    -> <unk> token timestamps as trigger
    -> crop local region (ts +/- pad, merged)
    -> EN candidate ASR verifies the region
    -> hit a ground-truth English word == detected code-switch.

Dataset and models: see README.txt in this folder.
"""

from pathlib import Path
import argparse
import csv
import re
import time

import numpy as np
import soundfile as sf
import sherpa_onnx

BASE_DIR = Path(__file__).resolve().parent

PROJECT_ROOT_DEFAULT = Path(r"D:\dev\projects\FourLangAsrMobile")
PC_MODELS_DEFAULT = Path(r"D:\dev\projects\fourlang_translation\asr_exp1\models")

ZH_MODEL_NAME = "sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01"
EN_MODEL_NAME = "sherpa-onnx-zipformer-gigaspeech-2023-12-12"

EXP4_DIR_NAME = "codeswitch_natural_exp4"
MANIFEST_CSV = "ascend_natural_manifest.csv"

# Same flush tail as the Android ChineseSherpaEngine (0.66 s of silence).
TAIL_PAD_SEC = 0.66

UNK_TOKEN = "<unk>"

UTTERANCES_CSV = BASE_DIR / "unk_exp5_utterances.csv"
REGIONS_CSV = BASE_DIR / "unk_exp5_regions.csv"
SUMMARY_CSV = BASE_DIR / "unk_exp5_summary.csv"


def load_manifest(exp4_dir: Path):
    rows = []
    with open(exp4_dir / MANIFEST_CSV, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def load_audio(path: Path):
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim == 2:
        samples = samples.mean(axis=1).astype(np.float32)
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    if sample_rate != 16000:
        raise RuntimeError(f"{path}: expected 16000 Hz, got {sample_rate}")
    return samples


def build_zh_recognizer(zh_model_dir: Path):
    """Same model/config as the Android ChineseSherpaEngine and PC EXP4-A."""
    return sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
        tokens=str(zh_model_dir / "tokens.txt"),
        model=str(zh_model_dir / "model.int8.onnx"),
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        enable_endpoint_detection=False,
        provider="cpu",
        debug=False,
    )


def build_en_recognizer(en_model_dir: Path):
    """Same model/config as the Android EnglishSherpaEngine and PC EXP4-B."""
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(en_model_dir / "encoder-epoch-30-avg-1.int8.onnx"),
        decoder=str(en_model_dir / "decoder-epoch-30-avg-1.int8.onnx"),
        joiner=str(en_model_dir / "joiner-epoch-30-avg-1.int8.onnx"),
        tokens=str(en_model_dir / "tokens.txt"),
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        provider="cpu",
        debug=False,
    )


def run_zh_pass(recognizer, samples):
    """Whole-utterance streaming CTC decode; returns (text, unk_times_sec)."""
    stream = recognizer.create_stream()
    stream.accept_waveform(16000, samples)
    stream.accept_waveform(16000, np.zeros(int(TAIL_PAD_SEC * 16000), dtype=np.float32))
    stream.input_finished()
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    result = recognizer.get_result_all(stream)

    text = result.text.strip()

    unk_times = []
    times = list(getattr(result, "timestamps", []) or [])
    tokens = list(getattr(result, "tokens", []) or [])
    if times and len(tokens) == len(times):
        # Older builds report frame indices instead of seconds.
        if times and max(times) > samples.size / 16000.0 + 2.0:
            times = [t * 0.01 for t in times]
        for token, ts in zip(tokens, times):
            if token == UNK_TOKEN:
                unk_times.append(float(ts))

    return text, unk_times


def run_en_pass(recognizer, samples, start_sec, end_sec):
    """Decode one audio region; returns (texts, [text per candidate window]).

    The region is decoded whole AND split into overlapping sub-windows
    (left / center / right). A code-switched word surrounded by Chinese is
    often only recovered from one of these crops, so any candidate hit
    counts. Region < 0.5 s is decoded once only.
    """
    start = max(0, int(start_sec * 16000))
    end = min(samples.size, int(end_sec * 16000))
    if end - start < 1600:  # <0.1 s, nothing to verify
        return []

    candidates = [(start, end)]
    if end - start >= int(0.5 * 16000):
        mid = (start + end) // 2
        candidates = [
            (start, end),
            (start, mid),
            (mid, end),
        ]

    texts = []
    for c_start, c_end in candidates:
        if c_end - c_start < 1600:
            continue
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, samples[c_start:c_end])
        recognizer.decode_streams([stream])
        text = stream.result.text.strip()
        if text:
            texts.append(text)
    return texts


def merge_regions(unk_times, pad_sec, duration_sec, max_gap_sec=0.2):
    """Center a window on each <unk>; merge overlaps and near-neighbours."""
    spans = []
    for ts in unk_times:
        spans.append((max(0.0, ts - pad_sec), min(duration_sec, ts + pad_sec)))
    if not spans:
        return []
    spans.sort()
    merged = [list(spans[0])]
    for start, end in spans[1:]:
        if start <= merged[-1][1] + max_gap_sec:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def en_words(text):
    return set(re.findall(r"[a-z']+", text.lower()))


def word_hit(en_texts, gt_words):
    """Match ground-truth words against EN candidate outputs.

    len(gt) >= 2: substring containment either way, so "smart phone" in the
        ground truth is recovered from EN "SMARTPHONE" and vice versa
        (different BPE/word segmentation granularity).
    len(gt) == 1 (e.g. letter acronyms "C R P"): exact token match only,
        because substring matching on single letters is meaningless.
    """
    gt = [w for w in gt_words if w]
    if not gt:
        return set(), False
    single = {w for w in gt if len(w) == 1}
    multi = {w for w in gt if len(w) > 1}

    hit = set()
    for text in en_texts:
        flat = re.sub(r"[^a-z]", "", text.lower())
        for w in multi:
            if w in flat:
                hit.add(w)
        toks = en_words(text)
        for w in single:
            if w in toks:
                hit.add(w)
    return hit, bool(hit)


def gt_words(field):
    return [w for w in re.split(r"[^A-Za-z']+", (field or "").lower()) if w]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT_DEFAULT)
    parser.add_argument("--pc-models", type=Path, default=PC_MODELS_DEFAULT)
    parser.add_argument("--pads", default="0.30,0.45,0.60")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    pads = [float(p) for p in args.pads.split(",")]

    exp4_dir = args.project_root / "experiments" / EXP4_DIR_NAME
    zh_model_dir = args.pc_models / ZH_MODEL_NAME
    en_model_dir = args.pc_models / EN_MODEL_NAME

    for path in (zh_model_dir / "model.int8.onnx", en_model_dir / "encoder-epoch-30-avg-1.int8.onnx"):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = load_manifest(exp4_dir)
    if args.limit:
        manifest = manifest[: args.limit]

    print("=" * 80)
    print("Code-switch UNK-Trigger EXP5")
    print("=" * 80)
    print(f"ZH model : {zh_model_dir}")
    print(f"EN model : {en_model_dir}")
    print(f"Clips    : {len(manifest)}   pads: {pads}")

    zh_recognizer = build_zh_recognizer(zh_model_dir)
    en_recognizer = build_en_recognizer(en_model_dir)

    # ---- Stage 1: one cached ZH pass per clip --------------------------
    zh_passes = []
    zh_sec_total = 0.0
    for row in manifest:
        audio_path = exp4_dir / row["file"]
        samples = load_audio(audio_path)
        duration = samples.size / 16000.0

        t0 = time.perf_counter()
        text, unk_times = run_zh_pass(zh_recognizer, samples)
        zh_sec_total += time.perf_counter() - t0

        zh_passes.append(
            {
                "row": row,
                "samples": samples,
                "duration": duration,
                "zh_text": text,
                "unk_times": unk_times,
            }
        )

    unk_counts = {
        cat: sum(
            1
            for p in zh_passes
            if p["row"]["category"] == cat and p["unk_times"]
        )
        for cat in ("positive_zh_en", "negative_zh", "negative_en")
    }
    print("\n[Stage 1] ZH CTC pass done. clips with >=1 <unk>:")
    for cat, count in unk_counts.items():
        total = sum(1 for p in zh_passes if p["row"]["category"] == cat)
        print(f"  {cat:16s}: {count}/{total}")

    # ---- Stage 2: pad sweep over regions + EN verification -------------
    utterance_rows = []
    region_rows = []
    summary_rows = []
    en_sec_total = 0.0

    for pad in pads:
        stats = {
            cat: {
                "total": 0,
                "triggered": 0,
                "detected": 0,
                "gt_word_total": 0,
                "gt_word_hit": 0,
                "en_word_false": 0,
                "regions": 0,
            }
            for cat in ("positive_zh_en", "negative_zh", "negative_en")
        }

        for p in zh_passes:
            row = p["row"]
            cat = row["category"]
            words_gt = gt_words(row.get("english_words", ""))
            regions = merge_regions(p["unk_times"], pad, p["duration"])

            en_texts_all = []
            hit = set()
            false_en_word = False

            for start, end in regions:
                t0 = time.perf_counter()
                texts = run_en_pass(en_recognizer, p["samples"], start, end)
                en_sec_total += time.perf_counter() - t0
                en_texts_all.extend(texts)

                reg_hit, _ = word_hit(texts, words_gt)
                hit |= reg_hit
                if cat == "negative_zh" and any(en_words(t) for t in texts):
                    false_en_word = True

                region_rows.append(
                    {
                        "pad_sec": pad,
                        "category": cat,
                        "id": row["id"],
                        "region_start_sec": round(start, 3),
                        "region_end_sec": round(end, 3),
                        "en_text": " | ".join(texts),
                    }
                )

            s = stats[cat]
            s["total"] += 1
            s["regions"] += len(regions)
            if regions:
                s["triggered"] += 1
            if hit:
                s["detected"] += 1
            s["gt_word_total"] += len(set(words_gt))
            s["gt_word_hit"] += len(hit)
            if cat == "negative_zh" and false_en_word:
                s["en_word_false"] += 1

            utterance_rows.append(
                {
                    "pad_sec": pad,
                    "category": cat,
                    "id": row["id"],
                    "transcription": row.get("transcription", ""),
                    "zh_text": p["zh_text"],
                    "unk_count": len(p["unk_times"]),
                    "region_count": len(regions),
                    "gt_words": " ".join(sorted(set(words_gt))),
                    "recovered_words": " ".join(sorted(hit)),
                    "detected": int(bool(hit)),
                    "en_texts": " | ".join(en_texts_all),
                }
            )

        pos = stats["positive_zh_en"]
        neg = stats["negative_zh"]
        summary_rows.append(
            {
                "pad_sec": pad,
                "positive_samples": pos["total"],
                "detected_rate": round(pos["detected"] / pos["total"], 4) if pos["total"] else 0.0,
                "word_recovery_rate": (
                    round(pos["gt_word_hit"] / pos["gt_word_total"], 4) if pos["gt_word_total"] else 0.0
                ),
                "regions_per_positive": (
                    round(pos["regions"] / pos["total"], 3) if pos["total"] else 0.0
                ),
                "negative_zh_samples": neg["total"],
                "false_island_rate": (
                    round(neg["triggered"] / neg["total"], 4) if neg["total"] else 0.0
                ),
                "en_word_false_rate": (
                    round(neg["en_word_false"] / neg["total"], 4) if neg["total"] else 0.0
                ),
                "zh_pass_sec": round(zh_sec_total, 2),
                "en_pass_sec": round(en_sec_total, 2),
            }
        )

    write_csv(UTTERANCES_CSV, utterance_rows)
    write_csv(REGIONS_CSV, region_rows)
    write_csv(SUMMARY_CSV, summary_rows)

    print("\n[Stage 2] pad sweep summary")
    for r in summary_rows:
        print(
            f"  pad={r['pad_sec']:.2f}  detected={r['detected_rate']:.1%}  "
            f"wordRecovery={r['word_recovery_rate']:.1%}  "
            f"falseIsland={r['false_island_rate']:.1%}  "
            f"enWordFalse={r['en_word_false_rate']:.1%}"
        )

    print(f"\nZH pass total {zh_sec_total:.1f}s, EN verify total {en_sec_total:.1f}s")
    print("\nWrote:")
    for path in (UTTERANCES_CSV, REGIONS_CSV, SUMMARY_CSV):
        print(f"  {path}")
    print("=" * 80)
    print("RECOMMENDED CONFIG: pick the pad with detected >= 0.70 and the")
    print("lowest false_island_rate; verify against the pass criteria in README.txt")


def write_csv(path: Path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
