from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from ai_edge_litert.interpreter import Interpreter


PROJECT_ROOT_DEFAULT = Path(r"D:\dev\projects\FourLangAsrMobile")

SAMPLE_RATE = 16000

EXP_DIR_REL = (
    Path("experiments")
    / "codeswitch_ascend"
)

MODEL_REL = (
    Path("experiments")
    / "lid"
    / "converted_tf"
    / "voxlingua107_float32.tflite"
)

MANIFEST_NAME = "ascend_natural_manifest.csv"

WINDOW_RESULTS_NAME = (
    "shortwin_windows.csv"
)

SUMMARY_NAME = (
    "shortwin_summary.csv"
)

UTTERANCE_NAME = (
    "shortwin_utterances.csv"
)

MODEL_INPUT_SAMPLES = 80000

LANGUAGES = [
    "zh",
    "en",
    "ru",
    "uz",
]

TARGET_INDEX = {
    "en": 20,
    "ru": 77,
    "uz": 101,
    "zh": 106,
}

# Keep the sweep small and interpretable.
# Each configuration is inferred only once.
WINDOW_CONFIGS = [
    # name, window_sec, hop_sec
    ("w050_h0125", 0.50, 0.125),
    ("w075_h0125", 0.75, 0.125),
    ("w100_h0250", 1.00, 0.250),
    ("w125_h0250", 1.25, 0.250),  # old baseline
]

# Short-window thresholds.
THRESHOLDS = [
    # min foreign probability, min (Pforeign - Pmain)
    (0.40, 0.00),
    (0.45, 0.05),
    (0.50, 0.10),
    (0.55, 0.15),
    (0.60, 0.20),
]

# We test both:
#   support=1 -> sensitive to a single very short foreign word
#   support=2 -> more conservative
SUPPORT_OPTIONS = [
    1,
    2,
]

# A candidate still needs main-language evidence somewhere before and after it,
# but much less strict than the previous 1.25-second sandwich detector.
MAIN_CONTEXT_MIN_PROB = 0.45
MAIN_CONTEXT_MIN_ADV = 0.02

# Do not accept an island exactly at utterance start/end.
EDGE_GAP_SEC = 0.15

# Product scope = a short inserted word or two words.
MAX_ISLAND_SEC = 1.60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Natural EXP4 short-window sweep for detecting "
            "one or two foreign words inside a main-language utterance."
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT_DEFAULT,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 = all rows; >0 = debug subset.",
    )

    return parser.parse_args()


def read_csv(path: Path) -> List[dict]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def load_audio(path: Path) -> np.ndarray:
    x, sr = sf.read(
        path,
        dtype="float32",
        always_2d=False,
    )

    x = np.asarray(
        x,
        dtype=np.float32,
    )

    if x.ndim == 2:
        x = x.mean(axis=1).astype(np.float32)

    if sr != SAMPLE_RATE:
        raise RuntimeError(
            f"{path}: expected 16000 Hz, got {sr}"
        )

    return np.ascontiguousarray(
        x,
        dtype=np.float32,
    )


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(
        x,
        dtype=np.float64,
    )
    x = x - np.max(x)
    y = np.exp(x)
    return y / np.sum(y)


def create_interpreter(
    model_path: Path,
    threads: int,
) -> Interpreter:
    interpreter = Interpreter(
        model_path=str(model_path),
        num_threads=max(1, threads),
    )
    interpreter.allocate_tensors()

    inp = interpreter.get_input_details()[0]
    shape = tuple(
        int(v)
        for v in inp["shape"]
    )

    if shape != (
        1,
        MODEL_INPUT_SAMPLES,
    ):
        raise RuntimeError(
            f"Unexpected TFLite input shape: {shape}"
        )

    return interpreter


def make_model_input(
    local_audio: np.ndarray,
) -> np.ndarray:
    x = np.asarray(
        local_audio,
        dtype=np.float32,
    )

    if len(x) > MODEL_INPUT_SAMPLES:
        start = (
            len(x) -
            MODEL_INPUT_SAMPLES
        ) // 2

        return np.ascontiguousarray(
            x[
                start:
                start + MODEL_INPUT_SAMPLES
            ],
            dtype=np.float32,
        )

    if len(x) == MODEL_INPUT_SAMPLES:
        return np.ascontiguousarray(
            x,
            dtype=np.float32,
        )

    out = np.zeros(
        MODEL_INPUT_SAMPLES,
        dtype=np.float32,
    )

    left = (
        MODEL_INPUT_SAMPLES -
        len(x)
    ) // 2

    out[
        left:
        left + len(x)
    ] = x

    return out


def infer_probs(
    interpreter: Interpreter,
    local_audio: np.ndarray,
) -> Dict[str, float]:
    x = make_model_input(
        local_audio
    )

    inp = interpreter.get_input_details()[0]

    interpreter.set_tensor(
        inp["index"],
        x[None, :].astype(
            inp["dtype"],
            copy=False,
        ),
    )

    interpreter.invoke()

    logits = None

    for detail in interpreter.get_output_details():
        y = interpreter.get_tensor(
            detail["index"]
        )
        flat = np.asarray(y).reshape(-1)

        if flat.size == 107:
            logits = flat.astype(
                np.float32,
                copy=False,
            )
            break

    if logits is None:
        raise RuntimeError(
            "107-class logits output not found."
        )

    four = np.asarray(
        [
            logits[TARGET_INDEX[lang]]
            for lang in LANGUAGES
        ],
        dtype=np.float64,
    )

    p = softmax(four)

    return {
        lang: float(p[i])
        for i, lang in enumerate(LANGUAGES)
    }


def iter_windows(
    audio: np.ndarray,
    window_sec: float,
    hop_sec: float,
):
    window_n = int(
        round(
            window_sec *
            SAMPLE_RATE
        )
    )

    hop_n = int(
        round(
            hop_sec *
            SAMPLE_RATE
        )
    )

    n = len(audio)

    if n <= window_n:
        starts = [0]
    else:
        starts = list(
            range(
                0,
                n - window_n + 1,
                hop_n,
            )
        )

        final_start = n - window_n

        if starts[-1] != final_start:
            starts.append(final_start)

    for start in starts:
        end = min(
            n,
            start + window_n,
        )

        local = audio[start:end]

        if len(local) < window_n:
            padded = np.zeros(
                window_n,
                dtype=np.float32,
            )

            left = (
                window_n -
                len(local)
            ) // 2

            padded[
                left:
                left + len(local)
            ] = local

            local = padded

        center_sec = (
            start +
            min(
                window_n,
                n,
            ) / 2.0
        ) / SAMPLE_RATE

        yield (
            start / SAMPLE_RATE,
            end / SAMPLE_RATE,
            center_sec,
            local,
        )


def category_info(
    category: str,
) -> Tuple[str, str]:
    if category == "positive_zh_en":
        return "zh", "en"

    if category == "negative_zh":
        return "zh", ""

    if category == "negative_en":
        return "en", ""

    raise ValueError(
        f"Unexpected category: {category}"
    )


def best_non_main(
    row: dict,
    main_lang: str,
) -> Tuple[
    str,
    float,
    float,
    float,
]:
    other = [
        lang
        for lang in LANGUAGES
        if lang != main_lang
    ]

    best = max(
        other,
        key=lambda lang:
            row[f"p_{lang}"],
    )

    p_foreign = row[f"p_{best}"]
    p_main = row[f"p_{main_lang}"]

    return (
        best,
        p_foreign,
        p_main,
        p_foreign - p_main,
    )


def main_is_stable(
    row: dict,
    main_lang: str,
) -> bool:
    p_main = row[f"p_{main_lang}"]

    best_other = max(
        row[f"p_{lang}"]
        for lang in LANGUAGES
        if lang != main_lang
    )

    return (
        p_main >= MAIN_CONTEXT_MIN_PROB
        and
        p_main - best_other >=
        MAIN_CONTEXT_MIN_ADV
    )


def build_candidate_runs(
    rows: List[dict],
    main_lang: str,
    min_foreign_prob: float,
    min_evidence: float,
    support: int,
    hop_sec: float,
) -> List[dict]:
    accepted = []

    for row in rows:
        (
            foreign_lang,
            p_foreign,
            p_main,
            evidence,
        ) = best_non_main(
            row,
            main_lang,
        )

        if (
            p_foreign <
            min_foreign_prob
            or
            evidence <
            min_evidence
        ):
            continue

        accepted.append(
            {
                "row": row,
                "lang": foreign_lang,
                "p_foreign": p_foreign,
                "p_main": p_main,
                "evidence": evidence,
            }
        )

    runs = []
    current = []

    max_gap = (
        hop_sec *
        1.6
    )

    for item in accepted:
        if not current:
            current = [item]
            continue

        same_lang = (
            item["lang"] ==
            current[-1]["lang"]
        )

        gap = (
            item["row"]["center_sec"] -
            current[-1]["row"]["center_sec"]
        )

        if (
            same_lang
            and
            gap <= max_gap
        ):
            current.append(item)
        else:
            runs.append(current)
            current = [item]

    if current:
        runs.append(current)

    candidates = []

    for run in runs:
        if len(run) < support:
            continue

        centers = np.asarray(
            [
                item["row"]["center_sec"]
                for item in run
            ],
            dtype=np.float64,
        )

        evidences = np.asarray(
            [
                item["evidence"]
                for item in run
            ],
            dtype=np.float64,
        )

        weights = np.maximum(
            evidences,
            0.001,
        )

        center = float(
            np.average(
                centers,
                weights=weights,
            )
        )

        start_sec = (
            float(centers.min()) -
            hop_sec / 2.0
        )

        end_sec = (
            float(centers.max()) +
            hop_sec / 2.0
        )

        mean_probs = {
            lang:
                float(
                    np.mean(
                        [
                            item["row"][f"p_{lang}"]
                            for item in run
                        ]
                    )
                )
            for lang in LANGUAGES
        }

        foreign_rank = sorted(
            [
                lang
                for lang in LANGUAGES
                if lang != main_lang
            ],
            key=lambda lang:
                mean_probs[lang],
            reverse=True,
        )

        candidates.append(
            {
                "lang": run[0]["lang"],
                "support": len(run),
                "center_sec": center,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec":
                    end_sec - start_sec,
                "score":
                    float(
                        evidences.mean() *
                        math.sqrt(
                            len(run)
                        )
                    ),
                "foreign_top1":
                    foreign_rank[0],
                "foreign_top2":
                    foreign_rank[1],
                "p_top1":
                    mean_probs[
                        foreign_rank[0]
                    ],
                "p_top2":
                    mean_probs[
                        foreign_rank[1]
                    ],
            }
        )

    return candidates


def has_main_before(
    rows: List[dict],
    main_lang: str,
    candidate: dict,
) -> bool:
    candidates = [
        row
        for row in rows
        if (
            row["center_sec"] <
            candidate["start_sec"]
        )
    ]

    candidates.sort(
        key=lambda row:
            row["center_sec"],
        reverse=True,
    )

    return any(
        main_is_stable(
            row,
            main_lang,
        )
        for row in candidates[:4]
    )


def has_main_after(
    rows: List[dict],
    main_lang: str,
    candidate: dict,
) -> bool:
    candidates = [
        row
        for row in rows
        if (
            row["center_sec"] >
            candidate["end_sec"]
        )
    ]

    candidates.sort(
        key=lambda row:
            row["center_sec"]
    )

    return any(
        main_is_stable(
            row,
            main_lang,
        )
        for row in candidates[:4]
    )


def choose_candidate(
    rows: List[dict],
    duration_sec: float,
    main_lang: str,
    min_foreign_prob: float,
    min_evidence: float,
    support: int,
    hop_sec: float,
) -> Optional[dict]:
    runs = build_candidate_runs(
        rows,
        main_lang,
        min_foreign_prob,
        min_evidence,
        support,
        hop_sec,
    )

    accepted = []

    for candidate in runs:
        if (
            candidate["duration_sec"] >
            MAX_ISLAND_SEC
        ):
            continue

        if (
            candidate["start_sec"] <
            EDGE_GAP_SEC
        ):
            continue

        if (
            duration_sec -
            candidate["end_sec"] <
            EDGE_GAP_SEC
        ):
            continue

        if not has_main_before(
            rows,
            main_lang,
            candidate,
        ):
            continue

        if not has_main_after(
            rows,
            main_lang,
            candidate,
        ):
            continue

        accepted.append(candidate)

    if not accepted:
        return None

    accepted.sort(
        key=lambda item: (
            item["score"],
            item["support"],
            item["p_top1"],
        ),
        reverse=True,
    )

    return accepted[0]


def main() -> None:
    args = parse_args()

    project_root = (
        args.project_root
        .resolve()
    )

    exp_dir = (
        project_root /
        EXP_DIR_REL
    )

    manifest_path = (
        exp_dir /
        MANIFEST_NAME
    )

    model_path = (
        project_root /
        MODEL_REL
    )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing manifest: {manifest_path}"
        )

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Missing TFLite model: {model_path}"
        )

    manifest = read_csv(
        manifest_path
    )

    if args.limit > 0:
        manifest = manifest[
            :args.limit
        ]

    interpreter = create_interpreter(
        model_path,
        args.threads,
    )

    print("=" * 100)
    print("NATURAL EXP4 - SHORT WINDOW SWEEP")
    print("=" * 100)
    print(f"Samples : {len(manifest)}")
    print(f"Threads : {args.threads}")
    print()
    print("Window configs:")

    for name, win, hop in WINDOW_CONFIGS:
        print(
            f"  {name}: "
            f"window={win:.3f}s "
            f"hop={hop:.3f}s"
        )

    print()
    print(
        "Each window configuration is inferred only once; "
        "all thresholds/support options reuse cached probabilities."
    )
    print()

    cache = {}
    all_window_rows = []

    # ============================================================
    # One inference pass per window configuration
    # ============================================================
    for (
        config_name,
        window_sec,
        hop_sec,
    ) in WINDOW_CONFIGS:
        print()
        print("=" * 100)
        print(
            f"INFERENCE: {config_name}"
        )
        print("=" * 100)

        for index, item in enumerate(
            manifest,
            start=1,
        ):
            (
                main_lang,
                expected_foreign,
            ) = category_info(
                item["category"]
            )

            audio_path = (
                exp_dir /
                Path(item["file"])
            )

            audio = load_audio(
                audio_path
            )

            rows = []

            for (
                start_sec,
                end_sec,
                center_sec,
                local,
            ) in iter_windows(
                audio,
                window_sec,
                hop_sec,
            ):
                probs = infer_probs(
                    interpreter,
                    local,
                )

                row = {
                    "config":
                        config_name,
                    "window_sec":
                        window_sec,
                    "hop_sec":
                        hop_sec,
                    "sample_key":
                        (
                            f"{item['split']}:"
                            f"{item['id']}"
                        ),
                    "category":
                        item["category"],
                    "main_lang":
                        main_lang,
                    "expected_foreign":
                        expected_foreign,
                    "start_sec":
                        start_sec,
                    "end_sec":
                        end_sec,
                    "center_sec":
                        center_sec,
                    "p_zh":
                        probs["zh"],
                    "p_en":
                        probs["en"],
                    "p_ru":
                        probs["ru"],
                    "p_uz":
                        probs["uz"],
                }

                rows.append(row)
                all_window_rows.append(
                    row
                )

            sample_key = (
                f"{item['split']}:"
                f"{item['id']}"
            )

            cache[
                (
                    config_name,
                    sample_key,
                )
            ] = {
                "rows":
                    rows,
                "duration_sec":
                    len(audio) /
                    SAMPLE_RATE,
                "item":
                    item,
                "main_lang":
                    main_lang,
                "expected_foreign":
                    expected_foreign,
            }

            print(
                f"{index:02d}/{len(manifest):02d} | "
                f"{item['category']:14s} | "
                f"{len(rows):2d} windows | "
                f"{item['transcription']}"
            )

    # Save raw windows once.
    window_path = (
        exp_dir /
        WINDOW_RESULTS_NAME
    )

    with window_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                all_window_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            all_window_rows
        )

    summary_rows = []
    utterance_rows = []

    print()
    print("=" * 100)
    print("SHORT-WINDOW SWEEP")
    print("=" * 100)

    for (
        config_name,
        window_sec,
        hop_sec,
    ) in WINDOW_CONFIGS:

        for support in SUPPORT_OPTIONS:

            for (
                min_foreign_prob,
                min_evidence,
            ) in THRESHOLDS:

                pos_total = 0
                pos_detect = 0
                pos_top1 = 0
                pos_top2 = 0
                pos_wrong = 0
                pos_miss = 0

                neg_total = 0
                neg_false = 0
                neg_zh_total = 0
                neg_zh_false = 0
                neg_en_total = 0
                neg_en_false = 0

                for item in manifest:
                    sample_key = (
                        f"{item['split']}:"
                        f"{item['id']}"
                    )

                    payload = cache[
                        (
                            config_name,
                            sample_key,
                        )
                    ]

                    candidate = choose_candidate(
                        payload["rows"],
                        payload["duration_sec"],
                        payload["main_lang"],
                        min_foreign_prob,
                        min_evidence,
                        support,
                        hop_sec,
                    )

                    detected = (
                        candidate is not None
                    )

                    category = item["category"]

                    if category == "positive_zh_en":
                        pos_total += 1

                        if detected:
                            pos_detect += 1

                            if (
                                candidate["foreign_top1"]
                                == "en"
                            ):
                                pos_top1 += 1
                            else:
                                pos_wrong += 1

                            if (
                                "en"
                                in {
                                    candidate["foreign_top1"],
                                    candidate["foreign_top2"],
                                }
                            ):
                                pos_top2 += 1
                        else:
                            pos_miss += 1

                    else:
                        neg_total += 1

                        if detected:
                            neg_false += 1

                        if category == "negative_zh":
                            neg_zh_total += 1

                            if detected:
                                neg_zh_false += 1

                        elif category == "negative_en":
                            neg_en_total += 1

                            if detected:
                                neg_en_false += 1

                    utterance_rows.append(
                        {
                            "config":
                                config_name,
                            "window_sec":
                                window_sec,
                            "hop_sec":
                                hop_sec,
                            "support":
                                support,
                            "min_foreign_prob":
                                min_foreign_prob,
                            "min_evidence":
                                min_evidence,
                            "sample_key":
                                sample_key,
                            "category":
                                category,
                            "transcription":
                                item["transcription"],
                            "english_words":
                                item.get(
                                    "english_words",
                                    "",
                                ),
                            "detected":
                                detected,
                            "foreign_top1":
                                (
                                    ""
                                    if not detected
                                    else
                                    candidate["foreign_top1"]
                                ),
                            "foreign_top2":
                                (
                                    ""
                                    if not detected
                                    else
                                    candidate["foreign_top2"]
                                ),
                            "support_found":
                                (
                                    ""
                                    if not detected
                                    else
                                    candidate["support"]
                                ),
                            "pred_start_sec":
                                (
                                    ""
                                    if not detected
                                    else
                                    candidate["start_sec"]
                                ),
                            "pred_end_sec":
                                (
                                    ""
                                    if not detected
                                    else
                                    candidate["end_sec"]
                                ),
                        }
                    )

                detect_rate = (
                    pos_detect /
                    pos_total
                )

                top1_rate = (
                    pos_top1 /
                    pos_total
                )

                top2_rate = (
                    pos_top2 /
                    pos_total
                )

                wrong_rate = (
                    pos_wrong /
                    pos_total
                )

                miss_rate = (
                    pos_miss /
                    pos_total
                )

                false_rate = (
                    neg_false /
                    neg_total
                )

                false_zh = (
                    neg_zh_false /
                    neg_zh_total
                    if neg_zh_total
                    else 0.0
                )

                false_en = (
                    neg_en_false /
                    neg_en_total
                    if neg_en_total
                    else 0.0
                )

                summary = {
                    "config":
                        config_name,
                    "window_sec":
                        window_sec,
                    "hop_sec":
                        hop_sec,
                    "support":
                        support,
                    "min_foreign_prob":
                        min_foreign_prob,
                    "min_evidence":
                        min_evidence,
                    "positive_samples":
                        pos_total,
                    "detected_rate":
                        detect_rate,
                    "foreign_top1_rate":
                        top1_rate,
                    "foreign_top2_rate":
                        top2_rate,
                    "wrong_top1_rate":
                        wrong_rate,
                    "miss_rate":
                        miss_rate,
                    "negative_samples":
                        neg_total,
                    "false_island_rate":
                        false_rate,
                    "false_island_zh_rate":
                        false_zh,
                    "false_island_en_rate":
                        false_en,
                }

                summary_rows.append(
                    summary
                )

                print(
                    f"{config_name:12s} | "
                    f"support={support} | "
                    f"Pforeign>={min_foreign_prob:.2f} "
                    f"evidence>={min_evidence:.2f} | "
                    f"detect={detect_rate*100:6.2f}% | "
                    f"Top1EN={top1_rate*100:6.2f}% | "
                    f"Top2EN={top2_rate*100:6.2f}% | "
                    f"wrong={wrong_rate*100:5.2f}% | "
                    f"falseIsland={false_rate*100:5.2f}%"
                )

    utterance_path = (
        exp_dir /
        UTTERANCE_NAME
    )

    summary_path = (
        exp_dir /
        SUMMARY_NAME
    )

    with utterance_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                utterance_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            utterance_rows
        )

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                summary_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    # Safety-first recommendation:
    safe = [
        row
        for row in summary_rows
        if (
            row["false_island_rate"] <= 0.05
            and
            row["wrong_top1_rate"] <= 0.10
        )
    ]

    pool = (
        safe
        if safe
        else summary_rows
    )

    def score(row: dict):
        return (
            row["foreign_top2_rate"],
            row["detected_rate"],
            row["foreign_top1_rate"],
            -row["false_island_rate"],
            -row["wrong_top1_rate"],
        )

    best = max(
        pool,
        key=score,
    )

    print()
    print("=" * 100)
    print("RECOMMENDED SHORT-WINDOW CONFIG")
    print("=" * 100)

    for key in [
        "config",
        "window_sec",
        "hop_sec",
        "support",
        "min_foreign_prob",
        "min_evidence",
        "detected_rate",
        "foreign_top1_rate",
        "foreign_top2_rate",
        "wrong_top1_rate",
        "miss_rate",
        "false_island_rate",
        "false_island_zh_rate",
        "false_island_en_rate",
    ]:
        value = best[key]

        if "rate" in key:
            print(
                f"{key:24s}: "
                f"{value*100:.2f}%"
            )
        elif isinstance(
            value,
            float,
        ):
            print(
                f"{key:24s}: "
                f"{value:.4f}"
            )
        else:
            print(
                f"{key:24s}: "
                f"{value}"
            )

    print()
    print("Decision:")
    print("  target detect / Top-2 >= 70-80%")
    print("  target falseIsland <= 5%")
    print()
    print(
        "If NO configuration reaches both targets, "
        "stop using VoxLingua as the primary detector for single-word "
        "code-switch and pivot to ASR-anomaly-triggered verification."
    )
    print()
    print(f"Windows   : {window_path}")
    print(f"Utterance : {utterance_path}")
    print(f"Summary   : {summary_path}")


if __name__ == "__main__":
    main()
