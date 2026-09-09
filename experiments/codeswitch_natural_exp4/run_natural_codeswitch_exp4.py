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
EXP_REL = Path("experiments") / "codeswitch_natural_exp4"

MODEL_REL = (
    Path("experiments")
    / "lid_exp4_tflite"
    / "converted_tf"
    / "voxlingua107_float32.tflite"
)

MANIFEST_NAME = "ascend_natural_manifest.csv"
UTTERANCE_RESULTS_NAME = "natural_exp4_utterance_results.csv"
SUMMARY_NAME = "natural_exp4_summary.csv"

WINDOW_SEC = 1.25
HOP_SEC = 0.25
MODEL_INPUT_SAMPLES = 80000

LANGUAGES = ["zh", "en", "ru", "uz"]

TARGET_INDEX = {
    "en": 20,
    "ru": 77,
    "uz": 101,
    "zh": 106,
}

FOREIGN_THRESHOLDS = [
    (0.40, 0.05),
    (0.45, 0.10),
    (0.50, 0.15),
    (0.55, 0.20),
    (0.60, 0.25),
]

SANDWICH_RULES = [
    {
        "name": "relaxed",
        "main_prob": 0.45,
        "main_adv": 0.05,
        "edge_gap_sec": 0.30,
        "max_island_sec": 1.80,
    },
    {
        "name": "balanced",
        "main_prob": 0.50,
        "main_adv": 0.10,
        "edge_gap_sec": 0.40,
        "max_island_sec": 1.70,
    },
    {
        "name": "strict",
        "main_prob": 0.55,
        "main_adv": 0.15,
        "edge_gap_sec": 0.50,
        "max_island_sec": 1.60,
    },
]

MIN_FOREIGN_WINDOWS = 2
MAX_CENTER_GAP_SEC = 0.40
REJECT_MULTI_FOREIGN = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Natural Chinese-English code-switch benchmark using "
            "VoxLingua107 TFLite + MAIN-FOREIGN-MAIN sandwich logic."
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
        help="TFLite CPU threads. Default: 4",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Debug only. 0 = use all manifest rows.",
    )

    return parser.parse_args()


def read_manifest(path: Path) -> List[dict]:
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

    x = np.asarray(x, dtype=np.float32)

    if x.ndim == 2:
        x = x.mean(axis=1).astype(np.float32)

    if sr != SAMPLE_RATE:
        raise RuntimeError(
            f"{path}: expected {SAMPLE_RATE} Hz, got {sr}"
        )

    return np.ascontiguousarray(
        x,
        dtype=np.float32,
    )


def softmax(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)


def create_interpreter(
    model_path: Path,
    threads: int,
) -> Interpreter:
    interpreter = Interpreter(
        model_path=str(model_path),
        num_threads=max(1, threads),
    )
    interpreter.allocate_tensors()

    input_detail = interpreter.get_input_details()[0]
    shape = tuple(
        int(v)
        for v in input_detail["shape"]
    )

    expected = (
        1,
        MODEL_INPUT_SAMPLES,
    )

    if shape != expected:
        raise RuntimeError(
            f"Unexpected TFLite input shape {shape}; "
            f"expected {expected}"
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
    x = make_model_input(local_audio)

    input_detail = interpreter.get_input_details()[0]

    interpreter.set_tensor(
        input_detail["index"],
        x[None, :].astype(
            input_detail["dtype"],
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
            "Could not locate 107-class logits."
        )

    selected = np.asarray(
        [
            logits[
                TARGET_INDEX[language]
            ]
            for language in LANGUAGES
        ],
        dtype=np.float64,
    )

    p = softmax(selected)

    return {
        language: float(p[i])
        for i, language in enumerate(
            LANGUAGES
        )
    }


def iter_windows(audio: np.ndarray):
    window_samples = int(
        WINDOW_SEC * SAMPLE_RATE
    )
    hop_samples = int(
        HOP_SEC * SAMPLE_RATE
    )

    n = len(audio)

    if n <= window_samples:
        starts = [0]
    else:
        starts = list(
            range(
                0,
                n - window_samples + 1,
                hop_samples,
            )
        )

        final_start = (
            n - window_samples
        )

        if starts[-1] != final_start:
            starts.append(final_start)

    for start in starts:
        end = min(
            n,
            start + window_samples,
        )

        local = audio[
            start:end
        ]

        if len(local) < window_samples:
            padded = np.zeros(
                window_samples,
                dtype=np.float32,
            )

            left = (
                window_samples -
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
                window_samples,
                n,
            ) / 2.0
        ) / SAMPLE_RATE

        yield (
            start / SAMPLE_RATE,
            end / SAMPLE_RATE,
            center_sec,
            local,
        )


def make_window_rows(
    interpreter: Interpreter,
    audio: np.ndarray,
) -> List[dict]:
    rows = []

    for (
        start_sec,
        end_sec,
        center_sec,
        local,
    ) in iter_windows(audio):
        probs = infer_probs(
            interpreter,
            local,
        )

        rows.append(
            {
                "start_sec": start_sec,
                "end_sec": end_sec,
                "center_sec": center_sec,
                "p_zh": probs["zh"],
                "p_en": probs["en"],
                "p_ru": probs["ru"],
                "p_uz": probs["uz"],
            }
        )

    return rows


def best_non_main(
    row: dict,
    main_lang: str,
) -> Tuple[
    str,
    float,
    float,
    float,
]:
    foreign_languages = [
        language
        for language in LANGUAGES
        if language != main_lang
    ]

    foreign_lang = max(
        foreign_languages,
        key=lambda language:
            row[
                f"p_{language}"
            ],
    )

    p_foreign = (
        row[
            f"p_{foreign_lang}"
        ]
    )

    p_main = (
        row[
            f"p_{main_lang}"
        ]
    )

    evidence = (
        p_foreign -
        p_main
    )

    return (
        foreign_lang,
        p_foreign,
        p_main,
        evidence,
    )


def main_strength(
    row: dict,
    main_lang: str,
) -> Tuple[
    float,
    float,
]:
    p_main = (
        row[
            f"p_{main_lang}"
        ]
    )

    best_other = max(
        row[
            f"p_{language}"
        ]
        for language in LANGUAGES
        if language != main_lang
    )

    return (
        p_main,
        p_main -
        best_other,
    )


def build_foreign_runs(
    rows: List[dict],
    main_lang: str,
    min_foreign_prob: float,
    min_evidence: float,
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

    for item in accepted:
        if not current:
            current = [item]
            continue

        same_language = (
            item["lang"] ==
            current[-1]["lang"]
        )

        gap = (
            item["row"]["center_sec"] -
            current[-1]["row"]["center_sec"]
        )

        if (
            same_language
            and
            gap <=
            MAX_CENTER_GAP_SEC
        ):
            current.append(item)
        else:
            runs.append(current)
            current = [item]

    if current:
        runs.append(current)

    candidates = []

    for run in runs:
        if (
            len(run) <
            MIN_FOREIGN_WINDOWS
        ):
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
            float(
                centers.min()
            ) -
            HOP_SEC / 2.0
        )

        end_sec = (
            float(
                centers.max()
            ) +
            HOP_SEC / 2.0
        )

        mean_probs = {
            language:
                float(
                    np.mean(
                        [
                            item["row"][
                                f"p_{language}"
                            ]
                            for item in run
                        ]
                    )
                )
            for language in LANGUAGES
        }

        foreign_ranking = sorted(
            [
                language
                for language in LANGUAGES
                if language != main_lang
            ],
            key=lambda language:
                mean_probs[language],
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
                "mean_evidence":
                    float(
                        evidences.mean()
                    ),
                "score":
                    float(
                        evidences.mean() *
                        math.sqrt(
                            len(run)
                        )
                    ),
                "foreign_top1":
                    foreign_ranking[0],
                "foreign_top2":
                    foreign_ranking[1],
                "p_top1":
                    mean_probs[
                        foreign_ranking[0]
                    ],
                "p_top2":
                    mean_probs[
                        foreign_ranking[1]
                    ],
            }
        )

    return candidates


def has_stable_main_before(
    rows: List[dict],
    main_lang: str,
    candidate: dict,
    rule: dict,
) -> bool:
    before = [
        row
        for row in rows
        if (
            row["center_sec"] <
            candidate["start_sec"]
        )
    ]

    before.sort(
        key=lambda row:
            row["center_sec"],
        reverse=True,
    )

    for row in before[:3]:
        (
            p_main,
            advantage,
        ) = main_strength(
            row,
            main_lang,
        )

        if (
            p_main >=
            rule["main_prob"]
            and
            advantage >=
            rule["main_adv"]
        ):
            return True

    return False


def has_stable_main_after(
    rows: List[dict],
    main_lang: str,
    candidate: dict,
    rule: dict,
) -> bool:
    after = [
        row
        for row in rows
        if (
            row["center_sec"] >
            candidate["end_sec"]
        )
    ]

    after.sort(
        key=lambda row:
            row["center_sec"]
    )

    for row in after[:3]:
        (
            p_main,
            advantage,
        ) = main_strength(
            row,
            main_lang,
        )

        if (
            p_main >=
            rule["main_prob"]
            and
            advantage >=
            rule["main_adv"]
        ):
            return True

    return False


def sandwich_accept(
    utterance_duration: float,
    rows: List[dict],
    main_lang: str,
    candidate: dict,
    rule: dict,
) -> bool:
    if (
        candidate["duration_sec"] >
        rule["max_island_sec"]
    ):
        return False

    edge_gap = (
        rule["edge_gap_sec"]
    )

    if (
        candidate["start_sec"] <
        edge_gap
    ):
        return False

    if (
        utterance_duration -
        candidate["end_sec"] <
        edge_gap
    ):
        return False

    if not has_stable_main_before(
        rows,
        main_lang,
        candidate,
        rule,
    ):
        return False

    if not has_stable_main_after(
        rows,
        main_lang,
        candidate,
        rule,
    ):
        return False

    return True


def choose_candidate(
    utterance_duration: float,
    rows: List[dict],
    main_lang: str,
    min_foreign_prob: float,
    min_evidence: float,
    rule: dict,
) -> Optional[dict]:
    candidates = build_foreign_runs(
        rows,
        main_lang,
        min_foreign_prob,
        min_evidence,
    )

    accepted = [
        candidate
        for candidate in candidates
        if sandwich_accept(
            utterance_duration,
            rows,
            main_lang,
            candidate,
            rule,
        )
    ]

    if not accepted:
        return None

    if REJECT_MULTI_FOREIGN:
        detected_languages = {
            candidate["lang"]
            for candidate in accepted
        }

        if len(detected_languages) > 1:
            return None

    accepted.sort(
        key=lambda candidate: (
            candidate["score"],
            candidate["support"],
            candidate["p_top1"],
        ),
        reverse=True,
    )

    return accepted[0]


def category_semantics(
    category: str,
) -> Tuple[
    str,
    str,
]:
    if category == "positive_zh_en":
        return (
            "zh",
            "en",
        )

    if category == "negative_zh":
        return (
            "zh",
            "",
        )

    if category == "negative_en":
        return (
            "en",
            "",
        )

    raise ValueError(
        f"Unexpected manifest category: {category}"
    )


def main() -> None:
    args = parse_args()

    project_root = (
        args.project_root
        .resolve()
    )

    base_dir = (
        project_root /
        EXP_REL
    )

    manifest_path = (
        base_dir /
        MANIFEST_NAME
    )

    model_path = (
        project_root /
        MODEL_REL
    )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing ASCEND manifest:\n{manifest_path}\n\n"
            "Run prepare_ascend_natural_exp4.py first."
        )

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Missing tested TFLite model:\n{model_path}"
        )

    manifest = read_manifest(
        manifest_path
    )

    if args.limit > 0:
        manifest = manifest[:args.limit]

    counts = defaultdict(int)

    for item in manifest:
        counts[item["category"]] += 1

    print("=" * 100)
    print("NATURAL CODE-SWITCH EXP4")
    print("=" * 100)
    print(f"Manifest : {manifest_path}")
    print(f"Model    : {model_path}")
    print(f"Threads  : {args.threads}")
    print(f"Window   : {WINDOW_SEC:.2f}s")
    print(f"Hop      : {HOP_SEC:.2f}s")
    print()
    print("Dataset counts:")

    for key in [
        "positive_zh_en",
        "negative_zh",
        "negative_en",
    ]:
        print(
            f"  {key:15s}: "
            f"{counts[key]}"
        )

    print()
    print(
        "NOTE: this ASCEND subset has transcript-level labels but "
        "no word-level timestamps."
    )
    print(
        "Natural EXP4 therefore scores detection, Top-1/Top-2 and "
        "false-island rate; not boundary MAE/IoU."
    )
    print()

    interpreter = create_interpreter(
        model_path,
        args.threads,
    )

    sample_cache = {}
    total_windows = 0

    print("=" * 100)
    print("ONE-TIME WINDOW INFERENCE")
    print("=" * 100)

    for index, item in enumerate(
        manifest,
        start=1,
    ):
        category = item["category"]

        (
            main_lang,
            expected_foreign,
        ) = category_semantics(
            category
        )

        audio_path = (
            base_dir /
            Path(item["file"])
        )

        if not audio_path.is_file():
            raise FileNotFoundError(
                audio_path
            )

        audio = load_audio(
            audio_path
        )

        duration_sec = (
            len(audio) /
            SAMPLE_RATE
        )

        rows = make_window_rows(
            interpreter,
            audio,
        )

        total_windows += len(rows)

        sample_key = (
            f"{item['split']}:"
            f"{item['id']}"
        )

        sample_cache[sample_key] = {
            "item": item,
            "category": category,
            "main_lang": main_lang,
            "expected_foreign":
                expected_foreign,
            "duration_sec":
                duration_sec,
            "rows": rows,
        }

        print(
            f"{index:02d}/{len(manifest):02d} | "
            f"{category:14s} | "
            f"{duration_sec:4.2f}s | "
            f"{len(rows):2d} windows | "
            f"{item['transcription']}"
        )

    print()
    print(
        f"Total TFLite windows: "
        f"{total_windows}"
    )
    print()

    utterance_results = []
    summary_rows = []

    print("=" * 100)
    print("NATURAL SANDWICH SWEEP")
    print("=" * 100)

    for rule in SANDWICH_RULES:
        for (
            min_foreign_prob,
            min_evidence,
        ) in FOREIGN_THRESHOLDS:

            pos_total = 0
            pos_detected = 0
            pos_top1 = 0
            pos_top2 = 0
            pos_wrong_top1 = 0
            pos_miss = 0

            neg_total = 0
            neg_false = 0

            neg_zh_total = 0
            neg_zh_false = 0

            neg_en_total = 0
            neg_en_false = 0

            for sample_key, payload in sample_cache.items():
                category = (
                    payload["category"]
                )

                candidate = choose_candidate(
                    payload["duration_sec"],
                    payload["rows"],
                    payload["main_lang"],
                    min_foreign_prob,
                    min_evidence,
                    rule,
                )

                detected = (
                    candidate
                    is not None
                )

                if category == "positive_zh_en":
                    pos_total += 1

                    if detected:
                        pos_detected += 1

                        if (
                            candidate["foreign_top1"]
                            == "en"
                        ):
                            pos_top1 += 1
                        else:
                            pos_wrong_top1 += 1

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

                utterance_results.append(
                    {
                        "rule":
                            rule["name"],
                        "min_foreign_prob":
                            min_foreign_prob,
                        "min_evidence":
                            min_evidence,
                        "sample_key":
                            sample_key,
                        "category":
                            category,
                        "main_lang":
                            payload["main_lang"],
                        "expected_foreign":
                            payload["expected_foreign"],
                        "transcription":
                            payload["item"]["transcription"],
                        "english_words":
                            payload["item"].get(
                                "english_words",
                                "",
                            ),
                        "detected":
                            detected,
                        "detected_lang":
                            (
                                ""
                                if not detected
                                else candidate["lang"]
                            ),
                        "foreign_top1":
                            (
                                ""
                                if not detected
                                else candidate["foreign_top1"]
                            ),
                        "foreign_top2":
                            (
                                ""
                                if not detected
                                else candidate["foreign_top2"]
                            ),
                        "support":
                            (
                                ""
                                if not detected
                                else candidate["support"]
                            ),
                        "pred_start_sec":
                            (
                                ""
                                if not detected
                                else candidate["start_sec"]
                            ),
                        "pred_end_sec":
                            (
                                ""
                                if not detected
                                else candidate["end_sec"]
                            ),
                    }
                )

            detected_rate = (
                pos_detected / pos_total
                if pos_total
                else 0.0
            )

            top1_rate = (
                pos_top1 / pos_total
                if pos_total
                else 0.0
            )

            top2_rate = (
                pos_top2 / pos_total
                if pos_total
                else 0.0
            )

            wrong_top1_rate = (
                pos_wrong_top1 / pos_total
                if pos_total
                else 0.0
            )

            miss_rate = (
                pos_miss / pos_total
                if pos_total
                else 0.0
            )

            false_island_rate = (
                neg_false / neg_total
                if neg_total
                else 0.0
            )

            false_zh_rate = (
                neg_zh_false / neg_zh_total
                if neg_zh_total
                else 0.0
            )

            false_en_rate = (
                neg_en_false / neg_en_total
                if neg_en_total
                else 0.0
            )

            summary = {
                "rule": rule["name"],
                "main_prob": rule["main_prob"],
                "main_adv": rule["main_adv"],
                "edge_gap_sec":
                    rule["edge_gap_sec"],
                "max_island_sec":
                    rule["max_island_sec"],
                "min_foreign_prob":
                    min_foreign_prob,
                "min_evidence":
                    min_evidence,
                "positive_samples":
                    pos_total,
                "detected_rate":
                    detected_rate,
                "foreign_top1_rate":
                    top1_rate,
                "foreign_top2_rate":
                    top2_rate,
                "wrong_top1_rate":
                    wrong_top1_rate,
                "miss_rate":
                    miss_rate,
                "negative_samples":
                    neg_total,
                "false_island_rate":
                    false_island_rate,
                "false_island_zh_rate":
                    false_zh_rate,
                "false_island_en_rate":
                    false_en_rate,
            }

            summary_rows.append(summary)

            print(
                f"{rule['name']:8s} | "
                f"Pforeign>={min_foreign_prob:.2f} "
                f"evidence>={min_evidence:.2f} | "
                f"detect={detected_rate*100:6.2f}% | "
                f"Top1EN={top1_rate*100:6.2f}% | "
                f"Top2EN={top2_rate*100:6.2f}% | "
                f"wrong1={wrong_top1_rate*100:5.2f}% | "
                f"falseIsland={false_island_rate*100:5.2f}%"
            )

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

    def recommendation_key(row: dict):
        return (
            row["foreign_top2_rate"],
            row["foreign_top1_rate"],
            row["detected_rate"],
            -row["false_island_rate"],
            -row["wrong_top1_rate"],
        )

    best = max(
        pool,
        key=recommendation_key,
    )

    utterance_path = (
        base_dir /
        UTTERANCE_RESULTS_NAME
    )

    summary_path = (
        base_dir /
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
                utterance_results[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            utterance_results
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

    print()
    print("=" * 100)
    print("RECOMMENDED CONFIG")
    print("=" * 100)

    for key in [
        "rule",
        "main_prob",
        "main_adv",
        "edge_gap_sec",
        "max_island_sec",
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
        elif isinstance(value, float):
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
    print("Natural EXP4 decision targets:")
    print("  false_island_rate <= 5%")
    print("  foreign_top2_rate >= 80-85%")
    print("  foreign_top1_rate: observe, not hard-gate yet")
    print()
    print(
        "If the real foreign language is in Top-2, "
        "the next stage can run both candidate ASRs on only "
        "the small island and let ASR quality decide."
    )
    print()
    print(
        f"Utterance results: "
        f"{utterance_path}"
    )
    print(
        f"Summary          : "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()
