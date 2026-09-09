from pathlib import Path
import csv
import math
import time
from collections import defaultdict

import numpy as np
import soundfile as sf
from ai_edge_litert.interpreter import Interpreter


PROJECT_ROOT = Path(
    r"D:\dev\projects\FourLangAsrMobile"
)

EXP1_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "codeswitch_island_exp1"
)

BASE_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "codeswitch_island_exp2"
)

MODEL = (
    PROJECT_ROOT
    / "experiments"
    / "lid_exp4_tflite"
    / "converted_tf"
    / "voxlingua107_float32.tflite"
)

POS_DIR = (
    EXP1_DIR /
    "island_data"
)

POS_MANIFEST = (
    EXP1_DIR /
    "island_manifest.csv"
)

NEG_DIR = (
    BASE_DIR /
    "negative_data"
)

NEG_MANIFEST = (
    BASE_DIR /
    "negative_manifest.csv"
)

WINDOW_RESULTS = (
    BASE_DIR /
    "island_exp2_window_results.csv"
)

UTTERANCE_RESULTS = (
    BASE_DIR /
    "island_exp2_utterance_results.csv"
)

SUMMARY_CSV = (
    BASE_DIR /
    "island_exp2_summary.csv"
)

SAMPLE_RATE = 16000

# Product-oriented two-stage detector:
#
# Stage A: sparse coarse scan across the whole utterance.
# Stage B: only around a coarse foreign candidate, run a local fine scan.
#
# This is much cheaper than EXP1's dense 0.25 s hop over the entire utterance.

COARSE_WINDOW_SEC = 1.25
COARSE_HOP_SEC = 0.75

FINE_WINDOW_SEC = 1.25
FINE_HOP_SEC = 0.25
REFINE_RADIUS_SEC = 0.75

MODEL_WINDOW_SAMPLES = 80000

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

# Sweep probability-evidence thresholds.
#
# foreign_prob = P(best non-main language)
# evidence     = foreign_prob - P(main)
#
# Unlike EXP1, we do NOT discard all information into a hard top-1 label.
THRESHOLDS = [
    (0.40, 0.05),
    (0.45, 0.10),
    (0.50, 0.15),
    (0.55, 0.20),
    (0.60, 0.25),
]

# Final island must contain at least two adjacent fine windows
# supporting the same foreign language.
MIN_FINE_SUPPORT = 2


def load_audio(
    path: Path,
):
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
        x = (
            x.mean(axis=1)
            .astype(np.float32)
        )

    if sr != SAMPLE_RATE:
        raise RuntimeError(
            f"{path}: expected "
            f"{SAMPLE_RATE} Hz, got {sr}"
        )

    return np.ascontiguousarray(
        x,
        dtype=np.float32,
    )


def softmax(
    x,
):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    x = x - np.max(x)

    y = np.exp(x)

    return y / np.sum(y)


def classify_4way(
    logits,
):
    selected = np.asarray(
        [
            logits[
                TARGET_INDEX[lang]
            ]
            for lang in LANGUAGES
        ],
        dtype=np.float64,
    )

    p = softmax(
        selected
    )

    return {
        lang:
            float(
                p[i]
            )
        for i, lang in enumerate(
            LANGUAGES
        )
    }


def make_model_input(
    local_audio,
):
    local_audio = np.asarray(
        local_audio,
        dtype=np.float32,
    )

    if (
        len(local_audio) >
        MODEL_WINDOW_SAMPLES
    ):

        start = (
            len(local_audio) -
            MODEL_WINDOW_SAMPLES
        ) // 2

        return np.ascontiguousarray(
            local_audio[
                start:
                start +
                MODEL_WINDOW_SAMPLES
            ],
            dtype=np.float32,
        )

    if (
        len(local_audio) ==
        MODEL_WINDOW_SAMPLES
    ):

        return np.ascontiguousarray(
            local_audio,
            dtype=np.float32,
        )

    out = np.zeros(
        MODEL_WINDOW_SAMPLES,
        dtype=np.float32,
    )

    left = (
        MODEL_WINDOW_SAMPLES -
        len(local_audio)
    ) // 2

    out[
        left:
        left +
        len(local_audio)
    ] = local_audio

    return out


def create_interpreter():

    interpreter = Interpreter(
        model_path=str(
            MODEL
        ),
        num_threads=1,
    )

    interpreter.allocate_tensors()

    inp = (
        interpreter
        .get_input_details()[0]
    )

    shape = tuple(
        int(v)
        for v in inp["shape"]
    )

    if shape != (
        1,
        MODEL_WINDOW_SAMPLES,
    ):
        raise RuntimeError(
            f"Unexpected model input "
            f"shape {shape}"
        )

    return interpreter


def infer(
    interpreter,
    local_audio,
):
    x = make_model_input(
        local_audio
    )

    input_detail = (
        interpreter
        .get_input_details()[0]
    )

    interpreter.set_tensor(
        input_detail["index"],
        x[None, :].astype(
            input_detail["dtype"],
            copy=False,
        ),
    )

    start = time.perf_counter()

    interpreter.invoke()

    elapsed = (
        time.perf_counter() -
        start
    )

    logits = None

    for detail in (
        interpreter
        .get_output_details()
    ):

        y = interpreter.get_tensor(
            detail["index"]
        )

        flat = np.asarray(
            y
        ).reshape(-1)

        if flat.size == 107:
            logits = flat.astype(
                np.float32,
                copy=False,
            )
            break

    if logits is None:
        raise RuntimeError(
            "Could not find "
            "107-class logits output."
        )

    probs = classify_4way(
        logits
    )

    probs["infer_sec"] = (
        elapsed
    )

    return probs


def iter_windows(
    audio,
    window_sec,
    hop_sec,
    region_start_sec=None,
    region_end_sec=None,
):
    window_samples = int(
        window_sec *
        SAMPLE_RATE
    )

    hop_samples = int(
        hop_sec *
        SAMPLE_RATE
    )

    n = len(audio)

    region_start = (
        0
        if region_start_sec is None
        else max(
            0,
            int(
                region_start_sec *
                SAMPLE_RATE
            ),
        )
    )

    region_end = (
        n
        if region_end_sec is None
        else min(
            n,
            int(
                region_end_sec *
                SAMPLE_RATE
            ),
        )
    )

    # Window starts are constrained so centers lie in the requested region.
    first_start = max(
        0,
        region_start -
        window_samples // 2,
    )

    last_start = min(
        max(
            0,
            n -
            window_samples,
        ),
        max(
            0,
            region_end -
            window_samples // 2,
        ),
    )

    starts = list(
        range(
            first_start,
            last_start + 1,
            hop_samples,
        )
    )

    if not starts:
        starts = [
            max(
                0,
                min(
                    n -
                    window_samples,
                    first_start,
                ),
            )
        ]

    # Ensure the right edge gets a window too.
    if (
        starts[-1] !=
        last_start
    ):
        starts.append(
            last_start
        )

    starts = sorted(
        set(
            starts
        )
    )

    for start in starts:

        end = min(
            n,
            start +
            window_samples,
        )

        local = audio[
            start:end
        ]

        if (
            len(local) <
            window_samples
        ):
            padded = np.zeros(
                window_samples,
                dtype=np.float32,
            )

            padded[
                :len(local)
            ] = local

            local = padded

        center_sec = (
            start +
            window_samples /
            2.0
        ) / SAMPLE_RATE

        yield (
            start,
            end,
            center_sec,
            local,
        )


def evaluate_window(
    row,
    main_lang,
):
    foreign_langs = [
        lang
        for lang in LANGUAGES
        if lang != main_lang
    ]

    best_foreign = max(
        foreign_langs,
        key=lambda lang:
            row[
                f"p_{lang}"
            ],
    )

    p_main = row[
        f"p_{main_lang}"
    ]

    p_foreign = row[
        f"p_{best_foreign}"
    ]

    evidence = (
        p_foreign -
        p_main
    )

    return (
        best_foreign,
        p_foreign,
        p_main,
        evidence,
    )


def threshold_pass(
    p_foreign,
    evidence,
    min_foreign_prob,
    min_evidence,
):
    return (
        p_foreign >=
        min_foreign_prob
        and
        evidence >=
        min_evidence
    )


def run_coarse_scan(
    interpreter,
    audio,
    main_lang,
    cache_rows,
    sample_key,
):
    rows = []

    for (
        start,
        end,
        center_sec,
        local,
    ) in iter_windows(
        audio,
        COARSE_WINDOW_SEC,
        COARSE_HOP_SEC,
    ):

        probs = infer(
            interpreter,
            local,
        )

        row = {
            "sample_key":
                sample_key,
            "stage":
                "coarse",
            "start_sec":
                start /
                SAMPLE_RATE,
            "end_sec":
                end /
                SAMPLE_RATE,
            "center_sec":
                center_sec,
            "main_lang":
                main_lang,
            "p_zh":
                probs["zh"],
            "p_en":
                probs["en"],
            "p_ru":
                probs["ru"],
            "p_uz":
                probs["uz"],
            "infer_ms":
                probs[
                    "infer_sec"
                ] *
                1000.0,
        }

        rows.append(
            row
        )

        cache_rows.append(
            row
        )

    return rows


def choose_coarse_trigger(
    coarse_rows,
    main_lang,
    min_foreign_prob,
    min_evidence,
):
    candidates = []

    for row in coarse_rows:

        (
            foreign_lang,
            p_foreign,
            p_main,
            evidence,
        ) = evaluate_window(
            row,
            main_lang,
        )

        if not threshold_pass(
            p_foreign,
            evidence,
            min_foreign_prob,
            min_evidence,
        ):
            continue

        candidates.append(
            {
                "foreign_lang":
                    foreign_lang,
                "center_sec":
                    row[
                        "center_sec"
                    ],
                "p_foreign":
                    p_foreign,
                "p_main":
                    p_main,
                "evidence":
                    evidence,
            }
        )

    if not candidates:
        return None

    # Highest evidence first.
    candidates.sort(
        key=lambda x: (
            x["evidence"],
            x["p_foreign"],
        ),
        reverse=True,
    )

    return candidates[0]


def run_fine_scan(
    interpreter,
    audio,
    main_lang,
    trigger,
    cache_rows,
    sample_key,
):
    region_start = (
        trigger["center_sec"] -
        REFINE_RADIUS_SEC
    )

    region_end = (
        trigger["center_sec"] +
        REFINE_RADIUS_SEC
    )

    rows = []

    for (
        start,
        end,
        center_sec,
        local,
    ) in iter_windows(
        audio,
        FINE_WINDOW_SEC,
        FINE_HOP_SEC,
        region_start_sec=
            region_start,
        region_end_sec=
            region_end,
    ):

        probs = infer(
            interpreter,
            local,
        )

        row = {
            "sample_key":
                sample_key,
            "stage":
                "fine",
            "start_sec":
                start /
                SAMPLE_RATE,
            "end_sec":
                end /
                SAMPLE_RATE,
            "center_sec":
                center_sec,
            "main_lang":
                main_lang,
            "p_zh":
                probs["zh"],
            "p_en":
                probs["en"],
            "p_ru":
                probs["ru"],
            "p_uz":
                probs["uz"],
            "infer_ms":
                probs[
                    "infer_sec"
                ] *
                1000.0,
        }

        rows.append(
            row
        )

        cache_rows.append(
            row
        )

    return rows


def aggregate_fine(
    fine_rows,
    main_lang,
    min_foreign_prob,
    min_evidence,
):
    by_lang = defaultdict(
        list
    )

    for row in fine_rows:

        (
            foreign_lang,
            p_foreign,
            p_main,
            evidence,
        ) = evaluate_window(
            row,
            main_lang,
        )

        if threshold_pass(
            p_foreign,
            evidence,
            min_foreign_prob,
            min_evidence,
        ):
            by_lang[
                foreign_lang
            ].append(
                {
                    "row":
                        row,
                    "p_foreign":
                        p_foreign,
                    "evidence":
                        evidence,
                }
            )

    candidates = []

    for foreign_lang, items in (
        by_lang.items()
    ):

        items = sorted(
            items,
            key=lambda x:
                x["row"][
                    "center_sec"
                ],
        )

        # Find longest sequence with gap <= 1.5 fine hops.
        current = []
        runs = []

        for item in items:

            if not current:
                current = [
                    item
                ]
                continue

            gap = (
                item[
                    "row"
                ][
                    "center_sec"
                ] -
                current[-1][
                    "row"
                ][
                    "center_sec"
                ]
            )

            if (
                gap <=
                FINE_HOP_SEC *
                1.5
            ):
                current.append(
                    item
                )
            else:
                runs.append(
                    current
                )
                current = [
                    item
                ]

        if current:
            runs.append(
                current
            )

        for run in runs:

            if (
                len(run) <
                MIN_FINE_SUPPORT
            ):
                continue

            centers = [
                x["row"][
                    "center_sec"
                ]
                for x in run
            ]

            evidences = [
                x["evidence"]
                for x in run
            ]

            foreign_probs = [
                x["p_foreign"]
                for x in run
            ]

            # Weighted center gives smoother timing than top-1 hard runs.
            weights = np.asarray(
                [
                    max(
                        0.001,
                        e,
                    )
                    for e in evidences
                ],
                dtype=np.float64,
            )

            center = float(
                np.average(
                    np.asarray(
                        centers,
                        dtype=np.float64,
                    ),
                    weights=weights,
                )
            )

            start_sec = (
                min(centers) -
                FINE_HOP_SEC /
                2.0
            )

            end_sec = (
                max(centers) +
                FINE_HOP_SEC /
                2.0
            )

            score = float(
                np.mean(
                    evidences
                ) *
                math.sqrt(
                    len(run)
                )
            )

            candidates.append(
                {
                    "lang":
                        foreign_lang,
                    "support":
                        len(run),
                    "center_sec":
                        center,
                    "start_sec":
                        start_sec,
                    "end_sec":
                        end_sec,
                    "mean_evidence":
                        float(
                            np.mean(
                                evidences
                            )
                        ),
                    "mean_foreign_prob":
                        float(
                            np.mean(
                                foreign_probs
                            )
                        ),
                    "score":
                        score,
                }
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: (
            x["score"],
            x["support"],
            x["mean_foreign_prob"],
        ),
        reverse=True,
    )

    return candidates[0]


def interval_iou(
    a0,
    a1,
    b0,
    b1,
):
    inter = max(
        0.0,
        min(a1, b1) -
        max(a0, b0),
    )

    union = (
        max(a1, b1) -
        min(a0, b0)
    )

    if union <= 0.0:
        return 0.0

    return (
        inter /
        union
    )


def load_manifests():
    with POS_MANIFEST.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        positives = list(
            csv.DictReader(f)
        )

    with NEG_MANIFEST.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        negatives = list(
            csv.DictReader(f)
        )

    samples = []

    for item in positives:
        samples.append(
            {
                "kind":
                    "positive",
                "sample_key":
                    "P" +
                    str(
                        item["id"]
                    ),
                "file":
                    item["file"],
                "path":
                    POS_DIR /
                    item["file"],
                "main_lang":
                    item[
                        "main_lang"
                    ],
                "foreign_lang":
                    item[
                        "foreign_lang"
                    ],
                "true_start_sec":
                    float(
                        item[
                            "island_start_sec"
                        ]
                    ),
                "true_end_sec":
                    float(
                        item[
                            "island_end_sec"
                        ]
                    ),
            }
        )

    for item in negatives:
        samples.append(
            {
                "kind":
                    "negative",
                "sample_key":
                    "N" +
                    str(
                        item["id"]
                    ),
                "file":
                    item["file"],
                "path":
                    NEG_DIR /
                    item["file"],
                "main_lang":
                    item[
                        "main_lang"
                    ],
                "foreign_lang":
                    "",
                "true_start_sec":
                    math.nan,
                "true_end_sec":
                    math.nan,
            }
        )

    return samples


def main():

    for required in [
        MODEL,
        POS_MANIFEST,
        NEG_MANIFEST,
    ]:
        if not required.is_file():
            raise FileNotFoundError(
                required
            )

    samples = (
        load_manifests()
    )

    interpreter = (
        create_interpreter()
    )

    # Cache coarse inference once per sample.
    coarse_cache = {}
    all_window_rows = []

    total_infer_sec = 0.0
    total_inferences = 0

    print("=" * 100)
    print(
        "CODE-SWITCH ISLAND EXP2"
    )
    print("=" * 100)
    print(
        f"Positive samples : "
        f"{sum(1 for s in samples if s['kind']=='positive')}"
    )
    print(
        f"Negative samples : "
        f"{sum(1 for s in samples if s['kind']=='negative')}"
    )
    print(
        f"Coarse scan      : "
        f"{COARSE_WINDOW_SEC:.2f}s / "
        f"hop {COARSE_HOP_SEC:.2f}s"
    )
    print(
        f"Fine scan        : "
        f"{FINE_WINDOW_SEC:.2f}s / "
        f"hop {FINE_HOP_SEC:.2f}s / "
        f"radius {REFINE_RADIUS_SEC:.2f}s"
    )
    print()

    for sample in samples:

        audio = load_audio(
            sample["path"]
        )

        before = len(
            all_window_rows
        )

        rows = run_coarse_scan(
            interpreter,
            audio,
            sample[
                "main_lang"
            ],
            all_window_rows,
            sample[
                "sample_key"
            ],
        )

        after = len(
            all_window_rows
        )

        for row in (
            all_window_rows[
                before:after
            ]
        ):
            total_infer_sec += (
                row[
                    "infer_ms"
                ] /
                1000.0
            )
            total_inferences += 1

        coarse_cache[
            sample[
                "sample_key"
            ]
        ] = {
            "audio":
                audio,
            "rows":
                rows,
        }

    utterance_rows = []
    summary_rows = []

    print()
    print("=" * 100)
    print(
        "THRESHOLD SWEEP"
    )
    print("=" * 100)

    for (
        min_foreign_prob,
        min_evidence,
    ) in THRESHOLDS:

        pos_total = 0
        pos_correct = 0
        pos_wrong = 0
        pos_miss = 0

        neg_total = 0
        neg_false_island = 0

        center_errors = []
        ious = []

        fine_inferences_this_config = 0
        triggered_samples = 0

        for sample in samples:

            cache = coarse_cache[
                sample[
                    "sample_key"
                ]
            ]

            audio = cache[
                "audio"
            ]

            coarse_rows = cache[
                "rows"
            ]

            trigger = (
                choose_coarse_trigger(
                    coarse_rows,
                    sample[
                        "main_lang"
                    ],
                    min_foreign_prob,
                    min_evidence,
                )
            )

            candidate = None
            fine_rows = []

            if trigger is not None:

                triggered_samples += 1

                before = len(
                    all_window_rows
                )

                fine_rows = run_fine_scan(
                    interpreter,
                    audio,
                    sample[
                        "main_lang"
                    ],
                    trigger,
                    all_window_rows,
                    sample[
                        "sample_key"
                    ],
                )

                after = len(
                    all_window_rows
                )

                new_rows = (
                    all_window_rows[
                        before:after
                    ]
                )

                fine_inferences_this_config += (
                    len(
                        new_rows
                    )
                )

                for row in new_rows:
                    total_infer_sec += (
                        row[
                            "infer_ms"
                        ] /
                        1000.0
                    )
                    total_inferences += 1

                candidate = (
                    aggregate_fine(
                        fine_rows,
                        sample[
                            "main_lang"
                        ],
                        min_foreign_prob,
                        min_evidence,
                    )
                )

            if (
                sample[
                    "kind"
                ] ==
                "positive"
            ):

                pos_total += 1

                correct = (
                    candidate is not None
                    and
                    candidate[
                        "lang"
                    ] ==
                    sample[
                        "foreign_lang"
                    ]
                )

                wrong = (
                    candidate is not None
                    and
                    candidate[
                        "lang"
                    ] !=
                    sample[
                        "foreign_lang"
                    ]
                )

                if correct:
                    pos_correct += 1

                    true_center = (
                        sample[
                            "true_start_sec"
                        ] +
                        sample[
                            "true_end_sec"
                        ]
                    ) / 2.0

                    center_errors.append(
                        abs(
                            candidate[
                                "center_sec"
                            ] -
                            true_center
                        )
                    )

                    ious.append(
                        interval_iou(
                            candidate[
                                "start_sec"
                            ],
                            candidate[
                                "end_sec"
                            ],
                            sample[
                                "true_start_sec"
                            ],
                            sample[
                                "true_end_sec"
                            ],
                        )
                    )

                elif wrong:
                    pos_wrong += 1

                else:
                    pos_miss += 1

            else:

                neg_total += 1

                if (
                    candidate is not None
                ):
                    neg_false_island += 1

            utterance_rows.append(
                {
                    "min_foreign_prob":
                        min_foreign_prob,
                    "min_evidence":
                        min_evidence,
                    "sample_key":
                        sample[
                            "sample_key"
                        ],
                    "kind":
                        sample[
                            "kind"
                        ],
                    "file":
                        sample[
                            "file"
                        ],
                    "main_lang":
                        sample[
                            "main_lang"
                        ],
                    "foreign_lang":
                        sample[
                            "foreign_lang"
                        ],
                    "coarse_triggered":
                        trigger is not None,
                    "trigger_lang":
                        (
                            ""
                            if trigger is None
                            else
                            trigger[
                                "foreign_lang"
                            ]
                        ),
                    "trigger_center_sec":
                        (
                            ""
                            if trigger is None
                            else
                            trigger[
                                "center_sec"
                            ]
                        ),
                    "detected":
                        candidate is not None,
                    "detected_lang":
                        (
                            ""
                            if candidate is None
                            else
                            candidate[
                                "lang"
                            ]
                        ),
                    "support":
                        (
                            ""
                            if candidate is None
                            else
                            candidate[
                                "support"
                            ]
                        ),
                    "mean_evidence":
                        (
                            ""
                            if candidate is None
                            else
                            candidate[
                                "mean_evidence"
                            ]
                        ),
                    "pred_center_sec":
                        (
                            ""
                            if candidate is None
                            else
                            candidate[
                                "center_sec"
                            ]
                        ),
                    "correct":
                        (
                            sample[
                                "kind"
                            ] ==
                            "positive"
                            and
                            candidate is not None
                            and
                            candidate[
                                "lang"
                            ] ==
                            sample[
                                "foreign_lang"
                            ]
                        ),
                }
            )

        correct_rate = (
            pos_correct /
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

        false_island_rate = (
            neg_false_island /
            neg_total
        )

        center_mae = (
            float(
                np.mean(
                    center_errors
                )
            )
            if center_errors
            else math.nan
        )

        mean_iou = (
            float(
                np.mean(
                    ious
                )
            )
            if ious
            else math.nan
        )

        # Product cost estimate:
        # coarse windows are shared baseline; fine windows are extra only
        # on triggered samples.
        avg_fine_windows = (
            fine_inferences_this_config /
            len(samples)
        )

        summary = {
            "min_foreign_prob":
                min_foreign_prob,
            "min_evidence":
                min_evidence,
            "positive_samples":
                pos_total,
            "correct_foreign_rate":
                correct_rate,
            "wrong_foreign_rate":
                wrong_rate,
            "miss_rate":
                miss_rate,
            "negative_samples":
                neg_total,
            "false_island_rate":
                false_island_rate,
            "center_mae_sec":
                center_mae,
            "mean_iou":
                mean_iou,
            "triggered_sample_rate":
                triggered_samples /
                len(samples),
            "avg_fine_windows_per_sample":
                avg_fine_windows,
        }

        summary_rows.append(
            summary
        )

        print(
            f"Pforeign>={min_foreign_prob:.2f} "
            f"evidence>={min_evidence:.2f} | "
            f"correct={correct_rate*100:6.2f}% | "
            f"wrong={wrong_rate*100:5.2f}% | "
            f"miss={miss_rate*100:5.2f}% | "
            f"falseIsland={false_island_rate*100:5.2f}% | "
            f"centerMAE={center_mae:.3f}s | "
            f"IoU={mean_iou:.3f} | "
            f"trigger={(
                triggered_samples /
                len(samples)
            )*100:5.1f}% | "
            f"fine/sample={avg_fine_windows:.2f}"
        )

    # Save window-level output after the whole sweep.
    if all_window_rows:
        with WINDOW_RESULTS.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    all_window_rows[0]
                    .keys()
                ),
            )

            writer.writeheader()
            writer.writerows(
                all_window_rows
            )

    with UTTERANCE_RESULTS.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                utterance_rows[0]
                .keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            utterance_rows
        )

    with SUMMARY_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                summary_rows[0]
                .keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    print()
    print("=" * 100)
    print(
        "RECOMMENDED CONFIG"
    )
    print("=" * 100)

    # Conservative recommendation:
    # 1) false island <= 5%
    # 2) maximize correct foreign rate
    # 3) minimize wrong foreign rate
    # 4) minimize trigger/fine cost
    safe = [
        row
        for row in summary_rows
        if (
            row[
                "false_island_rate"
            ] <= 0.05
        )
    ]

    pool = (
        safe
        if safe
        else summary_rows
    )

    def sort_key(
        row,
    ):
        mae = (
            row[
                "center_mae_sec"
            ]
        )

        if math.isnan(
            mae
        ):
            mae = 999.0

        return (
            row[
                "correct_foreign_rate"
            ],
            -row[
                "wrong_foreign_rate"
            ],
            -row[
                "false_island_rate"
            ],
            -row[
                "avg_fine_windows_per_sample"
            ],
            -mae,
        )

    best = max(
        pool,
        key=sort_key,
    )

    for key in [
        "min_foreign_prob",
        "min_evidence",
        "correct_foreign_rate",
        "wrong_foreign_rate",
        "miss_rate",
        "false_island_rate",
        "center_mae_sec",
        "mean_iou",
        "triggered_sample_rate",
        "avg_fine_windows_per_sample",
    ]:
        value = best[
            key
        ]

        if "rate" in key:
            print(
                f"{key:28s}: "
                f"{value*100:.2f}%"
            )
        elif isinstance(
            value,
            float,
        ):
            print(
                f"{key:28s}: "
                f"{value:.4f}"
            )
        else:
            print(
                f"{key:28s}: "
                f"{value}"
            )

    print()
    print("=" * 100)
    print(
        "RUNTIME"
    )
    print("=" * 100)
    print(
        f"Total TFLite inferences : "
        f"{total_inferences}"
    )
    print(
        f"Total inference time    : "
        f"{total_infer_sec:.2f}s"
    )

    if total_inferences:
        print(
            f"Mean per inference      : "
            f"{(
                total_infer_sec /
                total_inferences
            )*1000:.1f} ms"
        )

    print()
    print(
        f"Window CSV    : "
        f"{WINDOW_RESULTS}"
    )
    print(
        f"Utterance CSV : "
        f"{UTTERANCE_RESULTS}"
    )
    print(
        f"Summary CSV   : "
        f"{SUMMARY_CSV}"
    )


if __name__ == "__main__":
    main()
