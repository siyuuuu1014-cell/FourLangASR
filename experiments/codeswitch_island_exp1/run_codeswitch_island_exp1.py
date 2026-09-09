from pathlib import Path
import csv
import math
import time

import numpy as np
import soundfile as sf
from ai_edge_litert.interpreter import Interpreter


PROJECT_ROOT = Path(
    r"D:\dev\projects\FourLangAsrMobile"
)

BASE_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "codeswitch_island_exp1"
)

MODEL = (
    PROJECT_ROOT
    / "experiments"
    / "lid_exp4_tflite"
    / "converted_tf"
    / "voxlingua107_float32.tflite"
)

DATA_DIR = (
    BASE_DIR /
    "island_data"
)

MANIFEST = (
    BASE_DIR /
    "island_manifest.csv"
)

WINDOW_RESULTS = (
    BASE_DIR /
    "island_exp1_window_results.csv"
)

UTTERANCE_RESULTS = (
    BASE_DIR /
    "island_exp1_utterance_results.csv"
)

SUMMARY_CSV = (
    BASE_DIR /
    "island_exp1_summary.csv"
)

SAMPLE_RATE = 16000

# We only sweep two useful sizes in EXP1 to keep runtime manageable.
WINDOW_SIZES_SEC = [
    1.0,
    1.25,
]

HOP_SEC = 0.25

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

THRESHOLDS = [
    (0.45, 0.10),
    (0.55, 0.15),
    (0.65, 0.20),
    (0.75, 0.25),
]

# Two consecutive foreign windows are required.
MIN_FOREIGN_RUN = 2


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

    order = np.argsort(
        -p
    )

    best_i = int(
        order[0]
    )

    second_i = int(
        order[1]
    )

    return {
        "lang":
            LANGUAGES[best_i],
        "conf":
            float(p[best_i]),
        "second_lang":
            LANGUAGES[second_i],
        "second_conf":
            float(p[second_i]),
        "margin":
            float(
                p[best_i] -
                p[second_i]
            ),
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

    result = classify_4way(
        logits
    )

    result["infer_sec"] = (
        elapsed
    )

    return result


def iter_windows(
    audio,
    window_sec,
):
    window_samples = int(
        window_sec *
        SAMPLE_RATE
    )

    hop_samples = int(
        HOP_SEC *
        SAMPLE_RATE
    )

    n = len(audio)

    starts = list(
        range(
            0,
            max(
                1,
                n -
                window_samples +
                1,
            ),
            hop_samples,
        )
    )

    last_start = max(
        0,
        n -
        window_samples,
    )

    if (
        not starts or
        starts[-1] !=
        last_start
    ):
        starts.append(
            last_start
        )

    starts = sorted(
        set(starts)
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

        if len(local) < window_samples:

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


def gated_label(
    row,
    min_conf,
    min_margin,
):
    if (
        row["conf"] <
        min_conf or
        row["margin"] <
        min_margin
    ):
        return "?"

    return row["pred"]


def find_runs(
    labels,
):
    runs = []

    start = 0

    while start < len(labels):

        label = labels[start]

        end = start + 1

        while (
            end < len(labels)
            and labels[end] == label
        ):
            end += 1

        runs.append(
            (
                label,
                start,
                end,
            )
        )

        start = end

    return runs


def detect_foreign_island(
    rows,
    main_lang,
    min_conf,
    min_margin,
):
    labels = [
        gated_label(
            row,
            min_conf,
            min_margin,
        )
        for row in rows
    ]

    candidates = []

    for (
        label,
        start,
        end,
    ) in find_runs(
        labels
    ):

        if label in {
            "?",
            main_lang,
        }:
            continue

        run_len = (
            end - start
        )

        if (
            run_len <
            MIN_FOREIGN_RUN
        ):
            continue

        first_center = (
            rows[start]
            ["center_sec"]
        )

        last_center = (
            rows[end - 1]
            ["center_sec"]
        )

        estimated_start = (
            first_center -
            HOP_SEC / 2.0
        )

        estimated_end = (
            last_center +
            HOP_SEC / 2.0
        )

        mean_conf = float(
            np.mean(
                [
                    rows[i]["conf"]
                    for i in range(
                        start,
                        end,
                    )
                ]
            )
        )

        mean_margin = float(
            np.mean(
                [
                    rows[i]["margin"]
                    for i in range(
                        start,
                        end,
                    )
                ]
            )
        )

        candidates.append(
            {
                "lang":
                    label,
                "start_idx":
                    start,
                "end_idx":
                    end,
                "run_len":
                    run_len,
                "start_sec":
                    estimated_start,
                "end_sec":
                    estimated_end,
                "center_sec":
                    (
                        estimated_start +
                        estimated_end
                    ) / 2.0,
                "mean_conf":
                    mean_conf,
                "mean_margin":
                    mean_margin,
            }
        )

    if not candidates:
        return None, labels

    # Prefer longest stable run, then stronger confidence.
    candidates.sort(
        key=lambda x: (
            x["run_len"],
            x["mean_conf"],
            x["mean_margin"],
        ),
        reverse=True,
    )

    return (
        candidates[0],
        labels,
    )


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

    return inter / union


def main():

    if not MODEL.is_file():
        raise FileNotFoundError(
            MODEL
        )

    if not MANIFEST.is_file():
        raise FileNotFoundError(
            MANIFEST
        )

    with MANIFEST.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        manifest = list(
            csv.DictReader(f)
        )

    interpreter = (
        create_interpreter()
    )

    all_window_rows = []
    cache = {}

    total_infer_sec = 0.0
    total_windows = 0

    print("=" * 96)
    print(
        "CODE-SWITCH ISLAND EXP1"
    )
    print("=" * 96)
    print(
        f"Samples    : "
        f"{len(manifest)}"
    )
    print(
        f"Windows    : "
        f"{WINDOW_SIZES_SEC}"
    )
    print(
        f"Hop        : "
        f"{HOP_SEC:.2f}s"
    )
    print(
        f"Min run    : "
        f"{MIN_FOREIGN_RUN}"
    )
    print()

    for window_sec in (
        WINDOW_SIZES_SEC
    ):

        print()
        print(
            f"### WINDOW "
            f"{window_sec:.2f}s"
        )

        for item in manifest:

            audio = load_audio(
                DATA_DIR /
                item["file"]
            )

            rows = []

            for (
                start,
                end,
                center_sec,
                local,
            ) in iter_windows(
                audio,
                window_sec,
            ):

                result = infer(
                    interpreter,
                    local,
                )

                total_infer_sec += (
                    result["infer_sec"]
                )

                total_windows += 1

                row = {
                    "utterance_id":
                        int(item["id"]),
                    "file":
                        item["file"],
                    "window_sec":
                        window_sec,
                    "start_sec":
                        start /
                        SAMPLE_RATE,
                    "end_sec":
                        end /
                        SAMPLE_RATE,
                    "center_sec":
                        center_sec,
                    "main_lang":
                        item[
                            "main_lang"
                        ],
                    "foreign_lang":
                        item[
                            "foreign_lang"
                        ],
                    "island_start_sec":
                        float(
                            item[
                                "island_start_sec"
                            ]
                        ),
                    "island_end_sec":
                        float(
                            item[
                                "island_end_sec"
                            ]
                        ),
                    "pred":
                        result[
                            "lang"
                        ],
                    "conf":
                        result[
                            "conf"
                        ],
                    "second_lang":
                        result[
                            "second_lang"
                        ],
                    "second_conf":
                        result[
                            "second_conf"
                        ],
                    "margin":
                        result[
                            "margin"
                        ],
                    "infer_ms":
                        result[
                            "infer_sec"
                        ] *
                        1000.0,
                }

                rows.append(
                    row
                )

                all_window_rows.append(
                    row
                )

            cache[
                (
                    int(
                        item["id"]
                    ),
                    window_sec,
                )
            ] = {
                "meta":
                    item,
                "rows":
                    rows,
            }

            raw_seq = " ".join(
                r["pred"].upper()
                for r in rows
            )

            print(
                f"{item['id']:>3s} | "
                f"{item['main_lang'].upper()} "
                f"[{item['foreign_lang'].upper()}] "
                f"{item['foreign_duration_sec']}s | "
                f"{raw_seq}"
            )

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

    utterance_rows = []
    summary_rows = []

    print()
    print("=" * 96)
    print(
        "THRESHOLD SWEEP"
    )
    print("=" * 96)

    for window_sec in (
        WINDOW_SIZES_SEC
    ):

        for (
            min_conf,
            min_margin,
        ) in THRESHOLDS:

            total = 0
            detected_any = 0
            correct_lang = 0
            wrong_lang = 0
            missed = 0

            false_alarm_windows = 0
            pure_main_windows = 0

            center_errors = []
            ious = []

            for item in manifest:

                key = (
                    int(item["id"]),
                    window_sec,
                )

                rows = (
                    cache[key]
                    ["rows"]
                )

                main_lang = (
                    item[
                        "main_lang"
                    ]
                )

                foreign_lang = (
                    item[
                        "foreign_lang"
                    ]
                )

                true_start = float(
                    item[
                        "island_start_sec"
                    ]
                )

                true_end = float(
                    item[
                        "island_end_sec"
                    ]
                )

                true_center = (
                    true_start +
                    true_end
                ) / 2.0

                candidate, labels = (
                    detect_foreign_island(
                        rows,
                        main_lang,
                        min_conf,
                        min_margin,
                    )
                )

                total += 1

                if candidate is None:
                    missed += 1
                else:
                    detected_any += 1

                    if (
                        candidate["lang"] ==
                        foreign_lang
                    ):
                        correct_lang += 1

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
                                true_start,
                                true_end,
                            )
                        )
                    else:
                        wrong_lang += 1

                # False alarm windows:
                # only evaluate windows that do NOT overlap
                # the true foreign island at all.
                for row, label in zip(
                    rows,
                    labels,
                ):
                    no_overlap = (
                        row[
                            "end_sec"
                        ] <=
                        true_start or
                        row[
                            "start_sec"
                        ] >=
                        true_end
                    )

                    if not no_overlap:
                        continue

                    pure_main_windows += 1

                    if (
                        label not in {
                            "?",
                            main_lang,
                        }
                    ):
                        false_alarm_windows += 1

                utterance_rows.append(
                    {
                        "window_sec":
                            window_sec,
                        "min_conf":
                            min_conf,
                        "min_margin":
                            min_margin,
                        "utterance_id":
                            int(item["id"]),
                        "file":
                            item["file"],
                        "main_lang":
                            main_lang,
                        "foreign_lang":
                            foreign_lang,
                        "foreign_duration_sec":
                            float(
                                item[
                                    "foreign_duration_sec"
                                ]
                            ),
                        "detected":
                            candidate is not None,
                        "detected_lang":
                            (
                                ""
                                if candidate is None
                                else
                                candidate["lang"]
                            ),
                        "correct_foreign_lang":
                            (
                                candidate is not None
                                and
                                candidate["lang"] ==
                                foreign_lang
                            ),
                        "true_start_sec":
                            true_start,
                        "true_end_sec":
                            true_end,
                        "pred_start_sec":
                            (
                                ""
                                if candidate is None
                                else
                                candidate[
                                    "start_sec"
                                ]
                            ),
                        "pred_end_sec":
                            (
                                ""
                                if candidate is None
                                else
                                candidate[
                                    "end_sec"
                                ]
                            ),
                        "center_abs_error_sec":
                            (
                                ""
                                if (
                                    candidate is None
                                    or
                                    candidate["lang"] !=
                                    foreign_lang
                                )
                                else
                                abs(
                                    candidate[
                                        "center_sec"
                                    ] -
                                    true_center
                                )
                            ),
                        "iou":
                            (
                                ""
                                if (
                                    candidate is None
                                    or
                                    candidate["lang"] !=
                                    foreign_lang
                                )
                                else
                                interval_iou(
                                    candidate[
                                        "start_sec"
                                    ],
                                    candidate[
                                        "end_sec"
                                    ],
                                    true_start,
                                    true_end,
                                )
                            ),
                        "gated_sequence":
                            " ".join(
                                labels
                            ),
                    }
                )

            detect_rate = (
                detected_any /
                total
            )

            correct_rate = (
                correct_lang /
                total
            )

            wrong_rate = (
                wrong_lang /
                total
            )

            false_alarm_rate = (
                false_alarm_windows /
                pure_main_windows
                if pure_main_windows
                else 0.0
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

            summary = {
                "window_sec":
                    window_sec,
                "min_conf":
                    min_conf,
                "min_margin":
                    min_margin,
                "samples":
                    total,
                "detected_any_rate":
                    detect_rate,
                "correct_foreign_lang_rate":
                    correct_rate,
                "wrong_foreign_lang_rate":
                    wrong_rate,
                "miss_rate":
                    missed /
                    total,
                "false_alarm_window_rate":
                    false_alarm_rate,
                "center_mae_sec":
                    center_mae,
                "mean_iou":
                    mean_iou,
            }

            summary_rows.append(
                summary
            )

            print(
                f"win={window_sec:4.2f}s | "
                f"conf>={min_conf:.2f} "
                f"margin>={min_margin:.2f} | "
                f"detect={detect_rate*100:6.2f}% | "
                f"correct={correct_rate*100:6.2f}% | "
                f"wrong={wrong_rate*100:5.2f}% | "
                f"falseWin={false_alarm_rate*100:5.2f}% | "
                f"centerMAE={center_mae:.3f}s | "
                f"IoU={mean_iou:.3f}"
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
    print("=" * 96)
    print(
        "RECOMMENDED CONFIG"
    )
    print("=" * 96)

    # Conservative product-oriented recommendation:
    # first keep false-alarm windows <=5%, then maximize correct island rate,
    # then minimize wrong language rate, then center error.
    valid = [
        row
        for row in summary_rows
        if (
            row[
                "false_alarm_window_rate"
            ] <= 0.05
        )
    ]

    pool = (
        valid
        if valid
        else summary_rows
    )

    def sort_key(row):
        mae = row[
            "center_mae_sec"
        ]

        if math.isnan(mae):
            mae = 999.0

        return (
            row[
                "correct_foreign_lang_rate"
            ],
            -row[
                "wrong_foreign_lang_rate"
            ],
            -row[
                "false_alarm_window_rate"
            ],
            -mae,
        )

    best = max(
        pool,
        key=sort_key,
    )

    print(
        f"window      : "
        f"{best['window_sec']:.2f}s"
    )
    print(
        f"min_conf    : "
        f"{best['min_conf']:.2f}"
    )
    print(
        f"min_margin  : "
        f"{best['min_margin']:.2f}"
    )
    print(
        f"correct     : "
        f"{best['correct_foreign_lang_rate']*100:.2f}%"
    )
    print(
        f"wrong       : "
        f"{best['wrong_foreign_lang_rate']*100:.2f}%"
    )
    print(
        f"falseWin    : "
        f"{best['false_alarm_window_rate']*100:.2f}%"
    )
    print(
        f"center MAE  : "
        f"{best['center_mae_sec']:.3f}s"
    )
    print(
        f"mean IoU    : "
        f"{best['mean_iou']:.3f}"
    )

    print()
    print("=" * 96)
    print(
        "RUNTIME"
    )
    print("=" * 96)
    print(
        f"Windows     : "
        f"{total_windows}"
    )
    print(
        f"Total infer : "
        f"{total_infer_sec:.2f}s"
    )
    print(
        f"Mean/window : "
        f"{(
            total_infer_sec /
            total_windows
        )*1000:.1f} ms"
    )
    print()
    print(
        f"Window CSV  : "
        f"{WINDOW_RESULTS}"
    )
    print(
        f"Utterance   : "
        f"{UTTERANCE_RESULTS}"
    )
    print(
        f"Summary     : "
        f"{SUMMARY_CSV}"
    )


if __name__ == "__main__":
    main()
