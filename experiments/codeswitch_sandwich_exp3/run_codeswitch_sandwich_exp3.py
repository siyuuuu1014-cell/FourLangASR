from pathlib import Path
import csv
import math
from collections import defaultdict

import numpy as np


PROJECT_ROOT = Path(
    r"D:\dev\projects\FourLangAsrMobile"
)

EXP1_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "codeswitch_island_exp1"
)

EXP2_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "codeswitch_island_exp2"
)

BASE_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "codeswitch_sandwich_exp3"
)

POS_MANIFEST = (
    EXP1_DIR
    / "island_manifest.csv"
)

NEG_MANIFEST = (
    EXP2_DIR
    / "negative_manifest.csv"
)

EXP2_WINDOWS = (
    EXP2_DIR
    / "island_exp2_window_results.csv"
)

UTTERANCE_RESULTS = (
    BASE_DIR
    / "sandwich_exp3_utterance_results.csv"
)

SUMMARY_CSV = (
    BASE_DIR
    / "sandwich_exp3_summary.csv"
)

LANGUAGES = [
    "zh",
    "en",
    "ru",
    "uz",
]

FOREIGN_THRESHOLDS = [
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
        "edge_gap_sec": 0.40,
        "max_island_sec": 1.80,
    },
    {
        "name": "balanced",
        "main_prob": 0.50,
        "main_adv": 0.10,
        "edge_gap_sec": 0.50,
        "max_island_sec": 1.70,
    },
    {
        "name": "strict",
        "main_prob": 0.55,
        "main_adv": 0.15,
        "edge_gap_sec": 0.60,
        "max_island_sec": 1.60,
    },
]

MIN_FOREIGN_WINDOWS = 2
MAX_CENTER_GAP_SEC = 0.40
REJECT_MULTI_FOREIGN = True


def read_csv(path: Path):
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def load_samples():
    positives = read_csv(POS_MANIFEST)
    negatives = read_csv(NEG_MANIFEST)

    samples = {}

    for item in positives:
        key = "P" + str(item["id"])
        samples[key] = {
            "sample_key": key,
            "kind": "positive",
            "file": item["file"],
            "main_lang": item["main_lang"],
            "foreign_lang": item["foreign_lang"],
            "true_start_sec": float(item["island_start_sec"]),
            "true_end_sec": float(item["island_end_sec"]),
            "duration_sec": float(item["total_duration_sec"]),
        }

    for item in negatives:
        key = "N" + str(item["id"])
        samples[key] = {
            "sample_key": key,
            "kind": "negative",
            "file": item["file"],
            "main_lang": item["main_lang"],
            "foreign_lang": "",
            "true_start_sec": math.nan,
            "true_end_sec": math.nan,
            "duration_sec": float(item["duration_sec"]),
        }

    return samples


def load_window_cache():
    rows = read_csv(EXP2_WINDOWS)

    dedup = {}

    for row in rows:
        key = (
            row["sample_key"],
            row["stage"],
            round(float(row["center_sec"]), 6),
        )

        if key in dedup:
            continue

        dedup[key] = {
            "sample_key": row["sample_key"],
            "stage": row["stage"],
            "start_sec": float(row["start_sec"]),
            "end_sec": float(row["end_sec"]),
            "center_sec": float(row["center_sec"]),
            "main_lang": row["main_lang"],
            "p_zh": float(row["p_zh"]),
            "p_en": float(row["p_en"]),
            "p_ru": float(row["p_ru"]),
            "p_uz": float(row["p_uz"]),
        }

    by_sample = defaultdict(
        lambda: {
            "coarse": [],
            "fine": [],
        }
    )

    for row in dedup.values():
        by_sample[row["sample_key"]][row["stage"]].append(row)

    for payload in by_sample.values():
        payload["coarse"].sort(key=lambda r: r["center_sec"])
        payload["fine"].sort(key=lambda r: r["center_sec"])

    return by_sample


def best_non_main(row, main_lang):
    foreign_langs = [
        lang
        for lang in LANGUAGES
        if lang != main_lang
    ]

    foreign_lang = max(
        foreign_langs,
        key=lambda lang: row[f"p_{lang}"],
    )

    p_foreign = row[f"p_{foreign_lang}"]
    p_main = row[f"p_{main_lang}"]
    evidence = p_foreign - p_main

    return foreign_lang, p_foreign, p_main, evidence


def main_strength(row, main_lang):
    p_main = row[f"p_{main_lang}"]

    best_other = max(
        row[f"p_{lang}"]
        for lang in LANGUAGES
        if lang != main_lang
    )

    return p_main, p_main - best_other


def foreign_pass(
    row,
    main_lang,
    min_foreign_prob,
    min_evidence,
):
    (
        foreign_lang,
        p_foreign,
        p_main,
        evidence,
    ) = best_non_main(
        row,
        main_lang,
    )

    passed = (
        p_foreign >= min_foreign_prob
        and evidence >= min_evidence
    )

    return (
        passed,
        foreign_lang,
        p_foreign,
        p_main,
        evidence,
    )


def build_foreign_runs(
    fine_rows,
    main_lang,
    min_foreign_prob,
    min_evidence,
):
    accepted = []

    for row in fine_rows:
        (
            passed,
            foreign_lang,
            p_foreign,
            p_main,
            evidence,
        ) = foreign_pass(
            row,
            main_lang,
            min_foreign_prob,
            min_evidence,
        )

        if not passed:
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

        same_lang = item["lang"] == current[-1]["lang"]

        gap = (
            item["row"]["center_sec"]
            - current[-1]["row"]["center_sec"]
        )

        if same_lang and gap <= MAX_CENTER_GAP_SEC:
            current.append(item)
        else:
            runs.append(current)
            current = [item]

    if current:
        runs.append(current)

    candidates = []

    for run in runs:
        if len(run) < MIN_FOREIGN_WINDOWS:
            continue

        centers = np.asarray(
            [item["row"]["center_sec"] for item in run],
            dtype=np.float64,
        )

        evidences = np.asarray(
            [item["evidence"] for item in run],
            dtype=np.float64,
        )

        weights = np.maximum(evidences, 0.001)

        center = float(
            np.average(
                centers,
                weights=weights,
            )
        )

        start_sec = float(centers.min()) - 0.125
        end_sec = float(centers.max()) + 0.125

        candidates.append(
            {
                "lang": run[0]["lang"],
                "support": len(run),
                "center_sec": center,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": end_sec - start_sec,
                "mean_evidence": float(evidences.mean()),
                "mean_foreign_prob": float(
                    np.mean(
                        [item["p_foreign"] for item in run]
                    )
                ),
                "score": float(
                    evidences.mean() * math.sqrt(len(run))
                ),
            }
        )

    return candidates


def has_main_before(
    coarse_rows,
    main_lang,
    candidate,
    rule,
):
    usable = [
        row
        for row in coarse_rows
        if row["end_sec"] <= candidate["start_sec"]
    ]

    if not usable:
        return False

    usable.sort(
        key=lambda r: r["end_sec"],
        reverse=True,
    )

    for row in usable[:2]:
        p_main, advantage = main_strength(row, main_lang)

        if (
            p_main >= rule["main_prob"]
            and advantage >= rule["main_adv"]
        ):
            return True

    return False


def has_main_after(
    coarse_rows,
    main_lang,
    candidate,
    rule,
):
    usable = [
        row
        for row in coarse_rows
        if row["start_sec"] >= candidate["end_sec"]
    ]

    if not usable:
        return False

    usable.sort(
        key=lambda r: r["start_sec"]
    )

    for row in usable[:2]:
        p_main, advantage = main_strength(row, main_lang)

        if (
            p_main >= rule["main_prob"]
            and advantage >= rule["main_adv"]
        ):
            return True

    return False


def sandwich_accept(
    sample,
    coarse_rows,
    candidate,
    rule,
):
    if candidate["duration_sec"] > rule["max_island_sec"]:
        return False

    edge_gap = rule["edge_gap_sec"]

    if candidate["start_sec"] < edge_gap:
        return False

    if (
        sample["duration_sec"] - candidate["end_sec"]
        < edge_gap
    ):
        return False

    if not has_main_before(
        coarse_rows,
        sample["main_lang"],
        candidate,
        rule,
    ):
        return False

    if not has_main_after(
        coarse_rows,
        sample["main_lang"],
        candidate,
        rule,
    ):
        return False

    return True


def choose_candidate(
    sample,
    coarse_rows,
    fine_rows,
    min_foreign_prob,
    min_evidence,
    rule,
):
    runs = build_foreign_runs(
        fine_rows,
        sample["main_lang"],
        min_foreign_prob,
        min_evidence,
    )

    accepted = [
        candidate
        for candidate in runs
        if sandwich_accept(
            sample,
            coarse_rows,
            candidate,
            rule,
        )
    ]

    if not accepted:
        return None

    if REJECT_MULTI_FOREIGN:
        languages = {
            candidate["lang"]
            for candidate in accepted
        }

        if len(languages) > 1:
            return None

    accepted.sort(
        key=lambda c: (
            c["score"],
            c["support"],
            c["mean_foreign_prob"],
        ),
        reverse=True,
    )

    return accepted[0]


def interval_iou(
    a0,
    a1,
    b0,
    b1,
):
    inter = max(
        0.0,
        min(a1, b1) - max(a0, b0),
    )

    union = (
        max(a1, b1)
        - min(a0, b0)
    )

    if union <= 0:
        return 0.0

    return inter / union


def main():
    BASE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for required in [
        POS_MANIFEST,
        NEG_MANIFEST,
        EXP2_WINDOWS,
    ]:
        if not required.is_file():
            raise FileNotFoundError(required)

    samples = load_samples()
    windows = load_window_cache()

    print("=" * 100)
    print("CODE-SWITCH SANDWICH EXP3")
    print("=" * 100)
    print("NO NEW TFLITE INFERENCE")
    print("Reusing EXP2 cached probabilities.")
    print()
    print(
        "Positive samples : "
        f"{sum(1 for s in samples.values() if s['kind']=='positive')}"
    )
    print(
        "Negative samples : "
        f"{sum(1 for s in samples.values() if s['kind']=='negative')}"
    )
    print()

    utterance_rows = []
    summary_rows = []

    print("=" * 100)
    print("SANDWICH SWEEP")
    print("=" * 100)

    for rule in SANDWICH_RULES:
        for (
            min_foreign_prob,
            min_evidence,
        ) in FOREIGN_THRESHOLDS:

            pos_total = 0
            pos_correct = 0
            pos_wrong = 0
            pos_miss = 0

            neg_total = 0
            neg_false = 0

            center_errors = []
            ious = []

            for sample_key, sample in samples.items():
                payload = windows.get(
                    sample_key,
                    {
                        "coarse": [],
                        "fine": [],
                    },
                )

                candidate = choose_candidate(
                    sample,
                    payload["coarse"],
                    payload["fine"],
                    min_foreign_prob,
                    min_evidence,
                    rule,
                )

                if sample["kind"] == "positive":
                    pos_total += 1

                    correct = (
                        candidate is not None
                        and candidate["lang"]
                        == sample["foreign_lang"]
                    )

                    wrong = (
                        candidate is not None
                        and candidate["lang"]
                        != sample["foreign_lang"]
                    )

                    if correct:
                        pos_correct += 1

                        true_center = (
                            sample["true_start_sec"]
                            + sample["true_end_sec"]
                        ) / 2.0

                        center_errors.append(
                            abs(
                                candidate["center_sec"]
                                - true_center
                            )
                        )

                        ious.append(
                            interval_iou(
                                candidate["start_sec"],
                                candidate["end_sec"],
                                sample["true_start_sec"],
                                sample["true_end_sec"],
                            )
                        )

                    elif wrong:
                        pos_wrong += 1

                    else:
                        pos_miss += 1

                else:
                    neg_total += 1

                    if candidate is not None:
                        neg_false += 1

                utterance_rows.append(
                    {
                        "rule": rule["name"],
                        "min_foreign_prob":
                            min_foreign_prob,
                        "min_evidence":
                            min_evidence,
                        "sample_key":
                            sample_key,
                        "kind":
                            sample["kind"],
                        "file":
                            sample["file"],
                        "main_lang":
                            sample["main_lang"],
                        "foreign_lang":
                            sample["foreign_lang"],
                        "detected":
                            candidate is not None,
                        "detected_lang":
                            (
                                ""
                                if candidate is None
                                else candidate["lang"]
                            ),
                        "support":
                            (
                                ""
                                if candidate is None
                                else candidate["support"]
                            ),
                        "pred_start_sec":
                            (
                                ""
                                if candidate is None
                                else candidate["start_sec"]
                            ),
                        "pred_end_sec":
                            (
                                ""
                                if candidate is None
                                else candidate["end_sec"]
                            ),
                        "pred_center_sec":
                            (
                                ""
                                if candidate is None
                                else candidate["center_sec"]
                            ),
                        "duration_sec":
                            (
                                ""
                                if candidate is None
                                else candidate["duration_sec"]
                            ),
                        "correct":
                            (
                                sample["kind"] == "positive"
                                and candidate is not None
                                and candidate["lang"]
                                == sample["foreign_lang"]
                            ),
                    }
                )

            correct_rate = pos_correct / pos_total
            wrong_rate = pos_wrong / pos_total
            miss_rate = pos_miss / pos_total
            false_island_rate = neg_false / neg_total

            center_mae = (
                float(np.mean(center_errors))
                if center_errors
                else math.nan
            )

            mean_iou = (
                float(np.mean(ious))
                if ious
                else math.nan
            )

            summary = {
                "rule": rule["name"],
                "main_prob":
                    rule["main_prob"],
                "main_adv":
                    rule["main_adv"],
                "edge_gap_sec":
                    rule["edge_gap_sec"],
                "max_island_sec":
                    rule["max_island_sec"],
                "min_foreign_prob":
                    min_foreign_prob,
                "min_evidence":
                    min_evidence,
                "correct_foreign_rate":
                    correct_rate,
                "wrong_foreign_rate":
                    wrong_rate,
                "miss_rate":
                    miss_rate,
                "false_island_rate":
                    false_island_rate,
                "center_mae_sec":
                    center_mae,
                "mean_iou":
                    mean_iou,
            }

            summary_rows.append(summary)

            print(
                f"{rule['name']:8s} | "
                f"Pforeign>={min_foreign_prob:.2f} "
                f"evidence>={min_evidence:.2f} | "
                f"correct={correct_rate*100:6.2f}% | "
                f"wrong={wrong_rate*100:5.2f}% | "
                f"miss={miss_rate*100:5.2f}% | "
                f"falseIsland={false_island_rate*100:5.2f}% | "
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
                utterance_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(utterance_rows)

    with SUMMARY_CSV.open(
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
        writer.writerows(summary_rows)

    print()
    print("=" * 100)
    print("RECOMMENDED CONFIG")
    print("=" * 100)

    safe = [
        row
        for row in summary_rows
        if (
            row["false_island_rate"] <= 0.05
            and row["wrong_foreign_rate"] <= 0.05
        )
    ]

    pool = safe if safe else summary_rows

    def sort_key(row):
        mae = row["center_mae_sec"]

        if math.isnan(mae):
            mae = 999.0

        return (
            row["correct_foreign_rate"],
            -row["false_island_rate"],
            -row["wrong_foreign_rate"],
            -mae,
            row["mean_iou"],
        )

    best = max(
        pool,
        key=sort_key,
    )

    for key in [
        "rule",
        "main_prob",
        "main_adv",
        "edge_gap_sec",
        "max_island_sec",
        "min_foreign_prob",
        "min_evidence",
        "correct_foreign_rate",
        "wrong_foreign_rate",
        "miss_rate",
        "false_island_rate",
        "center_mae_sec",
        "mean_iou",
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
    print("Pass target:")
    print("  correct >= 75-80%")
    print("  wrong <= 5%")
    print("  falseIsland <= 5%")
    print("  center MAE <= 0.50s")
    print()
    print(
        f"Utterance CSV: "
        f"{UTTERANCE_RESULTS}"
    )
    print(
        f"Summary CSV  : "
        f"{SUMMARY_CSV}"
    )


if __name__ == "__main__":
    main()
