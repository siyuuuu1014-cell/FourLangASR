from pathlib import Path
import csv

import numpy as np
import soundfile as sf
import sherpa_onnx


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "speaker_embedding.onnx"
)

DATA_DIR = (
    BASE_DIR
    / "speaker_data"
)

RESULTS_CSV = (
    BASE_DIR
    / "speaker_results.csv"
)

SUMMARY_CSV = (
    BASE_DIR
    / "speaker_summary.csv"
)


# ============================================================
# Languages
# ============================================================

LANGUAGES = [
    "zh",
    "en",
    "ru",
    "uz",
]


# ============================================================
# Current candidate decision rule
# ============================================================

SCORE_THRESHOLD = 0.45
MARGIN_THRESHOLD = 0.08


# ============================================================
# Audio
# ============================================================

def load_audio(
    path,
):

    samples, sample_rate = sf.read(
        path,
        dtype="float32",
        always_2d=False,
    )

    samples = np.asarray(
        samples,
        dtype=np.float32,
    )

    if samples.ndim == 2:

        samples = samples.mean(
            axis=1,
            dtype=np.float32,
        )

    samples = np.ascontiguousarray(
        samples,
        dtype=np.float32,
    )

    if sample_rate != 16000:

        raise RuntimeError(
            f"{path}: "
            f"expected 16000 Hz, "
            f"got {sample_rate}"
        )

    return (
        samples,
        sample_rate,
    )


# ============================================================
# Embedding
# ============================================================

def normalize(
    x,
):

    x = np.asarray(
        x,
        dtype=np.float32,
    )

    norm = float(
        np.linalg.norm(x)
    )

    if norm < 1e-12:

        raise RuntimeError(
            "Zero embedding"
        )

    return (
        x / norm
    )


def compute_embedding(
    extractor,
    path,
):

    samples, sample_rate = (
        load_audio(path)
    )

    stream = (
        extractor
        .create_stream()
    )

    stream.accept_waveform(
        sample_rate=sample_rate,
        waveform=samples,
    )

    stream.input_finished()

    if not extractor.is_ready(
        stream
    ):

        raise RuntimeError(
            "Extractor not ready: "
            f"{path}"
        )

    embedding = (
        extractor.compute(
            stream
        )
    )

    return normalize(
        embedding
    )


def cosine(
    a,
    b,
):

    return float(
        np.dot(
            normalize(a),
            normalize(b),
        )
    )


# ============================================================
# Speaker discovery
# ============================================================

def discover_speakers(
    language_dir,
):

    speakers = []

    for path in sorted(
        language_dir.glob(
            "speaker_*"
        )
    ):

        if not path.is_dir():
            continue

        enroll = (
            path
            / "enroll"
        )

        test = (
            path
            / "test"
        )

        if (
            enroll.is_dir()
            and
            test.is_dir()
        ):

            speakers.append(
                path
            )

    return speakers


# ============================================================
# Enrollment
# ============================================================

def build_profiles(
    extractor,
    language,
):

    language_dir = (
        DATA_DIR
        / language
    )

    speakers = (
        discover_speakers(
            language_dir
        )
    )

    if len(speakers) != 4:

        raise RuntimeError(
            f"{language}: "
            f"expected 4 speakers, "
            f"got {len(speakers)}"
        )

    profiles = {}

    for speaker_dir in speakers:

        name = (
            speaker_dir.name
        )

        files = sorted(
            (
                speaker_dir
                / "enroll"
            ).glob(
                "*.wav"
            )
        )

        if not files:

            raise RuntimeError(
                f"{language}/{name}: "
                "no enroll wav"
            )

        embeddings = []

        for path in files:

            embeddings.append(
                compute_embedding(
                    extractor,
                    path,
                )
            )

        centroid = np.mean(
            np.stack(
                embeddings
            ),
            axis=0,
        )

        profiles[
            name
        ] = normalize(
            centroid
        )

    return (
        profiles,
        speakers,
    )


# ============================================================
# One test sample
# ============================================================

def evaluate_embedding(
    embedding,
    profiles,
):

    scores = {
        speaker:
            cosine(
                embedding,
                profile,
            )
        for speaker, profile
        in profiles.items()
    }

    ranking = sorted(
        scores.items(),
        key=lambda x:
            x[1],
        reverse=True,
    )

    best_name = (
        ranking[0][0]
    )

    best_score = (
        ranking[0][1]
    )

    second_name = (
        ranking[1][0]
    )

    second_score = (
        ranking[1][1]
    )

    margin = (
        best_score
        - second_score
    )

    accepted = (
        best_score
        >= SCORE_THRESHOLD
        and
        margin
        >= MARGIN_THRESHOLD
    )

    return {
        "scores":
            scores,

        "best_name":
            best_name,

        "best_score":
            best_score,

        "second_name":
            second_name,

        "second_score":
            second_score,

        "margin":
            margin,

        "accepted":
            accepted,
    }


# ============================================================
# Evaluate one language
# ============================================================

def evaluate_language(
    extractor,
    language,
):

    print()
    print(
        "=" * 80
    )

    print(
        f"LANGUAGE: "
        f"{language.upper()}"
    )

    print(
        "=" * 80
    )

    profiles, speakers = (
        build_profiles(
            extractor,
            language,
        )
    )

    rows = []

    pure_correct = 0
    accepted_correct = 0
    accepted_wrong = 0
    rejected_known = 0

    total = 0

    best_scores = []
    margins = []

    for speaker_dir in speakers:

        true_name = (
            speaker_dir.name
        )

        test_files = sorted(
            (
                speaker_dir
                / "test"
            ).glob(
                "*.wav"
            )
        )

        for path in test_files:

            total += 1

            embedding = (
                compute_embedding(
                    extractor,
                    path,
                )
            )

            result = (
                evaluate_embedding(
                    embedding,
                    profiles,
                )
            )

            top1_correct = (
                result[
                    "best_name"
                ]
                == true_name
            )

            if top1_correct:
                pure_correct += 1

            if result["accepted"]:

                if top1_correct:

                    accepted_correct += 1

                else:

                    accepted_wrong += 1

            else:

                rejected_known += 1

            best_scores.append(
                result[
                    "best_score"
                ]
            )

            margins.append(
                result[
                    "margin"
                ]
            )

            row = {
                "language":
                    language,

                "true":
                    true_name,

                "filename":
                    path.name,

                "path":
                    str(path),

                "best_name":
                    result[
                        "best_name"
                    ],

                "best_score":
                    result[
                        "best_score"
                    ],

                "second_name":
                    result[
                        "second_name"
                    ],

                "second_score":
                    result[
                        "second_score"
                    ],

                "margin":
                    result[
                        "margin"
                    ],

                "top1_correct":
                    top1_correct,

                "accepted":
                    result[
                        "accepted"
                    ],

                "decision":
                    (
                        result[
                            "best_name"
                        ]
                        if result[
                            "accepted"
                        ]
                        else "unknown"
                    ),
            }

            for speaker_name in sorted(
                profiles.keys()
            ):

                row[
                    f"sim_{speaker_name}"
                ] = (
                    result[
                        "scores"
                    ][
                        speaker_name
                    ]
                )

            rows.append(
                row
            )

            print(
                f"{true_name}/"
                f"{path.name} | "
                f"best="
                f"{result['best_name']} "
                f"{result['best_score']:.4f} | "
                f"margin="
                f"{result['margin']:.4f} | "
                f"{'ACCEPT' if result['accepted'] else 'REJECT'} | "
                f"{'OK' if top1_correct else 'WRONG'}"
            )

    pure_accuracy = (
        pure_correct
        / total
        if total
        else 0.0
    )

    accepted_correct_rate = (
        accepted_correct
        / total
        if total
        else 0.0
    )

    accepted_wrong_rate = (
        accepted_wrong
        / total
        if total
        else 0.0
    )

    reject_rate = (
        rejected_known
        / total
        if total
        else 0.0
    )

    summary = {
        "language":
            language,

        "total":
            total,

        "pure_correct":
            pure_correct,

        "pure_accuracy":
            pure_accuracy,

        "accepted_correct":
            accepted_correct,

        "accepted_correct_rate":
            accepted_correct_rate,

        "accepted_wrong":
            accepted_wrong,

        "accepted_wrong_rate":
            accepted_wrong_rate,

        "rejected_known":
            rejected_known,

        "reject_rate":
            reject_rate,

        "best_score_mean":
            float(
                np.mean(
                    best_scores
                )
            ),

        "best_score_min":
            float(
                np.min(
                    best_scores
                )
            ),

        "best_score_max":
            float(
                np.max(
                    best_scores
                )
            ),

        "margin_mean":
            float(
                np.mean(
                    margins
                )
            ),

        "margin_min":
            float(
                np.min(
                    margins
                )
            ),

        "margin_max":
            float(
                np.max(
                    margins
                )
            ),
    }

    print()
    print(
        f"Pure Top-1       : "
        f"{pure_correct}/"
        f"{total} "
        f"= "
        f"{pure_accuracy*100:.2f}%"
    )

    print(
        f"Fixed rule       : "
        f"score >= "
        f"{SCORE_THRESHOLD:.2f}, "
        f"margin >= "
        f"{MARGIN_THRESHOLD:.2f}"
    )

    print(
        f"Accepted correct : "
        f"{accepted_correct}/"
        f"{total} "
        f"= "
        f"{accepted_correct_rate*100:.2f}%"
    )

    print(
        f"Accepted wrong   : "
        f"{accepted_wrong}/"
        f"{total} "
        f"= "
        f"{accepted_wrong_rate*100:.2f}%"
    )

    print(
        f"Known rejected   : "
        f"{rejected_known}/"
        f"{total} "
        f"= "
        f"{reject_rate*100:.2f}%"
    )

    print(
        f"Best score mean  : "
        f"{summary['best_score_mean']:.4f}"
    )

    print(
        f"Margin mean      : "
        f"{summary['margin_mean']:.4f}"
    )

    return (
        rows,
        summary,
    )


# ============================================================
# Main
# ============================================================

def main():

    if not MODEL_PATH.is_file():

        raise FileNotFoundError(
            MODEL_PATH
        )

    if not DATA_DIR.is_dir():

        raise FileNotFoundError(
            DATA_DIR
        )

    config = (
        sherpa_onnx
        .SpeakerEmbeddingExtractorConfig(
            model=str(
                MODEL_PATH
            ),
            num_threads=1,
            provider="cpu",
            debug=False,
        )
    )

    if not config.validate():

        raise RuntimeError(
            "Invalid speaker embedding model"
        )

    extractor = (
        sherpa_onnx
        .SpeakerEmbeddingExtractor(
            config
        )
    )

    print(
        "=" * 80
    )

    print(
        "Speaker EXP4-A "
        "- Four Language Robustness"
    )

    print(
        "=" * 80
    )

    print()

    print(
        f"Embedding dim : "
        f"{extractor.dim}"
    )

    print(
        f"Score rule    : "
        f">= {SCORE_THRESHOLD}"
    )

    print(
        f"Margin rule   : "
        f">= {MARGIN_THRESHOLD}"
    )

    all_rows = []
    summaries = []

    for language in LANGUAGES:

        rows, summary = (
            evaluate_language(
                extractor,
                language,
            )
        )

        all_rows.extend(
            rows
        )

        summaries.append(
            summary
        )

    # ========================================================
    # Result CSV
    # ========================================================

    speaker_columns = [
        "sim_speaker_1",
        "sim_speaker_2",
        "sim_speaker_3",
        "sim_speaker_4",
    ]

    result_fields = [
        "language",
        "true",
        "filename",
        "path",
        *speaker_columns,
        "best_name",
        "best_score",
        "second_name",
        "second_score",
        "margin",
        "top1_correct",
        "accepted",
        "decision",
    ]

    with RESULTS_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=
                result_fields,
        )

        writer.writeheader()

        for row in all_rows:

            writer.writerow(
                {
                    field:
                        row.get(
                            field,
                            ""
                        )
                    for field
                    in result_fields
                }
            )

    # ========================================================
    # Summary CSV
    # ========================================================

    summary_fields = [
        "language",
        "total",
        "pure_correct",
        "pure_accuracy",
        "accepted_correct",
        "accepted_correct_rate",
        "accepted_wrong",
        "accepted_wrong_rate",
        "rejected_known",
        "reject_rate",
        "best_score_mean",
        "best_score_min",
        "best_score_max",
        "margin_mean",
        "margin_min",
        "margin_max",
    ]

    with SUMMARY_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=
                summary_fields,
        )

        writer.writeheader()

        writer.writerows(
            summaries
        )

    # ========================================================
    # Overall
    # ========================================================

    total = sum(
        x["total"]
        for x in summaries
    )

    pure_correct = sum(
        x["pure_correct"]
        for x in summaries
    )

    accepted_correct = sum(
        x["accepted_correct"]
        for x in summaries
    )

    accepted_wrong = sum(
        x["accepted_wrong"]
        for x in summaries
    )

    rejected = sum(
        x["rejected_known"]
        for x in summaries
    )

    print()
    print(
        "=" * 80
    )

    print(
        "OVERALL"
    )

    print(
        "=" * 80
    )

    print()

    print(
        f"Total test       : "
        f"{total}"
    )

    print(
        f"Pure Top-1       : "
        f"{pure_correct}/"
        f"{total} "
        f"= "
        f"{pure_correct/total*100:.2f}%"
    )

    print(
        f"Accepted correct : "
        f"{accepted_correct}/"
        f"{total} "
        f"= "
        f"{accepted_correct/total*100:.2f}%"
    )

    print(
        f"Accepted wrong   : "
        f"{accepted_wrong}/"
        f"{total} "
        f"= "
        f"{accepted_wrong/total*100:.2f}%"
    )

    print(
        f"Known rejected   : "
        f"{rejected}/"
        f"{total} "
        f"= "
        f"{rejected/total*100:.2f}%"
    )

    print()

    print(
        f"Results:"
    )

    print(
        RESULTS_CSV
    )

    print()

    print(
        f"Summary:"
    )

    print(
        SUMMARY_CSV
    )


if __name__ == "__main__":

    main()