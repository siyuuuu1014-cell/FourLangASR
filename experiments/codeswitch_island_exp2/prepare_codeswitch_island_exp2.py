from pathlib import Path
import csv
import random

import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(r"D:\dev\projects\FourLangAsrMobile")

SOURCE_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "speaker_exp1"
    / "speaker_data_exp4"
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

NEG_DIR = BASE_DIR / "negative_data"
NEG_MANIFEST = BASE_DIR / "negative_manifest.csv"

SAMPLE_RATE = 16000
LANGUAGES = ["zh", "en", "ru", "uz"]

# Pure single-language negative controls.
# 5 clips per language = 20 negatives.
NUM_NEG_PER_LANG = 5

# Normalize every negative control to a product-like utterance length.
NEGATIVE_SEC = 4.8
NEGATIVE_SAMPLES = int(NEGATIVE_SEC * SAMPLE_RATE)

RANDOM_SEED = 20260908


def load_mono_16k(path: Path) -> np.ndarray:
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

    return np.ascontiguousarray(x, dtype=np.float32)


def center_crop_or_pad(
    x: np.ndarray,
    target_samples: int,
) -> np.ndarray:

    if len(x) == target_samples:
        return x.copy()

    if len(x) > target_samples:
        start = (len(x) - target_samples) // 2

        return np.ascontiguousarray(
            x[start:start + target_samples],
            dtype=np.float32,
        )

    out = np.zeros(
        target_samples,
        dtype=np.float32,
    )

    left = (
        target_samples - len(x)
    ) // 2

    out[
        left:left + len(x)
    ] = x

    return out


def collect_wavs(
    lang: str,
):
    wavs = sorted(
        (SOURCE_DIR / lang)
        .glob(
            "speaker_*/test/*.wav"
        )
    )

    if len(wavs) < NUM_NEG_PER_LANG:
        raise RuntimeError(
            f"Not enough source WAVs for {lang}: "
            f"found {len(wavs)}"
        )

    return wavs


def main():

    random.seed(
        RANDOM_SEED
    )

    exp1_manifest = (
        EXP1_DIR /
        "island_manifest.csv"
    )

    if not exp1_manifest.is_file():
        raise FileNotFoundError(
            "EXP1 island dataset is required first: "
            f"{exp1_manifest}"
        )

    NEG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for old in NEG_DIR.glob(
        "*.wav"
    ):
        old.unlink()

    rows = []
    uid = 0

    for lang in LANGUAGES:

        wavs = collect_wavs(
            lang
        )

        random.shuffle(
            wavs
        )

        for i in range(
            NUM_NEG_PER_LANG
        ):
            uid += 1

            src = wavs[i]

            x = center_crop_or_pad(
                load_mono_16k(
                    src
                ),
                NEGATIVE_SAMPLES,
            )

            name = (
                f"neg_{uid:03d}_"
                f"{lang}_"
                f"{i+1:02d}.wav"
            )

            out_path = (
                NEG_DIR /
                name
            )

            sf.write(
                out_path,
                x,
                SAMPLE_RATE,
                subtype="PCM_16",
            )

            rows.append(
                {
                    "id": uid,
                    "file": name,
                    "main_lang": lang,
                    "duration_sec":
                        f"{len(x)/SAMPLE_RATE:.6f}",
                    "source":
                        str(src),
                }
            )

            print(
                f"{uid:03d} | "
                f"NEG {lang.upper()} | "
                f"{len(x)/SAMPLE_RATE:.2f}s"
            )

    with NEG_MANIFEST.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    print()
    print("=" * 88)
    print(
        "CODE-SWITCH ISLAND EXP2 "
        "NEGATIVE DATASET DONE"
    )
    print("=" * 88)
    print(
        f"Negatives : {len(rows)}"
    )
    print(
        f"WAV dir   : {NEG_DIR}"
    )
    print(
        f"Manifest  : {NEG_MANIFEST}"
    )
    print()
    print(
        "EXP1 positive island samples are reused from:"
    )
    print(
        EXP1_DIR / "island_data"
    )


if __name__ == "__main__":
    main()
