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

BASE_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "codeswitch_island_exp1"
)

OUT_DIR = BASE_DIR / "island_data"
MANIFEST = BASE_DIR / "island_manifest.csv"

SAMPLE_RATE = 16000
LANGUAGES = ["zh", "en", "ru", "uz"]

# Main-language context on both sides.
MAIN_LEFT_SEC = 2.0
MAIN_RIGHT_SEC = 2.0

# Two "foreign island" durations per ordered language pair.
# Roughly corresponds to one short word / one longer word or two short words.
FOREIGN_DURATIONS_SEC = [0.8, 1.2]

# Small crossfade to avoid an artificial hard click.
CROSSFADE_SEC = 0.03

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


def crossfade_concat(
    a: np.ndarray,
    b: np.ndarray,
    fade_samples: int,
) -> np.ndarray:

    if fade_samples <= 0:
        return np.concatenate(
            [a, b]
        ).astype(np.float32)

    n = min(
        fade_samples,
        len(a),
        len(b),
    )

    t = np.linspace(
        0.0,
        1.0,
        n,
        dtype=np.float32,
    )

    cross = (
        a[-n:] * (1.0 - t)
        + b[:n] * t
    )

    return np.concatenate(
        [
            a[:-n],
            cross,
            b[n:],
        ]
    ).astype(np.float32)


def collect_wavs(
    lang: str,
):
    wavs = sorted(
        (SOURCE_DIR / lang)
        .glob(
            "speaker_*/test/*.wav"
        )
    )

    if len(wavs) < 10:
        raise RuntimeError(
            f"Not enough WAV files for {lang}: "
            f"found {len(wavs)}"
        )

    return wavs


def main():

    random.seed(
        RANDOM_SEED
    )

    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(
            SOURCE_DIR
        )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for old in OUT_DIR.glob(
        "*.wav"
    ):
        old.unlink()

    source = {
        lang: collect_wavs(lang)
        for lang in LANGUAGES
    }

    left_n = int(
        MAIN_LEFT_SEC * SAMPLE_RATE
    )

    right_n = int(
        MAIN_RIGHT_SEC * SAMPLE_RATE
    )

    fade_n = int(
        CROSSFADE_SEC * SAMPLE_RATE
    )

    rows = []
    uid = 0

    for main_lang in LANGUAGES:

        for foreign_lang in LANGUAGES:

            if (
                main_lang ==
                foreign_lang
            ):
                continue

            main_pool = (
                source[main_lang][:]
            )

            foreign_pool = (
                source[foreign_lang][:]
            )

            random.shuffle(
                main_pool
            )

            random.shuffle(
                foreign_pool
            )

            for i, foreign_sec in enumerate(
                FOREIGN_DURATIONS_SEC
            ):

                uid += 1

                # Use DIFFERENT main-language source utterances
                # before and after the island.
                left_path = (
                    main_pool[
                        (2 * i) %
                        len(main_pool)
                    ]
                )

                right_path = (
                    main_pool[
                        (2 * i + 1) %
                        len(main_pool)
                    ]
                )

                foreign_path = (
                    foreign_pool[
                        i %
                        len(foreign_pool)
                    ]
                )

                left = center_crop_or_pad(
                    load_mono_16k(
                        left_path
                    ),
                    left_n,
                )

                foreign_n = int(
                    foreign_sec *
                    SAMPLE_RATE
                )

                foreign = center_crop_or_pad(
                    load_mono_16k(
                        foreign_path
                    ),
                    foreign_n,
                )

                right = center_crop_or_pad(
                    load_mono_16k(
                        right_path
                    ),
                    right_n,
                )

                first = crossfade_concat(
                    left,
                    foreign,
                    fade_n,
                )

                mixed = crossfade_concat(
                    first,
                    right,
                    fade_n,
                )

                # Approximate true island boundaries at the center
                # of each crossfade.
                island_start_sample = (
                    left_n -
                    fade_n // 2
                )

                island_end_sample = (
                    left_n +
                    foreign_n -
                    (3 * fade_n) // 2
                )

                island_start_sec = (
                    island_start_sample /
                    SAMPLE_RATE
                )

                island_end_sec = (
                    island_end_sample /
                    SAMPLE_RATE
                )

                name = (
                    f"{uid:03d}_"
                    f"{main_lang}_"
                    f"with_{foreign_lang}_"
                    f"{foreign_sec:.1f}s.wav"
                )

                out_path = (
                    OUT_DIR / name
                )

                sf.write(
                    out_path,
                    mixed,
                    SAMPLE_RATE,
                    subtype="PCM_16",
                )

                rows.append(
                    {
                        "id": uid,
                        "file": name,
                        "main_lang":
                            main_lang,
                        "foreign_lang":
                            foreign_lang,
                        "foreign_duration_sec":
                            f"{foreign_sec:.3f}",
                        "island_start_sec":
                            f"{island_start_sec:.6f}",
                        "island_end_sec":
                            f"{island_end_sec:.6f}",
                        "island_center_sec":
                            f"{(
                                (
                                    island_start_sec +
                                    island_end_sec
                                ) / 2.0
                            ):.6f}",
                        "total_duration_sec":
                            f"{(
                                len(mixed) /
                                SAMPLE_RATE
                            ):.6f}",
                        "left_source":
                            str(left_path),
                        "foreign_source":
                            str(
                                foreign_path
                            ),
                        "right_source":
                            str(right_path),
                    }
                )

                print(
                    f"{uid:03d} | "
                    f"main={main_lang.upper()} | "
                    f"island={foreign_lang.upper()} "
                    f"{foreign_sec:.1f}s | "
                    f"{island_start_sec:.2f}"
                    f"-{island_end_sec:.2f}s"
                )

    with MANIFEST.open(
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
        "CODE-SWITCH ISLAND EXP1 "
        "DATASET DONE"
    )
    print("=" * 88)
    print(
        f"Samples : {len(rows)}"
    )
    print(
        f"WAV dir : {OUT_DIR}"
    )
    print(
        f"Manifest: {MANIFEST}"
    )


if __name__ == "__main__":
    main()
