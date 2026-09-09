from __future__ import annotations

import argparse
import csv
import io
import os
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


PROJECT_ROOT_DEFAULT = Path(r"D:\dev\projects\FourLangAsrMobile")

DATASET_NAME = "CAiRE/ASCEND"
DATASET_CONFIG = "main"
SPLITS = ("validation", "test", "train")

SAMPLE_RATE = 16000

# Product-oriented first pass.
POSITIVE_TARGET = 30
NEGATIVE_ZH_TARGET = 15
NEGATIVE_EN_TARGET = 15

MIN_DURATION_SEC = 1.5
MAX_DURATION_SEC = 7.5

# "Chinese main sentence + a small English island"
MIN_CHINESE_CHARS = 4
MIN_ENGLISH_WORDS = 1
MAX_ENGLISH_WORDS = 3
MAX_ENGLISH_LETTERS = 24

CHINESE_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
EN_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

# A single English island may contain spaces/punctuation,
# but no Chinese characters between its first and last English word.
ALLOWED_INSIDE_ENGLISH_ISLAND_RE = re.compile(
    r"^[A-Za-z\s'’\-.,!?]+$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream a small natural Chinese-English code-switch subset "
            "from CAiRE/ASCEND without downloading the full dataset."
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT_DEFAULT,
    )

    parser.add_argument(
        "--endpoint",
        default="https://hf-mirror.com",
        help=(
            "Hugging Face endpoint. Default uses hf-mirror.com. "
            "Use https://huggingface.co if the mirror fails."
        ),
    )

    parser.add_argument(
        "--positive",
        type=int,
        default=POSITIVE_TARGET,
    )

    parser.add_argument(
        "--negative-zh",
        type=int,
        default=NEGATIVE_ZH_TARGET,
    )

    parser.add_argument(
        "--negative-en",
        type=int,
        default=NEGATIVE_EN_TARGET,
    )

    return parser.parse_args()


def configure_huggingface(endpoint: str) -> None:
    # Must happen before importing datasets/huggingface_hub.
    os.environ["HF_ENDPOINT"] = endpoint
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    # Avoid Xet-specific transfer problems on some Windows/network setups.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def count_chinese(text: str) -> int:
    return len(CHINESE_RE.findall(text))


def english_matches(text: str):
    return list(EN_WORD_RE.finditer(text))


def is_single_internal_english_island(text: str) -> Tuple[bool, Dict[str, object]]:
    """
    Select real examples shaped roughly like:

        中文 ... English/English phrase ... 中文

    Conditions:
    - Chinese-dominant
    - 1-3 English words
    - English island is internal, not only at utterance edge
    - only one contiguous English island
    """
    text = text.strip()

    matches = english_matches(text)
    zh_count = count_chinese(text)

    if not (
        MIN_ENGLISH_WORDS
        <= len(matches)
        <= MAX_ENGLISH_WORDS
    ):
        return False, {}

    if zh_count < MIN_CHINESE_CHARS:
        return False, {}

    english_letters = sum(
        len(re.sub(r"[^A-Za-z]", "", m.group(0)))
        for m in matches
    )

    if english_letters > MAX_ENGLISH_LETTERS:
        return False, {}

    first = matches[0]
    last = matches[-1]

    before = text[: first.start()]
    after = text[last.end() :]

    # Sandwich requirement: Chinese before AND after the English island.
    if count_chinese(before) < 1:
        return False, {}

    if count_chinese(after) < 1:
        return False, {}

    island_text = text[first.start() : last.end()]

    # If Chinese appears between English words, then we have multiple islands.
    if CHINESE_RE.search(island_text):
        return False, {}

    # Keep the first experiment simple and interpretable.
    if not ALLOWED_INSIDE_ENGLISH_ISLAND_RE.match(island_text):
        return False, {}

    # Ensure Chinese is the dominant transcript component.
    # This is intentionally simple; it is only a dataset filter.
    if zh_count < 2 * len(matches):
        return False, {}

    return True, {
        "english_words": " ".join(
            m.group(0)
            for m in matches
        ),
        "english_word_count": len(matches),
        "chinese_char_count": zh_count,
        "island_char_start": first.start(),
        "island_char_end": last.end(),
    }


def row_duration(row: dict) -> float:
    value = row.get("duration")

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def is_duration_ok(row: dict) -> bool:
    d = row_duration(row)
    return MIN_DURATION_SEC <= d <= MAX_DURATION_SEC


def read_audio_payload(
    audio_obj: dict,
):
    """
    Read Audio(decode=False) without torchcodec.

    Hugging Face parquet audio commonly gives:
      {"bytes": <encoded audio bytes>, "path": "...wav"}

    We decode with soundfile ourselves, so this script does NOT require
    torchcodec.
    """
    import numpy as np
    import soundfile as sf

    if not isinstance(audio_obj, dict):
        raise RuntimeError(
            f"Unexpected audio object: {type(audio_obj)}"
        )

    raw_bytes = audio_obj.get("bytes")
    audio_path = audio_obj.get("path")

    if raw_bytes:
        samples, sr = sf.read(
            io.BytesIO(raw_bytes),
            dtype="float32",
            always_2d=False,
        )
    elif audio_path and Path(audio_path).is_file():
        samples, sr = sf.read(
            audio_path,
            dtype="float32",
            always_2d=False,
        )
    else:
        raise RuntimeError(
            "Audio row contains neither embedded bytes nor a readable local path. "
            "Try the official endpoint with:\n"
            "  --endpoint https://huggingface.co"
        )

    samples = np.asarray(
        samples,
        dtype=np.float32,
    )

    if samples.ndim == 2:
        samples = samples.mean(
            axis=1
        ).astype(
            np.float32
        )

    if sr != SAMPLE_RATE:
        raise RuntimeError(
            f"ASCEND sample rate changed unexpectedly: {sr} Hz "
            f"(expected {SAMPLE_RATE} Hz)"
        )

    return samples, sr


def save_as_wav(
    audio_obj: dict,
    destination: Path,
) -> None:
    import soundfile as sf

    samples, sr = read_audio_payload(
        audio_obj
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sf.write(
        destination,
        samples,
        sr,
        subtype="PCM_16",
    )


def stream_split(
    split_name: str,
):
    from datasets import Audio, load_dataset

    print()
    print(
        f"Opening ASCEND split '{split_name}' "
        f"in streaming mode..."
    )

    ds = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=split_name,
        streaming=True,
    )

    # Critical:
    # decode=False prevents datasets from invoking torchcodec.
    ds = ds.cast_column(
        "audio",
        Audio(
            sampling_rate=SAMPLE_RATE,
            decode=False,
        ),
    )

    return ds


def main() -> None:
    args = parse_args()

    configure_huggingface(
        args.endpoint
    )

    # Import only after HF endpoint is configured.
    import datasets  # noqa: F401
    import numpy  # noqa: F401
    import soundfile  # noqa: F401

    base_dir = (
        args.project_root
        / "experiments"
        / "codeswitch_natural_exp4"
    )

    positive_dir = (
        base_dir
        / "audio"
        / "positive_zh_en"
    )

    negative_zh_dir = (
        base_dir
        / "audio"
        / "negative_zh"
    )

    negative_en_dir = (
        base_dir
        / "audio"
        / "negative_en"
    )

    manifest_path = (
        base_dir
        / "ascend_natural_manifest.csv"
    )

    base_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for directory in [
        positive_dir,
        negative_zh_dir,
        negative_en_dir,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Clean only files previously generated by this script.
        for old in directory.glob(
            "ascend_*.wav"
        ):
            old.unlink()

    target = {
        "positive_zh_en":
            args.positive,
        "negative_zh":
            args.negative_zh,
        "negative_en":
            args.negative_en,
    }

    counts = {
        key: 0
        for key in target
    }

    rows = []
    seen_ids = set()

    def complete() -> bool:
        return all(
            counts[key] >=
            target[key]
            for key in target
        )

    print("=" * 88)
    print(
        "ASCEND NATURAL CODE-SWITCH SUBSET"
    )
    print("=" * 88)
    print(
        f"Endpoint  : {args.endpoint}"
    )
    print(
        "Mode      : streaming=True "
        "(do not download full ~1.2 GB corpus)"
    )
    print(
        f"Positive  : {target['positive_zh_en']} "
        "Chinese-main + 1-3 English words"
    )
    print(
        f"NegativeZH: {target['negative_zh']}"
    )
    print(
        f"NegativeEN: {target['negative_en']}"
    )

    for split_name in SPLITS:
        if complete():
            break

        ds = stream_split(
            split_name
        )

        scanned = 0

        for row in ds:
            scanned += 1

            row_id = str(
                row.get("id", "")
            )

            unique_id = (
                split_name,
                row_id,
            )

            if unique_id in seen_ids:
                continue

            seen_ids.add(
                unique_id
            )

            transcript = str(
                row.get(
                    "transcription",
                    "",
                )
            ).strip()

            language = str(
                row.get(
                    "language",
                    "",
                )
            ).lower()

            if not transcript:
                continue

            if not is_duration_ok(
                row
            ):
                continue

            category: Optional[str] = None
            metadata: Dict[str, object] = {}

            if (
                language == "mixed"
                and
                counts[
                    "positive_zh_en"
                ] <
                target[
                    "positive_zh_en"
                ]
            ):
                ok, metadata = (
                    is_single_internal_english_island(
                        transcript
                    )
                )

                if ok:
                    category = (
                        "positive_zh_en"
                    )

            elif (
                language == "zh"
                and
                counts[
                    "negative_zh"
                ] <
                target[
                    "negative_zh"
                ]
            ):
                # Strict pure-Chinese negative:
                # no Latin word token.
                if (
                    count_chinese(
                        transcript
                    ) >=
                    MIN_CHINESE_CHARS
                    and
                    not EN_WORD_RE.search(
                        transcript
                    )
                ):
                    category = (
                        "negative_zh"
                    )

            elif (
                language == "en"
                and
                counts[
                    "negative_en"
                ] <
                target[
                    "negative_en"
                ]
            ):
                # Strict pure-English negative:
                # no Chinese character.
                if (
                    len(
                        english_matches(
                            transcript
                        )
                    ) >= 3
                    and
                    not CHINESE_RE.search(
                        transcript
                    )
                ):
                    category = (
                        "negative_en"
                    )

            if category is None:
                continue

            index = (
                counts[category]
                + 1
            )

            if category == "positive_zh_en":
                out_dir = (
                    positive_dir
                )
            elif category == "negative_zh":
                out_dir = (
                    negative_zh_dir
                )
            else:
                out_dir = (
                    negative_en_dir
                )

            filename = (
                f"ascend_{category}_"
                f"{index:03d}_"
                f"{split_name}_"
                f"{row_id}.wav"
            )

            destination = (
                out_dir
                / filename
            )

            try:
                save_as_wav(
                    row["audio"],
                    destination,
                )
            except Exception as exc:
                print(
                    f"  audio save failed "
                    f"{split_name}/{row_id}: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue

            counts[category] += 1

            manifest_row = {
                "category":
                    category,
                "split":
                    split_name,
                "id":
                    row_id,
                "file":
                    str(
                        destination
                        .relative_to(
                            base_dir
                        )
                    ),
                "duration_sec":
                    f"{row_duration(row):.3f}",
                "language_label":
                    language,
                "transcription":
                    transcript,
                "english_words":
                    metadata.get(
                        "english_words",
                        "",
                    ),
                "english_word_count":
                    metadata.get(
                        "english_word_count",
                        "",
                    ),
                "chinese_char_count":
                    metadata.get(
                        "chinese_char_count",
                        "",
                    ),
                "speaker_id":
                    row.get(
                        "original_speaker_id",
                        "",
                    ),
                "session_id":
                    row.get(
                        "session_id",
                        "",
                    ),
                "topic":
                    row.get(
                        "topic",
                        "",
                    ),
            }

            rows.append(
                manifest_row
            )

            print(
                f"  {category:14s} "
                f"{counts[category]:02d}/"
                f"{target[category]:02d} | "
                f"{row_duration(row):4.2f}s | "
                f"{transcript}"
            )

            if complete():
                break

        print(
            f"Scanned {scanned} rows "
            f"from {split_name}."
        )

    fieldnames = [
        "category",
        "split",
        "id",
        "file",
        "duration_sec",
        "language_label",
        "transcription",
        "english_words",
        "english_word_count",
        "chinese_char_count",
        "speaker_id",
        "session_id",
        "topic",
    ]

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    print()
    print("=" * 88)
    print(
        "DONE"
        if complete()
        else
        "PARTIAL RESULT"
    )
    print("=" * 88)
    print(
        f"positive_zh_en : "
        f"{counts['positive_zh_en']}/"
        f"{target['positive_zh_en']}"
    )
    print(
        f"negative_zh    : "
        f"{counts['negative_zh']}/"
        f"{target['negative_zh']}"
    )
    print(
        f"negative_en    : "
        f"{counts['negative_en']}/"
        f"{target['negative_en']}"
    )
    print(
        f"Manifest       : "
        f"{manifest_path}"
    )
    print(
        f"Audio root     : "
        f"{base_dir / 'audio'}"
    )

    if not complete():
        print()
        print(
            "If the mirror returned too few rows or audio payload errors, "
            "rerun with:"
        )
        print(
            "  python .\\prepare_ascend_natural_exp4.py "
            "--endpoint https://huggingface.co"
        )


if __name__ == "__main__":
    main()
