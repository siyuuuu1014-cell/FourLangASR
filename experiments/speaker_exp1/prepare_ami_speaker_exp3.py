from pathlib import Path
import csv
import xml.etree.ElementTree as ET

import numpy as np
import soundfile as sf


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MEETING = "ES2003a"

WORDS_DIR = BASE_DIR / "words"

AUDIO_PATH = (
    BASE_DIR
    / "ami_test"
    / f"{MEETING}.Mix-Headset.wav"
)

OUTPUT_DIR = (
    BASE_DIR
    / "speaker_unknown_exp3"
)

MANIFEST_PATH = (
    BASE_DIR
    / "speaker_exp3_unknown_manifest.csv"
)


# ============================================================
# Speakers
# ============================================================

SPEAKERS = [
    "A",
    "B",
    "C",
    "D",
]


# ============================================================
# Parameters
# ============================================================

# 每个完全陌生的 Speaker 取 8 条
NUM_TEST_PER_SPEAKER = 8

MIN_SEGMENT_SEC = 2.0
TARGET_SEGMENT_SEC = 3.5
MAX_SEGMENT_SEC = 5.5

MAX_WORD_GAP_SEC = 1.0

ACTIVITY_MERGE_GAP_SEC = 0.30

# 严格排除其他人同时讲话
MAX_OTHER_OVERLAP_SEC = 0.10
MAX_OTHER_OVERLAP_RATIO = 0.03

# 两个最终样本至少相隔 3 秒
MIN_SELECTED_GAP_SEC = 3.0

AUDIO_PADDING_SEC = 0.08

SAMPLE_RATE_REQUIRED = 16000


# ============================================================
# XML helpers
# ============================================================

def local_tag(tag):

    if "}" in tag:
        return tag.split("}", 1)[1]

    return tag


def load_word_intervals(
    speaker,
):

    path = (
        WORDS_DIR
        / f"{MEETING}.{speaker}.words.xml"
    )

    if not path.is_file():

        raise FileNotFoundError(
            "\n找不到标注：\n"
            f"{path}\n"
        )

    tree = ET.parse(path)
    root = tree.getroot()

    intervals = []

    for element in root.iter():

        if local_tag(element.tag) != "w":
            continue

        start_text = element.attrib.get(
            "starttime"
        )

        end_text = element.attrib.get(
            "endtime"
        )

        if (
            start_text is None
            or end_text is None
        ):
            continue

        try:
            start = float(start_text)
            end = float(end_text)

        except ValueError:
            continue

        if end <= start:
            continue

        intervals.append(
            (
                start,
                end,
            )
        )

    intervals.sort(
        key=lambda x: x[0]
    )

    if not intervals:

        raise RuntimeError(
            f"Speaker {speaker} "
            "没有有效 timestamp"
        )

    return intervals


# ============================================================
# Activity
# ============================================================

def merge_intervals(
    intervals,
    max_gap,
):

    if not intervals:
        return []

    merged = []

    current_start = intervals[0][0]
    current_end = intervals[0][1]

    for start, end in intervals[1:]:

        if (
            start - current_end
            <= max_gap
        ):

            current_end = max(
                current_end,
                end,
            )

        else:

            merged.append(
                (
                    current_start,
                    current_end,
                )
            )

            current_start = start
            current_end = end

    merged.append(
        (
            current_start,
            current_end,
        )
    )

    return merged


def overlap_duration(
    start,
    end,
    intervals,
):

    total = 0.0

    for other_start, other_end in intervals:

        if other_end <= start:
            continue

        if other_start >= end:
            break

        overlap_start = max(
            start,
            other_start,
        )

        overlap_end = min(
            end,
            other_end,
        )

        if overlap_end > overlap_start:

            total += (
                overlap_end
                - overlap_start
            )

    return total


# ============================================================
# Candidate generation
# ============================================================

def build_candidates(
    word_intervals,
):

    candidates = []

    n = len(word_intervals)

    for i in range(n):

        start = word_intervals[i][0]
        end = word_intervals[i][1]

        previous_end = end

        best = None

        initial_duration = (
            end - start
        )

        if (
            MIN_SEGMENT_SEC
            <= initial_duration
            <= MAX_SEGMENT_SEC
        ):

            best = {
                "start": start,
                "end": end,
                "duration": initial_duration,
                "distance": abs(
                    initial_duration
                    - TARGET_SEGMENT_SEC
                ),
            }

        for j in range(
            i + 1,
            n,
        ):

            next_start = (
                word_intervals[j][0]
            )

            next_end = (
                word_intervals[j][1]
            )

            gap = (
                next_start
                - previous_end
            )

            if (
                gap > MAX_WORD_GAP_SEC
            ):
                break

            duration = (
                next_end - start
            )

            if (
                duration
                > MAX_SEGMENT_SEC
            ):
                break

            previous_end = next_end

            if (
                duration
                >= MIN_SEGMENT_SEC
            ):

                distance = abs(
                    duration
                    - TARGET_SEGMENT_SEC
                )

                if (
                    best is None
                    or distance
                    < best["distance"]
                ):

                    best = {
                        "start": start,
                        "end": next_end,
                        "duration": duration,
                        "distance": distance,
                    }

        if best is not None:

            candidates.append(
                {
                    "start":
                        best["start"],
                    "end":
                        best["end"],
                    "duration":
                        best["duration"],
                }
            )

    candidates.sort(
        key=lambda x: (
            x["start"],
            x["end"],
        )
    )

    # 去掉非常接近的重复 candidate
    deduplicated = []

    for item in candidates:

        duplicate = False

        for previous in (
            deduplicated[-4:]
        ):

            if (
                abs(
                    item["start"]
                    - previous["start"]
                ) < 0.25
                and
                abs(
                    item["end"]
                    - previous["end"]
                ) < 0.25
            ):

                duplicate = True
                break

        if not duplicate:

            deduplicated.append(
                item
            )

    return deduplicated


# ============================================================
# Remove overlap
# ============================================================

def filter_clean_candidates(
    speaker,
    candidates,
    activity_by_speaker,
):

    clean = []

    for item in candidates:

        start = item["start"]
        end = item["end"]

        duration = (
            end - start
        )

        other_overlap = 0.0

        for other in SPEAKERS:

            if other == speaker:
                continue

            other_overlap += (
                overlap_duration(
                    start,
                    end,
                    activity_by_speaker[
                        other
                    ],
                )
            )

        ratio = (
            other_overlap
            / max(
                duration,
                1e-9,
            )
        )

        if (
            other_overlap
            > MAX_OTHER_OVERLAP_SEC
        ):
            continue

        if (
            ratio
            > MAX_OTHER_OVERLAP_RATIO
        ):
            continue

        result = dict(item)

        result[
            "other_overlap"
        ] = other_overlap

        result[
            "overlap_ratio"
        ] = ratio

        clean.append(
            result
        )

    return clean


# ============================================================
# Spacing
# ============================================================

def space_candidates(
    candidates,
):

    if not candidates:
        return []

    candidates = sorted(
        candidates,
        key=lambda x:
            x["start"],
    )

    spaced = []

    for item in candidates:

        if not spaced:

            spaced.append(item)
            continue

        if (
            item["start"]
            - spaced[-1]["start"]
            >= MIN_SELECTED_GAP_SEC
        ):

            spaced.append(
                item
            )

    return spaced


# ============================================================
# Even selection
# ============================================================

def select_evenly(
    candidates,
    wanted,
):

    if len(candidates) <= wanted:
        return list(candidates)

    indices = np.linspace(
        0,
        len(candidates) - 1,
        num=wanted,
    )

    indices = (
        np.rint(indices)
        .astype(int)
        .tolist()
    )

    selected = []

    used = set()

    for index in indices:

        if index in used:
            continue

        used.add(index)

        selected.append(
            candidates[index]
        )

    if len(selected) < wanted:

        for index, item in enumerate(
            candidates
        ):

            if index in used:
                continue

            selected.append(item)
            used.add(index)

            if (
                len(selected)
                >= wanted
            ):
                break

    selected.sort(
        key=lambda x:
            x["start"]
    )

    return selected[:wanted]


# ============================================================
# Audio
# ============================================================

def load_audio():

    if not AUDIO_PATH.is_file():

        raise FileNotFoundError(
            AUDIO_PATH
        )

    audio, sample_rate = sf.read(
        AUDIO_PATH,
        dtype="float32",
        always_2d=False,
    )

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    if audio.ndim == 2:

        audio = audio.mean(
            axis=1,
            dtype=np.float32,
        )

    audio = np.ascontiguousarray(
        audio,
        dtype=np.float32,
    )

    if (
        sample_rate
        != SAMPLE_RATE_REQUIRED
    ):

        raise RuntimeError(
            f"需要 16000 Hz，"
            f"当前 {sample_rate}"
        )

    return (
        audio,
        sample_rate,
    )


def save_segment(
    audio,
    sample_rate,
    item,
    output_path,
):

    audio_duration = (
        len(audio)
        / float(sample_rate)
    )

    start = max(
        0.0,
        item["start"]
        - AUDIO_PADDING_SEC,
    )

    end = min(
        audio_duration,
        item["end"]
        + AUDIO_PADDING_SEC,
    )

    start_sample = int(
        round(
            start
            * sample_rate
        )
    )

    end_sample = int(
        round(
            end
            * sample_rate
        )
    )

    samples = np.ascontiguousarray(
        audio[
            start_sample:
            end_sample
        ],
        dtype=np.float32,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sf.write(
        output_path,
        samples,
        sample_rate,
        subtype="PCM_16",
    )


def clear_output():

    if not OUTPUT_DIR.exists():
        return

    for wav in OUTPUT_DIR.rglob(
        "*.wav"
    ):
        wav.unlink()


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 78)
    print(
        "Speaker EXP3 - Unknown Dataset"
    )
    print("=" * 78)

    print()
    print(
        f"Meeting : {MEETING}"
    )

    print(
        f"Audio   : {AUDIO_PATH}"
    )

    print(
        f"Words   : {WORDS_DIR}"
    )

    print(
        f"Output  : {OUTPUT_DIR}"
    )

    words_by_speaker = {}
    activity_by_speaker = {}

    for speaker in SPEAKERS:

        words = (
            load_word_intervals(
                speaker
            )
        )

        words_by_speaker[
            speaker
        ] = words

        activity_by_speaker[
            speaker
        ] = merge_intervals(
            words,
            ACTIVITY_MERGE_GAP_SEC,
        )

        print(
            f"Speaker {speaker}: "
            f"{len(words)} words"
        )

    audio, sample_rate = (
        load_audio()
    )

    print()
    print(
        "Audio duration: "
        f"{len(audio) / sample_rate:.2f}s"
    )

    clear_output()

    manifest = []

    for speaker in SPEAKERS:

        raw = build_candidates(
            words_by_speaker[
                speaker
            ]
        )

        clean = (
            filter_clean_candidates(
                speaker,
                raw,
                activity_by_speaker,
            )
        )

        spaced = (
            space_candidates(
                clean
            )
        )

        selected = (
            select_evenly(
                spaced,
                NUM_TEST_PER_SPEAKER,
            )
        )

        print()
        print(
            f"Speaker {speaker}"
        )

        print(
            f"  Raw      : {len(raw)}"
        )

        print(
            f"  Clean    : {len(clean)}"
        )

        print(
            f"  Spaced   : {len(spaced)}"
        )

        print(
            f"  Selected : {len(selected)}"
        )

        if (
            len(selected)
            < NUM_TEST_PER_SPEAKER
        ):

            raise RuntimeError(
                f"Speaker {speaker} "
                f"只有 {len(selected)} "
                "个独立干净 segment，"
                f"需要 {NUM_TEST_PER_SPEAKER}"
            )

        for number, item in enumerate(
            selected,
            start=1,
        ):

            output_path = (
                OUTPUT_DIR
                / f"unknown_{speaker}"
                / f"{number:02d}.wav"
            )

            save_segment(
                audio,
                sample_rate,
                item,
                output_path,
            )

            manifest.append(
                {
                    "unknown_speaker":
                        f"unknown_{speaker}",
                    "meeting":
                        MEETING,
                    "filename":
                        output_path.name,
                    "start":
                        item["start"],
                    "end":
                        item["end"],
                    "duration":
                        item["duration"],
                    "other_overlap":
                        item[
                            "other_overlap"
                        ],
                    "overlap_ratio":
                        item[
                            "overlap_ratio"
                        ],
                }
            )

    with MANIFEST_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "unknown_speaker",
                "meeting",
                "filename",
                "start",
                "end",
                "duration",
                "other_overlap",
                "overlap_ratio",
            ],
        )

        writer.writeheader()

        writer.writerows(
            manifest
        )

    print()
    print("=" * 78)
    print("DONE")
    print("=" * 78)

    print()
    print(
        f"Unknown samples: "
        f"{len(SPEAKERS)} × "
        f"{NUM_TEST_PER_SPEAKER} "
        f"= "
        f"{len(SPEAKERS) * NUM_TEST_PER_SPEAKER}"
    )

    print(
        f"Output: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()