from pathlib import Path
import csv
import shutil
import subprocess
import tempfile

from datasets import (
    Audio,
    load_dataset,
)
import soundfile as sf


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = (
    BASE_DIR
    / "speaker_data_exp4"
)

MANIFEST_PATH = (
    BASE_DIR
    / "speaker_exp4_manifest.csv"
)


# ============================================================
# Experiment settings
# ============================================================

NUM_SPEAKERS = 4

NUM_ENROLL = 3
NUM_TEST = 5

NUM_REQUIRED = (
    NUM_ENROLL
    + NUM_TEST
)

SAMPLE_RATE = 16000

# 太短的片段不用于 speaker embedding
MIN_DURATION_SEC = 1.2

# 防止坏数据导致无限扫描
MAX_SCAN_RECORDS = 200000


# ============================================================
# Dataset sources
# ============================================================

LANG_CONFIGS = {

    "zh": {
        "dataset":
            "masuidrive/"
            "cv-corpus-17.0-zh-CN-client_id-grouped",

        "config":
            None,

        "split":
            "train",
    },

    "en": {
        "dataset":
            "fixie-ai/common_voice_17_0",

        "config":
            "en",

        "split":
            "train",
    },

    "ru": {
        "dataset":
            "fixie-ai/common_voice_17_0",

        "config":
            "ru",

        "split":
            "train",
    },

    "uz": {
        "dataset":
            "yakhyo/"
            "mozilla-common-voice-uzbek",

        "config":
            None,

        "split":
            "train",
    },
}


# ============================================================
# Dataset loading
# ============================================================

def load_stream(
    language,
):

    cfg = LANG_CONFIGS[
        language
    ]

    print()
    print(
        f"Loading {language}:"
    )

    print(
        f"  dataset = "
        f"{cfg['dataset']}"
    )

    print(
        f"  config  = "
        f"{cfg['config']}"
    )

    if cfg["config"] is None:

        dataset = load_dataset(
            cfg["dataset"],
            split=cfg["split"],
            streaming=True,
        )

    else:

        dataset = load_dataset(
            cfg["dataset"],
            cfg["config"],
            split=cfg["split"],
            streaming=True,
        )

    # 非常重要：
    # 禁用 torchcodec 自动解码。
    dataset = dataset.cast_column(
        "audio",
        Audio(
            decode=False
        ),
    )

    return dataset


# ============================================================
# Handle slightly different dataset layouts
# ============================================================

def iter_records(
    example,
):
    """
    标准 Common Voice：
        client_id = str
        audio     = dict

    同时兼容某些 grouped dataset
    可能返回 audio list。
    """

    client_id = example.get(
        "client_id"
    )

    audio = example.get(
        "audio"
    )

    if (
        client_id is None
        or audio is None
    ):
        return

    # --------------------------------------------------------
    # 普通格式
    # --------------------------------------------------------

    if isinstance(
        audio,
        dict,
    ):

        yield {
            "client_id":
                str(client_id),

            "audio":
                audio,
        }

        return

    # --------------------------------------------------------
    # grouped / sequence 格式
    # --------------------------------------------------------

    if isinstance(
        audio,
        (
            list,
            tuple,
        ),
    ):

        # client_id 也可能是 list
        if isinstance(
            client_id,
            (
                list,
                tuple,
            ),
        ):

            for cid, item in zip(
                client_id,
                audio,
            ):

                if isinstance(
                    item,
                    dict,
                ):

                    yield {
                        "client_id":
                            str(cid),

                        "audio":
                            item,
                    }

        else:

            for item in audio:

                if isinstance(
                    item,
                    dict,
                ):

                    yield {
                        "client_id":
                            str(client_id),

                        "audio":
                            item,
                    }


# ============================================================
# First pass:
# find 4 speakers that have at least 8 clips
# ============================================================

def find_candidate_speakers(
    language,
):

    dataset = load_stream(
        language
    )

    counts = {}

    selected = []

    processed = 0

    for example in dataset:

        for record in iter_records(
            example
        ):

            processed += 1

            cid = record[
                "client_id"
            ]

            counts[cid] = (
                counts.get(
                    cid,
                    0,
                )
                + 1
            )

            if (
                counts[cid]
                >= NUM_REQUIRED
                and
                cid not in selected
            ):

                selected.append(
                    cid
                )

                print(
                    f"  Found speaker "
                    f"{len(selected)}/"
                    f"{NUM_SPEAKERS}: "
                    f"{cid[:20]}..."
                )

                if (
                    len(selected)
                    >= NUM_SPEAKERS
                ):

                    return selected

            if (
                processed
                % 5000
                == 0
            ):

                print(
                    f"  scanned "
                    f"{processed} records..."
                )

            if (
                processed
                >= MAX_SCAN_RECORDS
            ):

                break

        if (
            processed
            >= MAX_SCAN_RECORDS
        ):

            break

    raise RuntimeError(
        f"\n{language}: "
        f"扫描 {processed} 条后，"
        f"只找到 {len(selected)} 个"
        f"至少拥有 {NUM_REQUIRED} 条"
        "录音的 speaker。\n"
    )


# ============================================================
# Audio conversion
# ============================================================

def get_suffix(
    audio_info,
):

    path = audio_info.get(
        "path"
    )

    if path:

        suffix = Path(
            path
        ).suffix

        if suffix:
            return suffix

    return ".audio"


def convert_audio_bytes(
    audio_info,
    output_path,
):

    audio_bytes = audio_info.get(
        "bytes"
    )

    if not audio_bytes:

        raise RuntimeError(
            "audio bytes is empty"
        )

    suffix = get_suffix(
        audio_info
    )

    with tempfile.TemporaryDirectory() as td:

        temp_dir = Path(td)

        input_path = (
            temp_dir
            / (
                "input"
                + suffix
            )
        )

        input_path.write_bytes(
            audio_bytes
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:

            raise RuntimeError(
                "ffmpeg failed:\n"
                + result.stderr
            )


def check_wav(
    path,
):

    info = sf.info(
        path
    )

    if info.samplerate != SAMPLE_RATE:

        return False, 0.0

    if info.channels != 1:

        return False, 0.0

    duration = (
        info.frames
        / float(
            info.samplerate
        )
    )

    if (
        duration
        < MIN_DURATION_SEC
    ):

        return False, duration

    return True, duration


# ============================================================
# Second pass:
# actually extract 8 valid clips per selected speaker
# ============================================================

def collect_speaker_audio(
    language,
    speaker_ids,
):

    dataset = load_stream(
        language
    )

    speaker_set = set(
        speaker_ids
    )

    collected = {
        cid: []
        for cid
        in speaker_ids
    }

    seen_paths = {
        cid: set()
        for cid
        in speaker_ids
    }

    processed = 0

    temp_output = (
        OUTPUT_DIR
        / "_tmp"
        / language
    )

    temp_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    for example in dataset:

        for record in iter_records(
            example
        ):

            processed += 1

            cid = record[
                "client_id"
            ]

            if (
                cid
                not in speaker_set
            ):
                continue

            if (
                len(
                    collected[cid]
                )
                >= NUM_REQUIRED
            ):
                continue

            audio_info = record[
                "audio"
            ]

            original_path = (
                audio_info.get(
                    "path"
                )
                or ""
            )

            if (
                original_path
                in seen_paths[cid]
            ):
                continue

            seen_paths[cid].add(
                original_path
            )

            speaker_index = (
                speaker_ids.index(
                    cid
                )
                + 1
            )

            candidate_number = (
                len(
                    collected[cid]
                )
                + 1
            )

            temp_wav = (
                temp_output
                / (
                    f"speaker_"
                    f"{speaker_index}_"
                    f"{candidate_number}.wav"
                )
            )

            try:

                convert_audio_bytes(
                    audio_info,
                    temp_wav,
                )

                valid, duration = (
                    check_wav(
                        temp_wav
                    )
                )

                if not valid:

                    try:
                        temp_wav.unlink()
                    except FileNotFoundError:
                        pass

                    continue

                collected[cid].append(
                    {
                        "temp_wav":
                            temp_wav,

                        "duration":
                            duration,

                        "original_path":
                            original_path,
                    }
                )

                print(
                    f"  {language} "
                    f"speaker_{speaker_index}: "
                    f"{len(collected[cid])}/"
                    f"{NUM_REQUIRED} "
                    f"({duration:.2f}s)"
                )

            except Exception as e:

                print(
                    "  Skip bad audio: "
                    f"{type(e).__name__}: "
                    f"{e}"
                )

            if all(
                len(
                    collected[x]
                )
                >= NUM_REQUIRED
                for x
                in speaker_ids
            ):

                return collected

            if (
                processed
                >= MAX_SCAN_RECORDS
            ):

                break

        if (
            processed
            >= MAX_SCAN_RECORDS
        ):

            break

    missing = {
        cid:
            len(collected[cid])
        for cid
        in speaker_ids
        if (
            len(collected[cid])
            < NUM_REQUIRED
        )
    }

    if missing:

        raise RuntimeError(
            f"\n{language}: "
            f"有效音频不足："
            f"{missing}"
        )

    return collected


# ============================================================
# Final dataset layout
# ============================================================

def save_final_dataset(
    language,
    speaker_ids,
    collected,
    manifest,
):

    for speaker_index, cid in enumerate(
        speaker_ids,
        start=1,
    ):

        clips = collected[
            cid
        ]

        if (
            len(clips)
            < NUM_REQUIRED
        ):

            raise RuntimeError(
                f"{language} "
                f"speaker_{speaker_index} "
                "clips insufficient"
            )

        # ----------------------------------------------------
        # First 3 -> enroll
        # Last 5  -> test
        # ----------------------------------------------------

        for index, item in enumerate(
            clips[
                :NUM_REQUIRED
            ]
        ):

            if index < NUM_ENROLL:

                split = "enroll"

                number = (
                    index + 1
                )

            else:

                split = "test"

                number = (
                    index
                    - NUM_ENROLL
                    + 1
                )

            output_path = (
                OUTPUT_DIR
                / language
                / (
                    f"speaker_"
                    f"{speaker_index}"
                )
                / split
                / f"{number:02d}.wav"
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            shutil.copy2(
                item["temp_wav"],
                output_path,
            )

            manifest.append(
                {
                    "language":
                        language,

                    "speaker":
                        (
                            f"speaker_"
                            f"{speaker_index}"
                        ),

                    "client_id":
                        cid,

                    "split":
                        split,

                    "filename":
                        output_path.name,

                    "duration":
                        item["duration"],

                    "source_path":
                        item[
                            "original_path"
                        ],
                }
            )


# ============================================================
# Cleanup
# ============================================================

def clear_old_output():

    if OUTPUT_DIR.exists():

        shutil.rmtree(
            OUTPUT_DIR
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "Speaker EXP4-A "
        "- Four Language Dataset"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "Languages:"
    )

    for language in LANG_CONFIGS:

        print(
            f"  {language}"
        )

    print()

    print(
        f"Per language:"
    )

    print(
        f"  speakers = "
        f"{NUM_SPEAKERS}"
    )

    print(
        f"  enroll   = "
        f"{NUM_ENROLL}"
    )

    print(
        f"  test     = "
        f"{NUM_TEST}"
    )

    print()

    clear_old_output()

    manifest = []

    for language in [
        "zh",
        "en",
        "ru",
        "uz",
    ]:

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

        # ----------------------------------------------------
        # Pass 1
        # ----------------------------------------------------

        speaker_ids = (
            find_candidate_speakers(
                language
            )
        )

        print()

        print(
            "Selected speakers:"
        )

        for i, cid in enumerate(
            speaker_ids,
            start=1,
        ):

            print(
                f"  speaker_{i}: "
                f"{cid}"
            )

        # ----------------------------------------------------
        # Pass 2
        # ----------------------------------------------------

        print()
        print(
            "Collecting audio..."
        )

        collected = (
            collect_speaker_audio(
                language,
                speaker_ids,
            )
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        save_final_dataset(
            language,
            speaker_ids,
            collected,
            manifest,
        )

    # --------------------------------------------------------
    # Remove temporary files
    # --------------------------------------------------------

    temp_dir = (
        OUTPUT_DIR
        / "_tmp"
    )

    if temp_dir.exists():

        shutil.rmtree(
            temp_dir
        )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    with MANIFEST_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "language",
                "speaker",
                "client_id",
                "split",
                "filename",
                "duration",
                "source_path",
            ],
        )

        writer.writeheader()

        writer.writerows(
            manifest
        )

    print()
    print(
        "=" * 80
    )

    print(
        "DONE"
    )

    print(
        "=" * 80
    )

    print()

    print(
        f"Dataset:"
    )

    print(
        OUTPUT_DIR
    )

    print()

    print(
        f"Manifest:"
    )

    print(
        MANIFEST_PATH
    )

    print()

    print(
        "Expected:"
    )

    print(
        "  4 languages"
    )

    print(
        "  4 speakers / language"
    )

    print(
        "  3 enroll / speaker"
    )

    print(
        "  5 test / speaker"
    )

    print()

    print(
        "下一步："
    )

    print(
        "python "
        ".\\run_speaker_exp4.py"
    )


if __name__ == "__main__":

    main()