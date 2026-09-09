from pathlib import Path
import argparse
import csv
import os
import shutil
import subprocess
import tempfile
import time

import requests
import soundfile as sf


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "speaker_data_exp4"

COMBINED_MANIFEST = (
    BASE_DIR / "speaker_exp4_manifest.csv"
)


# ============================================================
# Experiment settings
# ============================================================

NUM_SPEAKERS = 4

NUM_ENROLL = 3
NUM_TEST = 5

NUM_REQUIRED = (
    NUM_ENROLL + NUM_TEST
)

# 多收两条备用，防止某些音频太短/损坏
NUM_CANDIDATES = 10

SAMPLE_RATE = 16000

MIN_DURATION_SEC = 1.2

ROWS_PER_REQUEST = 100

MAX_SCAN_ROWS = 50000


# ============================================================
# Hugging Face Dataset Viewer
# ============================================================

API_BASE = (
    "https://datasets-server.huggingface.co"
)


LANG_CONFIGS = {
    "zh": {
        "dataset":
            "masuidrive/"
            "cv-corpus-17.0-zh-CN-client_id-grouped",
        "preferred_config": None,
        "split": "train",
    },

    "en": {
        "dataset":
            "fixie-ai/common_voice_17_0",
        "preferred_config": "en",
        "split": "train",
    },

    "ru": {
        "dataset":
            "fixie-ai/common_voice_17_0",
        "preferred_config": "ru",
        "split": "train",
    },

    "uz": {
        "dataset":
            "yakhyo/mozilla-common-voice-uzbek",
        "preferred_config": None,
        "split": "train",
    },
}


# ============================================================
# HTTP
# ============================================================

def make_session():

    session = requests.Session()

    token = os.environ.get("HF_TOKEN")

    if token:
        session.headers.update(
            {
                "Authorization":
                    f"Bearer {token}"
            }
        )

    session.headers.update(
        {
            "User-Agent":
                "FourLangAsrMobile-SpeakerEXP4/1.0"
        }
    )

    return session


SESSION = make_session()


def get_json(
    url,
    params=None,
    retries=8,
):

    last_error = None

    for attempt in range(
        1,
        retries + 1,
    ):

        try:

            response = SESSION.get(
                url,
                params=params,
                timeout=(20, 90),
            )

            response.raise_for_status()

            return response.json()

        except Exception as e:

            last_error = e

            wait = min(
                2 ** (attempt - 1),
                30,
            )

            print(
                f"  API retry "
                f"{attempt}/{retries}: "
                f"{type(e).__name__}: "
                f"{e}"
            )

            if attempt < retries:

                print(
                    f"  wait {wait}s..."
                )

                time.sleep(wait)

    raise RuntimeError(
        f"Dataset Viewer API failed: "
        f"{last_error}"
    )


# ============================================================
# Resolve config dynamically
# ============================================================

def resolve_config(
    language,
):

    cfg = LANG_CONFIGS[
        language
    ]

    data = get_json(
        f"{API_BASE}/splits",
        params={
            "dataset":
                cfg["dataset"],
        },
    )

    splits = data.get(
        "splits",
        []
    )

    if not splits:

        raise RuntimeError(
            f"{language}: "
            "Dataset Viewer returned no splits"
        )

    preferred = cfg[
        "preferred_config"
    ]

    target_split = cfg[
        "split"
    ]

    # --------------------------------------------------------
    # Preferred config: en / ru
    # --------------------------------------------------------

    if preferred is not None:

        for item in splits:

            if (
                item.get("config")
                == preferred
                and
                item.get("split")
                == target_split
            ):

                return (
                    preferred,
                    target_split,
                )

        raise RuntimeError(
            f"{language}: config "
            f"{preferred}/{target_split} "
            "not found"
        )

    # --------------------------------------------------------
    # Dataset without explicit config
    # --------------------------------------------------------

    for item in splits:

        if (
            item.get("split")
            == target_split
        ):

            return (
                item.get("config"),
                target_split,
            )

    raise RuntimeError(
        f"{language}: "
        f"split {target_split} not found"
    )


# ============================================================
# Get one page
# ============================================================

def get_rows_page(
    dataset,
    config,
    split,
    offset,
    length=ROWS_PER_REQUEST,
):

    params = {
        "dataset":
            dataset,

        "config":
            config,

        "split":
            split,

        "offset":
            offset,

        "length":
            length,
    }

    return get_json(
        f"{API_BASE}/rows",
        params=params,
    )


# ============================================================
# Normalize Dataset Viewer row formats
# ============================================================

def audio_src(
    audio,
):

    if not isinstance(
        audio,
        dict,
    ):
        return None

    src = audio.get(
        "src"
    )

    if src:
        return src

    # 兼容可能的其它结构
    src = audio.get(
        "url"
    )

    if src:
        return src

    return None


def iter_audio_records(
    row_data,
    row_idx,
):
    """
    支持：

    普通 Common Voice:
        client_id = str
        audio = dict

    grouped:
        client_id = str
        audio = list

    也兼容 client_id list + audio list
    """

    client_id = row_data.get(
        "client_id"
    )

    audio = row_data.get(
        "audio"
    )

    if (
        client_id is None
        or audio is None
    ):
        return

    # --------------------------------------------------------
    # Normal single audio
    # --------------------------------------------------------

    if isinstance(
        audio,
        dict,
    ):

        src = audio_src(
            audio
        )

        if src:

            yield {
                "client_id":
                    str(client_id),

                "src":
                    src,

                "row_idx":
                    row_idx,

                "item_idx":
                    0,
            }

        return

    # --------------------------------------------------------
    # Audio list
    # --------------------------------------------------------

    if isinstance(
        audio,
        list,
    ):

        # client_id is also list
        if isinstance(
            client_id,
            list,
        ):

            pairs = zip(
                client_id,
                audio,
            )

            for item_idx, (
                cid,
                audio_item,
            ) in enumerate(pairs):

                src = audio_src(
                    audio_item
                )

                if src:

                    yield {
                        "client_id":
                            str(cid),

                        "src":
                            src,

                        "row_idx":
                            row_idx,

                        "item_idx":
                            item_idx,
                    }

        else:

            # one client_id owns many audio clips
            for item_idx, audio_item in enumerate(
                audio
            ):

                src = audio_src(
                    audio_item
                )

                if src:

                    yield {
                        "client_id":
                            str(client_id),

                        "src":
                            src,

                        "row_idx":
                            row_idx,

                        "item_idx":
                            item_idx,
                    }


# ============================================================
# Find candidate speakers
# ============================================================

def find_speakers(
    language,
):

    cfg = LANG_CONFIGS[
        language
    ]

    config_name, split_name = (
        resolve_config(
            language
        )
    )

    print()
    print(
        f"Dataset : "
        f"{cfg['dataset']}"
    )

    print(
        f"Config  : "
        f"{config_name}"
    )

    print(
        f"Split   : "
        f"{split_name}"
    )

    print()
    print(
        "Scanning metadata only..."
    )

    candidates = {}

    selected = []

    offset = 0

    total_rows = None

    while (
        offset < MAX_SCAN_ROWS
    ):

        data = get_rows_page(
            dataset=
                cfg["dataset"],

            config=
                config_name,

            split=
                split_name,

            offset=
                offset,
        )

        rows = data.get(
            "rows",
            []
        )

        if total_rows is None:

            total_rows = data.get(
                "num_rows_total"
            )

            print(
                f"Total rows reported: "
                f"{total_rows}"
            )

        if not rows:

            break

        for row_item in rows:

            row_idx = row_item.get(
                "row_idx",
                offset,
            )

            row_data = row_item.get(
                "row",
                {}
            )

            for record in iter_audio_records(
                row_data,
                row_idx,
            ):

                cid = record[
                    "client_id"
                ]

                if cid not in candidates:

                    candidates[
                        cid
                    ] = []

                # Prevent duplicate item
                key = (
                    record["row_idx"],
                    record["item_idx"],
                )

                existing = {
                    (
                        x["row_idx"],
                        x["item_idx"],
                    )
                    for x
                    in candidates[cid]
                }

                if key in existing:
                    continue

                if (
                    len(
                        candidates[cid]
                    )
                    < NUM_CANDIDATES
                ):

                    candidates[
                        cid
                    ].append(
                        record
                    )

                if (
                    len(
                        candidates[cid]
                    )
                    >= NUM_REQUIRED
                    and
                    cid not in selected
                ):

                    selected.append(
                        cid
                    )

                    print(
                        f"  Candidate "
                        f"{len(selected)}/"
                        f"{NUM_SPEAKERS}: "
                        f"{cid[:24]}... "
                        f"clips="
                        f"{len(candidates[cid])}"
                    )

        offset += len(
            rows
        )

        if (
            offset % 1000
            == 0
        ):

            print(
                f"  scanned "
                f"{offset} rows..."
            )

        # Don't stop immediately at 8.
        # Try to gather reserve clips too.
        if (
            len(selected)
            >= NUM_SPEAKERS
        ):

            ready = True

            for cid in selected[
                :NUM_SPEAKERS
            ]:

                if (
                    len(
                        candidates[cid]
                    )
                    < NUM_CANDIDATES
                ):
                    ready = False
                    break

            if ready:
                break

        if (
            total_rows is not None
            and
            offset >= total_rows
        ):
            break

    usable = [
        cid
        for cid in selected
        if (
            len(candidates[cid])
            >= NUM_REQUIRED
        )
    ]

    if (
        len(usable)
        < NUM_SPEAKERS
    ):

        raise RuntimeError(
            f"{language}: only "
            f"{len(usable)} suitable "
            "speakers found"
        )

    speaker_ids = usable[
        :NUM_SPEAKERS
    ]

    print()
    print(
        "Selected:"
    )

    for i, cid in enumerate(
        speaker_ids,
        start=1,
    ):

        print(
            f"  speaker_{i}: "
            f"{cid} "
            f"({len(candidates[cid])} candidates)"
        )

    return (
        cfg["dataset"],
        config_name,
        split_name,
        speaker_ids,
        candidates,
    )


# ============================================================
# Refresh signed audio URL
# ============================================================

def refresh_record(
    dataset,
    config,
    split,
    record,
):

    data = get_rows_page(
        dataset=dataset,
        config=config,
        split=split,
        offset=record[
            "row_idx"
        ],
        length=1,
    )

    rows = data.get(
        "rows",
        []
    )

    if not rows:

        raise RuntimeError(
            "Unable to refresh row"
        )

    row_data = rows[0].get(
        "row",
        {}
    )

    fresh = list(
        iter_audio_records(
            row_data,
            record["row_idx"],
        )
    )

    wanted_idx = record[
        "item_idx"
    ]

    for item in fresh:

        if (
            item["item_idx"]
            == wanted_idx
            and
            item["client_id"]
            == record[
                "client_id"
            ]
        ):

            return item

    raise RuntimeError(
        "Unable to refresh audio URL"
    )


# ============================================================
# Download small audio clip
# ============================================================

def download_file(
    url,
    output_path,
    retries=5,
):

    last_error = None

    for attempt in range(
        1,
        retries + 1,
    ):

        try:

            with SESSION.get(
                url,
                stream=True,
                timeout=(20, 120),
            ) as response:

                response.raise_for_status()

                with output_path.open(
                    "wb"
                ) as f:

                    for chunk in (
                        response.iter_content(
                            chunk_size=
                                1024 * 256
                        )
                    ):

                        if chunk:
                            f.write(
                                chunk
                            )

            if (
                output_path.exists()
                and
                output_path.stat().st_size
                > 1000
            ):

                return

            raise RuntimeError(
                "Downloaded file too small"
            )

        except Exception as e:

            last_error = e

            try:
                output_path.unlink()
            except FileNotFoundError:
                pass

            if attempt < retries:

                wait = min(
                    2 ** (attempt - 1),
                    15,
                )

                print(
                    f"    audio retry "
                    f"{attempt}/{retries}, "
                    f"wait {wait}s"
                )

                time.sleep(
                    wait
                )

    raise RuntimeError(
        f"audio download failed: "
        f"{last_error}"
    )


# ============================================================
# FFmpeg conversion
# ============================================================

def convert_to_wav(
    source_path,
    wav_path,
):

    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",

        "-i",
        str(source_path),

        "-ar",
        str(SAMPLE_RATE),

        "-ac",
        "1",

        "-c:a",
        "pcm_s16le",

        str(wav_path),
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:

        raise RuntimeError(
            result.stderr
        )


def validate_wav(
    path,
):

    try:

        info = sf.info(
            path
        )

    except Exception:

        return False, 0.0

    if (
        info.samplerate
        != SAMPLE_RATE
    ):
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
# Download chosen speakers
# ============================================================

def prepare_language_audio(
    language,
    dataset,
    config,
    split,
    speaker_ids,
    candidates,
):

    staging = (
        OUTPUT_DIR
        / f"_staging_{language}"
    )

    if staging.exists():

        shutil.rmtree(
            staging
        )

    staging.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = []

    with tempfile.TemporaryDirectory() as td:

        temp_dir = Path(td)

        for speaker_number, cid in enumerate(
            speaker_ids,
            start=1,
        ):

            print()
            print(
                f"{language} speaker_"
                f"{speaker_number}"
            )

            valid_items = []

            for candidate_index, record in enumerate(
                candidates[cid],
                start=1,
            ):

                if (
                    len(valid_items)
                    >= NUM_REQUIRED
                ):
                    break

                source_path = (
                    temp_dir
                    / (
                        f"{language}_"
                        f"{speaker_number}_"
                        f"{candidate_index}.audio"
                    )
                )

                wav_temp = (
                    temp_dir
                    / (
                        f"{language}_"
                        f"{speaker_number}_"
                        f"{candidate_index}.wav"
                    )
                )

                # Refresh URL before download
                try:

                    fresh = refresh_record(
                        dataset,
                        config,
                        split,
                        record,
                    )

                    download_file(
                        fresh["src"],
                        source_path,
                    )

                    convert_to_wav(
                        source_path,
                        wav_temp,
                    )

                    valid, duration = (
                        validate_wav(
                            wav_temp
                        )
                    )

                    if not valid:

                        print(
                            f"  skip "
                            f"{candidate_index}: "
                            f"duration="
                            f"{duration:.2f}s"
                        )

                        continue

                    valid_items.append(
                        {
                            "wav":
                                wav_temp.read_bytes(),

                            "duration":
                                duration,

                            "row_idx":
                                record["row_idx"],

                            "item_idx":
                                record["item_idx"],
                        }
                    )

                    print(
                        f"  valid "
                        f"{len(valid_items)}/"
                        f"{NUM_REQUIRED}: "
                        f"{duration:.2f}s"
                    )

                except Exception as e:

                    print(
                        f"  skip "
                        f"{candidate_index}: "
                        f"{type(e).__name__}: "
                        f"{e}"
                    )

                finally:

                    try:
                        source_path.unlink()
                    except FileNotFoundError:
                        pass

                    try:
                        wav_temp.unlink()
                    except FileNotFoundError:
                        pass

            if (
                len(valid_items)
                < NUM_REQUIRED
            ):

                raise RuntimeError(
                    f"{language} speaker_"
                    f"{speaker_number}: "
                    f"only "
                    f"{len(valid_items)} "
                    "valid clips"
                )

            # ------------------------------------------------
            # 3 enroll + 5 test
            # ------------------------------------------------

            for index, item in enumerate(
                valid_items[
                    :NUM_REQUIRED
                ]
            ):

                if index < NUM_ENROLL:

                    split_name = "enroll"
                    number = index + 1

                else:

                    split_name = "test"

                    number = (
                        index
                        - NUM_ENROLL
                        + 1
                    )

                output_path = (
                    staging
                    / (
                        f"speaker_"
                        f"{speaker_number}"
                    )
                    / split_name
                    / f"{number:02d}.wav"
                )

                output_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                output_path.write_bytes(
                    item["wav"]
                )

                manifest.append(
                    {
                        "language":
                            language,

                        "speaker":
                            (
                                f"speaker_"
                                f"{speaker_number}"
                            ),

                        "client_id":
                            cid,

                        "split":
                            split_name,

                        "filename":
                            output_path.name,

                        "duration":
                            item[
                                "duration"
                            ],

                        "row_idx":
                            item[
                                "row_idx"
                            ],

                        "item_idx":
                            item[
                                "item_idx"
                            ],
                    }
                )

    # ========================================================
    # Only replace existing language after full success
    # ========================================================

    final_dir = (
        OUTPUT_DIR
        / language
    )

    if final_dir.exists():

        shutil.rmtree(
            final_dir
        )

    staging.rename(
        final_dir
    )

    return manifest


# ============================================================
# Save manifest per language
# ============================================================

MANIFEST_FIELDS = [
    "language",
    "speaker",
    "client_id",
    "split",
    "filename",
    "duration",
    "row_idx",
    "item_idx",
]


def save_language_manifest(
    language,
    rows,
):

    path = (
        BASE_DIR
        / (
            f"speaker_exp4_"
            f"manifest_{language}.csv"
        )
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=
                MANIFEST_FIELDS,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    rebuild_combined_manifest()

    return path


def rebuild_combined_manifest():

    all_rows = []

    for language in [
        "zh",
        "en",
        "ru",
        "uz",
    ]:

        path = (
            BASE_DIR
            / (
                f"speaker_exp4_"
                f"manifest_{language}.csv"
            )
        )

        if not path.exists():
            continue

        with path.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as f:

            reader = csv.DictReader(
                f
            )

            all_rows.extend(
                reader
            )

    if not all_rows:
        return

    with COMBINED_MANIFEST.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=
                MANIFEST_FIELDS,
        )

        writer.writeheader()

        writer.writerows(
            all_rows
        )


# ============================================================
# One language
# ============================================================

def prepare_one_language(
    language,
):

    print()
    print(
        "=" * 80
    )

    print(
        f"SPEAKER EXP4-A "
        f"- {language.upper()}"
    )

    print(
        "=" * 80
    )

    (
        dataset,
        config,
        split,
        speaker_ids,
        candidates,
    ) = find_speakers(
        language
    )

    rows = prepare_language_audio(
        language,
        dataset,
        config,
        split,
        speaker_ids,
        candidates,
    )

    manifest_path = (
        save_language_manifest(
            language,
            rows,
        )
    )

    print()
    print(
        "=" * 80
    )

    print(
        f"{language.upper()} DONE"
    )

    print(
        "=" * 80
    )

    print(
        f"WAV directory:"
    )

    print(
        OUTPUT_DIR / language
    )

    print(
        f"Manifest:"
    )

    print(
        manifest_path
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--language",
        choices=[
            "zh",
            "en",
            "ru",
            "uz",
            "all",
        ],
        default="all",
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.language == "all":

        languages = [
            "zh",
            "en",
            "ru",
            "uz",
        ]

    else:

        languages = [
            args.language
        ]

    for language in languages:

        prepare_one_language(
            language
        )

    print()
    print(
        "Finished requested languages."
    )


if __name__ == "__main__":

    main()