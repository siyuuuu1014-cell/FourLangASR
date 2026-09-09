from pathlib import Path
import csv
import json
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "voxlingua107"
MODEL_PATH = MODEL_DIR / "voxlingua107.onnx"
LANG_MAP_PATH = MODEL_DIR / "lang_map.json"

DATA_DIR = (
    BASE_DIR.parent
    / "speaker_exp1"
    / "speaker_data_exp4"
)

RESULTS_CSV = BASE_DIR / "lid_exp2_results.csv"
SUMMARY_CSV = BASE_DIR / "lid_exp2_summary.csv"

LANGUAGES = ["zh", "en", "ru", "uz"]
SAMPLE_RATE = 16000


def load_audio(path: Path):
    samples, sample_rate = sf.read(
        path,
        dtype="float32",
        always_2d=False,
    )

    samples = np.asarray(samples, dtype=np.float32)

    if samples.ndim == 2:
        samples = samples.mean(axis=1).astype(np.float32)

    samples = np.ascontiguousarray(samples, dtype=np.float32)

    if sample_rate != SAMPLE_RATE:
        raise RuntimeError(
            f"{path}: expected {SAMPLE_RATE} Hz, got {sample_rate}"
        )

    return samples, sample_rate


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits)
    exps = np.exp(logits)
    return (exps / np.sum(exps)).astype(np.float64)


def load_language_map():
    with LANG_MAP_PATH.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    by_index = {}

    if isinstance(raw, dict):
        for key, value in raw.items():
            idx = int(key)
            if not isinstance(value, dict):
                raise RuntimeError(
                    f"Unexpected lang_map entry at {key}: {value!r}"
                )
            iso = str(value["iso"]).strip().lower()
            name = str(value.get("name", iso))
            by_index[idx] = {
                "iso": iso,
                "name": name,
            }
    elif isinstance(raw, list):
        for idx, value in enumerate(raw):
            if not isinstance(value, dict):
                raise RuntimeError(
                    f"Unexpected lang_map entry at {idx}: {value!r}"
                )
            iso = str(value["iso"]).strip().lower()
            name = str(value.get("name", iso))
            by_index[idx] = {
                "iso": iso,
                "name": name,
            }
    else:
        raise RuntimeError(
            f"Unsupported lang_map JSON type: {type(raw).__name__}"
        )

    return by_index


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    if not LANG_MAP_PATH.is_file():
        raise FileNotFoundError(LANG_MAP_PATH)

    if not DATA_DIR.is_dir():
        raise FileNotFoundError(DATA_DIR)

    lang_map = load_language_map()

    iso_to_index = {
        entry["iso"]: idx
        for idx, entry in lang_map.items()
    }

    missing = [
        lang
        for lang in LANGUAGES
        if lang not in iso_to_index
    ]

    if missing:
        raise RuntimeError(
            f"Target languages missing from lang_map: {missing}"
        )

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1

    session = ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    input_meta = session.get_inputs()[0]
    input_name = input_meta.name

    output_names = [
        output.name
        for output in session.get_outputs()
    ]

    if "logits" not in output_names:
        raise RuntimeError(
            f"Expected output 'logits', got {output_names}"
        )

    target_indices = {
        lang: iso_to_index[lang]
        for lang in LANGUAGES
    }

    print("=" * 88)
    print("LID EXP2 - VoxLingua107 ECAPA ONNX")
    print("=" * 88)
    print(f"Model      : {MODEL_PATH}")
    print(f"Lang map   : {LANG_MAP_PATH}")
    print(f"Dataset    : {DATA_DIR}")
    print(f"Input      : {input_name} {input_meta.shape} {input_meta.type}")
    print(f"Outputs    : {output_names}")
    print(f"4-way idx  : {target_indices}")
    print()

    rows = []

    stats = {
        lang: {
            "total": 0,
            "full_correct": 0,
            "fourway_correct": 0,
            "audio_sec": 0.0,
            "elapsed_sec": 0.0,
        }
        for lang in LANGUAGES
    }

    full_confusion = {
        lang: {}
        for lang in LANGUAGES
    }

    fourway_confusion = {
        lang: {}
        for lang in LANGUAGES
    }

    for true_lang in LANGUAGES:
        print("=" * 88)
        print(f"LANGUAGE: {true_lang.upper()}")
        print("=" * 88)

        wav_files = sorted(
            (DATA_DIR / true_lang).glob(
                "speaker_*/test/*.wav"
            )
        )

        if not wav_files:
            raise RuntimeError(
                f"No test WAV found under: {DATA_DIR / true_lang}"
            )

        for wav_path in wav_files:
            samples, sample_rate = load_audio(wav_path)
            audio_sec = len(samples) / float(sample_rate)

            audio_input = samples[None, :]

            start = time.perf_counter()

            logits = session.run(
                ["logits"],
                {
                    input_name: audio_input
                },
            )[0]

            elapsed_sec = time.perf_counter() - start

            logits = np.asarray(logits)

            if logits.ndim == 2:
                logits = logits[0]
            else:
                logits = logits.reshape(-1)

            if len(logits) != len(lang_map):
                raise RuntimeError(
                    f"Unexpected logits size {len(logits)}; "
                    f"lang_map has {len(lang_map)} entries"
                )

            probs = softmax(logits)

            # -------------------------------------------------
            # 107-language top-1
            # -------------------------------------------------
            full_idx = int(np.argmax(probs))
            full_iso = lang_map[full_idx]["iso"]
            full_prob = float(probs[full_idx])
            full_correct = full_iso == true_lang

            # -------------------------------------------------
            # Four-language router:
            # mask to zh/en/ru/uz and re-normalize within them.
            # -------------------------------------------------
            target_logits = np.array(
                [
                    logits[target_indices[lang]]
                    for lang in LANGUAGES
                ],
                dtype=np.float64,
            )

            target_probs = softmax(target_logits)

            ranking = sorted(
                zip(LANGUAGES, target_probs),
                key=lambda item: item[1],
                reverse=True,
            )

            fourway_iso = ranking[0][0]
            fourway_conf = float(ranking[0][1])
            fourway_second_iso = ranking[1][0]
            fourway_second_conf = float(ranking[1][1])
            fourway_margin = (
                fourway_conf - fourway_second_conf
            )
            fourway_correct = fourway_iso == true_lang

            # Total probability mass assigned to our supported
            # four languages in the original 107-way softmax.
            target_mass = float(
                sum(
                    probs[target_indices[lang]]
                    for lang in LANGUAGES
                )
            )

            rtf = (
                elapsed_sec / audio_sec
                if audio_sec > 0
                else 0.0
            )

            stat = stats[true_lang]
            stat["total"] += 1
            stat["audio_sec"] += audio_sec
            stat["elapsed_sec"] += elapsed_sec

            if full_correct:
                stat["full_correct"] += 1

            if fourway_correct:
                stat["fourway_correct"] += 1

            full_confusion[true_lang][full_iso] = (
                full_confusion[true_lang].get(full_iso, 0) + 1
            )

            fourway_confusion[true_lang][fourway_iso] = (
                fourway_confusion[true_lang].get(fourway_iso, 0) + 1
            )

            rows.append(
                {
                    "true_lang": true_lang,
                    "speaker": wav_path.parent.parent.name,
                    "filename": wav_path.name,
                    "audio_sec": audio_sec,
                    "elapsed_ms": elapsed_sec * 1000.0,
                    "rtf": rtf,
                    "full_top1": full_iso,
                    "full_top1_prob": full_prob,
                    "full_correct": full_correct,
                    "fourway_top1": fourway_iso,
                    "fourway_conf": fourway_conf,
                    "fourway_second": fourway_second_iso,
                    "fourway_second_conf": fourway_second_conf,
                    "fourway_margin": fourway_margin,
                    "fourway_correct": fourway_correct,
                    "target_mass": target_mass,
                    "p_zh": float(target_probs[LANGUAGES.index("zh")]),
                    "p_en": float(target_probs[LANGUAGES.index("en")]),
                    "p_ru": float(target_probs[LANGUAGES.index("ru")]),
                    "p_uz": float(target_probs[LANGUAGES.index("uz")]),
                    "path": str(wav_path),
                }
            )

            print(
                f"{true_lang} | "
                f"107={full_iso:4s} {full_prob:6.3f} "
                f"{'OK' if full_correct else 'WRONG':5s} | "
                f"4way={fourway_iso:2s} "
                f"conf={fourway_conf:6.3f} "
                f"margin={fourway_margin:6.3f} "
                f"{'OK' if fourway_correct else 'WRONG':5s} | "
                f"mass={target_mass:6.3f} | "
                f"RTF={rtf:.4f}"
            )

    result_fields = [
        "true_lang",
        "speaker",
        "filename",
        "audio_sec",
        "elapsed_ms",
        "rtf",
        "full_top1",
        "full_top1_prob",
        "full_correct",
        "fourway_top1",
        "fourway_conf",
        "fourway_second",
        "fourway_second_conf",
        "fourway_margin",
        "fourway_correct",
        "target_mass",
        "p_zh",
        "p_en",
        "p_ru",
        "p_uz",
        "path",
    ]

    with RESULTS_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=result_fields,
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 88)
    print("SUMMARY")
    print("=" * 88)

    summary_rows = []

    total_all = 0
    full_correct_all = 0
    fourway_correct_all = 0
    total_audio = 0.0
    total_elapsed = 0.0

    for lang in LANGUAGES:
        stat = stats[lang]

        total = stat["total"]
        full_correct = stat["full_correct"]
        fourway_correct = stat["fourway_correct"]

        full_acc = (
            full_correct / total
            if total
            else 0.0
        )

        fourway_acc = (
            fourway_correct / total
            if total
            else 0.0
        )

        rtf = (
            stat["elapsed_sec"] / stat["audio_sec"]
            if stat["audio_sec"] > 0
            else 0.0
        )

        total_all += total
        full_correct_all += full_correct
        fourway_correct_all += fourway_correct
        total_audio += stat["audio_sec"]
        total_elapsed += stat["elapsed_sec"]

        print(
            f"{lang.upper():2s} | "
            f"107-way {full_correct:2d}/{total:2d} "
            f"= {full_acc * 100:6.2f}% | "
            f"4-way {fourway_correct:2d}/{total:2d} "
            f"= {fourway_acc * 100:6.2f}% | "
            f"RTF={rtf:.4f}"
        )
        print(
            f"     107 predictions : {full_confusion[lang]}"
        )
        print(
            f"     4-way predictions: {fourway_confusion[lang]}"
        )

        summary_rows.append(
            {
                "language": lang,
                "total": total,
                "full_correct": full_correct,
                "full_accuracy": full_acc,
                "fourway_correct": fourway_correct,
                "fourway_accuracy": fourway_acc,
                "audio_sec": stat["audio_sec"],
                "elapsed_sec": stat["elapsed_sec"],
                "rtf": rtf,
            }
        )

    overall_full = (
        full_correct_all / total_all
        if total_all
        else 0.0
    )

    overall_fourway = (
        fourway_correct_all / total_all
        if total_all
        else 0.0
    )

    overall_rtf = (
        total_elapsed / total_audio
        if total_audio > 0
        else 0.0
    )

    print()
    print(
        f"OVERALL 107-way : "
        f"{full_correct_all}/{total_all} "
        f"= {overall_full * 100:.2f}%"
    )
    print(
        f"OVERALL 4-way   : "
        f"{fourway_correct_all}/{total_all} "
        f"= {overall_fourway * 100:.2f}%"
    )
    print(f"Overall RTF     : {overall_rtf:.4f}")

    if overall_rtf > 0:
        print(
            f"Speed           : "
            f"{1.0 / overall_rtf:.2f}x realtime"
        )

    summary_rows.append(
        {
            "language": "overall",
            "total": total_all,
            "full_correct": full_correct_all,
            "full_accuracy": overall_full,
            "fourway_correct": fourway_correct_all,
            "fourway_accuracy": overall_fourway,
            "audio_sec": total_audio,
            "elapsed_sec": total_elapsed,
            "rtf": overall_rtf,
        }
    )

    summary_fields = [
        "language",
        "total",
        "full_correct",
        "full_accuracy",
        "fourway_correct",
        "fourway_accuracy",
        "audio_sec",
        "elapsed_sec",
        "rtf",
    ]

    with SUMMARY_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=summary_fields,
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print()
    print(f"Results : {RESULTS_CSV}")
    print(f"Summary : {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
