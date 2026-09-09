from pathlib import Path
import csv
import time
import numpy as np
import soundfile as sf
import sherpa_onnx

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "sherpa-onnx-whisper-tiny"
ENCODER = MODEL_DIR / "tiny-encoder.int8.onnx"
DECODER = MODEL_DIR / "tiny-decoder.int8.onnx"
DATA_DIR = BASE_DIR.parent / "speaker_exp1" / "speaker_data_exp4"
RESULTS_CSV = BASE_DIR / "lid_exp1_results.csv"
SUMMARY_CSV = BASE_DIR / "lid_exp1_summary.csv"

LANGUAGES = {"zh": "zh", "en": "en", "ru": "ru", "uz": "uz"}
TARGET_LANGS = set(LANGUAGES.values())

def load_audio(path: Path):
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1).astype(np.float32)
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    if sample_rate != 16000:
        raise RuntimeError(f"{path}: expected 16000 Hz, got {sample_rate}")
    return samples, sample_rate

def main():
    if not ENCODER.is_file():
        raise FileNotFoundError(ENCODER)
    if not DECODER.is_file():
        raise FileNotFoundError(DECODER)
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(DATA_DIR)

    print("=" * 80)
    print("LID EXP1 - Whisper Tiny INT8")
    print("=" * 80)
    print(f"Encoder : {ENCODER}")
    print(f"Decoder : {DECODER}")
    print(f"Dataset : {DATA_DIR}")

    config = sherpa_onnx.SpokenLanguageIdentificationConfig(
        whisper=sherpa_onnx.SpokenLanguageIdentificationWhisperConfig(
            encoder=str(ENCODER),
            decoder=str(DECODER),
        ),
        num_threads=1,
        debug=False,
        provider="cpu",
    )
    slid = sherpa_onnx.SpokenLanguageIdentification(config)

    rows = []
    language_stats = {
        lang: {"total": 0, "correct": 0, "elapsed": 0.0, "audio": 0.0}
        for lang in LANGUAGES
    }
    confusion = {lang: {} for lang in LANGUAGES}

    for folder_name, true_lang in LANGUAGES.items():
        print()
        print("=" * 80)
        print(f"LANGUAGE: {true_lang.upper()}")
        print("=" * 80)

        wav_files = sorted(
            (DATA_DIR / folder_name).glob("speaker_*/test/*.wav")
        )
        if not wav_files:
            raise RuntimeError(
                f"No test WAV found: {DATA_DIR / folder_name}"
            )

        for wav_path in wav_files:
            samples, sample_rate = load_audio(wav_path)
            audio_duration = len(samples) / float(sample_rate)

            stream = slid.create_stream()
            stream.accept_waveform(
                sample_rate=sample_rate,
                waveform=samples,
            )

            start = time.perf_counter()
            predicted_raw = slid.compute(stream)
            elapsed = time.perf_counter() - start

            predicted_raw = str(predicted_raw).strip().lower()
            predicted = (
                predicted_raw
                if predicted_raw in TARGET_LANGS
                else "other"
            )
            correct = predicted == true_lang
            rtf = elapsed / audio_duration if audio_duration > 0 else 0.0

            stat = language_stats[true_lang]
            stat["total"] += 1
            stat["audio"] += audio_duration
            stat["elapsed"] += elapsed
            if correct:
                stat["correct"] += 1

            confusion[true_lang][predicted] = (
                confusion[true_lang].get(predicted, 0) + 1
            )

            rows.append({
                "true_lang": true_lang,
                "predicted": predicted,
                "predicted_raw": predicted_raw,
                "correct": correct,
                "speaker": wav_path.parent.parent.name,
                "filename": wav_path.name,
                "audio_sec": audio_duration,
                "elapsed_ms": elapsed * 1000.0,
                "rtf": rtf,
                "path": str(wav_path),
            })

            print(
                f"{true_lang} -> {predicted_raw:5s} | "
                f"{'OK' if correct else 'WRONG':5s} | "
                f"{audio_duration:5.2f}s | "
                f"{elapsed * 1000:7.1f} ms | "
                f"RTF={rtf:.4f}"
            )

    result_fields = [
        "true_lang", "predicted", "predicted_raw", "correct",
        "speaker", "filename", "audio_sec", "elapsed_ms", "rtf", "path",
    ]
    with RESULTS_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=result_fields)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_all = 0
    correct_all = 0
    audio_all = 0.0
    elapsed_all = 0.0

    for lang in LANGUAGES:
        stat = language_stats[lang]
        total = stat["total"]
        correct = stat["correct"]
        accuracy = correct / total if total else 0.0
        rtf = (
            stat["elapsed"] / stat["audio"]
            if stat["audio"] > 0
            else 0.0
        )

        total_all += total
        correct_all += correct
        audio_all += stat["audio"]
        elapsed_all += stat["elapsed"]

        print(
            f"{lang.upper():2s} : {correct:2d}/{total:2d} "
            f"= {accuracy * 100:6.2f}% | RTF={rtf:.4f}"
        )
        print(f"     predictions: {confusion[lang]}")

        summary_rows.append({
            "language": lang,
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
            "audio_sec": stat["audio"],
            "elapsed_sec": stat["elapsed"],
            "rtf": rtf,
        })

    overall_accuracy = (
        correct_all / total_all if total_all else 0.0
    )
    overall_rtf = (
        elapsed_all / audio_all if audio_all > 0 else 0.0
    )

    print()
    print(
        f"OVERALL : {correct_all}/{total_all} "
        f"= {overall_accuracy * 100:.2f}%"
    )
    print(f"Overall RTF : {overall_rtf:.4f}")
    print(
        f"Speed       : {1.0 / overall_rtf:.2f}x realtime"
        if overall_rtf > 0
        else "Speed       : N/A"
    )

    summary_rows.append({
        "language": "overall",
        "correct": correct_all,
        "total": total_all,
        "accuracy": overall_accuracy,
        "audio_sec": audio_all,
        "elapsed_sec": elapsed_all,
        "rtf": overall_rtf,
    })

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "language", "correct", "total", "accuracy",
                "audio_sec", "elapsed_sec", "rtf",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print()
    print(f"Results : {RESULTS_CSV}")
    print(f"Summary : {SUMMARY_CSV}")

if __name__ == "__main__":
    main()
