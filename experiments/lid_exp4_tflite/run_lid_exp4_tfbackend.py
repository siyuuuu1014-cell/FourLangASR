from pathlib import Path
import csv
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf
from ai_edge_litert.interpreter import Interpreter


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

ONNX_MODEL = (
    PROJECT_ROOT
    / "experiments"
    / "lid_exp1-3"
    / "models"
    / "voxlingua107"
    / "voxlingua107.onnx"
)

TFLITE_DIR = BASE_DIR / "converted_tf"

DATA_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "speaker_exp4"
    / "speaker_data_exp4"
)

RESULTS_CSV = BASE_DIR / "lid_exp4_tfbackend_results.csv"
SUMMARY_CSV = BASE_DIR / "lid_exp4_tfbackend_summary.csv"

LANGUAGES = ["zh", "en", "ru", "uz"]

TARGET_INDEX = {
    "en": 20,
    "ru": 77,
    "uz": 101,
    "zh": 106,
}

SAMPLE_RATE = 16000
FIXED_SECONDS = 5.0
FIXED_SAMPLES = int(SAMPLE_RATE * FIXED_SECONDS)


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

    return samples


def make_fixed_window(samples: np.ndarray):
    """
    Convert every VAD utterance to exactly 5.0 s / 80000 samples.

    - Longer than 5 s: center crop.
    - Shorter than 5 s: symmetric zero pad.

    This matches the 1x80000 shape hint used during ONNX -> TFLite
    conversion and avoids dynamic-RESHAPE failures inside LiteRT.
    """
    n = len(samples)

    if n == FIXED_SAMPLES:
        return np.ascontiguousarray(samples, dtype=np.float32), "exact"

    if n > FIXED_SAMPLES:
        start = (n - FIXED_SAMPLES) // 2
        end = start + FIXED_SAMPLES
        return (
            np.ascontiguousarray(samples[start:end], dtype=np.float32),
            "center_crop",
        )

    total_pad = FIXED_SAMPLES - n
    left = total_pad // 2
    right = total_pad - left

    padded = np.pad(
        samples,
        (left, right),
        mode="constant",
        constant_values=0.0,
    ).astype(np.float32, copy=False)

    return np.ascontiguousarray(padded), "zero_pad"


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    y = np.exp(x)
    return y / np.sum(y)


def four_way(logits):
    selected = np.asarray(
        [logits[TARGET_INDEX[lang]] for lang in LANGUAGES],
        dtype=np.float64,
    )

    p = softmax(selected)
    order = np.argsort(-p)

    best_i = int(order[0])
    second_i = int(order[1])

    return {
        "lang": LANGUAGES[best_i],
        "conf": float(p[best_i]),
        "second": LANGUAGES[second_i],
        "second_conf": float(p[second_i]),
        "margin": float(p[best_i] - p[second_i]),
    }


def find_tflite_model():
    candidates = sorted(
        TFLITE_DIR.rglob("*_float32.tflite"),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )

    if not candidates:
        candidates = sorted(
            TFLITE_DIR.rglob("*.tflite"),
            key=lambda p: p.stat().st_size,
            reverse=True,
        )

    if not candidates:
        raise FileNotFoundError(
            f"No .tflite model found under {TFLITE_DIR}"
        )

    return candidates[0]


def prepare_interpreter(tflite_model: Path):
    interpreter = Interpreter(
        model_path=str(tflite_model),
        num_threads=1,
    )

    # IMPORTANT:
    # The converted graph was specialized around 1x80000 by the
    # converter's shape hint. Resize ONCE to that exact size.
    input_detail = interpreter.get_input_details()[0]

    interpreter.resize_tensor_input(
        input_detail["index"],
        [1, FIXED_SAMPLES],
        strict=False,
    )

    interpreter.allocate_tensors()

    input_detail = interpreter.get_input_details()[0]

    actual_shape = tuple(int(x) for x in input_detail["shape"])

    if actual_shape != (1, FIXED_SAMPLES):
        raise RuntimeError(
            f"Expected TFLite input shape (1,{FIXED_SAMPLES}), "
            f"got {actual_shape}"
        )

    return interpreter


def get_logits_from_tflite(interpreter, fixed_samples):
    input_detail = interpreter.get_input_details()[0]

    x = fixed_samples[None, :].astype(
        input_detail["dtype"],
        copy=False,
    )

    interpreter.set_tensor(
        input_detail["index"],
        x,
    )

    start = time.perf_counter()
    interpreter.invoke()
    elapsed = time.perf_counter() - start

    output_details = interpreter.get_output_details()

    logits = None

    for detail in output_details:
        value = interpreter.get_tensor(detail["index"])
        flat = np.asarray(value).reshape(-1)

        if flat.size == 107:
            logits = flat.astype(np.float32, copy=False)
            break

    if logits is None:
        output_shapes = [
            tuple(
                interpreter.get_tensor(d["index"]).shape
            )
            for d in output_details
        ]

        raise RuntimeError(
            "Could not find 107-class logits output. "
            f"TFLite outputs={output_shapes}"
        )

    return logits, elapsed


def main():
    if not ONNX_MODEL.is_file():
        raise FileNotFoundError(ONNX_MODEL)

    if not DATA_DIR.is_dir():
        raise FileNotFoundError(DATA_DIR)

    tflite_model = find_tflite_model()

    print("=" * 96)
    print("LID EXP4D - VoxLingua TF-backend TFLite fidelity test")
    print("=" * 96)
    print(f"ONNX         : {ONNX_MODEL}")
    print(f"TFLite       : {tflite_model}")
    print(f"Dataset      : {DATA_DIR}")
    print(f"Fixed window : {FIXED_SECONDS:.1f}s = {FIXED_SAMPLES} samples")
    print()

    onnx_session = ort.InferenceSession(
        str(ONNX_MODEL),
        providers=["CPUExecutionProvider"],
    )

    onnx_input_name = onnx_session.get_inputs()[0].name

    interpreter = prepare_interpreter(tflite_model)

    input_detail = interpreter.get_input_details()[0]

    print("TFLite input after one-time resize:")
    print(f"  name  : {input_detail['name']}")
    print(f"  shape : {input_detail['shape'].tolist()}")
    print(f"  dtype : {input_detail['dtype']}")
    print()

    rows = []

    stats = {
        lang: {
            "total": 0,
            "onnx_correct": 0,
            "tflite_correct": 0,
            "agree": 0,
            "audio_sec_original": 0.0,
            "tflite_sec": 0.0,
        }
        for lang in LANGUAGES
    }

    max_diffs = []

    for true_lang in LANGUAGES:
        print("=" * 96)
        print(f"LANGUAGE: {true_lang.upper()}")
        print("=" * 96)

        wav_files = sorted(
            (DATA_DIR / true_lang).glob(
                "speaker_*/test/*.wav"
            )
        )

        if not wav_files:
            raise RuntimeError(
                f"No WAV files: {DATA_DIR / true_lang}"
            )

        for wav_path in wav_files:
            samples = load_audio(wav_path)

            original_sec = len(samples) / float(SAMPLE_RATE)

            fixed, window_mode = make_fixed_window(samples)

            # Fair fidelity comparison:
            # ONNX and TFLite receive IDENTICAL fixed 5-second audio.
            onnx_logits = onnx_session.run(
                ["logits"],
                {
                    onnx_input_name:
                        fixed[None, :]
                },
            )[0]

            onnx_logits = (
                np.asarray(onnx_logits)
                .reshape(-1)
                .astype(np.float32)
            )

            tflite_logits, tflite_sec = (
                get_logits_from_tflite(
                    interpreter,
                    fixed,
                )
            )

            if onnx_logits.size != 107:
                raise RuntimeError(
                    f"Unexpected ONNX logits size: {onnx_logits.size}"
                )

            if tflite_logits.size != 107:
                raise RuntimeError(
                    f"Unexpected TFLite logits size: {tflite_logits.size}"
                )

            onnx_result = four_way(onnx_logits)
            tflite_result = four_way(tflite_logits)

            onnx_correct = (
                onnx_result["lang"] == true_lang
            )

            tflite_correct = (
                tflite_result["lang"] == true_lang
            )

            agree = (
                onnx_result["lang"]
                == tflite_result["lang"]
            )

            abs_diff = np.abs(
                onnx_logits - tflite_logits
            )

            max_abs_diff = float(np.max(abs_diff))
            mean_abs_diff = float(np.mean(abs_diff))

            max_diffs.append(max_abs_diff)

            # RTF relative to the actual 5 s LID window.
            tflite_rtf = (
                tflite_sec / FIXED_SECONDS
            )

            stat = stats[true_lang]

            stat["total"] += 1
            stat["audio_sec_original"] += original_sec
            stat["tflite_sec"] += tflite_sec

            if onnx_correct:
                stat["onnx_correct"] += 1

            if tflite_correct:
                stat["tflite_correct"] += 1

            if agree:
                stat["agree"] += 1

            rows.append(
                {
                    "true_lang": true_lang,
                    "speaker":
                        wav_path.parent.parent.name,
                    "filename":
                        wav_path.name,
                    "original_audio_sec":
                        original_sec,
                    "window_mode":
                        window_mode,
                    "fixed_audio_sec":
                        FIXED_SECONDS,
                    "onnx_lang":
                        onnx_result["lang"],
                    "onnx_conf4":
                        onnx_result["conf"],
                    "onnx_margin":
                        onnx_result["margin"],
                    "onnx_correct":
                        onnx_correct,
                    "tflite_lang":
                        tflite_result["lang"],
                    "tflite_conf4":
                        tflite_result["conf"],
                    "tflite_margin":
                        tflite_result["margin"],
                    "tflite_correct":
                        tflite_correct,
                    "prediction_agree":
                        agree,
                    "max_abs_logit_diff":
                        max_abs_diff,
                    "mean_abs_logit_diff":
                        mean_abs_diff,
                    "tflite_ms":
                        tflite_sec * 1000.0,
                    "tflite_rtf_5s":
                        tflite_rtf,
                    "path":
                        str(wav_path),
                }
            )

            print(
                f"{true_lang} | "
                f"{window_mode:11s} | "
                f"ONNX={onnx_result['lang']:2s} "
                f"{'OK' if onnx_correct else 'WRONG':5s} | "
                f"TFL={tflite_result['lang']:2s} "
                f"{'OK' if tflite_correct else 'WRONG':5s} | "
                f"agree={'Y' if agree else 'N'} | "
                f"diff={max_abs_diff:.5f} | "
                f"{tflite_sec*1000:.1f}ms"
            )

    fields = [
        "true_lang",
        "speaker",
        "filename",
        "original_audio_sec",
        "window_mode",
        "fixed_audio_sec",
        "onnx_lang",
        "onnx_conf4",
        "onnx_margin",
        "onnx_correct",
        "tflite_lang",
        "tflite_conf4",
        "tflite_margin",
        "tflite_correct",
        "prediction_agree",
        "max_abs_logit_diff",
        "mean_abs_logit_diff",
        "tflite_ms",
        "tflite_rtf_5s",
        "path",
    ]

    with RESULTS_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 96)
    print("SUMMARY")
    print("=" * 96)

    summary_rows = []

    total_all = 0
    onnx_correct_all = 0
    tflite_correct_all = 0
    agree_all = 0
    tflite_sec_all = 0.0

    for lang in LANGUAGES:
        s = stats[lang]

        total = s["total"]

        onnx_acc = s["onnx_correct"] / total
        tflite_acc = s["tflite_correct"] / total
        agreement = s["agree"] / total

        # Every inference processes a 5 s window.
        processed_sec = total * FIXED_SECONDS
        rtf = s["tflite_sec"] / processed_sec

        print(
            f"{lang.upper():2s} | "
            f"ONNX {s['onnx_correct']:2d}/{total:2d} "
            f"= {onnx_acc*100:6.2f}% | "
            f"TFLite {s['tflite_correct']:2d}/{total:2d} "
            f"= {tflite_acc*100:6.2f}% | "
            f"agree={agreement*100:6.2f}% | "
            f"RTF={rtf:.4f}"
        )

        total_all += total
        onnx_correct_all += s["onnx_correct"]
        tflite_correct_all += s["tflite_correct"]
        agree_all += s["agree"]
        tflite_sec_all += s["tflite_sec"]

        summary_rows.append(
            {
                "language": lang,
                "total": total,
                "onnx_correct":
                    s["onnx_correct"],
                "onnx_accuracy":
                    onnx_acc,
                "tflite_correct":
                    s["tflite_correct"],
                "tflite_accuracy":
                    tflite_acc,
                "agreement":
                    agreement,
                "tflite_sec":
                    s["tflite_sec"],
                "tflite_rtf_5s":
                    rtf,
            }
        )

    overall_onnx_acc = (
        onnx_correct_all / total_all
    )

    overall_tflite_acc = (
        tflite_correct_all / total_all
    )

    overall_agreement = (
        agree_all / total_all
    )

    total_processed_sec = (
        total_all * FIXED_SECONDS
    )

    overall_rtf = (
        tflite_sec_all /
        total_processed_sec
    )

    print()
    print(
        "OVERALL ONNX   : "
        f"{onnx_correct_all}/{total_all} "
        f"= {overall_onnx_acc*100:.2f}%"
    )

    print(
        "OVERALL TFLite : "
        f"{tflite_correct_all}/{total_all} "
        f"= {overall_tflite_acc*100:.2f}%"
    )

    print(
        "Prediction agree: "
        f"{agree_all}/{total_all} "
        f"= {overall_agreement*100:.2f}%"
    )

    print(
        f"TFLite RTF     : {overall_rtf:.4f}"
    )

    if overall_rtf > 0:
        print(
            f"TFLite speed   : "
            f"{1.0 / overall_rtf:.2f}x realtime"
        )

    if max_diffs:
        print(
            "Max logit diff : "
            f"{max(max_diffs):.6f}"
        )

        print(
            "Mean max diff  : "
            f"{np.mean(max_diffs):.6f}"
        )

    summary_rows.append(
        {
            "language": "overall",
            "total": total_all,
            "onnx_correct":
                onnx_correct_all,
            "onnx_accuracy":
                overall_onnx_acc,
            "tflite_correct":
                tflite_correct_all,
            "tflite_accuracy":
                overall_tflite_acc,
            "agreement":
                overall_agreement,
            "tflite_sec":
                tflite_sec_all,
            "tflite_rtf_5s":
                overall_rtf,
        }
    )

    summary_fields = [
        "language",
        "total",
        "onnx_correct",
        "onnx_accuracy",
        "tflite_correct",
        "tflite_accuracy",
        "agreement",
        "tflite_sec",
        "tflite_rtf_5s",
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
