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
    / "lid_exp1"
    / "models"
    / "voxlingua107"
    / "voxlingua107.onnx"
)

TFLITE_DIR = BASE_DIR / "converted"

DATA_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "speaker_exp1"
    / "speaker_data_exp4"
)

RESULTS_CSV = BASE_DIR / "lid_exp4_tflite_results.csv"
SUMMARY_CSV = BASE_DIR / "lid_exp4_tflite_summary.csv"

LANGUAGES = ["zh", "en", "ru", "uz"]

# Same class indices used in EXP2.
TARGET_INDEX = {
    "en": 20,
    "ru": 77,
    "uz": 101,
    "zh": 106,
}


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

    if sample_rate != 16000:
        raise RuntimeError(
            f"{path}: expected 16000 Hz, got {sample_rate}"
        )

    return samples, sample_rate


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


def get_logits_from_tflite(interpreter, samples):
    input_detail = interpreter.get_input_details()[0]
    input_index = input_detail["index"]

    desired_shape = np.array(
        [1, len(samples)],
        dtype=np.int32,
    )

    # Dynamic audio: resize to the exact VAD segment length.
    if not np.array_equal(
        np.asarray(input_detail["shape"], dtype=np.int32),
        desired_shape,
    ):
        interpreter.resize_tensor_input(
            input_index,
            desired_shape.tolist(),
            strict=False,
        )
        interpreter.allocate_tensors()

    input_detail = interpreter.get_input_details()[0]

    x = samples[None, :].astype(
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
        value = interpreter.get_tensor(
            detail["index"]
        )

        flat = np.asarray(value).reshape(-1)

        if flat.size == 107:
            logits = flat.astype(
                np.float32,
                copy=False,
            )
            break

    if logits is None:
        output_shapes = [
            tuple(
                interpreter
                .get_tensor(d["index"])
                .shape
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

    print("=" * 92)
    print("LID EXP4 - VoxLingua LiteRT/TFLite fidelity test")
    print("=" * 92)
    print(f"ONNX   : {ONNX_MODEL}")
    print(f"TFLite : {tflite_model}")
    print(f"Data   : {DATA_DIR}")
    print()

    onnx_session = ort.InferenceSession(
        str(ONNX_MODEL),
        providers=["CPUExecutionProvider"],
    )

    onnx_input_name = (
        onnx_session
        .get_inputs()[0]
        .name
    )

    interpreter = Interpreter(
        model_path=str(tflite_model),
        num_threads=1,
    )

    interpreter.allocate_tensors()

    input_detail = interpreter.get_input_details()[0]

    print("TFLite input:")
    print(f"  name            : {input_detail['name']}")
    print(f"  shape           : {input_detail['shape'].tolist()}")
    print(
        "  shape_signature : "
        f"{input_detail.get('shape_signature', []).tolist()}"
    )
    print(f"  dtype           : {input_detail['dtype']}")
    print()

    rows = []

    stats = {
        lang: {
            "total": 0,
            "onnx_correct": 0,
            "tflite_correct": 0,
            "agree": 0,
            "audio_sec": 0.0,
            "tflite_sec": 0.0,
        }
        for lang in LANGUAGES
    }

    diff_values = []

    for true_lang in LANGUAGES:
        print("=" * 92)
        print(f"LANGUAGE: {true_lang.upper()}")
        print("=" * 92)

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
            samples, sample_rate = load_audio(
                wav_path
            )

            audio_sec = (
                len(samples)
                / float(sample_rate)
            )

            onnx_logits = onnx_session.run(
                ["logits"],
                {
                    onnx_input_name:
                        samples[None, :]
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
                    samples,
                )
            )

            if onnx_logits.size != 107:
                raise RuntimeError(
                    "Unexpected ONNX logits size: "
                    f"{onnx_logits.size}"
                )

            if tflite_logits.size != 107:
                raise RuntimeError(
                    "Unexpected TFLite logits size: "
                    f"{tflite_logits.size}"
                )

            onnx_result = four_way(
                onnx_logits
            )

            tflite_result = four_way(
                tflite_logits
            )

            onnx_correct = (
                onnx_result["lang"]
                == true_lang
            )

            tflite_correct = (
                tflite_result["lang"]
                == true_lang
            )

            agree = (
                onnx_result["lang"]
                == tflite_result["lang"]
            )

            max_abs_diff = float(
                np.max(
                    np.abs(
                        onnx_logits
                        - tflite_logits
                    )
                )
            )

            mean_abs_diff = float(
                np.mean(
                    np.abs(
                        onnx_logits
                        - tflite_logits
                    )
                )
            )

            diff_values.append(
                max_abs_diff
            )

            rtf = (
                tflite_sec / audio_sec
                if audio_sec > 0
                else 0.0
            )

            stat = stats[true_lang]

            stat["total"] += 1
            stat["audio_sec"] += audio_sec
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
                        wav_path
                        .parent
                        .parent
                        .name,
                    "filename":
                        wav_path.name,
                    "audio_sec":
                        audio_sec,
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
                    "tflite_rtf":
                        rtf,
                    "path":
                        str(wav_path),
                }
            )

            print(
                f"{true_lang} | "
                f"ONNX={onnx_result['lang']:2s} "
                f"{'OK' if onnx_correct else 'WRONG':5s} | "
                f"TFL={tflite_result['lang']:2s} "
                f"{'OK' if tflite_correct else 'WRONG':5s} | "
                f"agree={'Y' if agree else 'N'} | "
                f"diff={max_abs_diff:.5f} | "
                f"RTF={rtf:.4f}"
            )

    fields = [
        "true_lang",
        "speaker",
        "filename",
        "audio_sec",
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
        "tflite_rtf",
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
    print("=" * 92)
    print("SUMMARY")
    print("=" * 92)

    summary_rows = []

    overall = {
        "total": 0,
        "onnx_correct": 0,
        "tflite_correct": 0,
        "agree": 0,
        "audio_sec": 0.0,
        "tflite_sec": 0.0,
    }

    for lang in LANGUAGES:
        s = stats[lang]

        total = s["total"]

        onnx_acc = (
            s["onnx_correct"] / total
        )

        tflite_acc = (
            s["tflite_correct"] / total
        )

        agreement = (
            s["agree"] / total
        )

        rtf = (
            s["tflite_sec"]
            / s["audio_sec"]
        )

        print(
            f"{lang.upper():2s} | "
            f"ONNX {s['onnx_correct']:2d}/{total:2d} "
            f"= {onnx_acc*100:6.2f}% | "
            f"TFLite {s['tflite_correct']:2d}/{total:2d} "
            f"= {tflite_acc*100:6.2f}% | "
            f"agree={agreement*100:6.2f}% | "
            f"RTF={rtf:.4f}"
        )

        for key in overall:
            overall[key] += s[key]

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
                "audio_sec":
                    s["audio_sec"],
                "tflite_sec":
                    s["tflite_sec"],
                "tflite_rtf":
                    rtf,
            }
        )

    total = overall["total"]

    overall_onnx_acc = (
        overall["onnx_correct"] / total
    )

    overall_tflite_acc = (
        overall["tflite_correct"] / total
    )

    overall_agreement = (
        overall["agree"] / total
    )

    overall_rtf = (
        overall["tflite_sec"]
        / overall["audio_sec"]
    )

    print()
    print(
        "OVERALL ONNX   : "
        f"{overall['onnx_correct']}/{total} "
        f"= {overall_onnx_acc*100:.2f}%"
    )

    print(
        "OVERALL TFLite : "
        f"{overall['tflite_correct']}/{total} "
        f"= {overall_tflite_acc*100:.2f}%"
    )

    print(
        "Prediction agree: "
        f"{overall['agree']}/{total} "
        f"= {overall_agreement*100:.2f}%"
    )

    print(
        f"TFLite RTF     : {overall_rtf:.4f}"
    )

    if overall_rtf > 0:
        print(
            f"TFLite speed   : "
            f"{1.0/overall_rtf:.2f}x realtime"
        )

    if diff_values:
        print(
            "Max logit diff : "
            f"{max(diff_values):.6f}"
        )

        print(
            "Mean max diff  : "
            f"{np.mean(diff_values):.6f}"
        )

    summary_rows.append(
        {
            "language": "overall",
            "total": total,
            "onnx_correct":
                overall["onnx_correct"],
            "onnx_accuracy":
                overall_onnx_acc,
            "tflite_correct":
                overall["tflite_correct"],
            "tflite_accuracy":
                overall_tflite_acc,
            "agreement":
                overall_agreement,
            "audio_sec":
                overall["audio_sec"],
            "tflite_sec":
                overall["tflite_sec"],
            "tflite_rtf":
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
        "audio_sec",
        "tflite_sec",
        "tflite_rtf",
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
