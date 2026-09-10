"""Uzbek MMS TTS PC probe: synthesize with sherpa-onnx from the locally
converted vits-mms uzb model and measure RTF/volume/validity.

Model: facebook/mms-tts models/uzb-script_cyrillic converted via the
official sherpa vits-mms.py pipeline (see README). The model expects
CYRILLIC script input - the app's ASR outputs Latin Uzbek, so Android
integration needs a Latin->Cyrillic transliteration step.
"""

import time
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf

BASE_DIR = Path(__file__).resolve().parent
MODEL_ONNX = BASE_DIR / "convert_uzb" / "model.onnx"
TOKENS = BASE_DIR / "convert_uzb" / "tokens.txt"
OUT_DIR = BASE_DIR / "out_wav_uzb"

SENTENCES = [
    "Салом, алейкум!",
    "Ассалому алайкум, мен Тошкентда яшайман.",
    "Сунъий интеллект ҳаётимизни ҳар куни ўзгартирмоқда.",
]


def build_tts():
    vits = sherpa_onnx.OfflineTtsVitsModelConfig(
        model=str(MODEL_ONNX),
        tokens=str(TOKENS),
    )
    model_config = sherpa_onnx.OfflineTtsModelConfig(
        vits=vits,
        num_threads=4,
        debug=False,
        provider="cpu",
    )
    config = sherpa_onnx.OfflineTtsConfig(
        model=model_config,
        max_num_sentences=1,
    )
    return sherpa_onnx.OfflineTts(config)


def main():
    for path in (MODEL_ONNX, TOKENS):
        if not path.is_file():
            raise FileNotFoundError(path)

    print("=" * 70)
    print("Uzbek MMS TTS probe - vits-mms uzb (cyrillic script)")
    print("=" * 70)
    print(f"model : {MODEL_ONNX}  {MODEL_ONNX.stat().st_size/1e6:.1f} MB")
    print(f"sherpa: {sherpa_onnx.__file__}")

    tts = build_tts()
    sr = tts.sample_rate
    print(f"sample rate: {sr}")

    OUT_DIR.mkdir(exist_ok=True)

    rows = []
    for i, text in enumerate(SENTENCES):
        t0 = time.perf_counter()
        gen = tts.generate(text, sid=0, speed=1.0)
        elapsed = time.perf_counter() - t0

        samples = np.asarray(gen.samples, dtype=np.float32)
        duration = len(samples) / sr
        rtf = elapsed / duration if duration > 0 else float("inf")
        rms = float(np.sqrt(np.mean(samples**2))) if len(samples) else 0.0

        wav = OUT_DIR / f"uzb_{i}.wav"
        sf.write(wav, samples, sr)

        print(f"\n[{i}] {text}")
        print(f"    audio {duration:.2f}s  synth {elapsed:.2f}s  "
              f"RTF {rtf:.3f}  rms {rms:.4f}  -> {wav.name}")
        rows.append((text, duration, elapsed, rtf, rms, wav))

    print("\n" + "=" * 70)
    avg = sum(r[3] for r in rows) / len(rows)
    print(f"summary: model {MODEL_ONNX.stat().st_size/1e6:.0f} MB, "
          f"avg RTF (PC, 4 threads) {avg:.3f}")
    print("wav files: " + str(OUT_DIR))
    print("NOTE: model input is CYRILLIC; latin->cyrillic needed for app.")


if __name__ == "__main__":
    main()
