"""TTS route B feasibility probe: sherpa-onnx Piper (Russian) on PC.

Goal: verify that an offline neural TTS engine (the same sherpa-onnx
runtime the app already ships) can synthesize Russian on-device-class
hardware, and measure the numbers that decide Android integration:

  - model size (asset budget)
  - synthesis RTF on CPU (orders of magnitude only; phone re-measured)
  - output validity (non-silent, plausible duration)

The final Android shape would be an OfflineNeuralTtsEngine implementing
tts/TtsEngine with this model; the upper pipeline (translation seam,
auto-speak, mic gate) would not change.

Model: vits-piper-ru_RU-irina-medium (sherpa-onnx tts-models release),
63 MB onnx + espeak-ng-data G2P data.
"""

import time
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "vits-piper-ru_RU-irina-medium"

ONNX = MODEL_DIR / "ru_RU-irina-medium.onnx"
TOKENS = MODEL_DIR / "tokens.txt"
DATA_DIR = MODEL_DIR / "espeak-ng-data"

OUT_WAV_DIR = BASE_DIR / "out_wav"

# Short / medium / long Russian sentences (includes a dictation-style one).
SENTENCES = [
    "Привет!",
    "Меня зовут Анна, я живу в Ташкенте.",
    "Дайте, пожалуйста, два билета на завтрашний поезд.",
    "Искусственный интеллект меняет нашу жизнь каждый день, и мы должны быть готовы к новым возможностям.",
]


def build_tts():
    vits = sherpa_onnx.OfflineTtsVitsModelConfig(
        model=str(ONNX),
        tokens=str(TOKENS),
        data_dir=str(DATA_DIR),
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
    for path in (ONNX, TOKENS):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(DATA_DIR)

    model_bytes = ONNX.stat().st_size
    espeak_bytes = sum(f.stat().st_size for f in DATA_DIR.rglob("*") if f.is_file())

    print("=" * 70)
    print("TTS route B probe - sherpa-onnx Piper ru_RU (irina-medium)")
    print("=" * 70)
    print(f"model   : {ONNX}  {model_bytes / 1e6:.1f} MB")
    print(f"espeak  : {espeak_bytes / 1e6:.1f} MB (G2P data)")
    print(f"sherpa  : {sherpa_onnx.__file__}")

    tts = build_tts()
    sample_rate = tts.sample_rate
    print(f"sample rate: {sample_rate}")

    OUT_WAV_DIR.mkdir(exist_ok=True)

    rows = []
    for i, text in enumerate(SENTENCES):
        t0 = time.perf_counter()
        generated = tts.generate(text, sid=0, speed=1.0)
        elapsed = time.perf_counter() - t0

        samples_np = np.asarray(generated.samples, dtype=np.float32)
        duration = len(samples_np) / sample_rate
        rtf = elapsed / duration if duration > 0 else float("inf")
        rms = float(np.sqrt(np.mean(samples_np**2))) if len(samples_np) else 0.0

        wav = OUT_WAV_DIR / f"sentence_{i}.wav"
        sf.write(wav, samples_np, sample_rate)

        print(f"\n[{i}] {text[:60]}")
        print(f"    audio {duration:.2f}s  synth {elapsed:.2f}s  "
              f"RTF {rtf:.3f}  rms {rms:.4f}  -> {wav.name}")
        rows.append((text, duration, elapsed, rtf, rms, wav))

    print("\n" + "=" * 70)
    avg_rtf = sum(r[3] for r in rows) / len(rows)
    print(f"summary: model {model_bytes/1e6:.0f}MB + espeak {espeak_bytes/1e6:.0f}MB, "
          f"avg RTF (PC, 4 threads) {avg_rtf:.3f}")
    print("RTF on a phone will be higher; audio quality needs a human listen.")
    print("wav files: " + str(OUT_WAV_DIR))


if __name__ == "__main__":
    main()
