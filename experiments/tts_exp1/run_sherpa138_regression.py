"""sherpa-onnx 1.13.8 upgrade risk assessment: run every model the app
uses through the new runtime on PC.

Covers:
  1. uz mms TTS (the unlock candidate)
  2. ru piper TTS (current offline TTS)
  3. zh online zipformer2 CTC ASR
  4. en offline transducer ASR (also the code-switch verifier)
  5. silero VAD

Uses the same models/assets the app ships, with the same decode configs.
"""

import time
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parents[1]
ASSETS = PROJECT / "app" / "src" / "main" / "assets"
EXP4 = PROJECT / "experiments" / "codeswitch_natural_exp4"

ZH_DIR = BASE / "models" / "zh_probe"
EN_DIR = BASE / "models" / "en_probe"

MODEL_ROOT = Path(r"D:\dev\projects\fourlang_translation\asr_exp1\models")
ZH_MODEL = MODEL_ROOT / "sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01"
EN_MODEL = MODEL_ROOT / "sherpa-onnx-zipformer-gigaspeech-2023-12-12"

RU_VITS = BASE / "models" / "vits-piper-ru_RU-irina-medium"
UZ_VITS = BASE / "convert_uzb"

results = []


def check(name, fn):
    t0 = time.perf_counter()
    try:
        fn()
        results.append((name, "PASS", time.perf_counter() - t0))
    except Exception as e:
        results.append((name, f"FAIL: {type(e).__name__}: {e}", time.perf_counter() - t0))


def load_wav(path):
    s, sr = sf.read(path, dtype="float32", always_2d=False)
    if s.ndim == 2:
        s = s.mean(axis=1).astype(np.float32)
    return np.ascontiguousarray(s, dtype=np.float32), sr


def uz_tts():
    vits = sherpa_onnx.OfflineTtsVitsModelConfig(
        model=str(UZ_VITS / "model.onnx"),
        tokens=str(UZ_VITS / "tokens.txt"),
    )
    tts = sherpa_onnx.OfflineTts(
        sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(vits=vits, num_threads=4, debug=False, provider="cpu")
        )
    )
    g = tts.generate("Салом, дунё!", sid=0, speed=1.0)
    assert len(g.samples) > 1000, "empty uz synthesis"


def ru_tts():
    vits = sherpa_onnx.OfflineTtsVitsModelConfig(
        model=str(RU_VITS / "ru_RU-irina-medium.onnx"),
        tokens=str(RU_VITS / "tokens.txt"),
        data_dir=str(RU_VITS / "espeak-ng-data"),
    )
    tts = sherpa_onnx.OfflineTts(
        sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(vits=vits, num_threads=4, debug=False, provider="cpu")
        )
    )
    g = tts.generate("Привет, как дела?", sid=0, speed=1.0)
    assert len(g.samples) > 1000, "empty ru synthesis"


def zh_asr():
    rec = sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
        tokens=str(ZH_MODEL / "tokens.txt"),
        model=str(ZH_MODEL / "model.int8.onnx"),
        num_threads=2, sample_rate=16000, feature_dim=80,
        decoding_method="greedy_search", enable_endpoint_detection=False,
        provider="cpu", debug=False,
    )
    # use one ASCEND pure-Chinese clip
    clip = EXP4 / "audio" / "negative_zh" / "ascend_negative_zh_001_validation_00006.wav"
    s, sr = load_wav(clip)
    st = rec.create_stream()
    st.accept_waveform(16000, s)
    st.accept_waveform(16000, np.zeros(int(0.66 * 16000), dtype=np.float32))
    st.input_finished()
    while rec.is_ready(st):
        rec.decode_stream(st)
    text = rec.get_result(st).strip()
    assert text, "empty zh ASR"


def en_asr():
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(EN_MODEL / "encoder-epoch-30-avg-1.int8.onnx"),
        decoder=str(EN_MODEL / "decoder-epoch-30-avg-1.int8.onnx"),
        joiner=str(EN_MODEL / "joiner-epoch-30-avg-1.int8.onnx"),
        tokens=str(EN_MODEL / "tokens.txt"),
        num_threads=2, sample_rate=16000, feature_dim=80,
        decoding_method="greedy_search", provider="cpu", debug=False,
    )
    clip = EXP4 / "audio" / "negative_en" / "ascend_negative_en_001_validation_00002.wav"
    s, sr = load_wav(clip)
    st = rec.create_stream()
    st.accept_waveform(16000, s)
    rec.decode_streams([st])
    assert st.result.text.strip(), "empty en ASR"


def vad():
    cfg = sherpa_onnx.VadModelConfig(
        silero_vad=sherpa_onnx.SileroVadModelConfig(
            model=str(ASSETS / "silero_vad.onnx"),
            threshold=0.5, min_silence_duration=0.6,
            min_speech_duration=0.25, window_size=512,
        ),
        sample_rate=16000, num_threads=1, provider="cpu", debug=False,
    )
    vad = sherpa_onnx.VoiceActivityDetector(cfg)
    rng = np.random.default_rng(0)
    vad.accept_waveform(rng.standard_normal(5120).astype(np.float32) * 0.01)
    vad.flush()


def main():
    print("sherpa", sherpa_onnx.__version__)
    check("uz mms TTS (unlock candidate)", uz_tts)
    check("ru piper TTS", ru_tts)
    check("zh online zipformer2 CTC ASR", zh_asr)
    check("en offline transducer ASR / code-switch verifier", en_asr)
    check("silero VAD", vad)
    print()
    for name, status, elapsed in results:
        print(f"{status:8s} {elapsed:6.2f}s  {name}")


if __name__ == "__main__":
    main()
