"""Reproduce the Android crash hypothesis on PC: concurrent sherpa native
activity (VAD-style frame loop) while an OfflineTts model is being
constructed. The Android crash log shows 'pthread_mutex_lock called on a
destroyed mutex' across several threads right when the offline TTS loads
mid-conversation (VAD still streaming frames).

If this reproduces a crash/abort, the fix must serialize native model
construction against streaming sherpa activity (e.g. preload the TTS
engine before the conversation starts).
"""

import threading
import time
import numpy as np
from pathlib import Path

import sherpa_onnx

MODEL = Path(r"D:\dev\projects\FourLangAsrMobile\experiments\tts\models\vits-piper-ru_RU-irina-medium")
SILERO = Path(r"D:\dev\projects\FourLangAsrMobile\app\src\main\assets\silero_vad.onnx")
STOP = threading.Event()
crashed = []


def build_vad():
    """Streaming silero VAD, like the Android VadEngine (sherpa java Vad)."""
    config = sherpa_onnx.VadModelConfig(
        silero_vad=sherpa_onnx.SileroVadModelConfig(
            model=str(SILERO),
            threshold=0.5,
            min_silence_duration=0.6,
            min_speech_duration=0.25,
            window_size=512,
        ),
        sample_rate=16000,
        num_threads=1,
        provider="cpu",
        debug=False,
    )
    return sherpa_onnx.VoiceActivityDetector(config)


def vad_loop():
    try:
        vad = build_vad()
        rng = np.random.default_rng(0)
        while not STOP.is_set():
            vad.accept_waveform(rng.standard_normal(512).astype(np.float32) * 0.01)
            while not vad.empty():
                vad.pop()
            time.sleep(0.001)
    except Exception as e:  # noqa
        crashed.append(f"vad: {type(e).__name__}: {e}")
        STOP.set()


def tts_loop(iterations):
    try:
        vits = sherpa_onnx.OfflineTtsVitsModelConfig(
            model=str(MODEL / "ru_RU-irina-medium.onnx"),
            tokens=str(MODEL / "tokens.txt"),
            data_dir=str(MODEL / "espeak-ng-data"),
        )
        mc = sherpa_onnx.OfflineTtsModelConfig(vits=vits, num_threads=2, debug=False, provider="cpu")
        cfg = sherpa_onnx.OfflineTtsConfig(model=mc)
        tts = sherpa_onnx.OfflineTts(cfg)
        for i in range(iterations):
            tts.generate("Привет, как дела?", sid=0, speed=1.0)
            if STOP.is_set():
                return
    except Exception as e:  # noqa
        crashed.append(f"tts: {type(e).__name__}: {e}")
        STOP.set()


def main():
    print("sherpa:", sherpa_onnx.__version__ if hasattr(sherpa_onnx, "__version__") else "?")
    ok = 0
    for round_no in range(3):
        STOP.clear()
        crashed.clear()
        t1 = threading.Thread(target=vad_loop, daemon=True)
        t1.start()
        time.sleep(1.0)  # let VAD native activity spin up
        t2 = threading.Thread(target=tts_loop, args=(20,), daemon=True)
        t2.start()
        t2.join(timeout=60)
        STOP.set()
        t1.join(timeout=5)
        if crashed:
            print(f"round {round_no}: CRASH/ERROR -> {crashed[:2]}")
            return
        print(f"round {round_no}: ok (tts constructed+synthesized while vad streaming)")
        ok += 1
    print("no reproduction on PC")


if __name__ == "__main__":
    main()
