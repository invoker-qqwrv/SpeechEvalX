"""Shared 16-kHz mono float32 preprocessing."""
from pathlib import Path


def read_audio(audio_path: Path):
    try:
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise RuntimeError("Audio input requires: pip install numpy soundfile scipy") from exc
    from math import gcd
    audio, rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if not len(audio): raise ValueError(f"Empty audio: {audio_path}")
    if rate != 16000:
        divisor = gcd(rate, 16000)
        audio = resample_poly(audio, 16000 // divisor, rate // divisor)
    return np.asarray(audio, dtype=np.float32)
