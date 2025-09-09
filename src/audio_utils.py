# src/audio_utils.py
from __future__ import annotations
import os
import numpy as np
import soundfile as sf
import resampy

TARGET_SR = 16000

def load_audio_mono(path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    """Load audio as mono float32 at target_sr."""
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != target_sr:
        audio = resampy.resample(audio, sr, target_sr)
    # normalize to [-1, 1]
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    maxv = max(1e-9, np.max(np.abs(audio)))
    return (audio / maxv) * 0.95

def write_wav(path: str, audio: np.ndarray, sr: int = TARGET_SR):
    sf.write(path, audio, sr)

def chunk_audio(audio: np.ndarray, sr: int = TARGET_SR, chunk_sec: float = 15.0, overlap_sec: float = 0.5):
    """Yield overlapping chunks in samples."""
    chunk = int(chunk_sec * sr)
    hop = int((chunk_sec - overlap_sec) * sr)
    n = len(audio)
    if n <= chunk:
        yield 0, audio
        return
    start = 0
    while start < n:
        end = min(n, start + chunk)
        yield start, audio[start:end]
        if end == n:
            break
        start += hop
