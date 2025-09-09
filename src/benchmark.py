# src/benchmark.py
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import jiwer
from audio_utils import load_audio_mono, TARGET_SR
from vosk_transcriber import VoskTranscriber
from whisper_transcriber import WhisperTranscriber

@dataclass
class ASRBenchmarkResult:
    engine: str
    text: str
    elapsed_sec: float
    language: Optional[str] = None
    wer: Optional[float] = None

# Substitua a função evaluate_wer existente por esta:

def evaluate_wer(hyp: str, ref: Optional[str]) -> Optional[float]:
    if not ref:
        return None
    # Normalizações idênticas para truth e hypothesis
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),         # remove vírgulas/pontos/apóstrofos
        jiwer.ReduceToListOfListOfWords()  # tokeniza em palavras
    ])
    try:
        return float(jiwer.wer(
            ref, hyp,
            truth_transform=transformation,
            hypothesis_transform=transformation
        ))
    except Exception:
        return None


def run_vosk(audio_path: str, vosk_model_path: str, reference_text: Optional[str] = None) -> ASRBenchmarkResult:
    audio = load_audio_mono(audio_path, TARGET_SR)
    asr = VoskTranscriber(vosk_model_path)
    t0 = time.time()
    out = asr.transcribe(audio)
    elapsed = time.time() - t0
    return ASRBenchmarkResult(
        engine="vosk",
        text=out["text"],
        elapsed_sec=elapsed,
        language=None,
        wer=evaluate_wer(out["text"], reference_text)
    )

def run_whisper(audio_path: str, model_size: str = "small", language: Optional[str] = None,
                reference_text: Optional[str] = None, device: Optional[str] = None) -> ASRBenchmarkResult:
    audio = load_audio_mono(audio_path, TARGET_SR)
    asr = WhisperTranscriber(model_size=model_size, device=device)
    t0 = time.time()
    out = asr.transcribe(audio, language=language)
    elapsed = time.time() - t0
    return ASRBenchmarkResult(
        engine="whisper-" + model_size,
        text=out["text"],
        elapsed_sec=elapsed,
        language=out.get("language"),
        wer=evaluate_wer(out["text"], reference_text)
    )

def pretty_print(res: ASRBenchmarkResult):
    print(f"Engine      : {res.engine}")
    print(f"Language    : {res.language}")
    print(f"Elapsed (s) : {res.elapsed_sec:.2f}")
    if res.wer is not None:
        print(f"WER         : {res.wer:.3f}")
    print("Transcript  :")
    print(res.text)
    print("-" * 60)
