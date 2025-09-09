# src/whisper_transcriber.py
from __future__ import annotations
from typing import Dict, Any
import numpy as np
import whisper
from audio_utils import TARGET_SR

WHISPER_SIZES = ["tiny", "base", "small", "medium", "large"]

class WhisperTranscriber:
    def __init__(self, model_size: str = "small", device: str | None = None, compute_type: str | None = None):
        """
        model_size: one of WHISPER_SIZES
        device: 'cuda' or 'cpu' (auto if None)
        compute_type: ignored by openai-whisper; kept for future compatibility
        """
        assert model_size in WHISPER_SIZES, f"model_size must be one of {WHISPER_SIZES}"
        self.model = whisper.load_model(model_size, device=device)

    def transcribe(self, audio: np.ndarray, language: str | None = None, task: str = "transcribe") -> Dict[str, Any]:
        """
        language: 'pt', 'en', etc. If None, Whisper will detect.
        task: 'transcribe' or 'translate'
        """
        # Whisper expects float32 16000 mono in numpy, that's okay
        result = self.model.transcribe(audio, language=language, task=task, fp16=False)
        # Normalize output similar to Vosk
        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "start": seg.get("start"),
                "end": seg.get("end"),
                "text": seg.get("text", "").strip()
            })
        return {
            "engine": "whisper",
            "text": result.get("text", "").strip(),
            "segments": segments,
            "language": result.get("language")
        }
