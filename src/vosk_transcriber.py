# src/vosk_transcriber.py
from __future__ import annotations
import json
import wave
import numpy as np
from typing import Optional, Dict, Any, List
from vosk import Model, KaldiRecognizer
from audio_utils import TARGET_SR, chunk_audio

def _float_to_int16_pcm(audio: np.ndarray) -> bytes:
    audio = np.clip(audio, -1.0, 1.0)
    audio_i16 = (audio * 32767.0).astype('<i2')
    return audio_i16.tobytes()

class VoskTranscriber:
    def __init__(self, model_path: str, sample_rate: int = TARGET_SR):
        self.model = Model(model_path)
        self.sample_rate = sample_rate

    def transcribe(self, audio: np.ndarray, chunk_sec: float = 15.0) -> Dict[str, Any]:
        """Returns dict with 'text' and 'segments' (word-level if available)."""
        recognizer = KaldiRecognizer(self.model, self.sample_rate)
        recognizer.SetWords(True)

        full_text: List[str] = []
        segments_all: List[Dict[str, Any]] = []

        for start, chunk in chunk_audio(audio, self.sample_rate, chunk_sec=chunk_sec):
            pcm = _float_to_int16_pcm(chunk)
            # Feed in small frames for better behavior
            frame_size = 4000  # bytes ~ 2000 samples
            for i in range(0, len(pcm), frame_size):
                part = pcm[i:i+frame_size]
                if recognizer.AcceptWaveform(part):
                    res = json.loads(recognizer.Result())
                    if res.get("text"):
                        full_text.append(res["text"])
                        if "result" in res:
                            segments_all.extend(res["result"])
            # flush partial at chunk end
            partial = json.loads(recognizer.FinalResult())
            if partial.get("text"):
                full_text.append(partial["text"])
                if "result" in partial:
                    segments_all.extend(partial["result"])

        text = " ".join(t.strip() for t in full_text if t.strip())
        return {"engine": "vosk", "text": text.strip(), "segments": segments_all}
