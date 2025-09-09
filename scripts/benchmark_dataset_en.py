# scripts/benchmark_dataset_en.py
from __future__ import annotations
import sys, time, csv
from pathlib import Path
from typing import Optional, Iterable

# Monta import para src/
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(SRC))

from audio_utils import load_audio_mono, TARGET_SR
from vosk_transcriber import VoskTranscriber
from whisper_transcriber import WhisperTranscriber
from benchmark import evaluate_wer  # usa a versão corrigida (jiwer v3)

def iter_wavs(folder: Path) -> Iterable[Path]:
    return sorted(folder.glob("*.wav"))

def read_ref(txt_path: Path) -> Optional[str]:
    return txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else None

def main():
    samples_en = ROOT / "samples" / "en"
    if not samples_en.exists():
        print("samples/en não encontrado. Rode prepare_librispeech_full.py antes.")
        return

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / "benchmark_en.csv"

    # === CONFIGURÁVEIS ===
    # Vosk
    vosk_model_path = ROOT / "models" / "vosk-en"
    use_vosk = vosk_model_path.exists()

    # Whisper
    whisper_size = "base"  # "tiny", "base", "small", "medium", "large"
    whisper_device = "cpu" # "cpu" ou "cuda"

    # Carrega modelos UMA vez
    vosk_asr = VoskTranscriber(str(vosk_model_path)) if use_vosk else None
    whisper_asr = WhisperTranscriber(model_size=whisper_size, device=whisper_device)

    rows = []
    n = 0
    t0_all = time.time()

    for wav_path in iter_wavs(samples_en):
        ref = read_ref(wav_path.with_suffix(".txt"))
        audio = load_audio_mono(str(wav_path), TARGET_SR)

        # Vosk
        if vosk_asr is not None:
            t0 = time.time()
            out_v = vosk_asr.transcribe(audio)
            elapsed_v = time.time() - t0
            wer_v = evaluate_wer(out_v.get("text", ""), ref)
            rows.append(["vosk", wav_path.name, f"{elapsed_v:.3f}", "" if wer_v is None else f"{wer_v:.6f}"])

        # Whisper
        t0 = time.time()
        out_w = whisper_asr.transcribe(audio, language="en")
        elapsed_w = time.time() - t0
        wer_w = evaluate_wer(out_w.get("text", ""), ref)
        rows.append([f"whisper-{whisper_size}", wav_path.name, f"{elapsed_w:.3f}", "" if wer_w is None else f"{wer_w:.6f}"])

        n += 1
        if n % 100 == 0:
            print(f"... processados {n} arquivos")

    total_elapsed = time.time() - t0_all

    # Grava CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["engine", "file", "elapsed_s", "wer"])
        wr.writerows(rows)

    print(f"✔ Benchmark salvo em: {out_csv}")
    print(f"Total áudios: {n} | Tempo total: {total_elapsed/60:.1f} min")

    # Sumário simples (médias)
    def mean(vals):
        vals = [float(x) for x in vals]
        return sum(vals)/len(vals) if vals else 0.0

    import statistics
    from collections import defaultdict

    buckets = defaultdict(lambda: {"elapsed": [], "wer": []})
    for eng, _, elapsed, wer in rows:
        try:
            buckets[eng]["elapsed"].append(float(elapsed))
            if wer != "":
                buckets[eng]["wer"].append(float(wer))
        except:
            pass

    print("\n=== MÉDIAS POR ENGINE ===")
    for eng, data in buckets.items():
        m_t = mean(data["elapsed"]) if data["elapsed"] else 0.0
        m_w = mean(data["wer"]) if data["wer"] else None
        if m_w is None:
            print(f"{eng:16s}  elapsed_avg={m_t:.3f}s  wer_avg=–")
        else:
            print(f"{eng:16s}  elapsed_avg={m_t:.3f}s  wer_avg={m_w:.3f}")

if __name__ == "__main__":
    main()
