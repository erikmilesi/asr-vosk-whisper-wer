# scripts/prepare_librispeech_full.py
from __future__ import annotations
from pathlib import Path
import shutil
import numpy as np
import soundfile as sf
import resampy
from datasets import load_dataset
import click

TARGET_SR = 16000
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "samples" / "en"
DEFAULT_OUT.mkdir(parents=True, exist_ok=True)

def ensure_mono_16k(audio, sr):
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = resampy.resample(audio, sr, TARGET_SR)
    audio = audio.astype(np.float32)
    maxv = max(1e-9, float(np.max(np.abs(audio))))
    return (audio / maxv) * 0.95

@click.command()
@click.option("--split",
              type=click.Choice(["test", "dev", "train"]),
              default="test", show_default=True,
              help="Split do LibriSpeech clean.")
@click.option("--max_items", type=int, default=None,
              help="Limitar quantidade de ARQUIVOS GERADOS (novos ou sobrescritos). Omitir para baixar tudo.")
@click.option("--out_dir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
              default=DEFAULT_OUT, show_default=True,
              help="Diretório de saída para WAV+TXT.")
@click.option("--overwrite",
              type=click.Choice(["skip", "force"]),
              default="skip", show_default=True,
              help="Comportamento quando WAV/TXT já existem: 'skip' mantém, 'force' sobrescreve.")
@click.option("--reset", is_flag=True, default=False,
              help="Apaga todos os .wav/.txt do out_dir ANTES de começar.")
def main(split: str, max_items: int | None, out_dir: Path, overwrite: str, reset: bool):
    """
    Baixa TODO o LibriSpeech (clean, split escolhido) via streaming e salva WAV+TXT em 'out_dir'.
    Retomável; com --overwrite force você regrava arquivos existentes.
    Com --reset você limpa o out_dir antes de iniciar.
    O limite (--max_items) passa a contar APENAS arquivos gerados agora (novos OU sobrescritos).
    """
    if reset:
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in list(out_dir.glob("*.wav")) + list(out_dir.glob("*.txt")):
            try:
                p.unlink()
            except Exception:
                pass
        click.secho(f"[reset] Limpei arquivos antigos em: {out_dir}", fg="yellow")

    print(f"Carregando LibriSpeech clean [{split}] em streaming.")
    ds = load_dataset("librispeech_asr", "clean", split=split, streaming=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0  # conta quantos ARQUIVOS (pares) foram gerados nesta execução

    for ex in ds:
        uid = ex.get("id")  # id único do HF
        if not uid:
            # fallback estável (não recomendado, mas garante um nome)
            uid = f"{split}_{saved:06d}"

        wav_path = out_dir / f"{uid}.wav"
        txt_path = out_dir / f"{uid}.txt"

        exists = wav_path.exists() and txt_path.exists()

        if exists and overwrite == "skip":
            # Pula sem contar para o limite; queremos limitar APENAS o que for gerado agora.
            continue

        # Extrai e normaliza
        arr = np.array(ex["audio"]["array"], dtype=np.float32)
        sr = int(ex["audio"]["sampling_rate"])
        arr = ensure_mono_16k(arr, sr)

        # Salva (sobrescrevendo se necessário)
        sf.write(str(wav_path), arr, TARGET_SR)
        txt_path.write_text(ex["text"].strip(), encoding="utf-8")

        saved += 1
        if max_items and saved >= max_items:
            break

    print(f"✔ Gerados {saved} pares WAV+TXT em: {out_dir}")
    if max_items:
        print(f"(Limite solicitado: {max_items}; overwrite={overwrite}; reset={reset})")

if __name__ == "__main__":
    main()
