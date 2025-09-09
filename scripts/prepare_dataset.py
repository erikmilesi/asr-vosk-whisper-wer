# scripts/prepare_dataset.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import soundfile as sf
import resampy
import click
from datasets import load_dataset
from datasets.utils.logging import get_logger

TARGET_SR = 16000
SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"

def ensure_mono_16k(audio, sr):
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = resampy.resample(audio, sr, TARGET_SR)
    audio = audio.astype(np.float32)
    maxv = max(1e-9, float(np.max(np.abs(audio))))
    return (audio / maxv) * 0.95

def save_pair(out_dir: Path, uid: str, audio_arr: np.ndarray, text: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_dir / f"{uid}.wav"), audio_arr, TARGET_SR)
    (out_dir / f"{uid}.txt").write_text(text.strip(), encoding="utf-8")

@click.command()
@click.option("--n_en", default=3, show_default=True, help="Amostras EN (streaming).")
@click.option("--n_pt", default=3, show_default=True, help="Amostras PT (streaming).")
@click.option(
    "--pt_source",
    type=click.Choice(["auto", "cv17", "cv13", "none"]),
    default="auto",
    show_default=True,
    help="Fonte PT: auto tenta CV17 e cai para CV13; 'none' pula PT."
)
def main(n_en: int, n_pt: int, pt_source: str):
    log = get_logger(__name__)

    # ===== ENGLISH (LibriSpeech) via streaming =====
    click.echo("Baixando LibriSpeech/test-clean (EN) - amostras leves (streaming)...")
    ds_en = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
    out_en = SAMPLES_DIR / "en"
    count = 0
    for ex in ds_en:
        if count >= n_en:
            break
        audio = np.array(ex["audio"]["array"], dtype=np.float32)
        sr = int(ex["audio"]["sampling_rate"])
        audio = ensure_mono_16k(audio, sr)
        text = ex["text"]
        uid = f"en_{count:03d}"
        save_pair(out_en, uid, audio, text)
        count += 1
    click.secho(f"✔ EN pronto em: {out_en}", fg="green")

    # ===== PORTUGUÊS (Common Voice) via streaming =====
    if n_pt <= 0 or pt_source == "none":
        click.secho("Pulado PT (por configuração).", fg="yellow")
        return

    def try_load_cv(version: str):
        return load_dataset(f"mozilla-foundation/common_voice_{version}", "pt",
                            split="test", streaming=True)

    click.echo("Baixando Common Voice (PT) - amostras leves (streaming)...")
    ds_pt = None
    if pt_source in ("auto", "cv17"):
        try:
            ds_pt = try_load_cv("17_0")
        except Exception as e:
            log.warning("Falha no CV 17.0 (provável repo gated). %s", e)

    if ds_pt is None and pt_source in ("auto", "cv13"):
        try:
            ds_pt = try_load_cv("13_0")
        except Exception as e:
            log.error("Falha também no CV 13.0: %s", e)

    if ds_pt is None:
        raise click.ClickException(
            "Não foi possível carregar Common Voice PT. "
            "Opções: (1) rodar 'huggingface-cli login' para CV 17.0; "
            "(2) usar '--pt_source cv13'; "
            "(3) '--pt_source none' para pular PT; "
            "(4) usar um .wav próprio em samples/pt."
        )

    out_pt = SAMPLES_DIR / "pt"
    count = 0
    for ex in ds_pt:
        sent = (ex.get("sentence") or "").strip()
        if not sent:
            continue
        if count >= n_pt:
            break
        audio = np.array(ex["audio"]["array"], dtype=np.float32)
        sr = int(ex["audio"]["sampling_rate"])
        audio = ensure_mono_16k(audio, sr)
        uid = f"pt_{count:03d}"
        save_pair(out_pt, uid, audio, sent)
        count += 1
    click.secho(f"✔ PT pronto em: {out_pt}", fg="green")

    click.secho("Tudo pronto! Os .wav e .txt foram salvos em samples/.", fg="cyan")

if __name__ == "__main__":
    main()
