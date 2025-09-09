# scripts/get_models.py
from __future__ import annotations
import os
import sys
import re
import shutil
import tarfile
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, Optional
import urllib.request
import click

"""
Script extremamente comentado para baixar e preparar modelos Vosk.

Como usar:
  python scripts/get_models.py --lang pt
  python scripts/get_models.py --lang en
  python scripts/get_models.py --url https://.../vosk-model-small-en-us-0.15.zip

Saída:
  Cria a pasta ./models/<apelido> com o conteúdo extraído do modelo.
  Exemplos de apelido: vosk-pt, vosk-en, ou nome derivado do arquivo.

ATENÇÃO:
- URLs podem mudar ao longo do tempo. Caso um link dê erro 404,
  atualize o dicionário DEFAULT_URLS abaixo com um link válido.

- Você pode consultar a lista de modelos em:
  https://alphacephei.com/vosk/models  (site oficial Vosk)
"""

# Diretório base dos modelos
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Catálogo de URLs padrão (ajuste se desejar versões específicas/superiores)
DEFAULT_URLS: Dict[str, str] = {
    # Inglês (modelo pequeno, bom para demos rápidas)
    "en": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
    # Português (pt-BR) – exemplo comum (pode haver modelos alternativos)
    "pt": "https://alphacephei.com/vosk/models/vosk-model-small-pt-0.3.zip"
}

def _download(url: str, dst: Path) -> Path:
    """
    Baixa um arquivo da internet para 'dst'. Retorna o caminho baixado.
    Usa urllib.request por simplicidade (sem dependências extras).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    click.echo(f"Baixando: {url}")
    with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f)
    click.echo(f"OK: {dst}")
    return dst

def _safe_extract_zip(src_zip: Path, dst_dir: Path) -> Path:
    """
    Extrai um ZIP com segurança para 'dst_dir'.
    Retorna a pasta raiz extraída (se houver um único diretório) ou 'dst_dir'.
    """
    with zipfile.ZipFile(src_zip, "r") as zf:
        zf.extractall(dst_dir)
    # Se o zip contiver uma pasta raiz única, usamos ela como retorno
    entries = list(dst_dir.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return dst_dir

def _safe_extract_tar(src_tar: Path, dst_dir: Path) -> Path:
    """
    Extrai um TAR/TAR.GZ com segurança para 'dst_dir'.
    Retorna a pasta raiz extraída (se houver um único diretório) ou 'dst_dir'.
    """
    with tarfile.open(src_tar, "r:*") as tf:
        tf.extractall(dst_dir)
    entries = list(dst_dir.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return dst_dir

def _derive_nickname_from_filename(filename: str) -> str:
    """
    Gera um apelido curto a partir do nome do arquivo, ex.:
      'vosk-model-small-en-us-0.15.zip' -> 'vosk-en'
      'vosk-model-small-pt-0.3.zip'     -> 'vosk-pt'
    """
    fn = filename.lower()
    if "pt" in fn:
        return "vosk-pt"
    if "en" in fn:
        return "vosk-en"
    # fallback genérico
    base = re.sub(r"\.(zip|tar\.gz|tgz)$", "", filename, flags=re.IGNORECASE)
    return base

def _validate_vosk_folder(path: Path) -> bool:
    """
    Checagem simples: um modelo Vosk costuma ter subpastas como 'am', 'conf', etc.
    Não é exaustivo, mas ajuda a detectar extrações erradas.
    """
    must_have = ["conf"]
    present = [p.name for p in path.iterdir() if p.is_dir()]
    return all(x in present for x in must_have)

@click.command()
@click.option("--lang", type=click.Choice(["en", "pt"]), default=None,
              help="Idioma alvo para baixar o modelo Vosk a partir do catálogo padrão.")
@click.option("--url", type=str, default=None,
              help="URL direta para um modelo Vosk (.zip ou .tar.gz). Sobrescreve --lang.")
@click.option("--nickname", type=str, default=None,
              help="Apelido da pasta destino em ./models (padrão: auto a partir do arquivo).")
def main(lang: Optional[str], url: Optional[str], nickname: Optional[str]):
    """
    Baixa e extrai um modelo Vosk.

    Exemplos:
      python scripts/get_models.py --lang pt
      python scripts/get_models.py --lang en
      python scripts/get_models.py --url https://.../vosk-model-small-en-us-0.15.zip --nickname vosk-en
    """
    if url is None:
        if lang is None:
            raise click.UsageError("Informe --lang (pt/en) OU --url <link direto>")
        if lang not in DEFAULT_URLS:
            raise click.UsageError(f"Idioma '{lang}' sem URL padrão configurada.")
        url = DEFAULT_URLS[lang]

    # Nome do arquivo local
    filename = url.split("/")[-1]
    tmp_dir = Path(tempfile.mkdtemp(prefix="vosk_dl_"))
    archive_path = tmp_dir / filename

    try:
        _download(url, archive_path)

        # Pasta temporária para extração
        work_dir = tmp_dir / "extract"
        work_dir.mkdir(parents=True, exist_ok=True)

        # Detecta extensão e extrai
        if filename.lower().endswith(".zip"):
            root = _safe_extract_zip(archive_path, work_dir)
        elif filename.lower().endswith((".tar.gz", ".tgz", ".tar")):
            root = _safe_extract_tar(archive_path, work_dir)
        else:
            raise click.ClickException("Formato não suportado. Use .zip ou .tar.gz")

        # Determina apelido (nome final da pasta) se não informado
        if nickname is None:
            nickname = _derive_nickname_from_filename(filename)

        final_dir = MODELS_DIR / nickname
        if final_dir.exists():
            click.echo(f"Removendo destino antigo: {final_dir}")
            shutil.rmtree(final_dir)

        # Move pasta extraída para ./models/<nickname>
        shutil.move(str(root), str(final_dir))

        # Validação simples de estrutura
        if not _validate_vosk_folder(final_dir):
            click.echo("AVISO: Não encontrei estrutura típica do Vosk (ex.: 'conf/'). Verifique o modelo baixado.")

        click.secho(f"✔ Modelo pronto em: {final_dir}", fg="green")
        click.echo("Use este caminho com --vosk_model_path nos scripts de transcrição.")

    finally:
        # Limpeza de temporários
        shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()

