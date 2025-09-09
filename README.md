# Vosk vs Whisper — Benchmark com WER (Portfólio do Projeto de Inteligência de Comunicações)

Este repositório demonstra, de forma **aberta e reprodutível**, a comparação entre os motores de ASR **Vosk** e **Whisper** usando a métrica **WER (Word Error Rate)** em amostras públicas.  
Ele faz parte do meu storytelling técnico sobre o desenvolvimento de um **sistema de Inteligência de Comunicações** composto por: **redução de ruído → transcrição de fala → detecção de palavras‑chave**.

> ⚠️ **Nota importante:** O **core proprietário** do meu pipeline (filtros de ruído, KWS avançado, otimizações) não está neste repositório por razões de **patente**. Aqui você encontra apenas o **benchmark de transcrição** com dados públicos.

---

## 🧱 Estrutura
```text
asr-vosk-whisper-wer/
├─ README.md
├─ requirements.txt
├─ scripts/
│  ├─ get_models.py
│  ├─ prepare_dataset.py
│  ├─ prepare_librispeech_full.py
│  ├─ benchmark_dataset_en.py
│  └─ show_stats.py
├─ models/               # (baixados via get_models.py)
├─ samples/
│  ├─ en/                # (LibriSpeech)
└─ results/
   └─ benchmark_en.csv   # (gerado)
```

---

## ⚙️ 1) Ambiente
```bash
conda create -n asr python=3.10 -y
conda activate asr
pip install -r requirements.txt
```

- As dependências principais incluem `vosk`, `openai-whisper`, `torch`, `jiwer`, `datasets`, `librosa`, `pandas` e `matplotlib`. Veja `requirements.txt` para a lista completa.

## ⬇️ 2) Baixar modelos Vosk
```bash
python scripts/get_models.py --lang en
```
- Os scripts criam pastas em `./models/` (ex.: `models/vosk-en`, `models/vosk-pt`).  
- Se o link padrão mudar, use `--url` para informar manualmente a URL do modelo Vosk.

## 🎧 3) Preparar dataset de teste leve (EN)
```bash
python scripts\prepare_librispeech_full.py --split test --max_items 50
```
- Baixa poucas amostras por **streaming**: `LibriSpeech` (EN).
- Gera até 50 pares WAV+TXT (normalizados para 16 kHz mono) em `samples/en`.
- Parâmetros úteis: `--overwrite skip|force`, `--reset` para limpar saídas.

## 🧪 4) Rodar benchmark (EN)
```bash
python scripts/benchmark_dataset_en.py
```
- Gera `results/benchmark_en.csv` com **engine, arquivo, tempo (s), WER**.

## 📊 5) Estatísticas rápidas
```bash
python scripts/show_stats.py
```
Saída esperada (exemplo):
```text
=== Estatísticas de Benchmark ===

→ Amostras por engine:
whisper-base    50
vosk            50

→ Tempo médio por engine (s):
engine
vosk            0.221
whisper-base    0.974

→ WER médio por engine:
engine
vosk            0.24
whisper-base    0.11
```

---

## 🧭 Roadmap (o que vem por aí)
- Comparativo PT‑BR (Common Voice) e análise de sotaques regionais;
- Trade‑off **latência × precisão** e fator de tempo real (RTF);
- Integração de **redução de ruído** antes da transcrição;
- Detecção de **palavras‑chave** (KWS) em pipeline.

---

## 🤝 Contribuições
- *Pull requests* são bem‑vindos para melhorar reprodutibilidade e benchmark de WER.
- O pipeline proprietário (ruído/KWS) não será aberto nesta fase.

## 📄 Licença
- Sem licença aberta neste momento. Conteúdo disponibilizado apenas para fins de **demonstração de benchmark**.
