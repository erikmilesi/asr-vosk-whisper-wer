# scripts/show_stats.py
import pandas as pd
from pathlib import Path

csv_path = Path("results/benchmark_en.csv")
df = pd.read_csv(csv_path)

# Converte colunas numéricas
df["elapsed_s"] = pd.to_numeric(df["elapsed_s"], errors="coerce")
df["wer"] = pd.to_numeric(df["wer"], errors="coerce")

print("=== Estatísticas de Benchmark ===\n")

print("→ Amostras por engine:")
print(df["engine"].value_counts(), "\n")

print("→ Tempo médio por engine (s):")
print(df.groupby("engine")["elapsed_s"].mean(), "\n")

print("→ WER médio por engine:")
df_w = df.dropna(subset=["wer"])
print(df_w.groupby("engine")["wer"].mean(), "\n")

print("→ WER por engine (estatísticas detalhadas):")
print(df_w.groupby("engine")["wer"].describe(), "\n")
