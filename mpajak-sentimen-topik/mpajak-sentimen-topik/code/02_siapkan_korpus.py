"""Menyaring hasil scraping dan memberi label fase. Keluaran: data/korpus_mpajak.csv (dipakai notebook)."""
import pandas as pd

df = pd.read_csv("mpajak_reviews.csv", parse_dates=["at"])      # hasil 01_scrape_mpajak.py
df["content"] = df["content"].fillna("").str.strip()
df = df[df["content"].str.len() > 3]                              # buang teks <= 3 karakter
batas = [pd.Timestamp(d) for d in ["2024-01-01", "2025-01-01", "2025-04-01", "2026-01-01", "2026-10-01"]]
df["fase"] = pd.cut(df["at"], bins=batas, right=False, labels=[1, 2, 3, 4])   # Okt 2026 dikeluarkan (belum lengkap)
df = df[df["fase"].notna()]
df.to_csv("data/korpus_mpajak.csv", index=False, encoding="utf-8-sig")
print(len(df), "ulasan;", df["fase"].value_counts().sort_index().to_dict())
