"""
Scraping ulasan Google Play Store untuk aplikasi M-Pajak (DJP).
Package ID: id.go.pajak.djp

Cara pakai (Google Colab atau lokal):
    pip install google-play-scraper pandas
    python scrape_mpajak.py --since 2024-01-01

Output:
    mpajak_reviews.csv   -> semua ulasan (tanpa nama pengguna, untuk privasi)
    mpajak_per_bulan.csv -> jumlah ulasan, jumlah yang berteks, dan rata-rata rating per bulan
"""
import argparse
import time

import pandas as pd
from google_play_scraper import Sort, reviews

APP_ID = "id.go.pajak.djp"


def scrape(since: pd.Timestamp, batch_size: int = 200, pause: float = 1.0) -> pd.DataFrame:
    rows, seen, token = [], set(), None
    while True:
        batch, token = reviews(
            APP_ID,
            lang="id",
            country="id",
            sort=Sort.NEWEST,
            count=batch_size,
            continuation_token=token,
        )
        if not batch:
            break

        for r in batch:
            if r["reviewId"] in seen:
                continue
            seen.add(r["reviewId"])
            rows.append(
                {
                    "reviewId": r["reviewId"],
                    "at": r["at"],
                    "score": r["score"],
                    "content": r["content"],
                    "thumbsUpCount": r["thumbsUpCount"],
                    "appVersion": r.get("appVersion"),
                    "replyContent": r.get("replyContent"),
                }
            )

        oldest = min(r["at"] for r in batch)
        print(f"Terkumpul {len(rows)} ulasan, terlama di batch ini: {oldest:%Y-%m-%d}")
        if pd.Timestamp(oldest) < since or token is None:
            break
        time.sleep(pause)

    df = pd.DataFrame(rows)
    df["at"] = pd.to_datetime(df["at"])
    return df[df["at"] >= since].sort_values("at").reset_index(drop=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bulan"] = df["at"].dt.to_period("M")
    df["ada_teks"] = df["content"].fillna("").str.strip().str.len() > 0
    out = df.groupby("bulan").agg(
        jumlah_ulasan=("reviewId", "count"),
        jumlah_berteks=("ada_teks", "sum"),
        rata_rata_rating=("score", "mean"),
    )
    out["rata_rata_rating"] = out["rata_rata_rating"].round(2)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", default="2024-01-01", help="tanggal mulai (YYYY-MM-DD)")
    args = parser.parse_args()

    data = scrape(pd.Timestamp(args.since))
    data.to_csv("mpajak_reviews.csv", index=False, encoding="utf-8-sig")

    ringkasan = summarize(data)
    ringkasan.to_csv("mpajak_per_bulan.csv", encoding="utf-8-sig")
    print(ringkasan.to_string())
    print(f"\nTotal: {len(data)} ulasan sejak {args.since}")
