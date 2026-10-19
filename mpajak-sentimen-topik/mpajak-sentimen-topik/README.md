# Analisis Longitudinal Sentimen dan Topik Keluhan Pengguna M-Pajak (2024–2026)

Repositori data dan kode untuk artikel **"Analisis Longitudinal Sentimen dan Topik Keluhan Pengguna Aplikasi M-Pajak Periode 2024–2026: Pendekatan IndoBERT dan BERTopic"**.

Ulasan Google Play aplikasi M-Pajak (`id.go.pajak.djp`) dianalisis dengan IndoBERT (klasifikasi sentimen), TF-IDF + SVM (pembanding), dan BERTopic (topik keluhan), lalu dibandingkan antar empat fase waktu.

## Isi repositori

```
data/        dataset_berlabel_1000.csv   1.000 ulasan berlabel manual (991 dipakai)
code/        01_scrape_mpajak.py         mengambil ulasan dari Google Play
             02_siapkan_korpus.py        menyaring + memberi label fase
             03_baseline_svm.py          SVM, validasi silang 5-lipat
             Notebook_IndoBERT_BERTopic_MPajak.ipynb   IndoBERT + BERTopic (Colab, GPU T4)
             04_analisis_tren_topik.py   uji statistik, tren, tema
             05_gambar.py                Gambar 1–4
labeling/    PEDOMAN_PELABELAN.md        aturan pelabelan
hasil_model/ keluaran notebook (prediksi IndoBERT per ulasan, topik)
hasil/       tabel/ dan gambar/ yang dipakai di artikel
```

## Fase penelitian

| Fase | Periode | n ulasan |
|---|---|---|
| 1 | Jan–Des 2024 | 1.743 |
| 2 | Jan–Mar 2025 (Coretax berlaku 1 Jan 2025) | 1.210 |
| 3 | Apr–Des 2025 | 1.056 |
| 4 | Jan–Sep 2026 | 1.823 |

Total 5.832 ulasan (5.963 hasil scraping tanggal 5 Okt 2026; 18 ulasan Okt 2026 dan 113 ulasan dengan teks ≤3 karakter dikeluarkan).

## Data berlabel

`data/dataset_berlabel_1000.csv`: kolom `ulasan`, `fase`, `score` (bintang, tidak dilihat saat melabeli), `sentimen_final` (positif/netral/negatif), `jenis` (`opini_politik` / `tidak_relevan` / kosong), `dipakai` (True = masuk evaluasi, 991 baris), `sumber` (`acak` = sampel acak ±200 per fase; tambahan = 200 ulasan berbintang 3–5 agar kelas positif/netral cukup), `pelabel`.
Kesepakatan antar-pelabel (silang buta, n=88): Cohen's κ = 0,87.
Kolom `reviewId` sengaja dibuang dari berkas ini; teks ulasan publik, tanpa nama pengguna.

## Cara menjalankan

Jalankan dari folder utama repositori.

```bash
pip install -r requirements.txt
python code/01_scrape_mpajak.py        # hasil: mpajak_reviews.csv (isi akan berbeda jika dijalankan di lain waktu)
python code/02_siapkan_korpus.py       # hasil: data/korpus_mpajak.csv
python code/03_baseline_svm.py         # macro-F1 SVM = 0,709
```

Lalu buka `code/Notebook_IndoBERT_BERTopic_MPajak.ipynb` di Google Colab (Runtime → T4 GPU), unggah `data/dataset_berlabel_1000.csv` dan `data/korpus_mpajak.csv`, jalankan semua sel, unduh `hasil_mpajak.zip`, ekstrak ke folder `hasil_model/`. Kemudian:

```bash
python code/04_analisis_tren_topik.py  # tabel di hasil/tabel/
python code/05_gambar.py               # gambar di hasil/gambar/
```

Untuk **mereplikasi angka artikel tanpa scraping ulang dan tanpa GPU**, `hasil_model/` dan `data/dataset_berlabel_1000.csv` sudah disertakan; langsung jalankan `03`, `04`, `05`.

## Pengaturan model

- **SVM**: TF-IDF unigram+bigram, `min_df=2`, `sublinear_tf`, `LinearSVC(C=1, class_weight="balanced")`.
- **IndoBERT** `indobenchmark/indobert-base-p1`: 4 epoch, lr 2e-5, batch 16, max_len 128, AdamW (wd 0,01), warmup 10%, weighted cross-entropy, seed 42.
- **Evaluasi**: `StratifiedKFold(5, shuffle=True, random_state=42)`, lipatan sama untuk kedua model; macro-F1 utama; bootstrap berpasangan 2.000×.
- **BERTopic**: `paraphrase-multilingual-MiniLM-L12-v2`, UMAP (5 komponen, 15 tetangga), HDBSCAN (`min_cluster_size=20`), `nr_topics=20`, pada ulasan prediksi negatif dengan ≥4 kata (n=4.256); topik dikelompokkan manual menjadi tema.

## Hasil utama

IndoBERT macro-F1 0,820 vs SVM 0,709 (selisih 0,112; CI 95% 0,057–0,168). Porsi ulasan negatif per fase: 84,5 → 87,4 → 85,6 → 89,7% (χ²(3)=23,10; p<0,001; V=0,063).

## Keterbatasan

Scraper `google-play-scraper` tidak resmi; hasil dapat berbeda antar waktu. Ulasan dipilih dari Google Play saja. Pelabelan oleh dua orang. Topik dari BERTopic sensitif terhadap pengaturan; 40,6% ulasan negatif masuk kelompok campuran/outlier.

## Sitasi

Rifky Fakhri Maulana Srg, Bobby Marbun, Muhammad Yoga Khodani, Rizaldi. (2026). *Analisis Longitudinal Sentimen dan Topik Keluhan Pengguna Aplikasi M-Pajak Periode 2024–2026*. GitHub. [URL REPOSITORI]
