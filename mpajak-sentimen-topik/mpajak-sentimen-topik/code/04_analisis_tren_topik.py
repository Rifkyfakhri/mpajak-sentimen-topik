import json, re, itertools
import numpy as np, pandas as pd
import scipy.stats as st
from sklearn.metrics import f1_score, classification_report

H = "hasil_model/"      # isi folder dari hasil_mpajak.zip (keluaran notebook)
O = "data/"
T = "hasil/tabel/"
import os
os.makedirs(T, exist_ok=True)
R = {}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


k = pd.read_csv(H + "korpus_hasil.csv", parse_dates=["at"])
LAB = ["negatif", "netral", "positif"]

# ---------- 1. tren per fase (prediksi IndoBERT, seluruh korpus)
tab = pd.crosstab(k["fase"], k["pred_indobert"])[LAB]
tab["n"] = tab.sum(1)
rows = []
for f, r in tab.iterrows():
    lo, hi = wilson(r["negatif"], r["n"])
    rows.append(dict(fase=f, n=int(r["n"]), negatif=int(r["negatif"]), netral=int(r["netral"]), positif=int(r["positif"]),
                     pct_negatif=100 * r["negatif"] / r["n"], ci_lo=100 * lo, ci_hi=100 * hi,
                     pct_netral=100 * r["netral"] / r["n"], pct_positif=100 * r["positif"] / r["n"]))
fase_df = pd.DataFrame(rows)
fase_df.to_csv(T + "T_sentimen_per_fase.csv", index=False)
print(fase_df.round(1).to_string())
chi3 = st.chi2_contingency(tab[LAB].values)
chi2b = st.chi2_contingency(np.c_[tab["negatif"], tab["n"] - tab["negatif"]])
N = tab["n"].sum()
V = np.sqrt(chi2b[0] / N)
R["chi_3kelas"] = dict(chi2=chi3[0], dof=chi3[2], p=chi3[1], V=float(np.sqrt(chi3[0] / (N * 2))))
R["chi_biner"] = dict(chi2=chi2b[0], dof=chi2b[2], p=chi2b[1], V=float(V))


def cochran_armitage(neg, n):
    x = np.arange(1, len(n) + 1, dtype=float)
    N = n.sum(); p = neg.sum() / N; xb = (n * x).sum() / N
    z = (neg * (x - xb)).sum() / np.sqrt(p * (1 - p) * (n * (x - xb) ** 2).sum())
    return z, 2 * (1 - st.norm.cdf(abs(z)))


z, p = cochran_armitage(tab["negatif"].values.astype(float), tab["n"].values.astype(float))
R["tren_CA"] = dict(z=z, p=p)
pw = []
for a, b in itertools.combinations([1, 2, 3, 4], 2):
    t = np.array([[tab.loc[a, "negatif"], tab.loc[a, "n"] - tab.loc[a, "negatif"]],
                  [tab.loc[b, "negatif"], tab.loc[b, "n"] - tab.loc[b, "negatif"]]])
    c = st.chi2_contingency(t, correction=False)
    pw.append(dict(a=a, b=b, selisih_pp=100 * (t[1, 0] / t[1].sum() - t[0, 0] / t[0].sum()), chi2=c[0], p=c[1], p_bonf=min(1, c[1] * 6)))
pw = pd.DataFrame(pw); pw.to_csv(T + "T_uji_pasangan_fase.csv", index=False)
print(pw.round(4).to_string())
print({kk: {a: round(float(b), 5) for a, b in v.items()} for kk, v in R.items()})

# ---------- 2. pemeriksaan silang: label manual sampel acak
g = pd.read_csv(H + "berlabel_oof.csv")
ac = g[g["sumber"] == "acak"]
gt = ac.groupby("fase")["sentimen_final"].agg(n="size", neg=lambda s: (s == "negatif").sum()).reset_index()
gt["pct"] = 100 * gt["neg"] / gt["n"]
gt["ci_lo"] = [100 * wilson(a, b)[0] for a, b in zip(gt["neg"], gt["n"])]
gt["ci_hi"] = [100 * wilson(a, b)[1] for a, b in zip(gt["neg"], gt["n"])]
gt.to_csv(T + "T_negatif_label_manual_sampel_acak.csv", index=False)
print(gt.round(1).to_string())
cg = st.chi2_contingency(np.c_[gt["neg"], gt["n"] - gt["neg"]])
R["chi_manual"] = dict(chi2=cg[0], dof=cg[2], p=cg[1])
z2, p2 = cochran_armitage(gt["neg"].values.astype(float), gt["n"].values.astype(float))
R["tren_CA_manual"] = dict(z=z2, p=p2)
print(R["chi_manual"], R["tren_CA_manual"])

# ---------- 3. perbandingan model
import re as _re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
def _bersih(t):
    t = str(t).lower(); t = _re.sub(r"http\S+", " ", t); t = _re.sub(r"[^a-z\s]", " ", t)
    t = _re.sub(r"(.)\1{2,}", r"\1\1", t); return _re.sub(r"\s+", " ", t).strip()
_d = pd.read_csv(O + "dataset_berlabel_1000.csv"); _d = _d[_d["dipakai"]].reset_index(drop=True)
_x = _d["ulasan"].map(_bersih); _y = _d["sentimen_final"].values
_o = np.empty(len(_d), dtype=object)
for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(_x, _y):
    _o[te] = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), LinearSVC(C=1.0, class_weight="balanced")).fit(_x.iloc[tr], _y[tr]).predict(_x.iloc[te])
svm = _d[["no", "fase", "sumber", "sentimen_final"]].assign(pred_svm=_o)
assert (svm["no"].values == g["no"].values).all()
rows = []
for nama, yt, yp in [("SVM + TF-IDF", svm["sentimen_final"], svm["pred_svm"]), ("IndoBERT", g["sentimen_final"], g["pred_indobert_oof"])]:
    f = f1_score(yt, yp, labels=LAB, average=None)
    rows.append(dict(model=nama, f1_negatif=f[0], f1_netral=f[1], f1_positif=f[2], macro_f1=f.mean(), akurasi=(yt == yp).mean()))
    sub = (g if nama == "IndoBERT" else svm)
    sub = sub[sub["sumber"] == "acak"]
    yp2 = sub["pred_indobert_oof"] if nama == "IndoBERT" else sub["pred_svm"]
    rows[-1]["macro_f1_acak"] = f1_score(sub["sentimen_final"], yp2, labels=LAB, average="macro")
mod = pd.DataFrame(rows); mod.to_csv(T + "T_perbandingan_model.csv", index=False)
print(mod.round(3).to_string())
# bootstrap beda macro-F1 (pasangan)
mm = svm.merge(g[["no", "pred_indobert_oof"]], on="no")
rng = np.random.default_rng(42); d = []
for _ in range(2000):
    i = rng.integers(0, len(mm), len(mm)); s = mm.iloc[i]
    d.append(f1_score(s["sentimen_final"], s["pred_indobert_oof"], average="macro") - f1_score(s["sentimen_final"], s["pred_svm"], average="macro"))
R["selisih_macroF1_boot"] = dict(mean=float(np.mean(d)), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)))
print(R["selisih_macroF1_boot"])

# ---------- 4. tema topik
TEMA = {
    "Verifikasi kode OTP, email, dan nomor HP": [1, 12],
    "Login dan kata sandi": [4, 16],
    "Pendaftaran dan aktivasi (NPWP, EFIN, akun)": [3, 5, 15],
    "Verifikasi wajah dan unggah foto": [14],
    "Perubahan NPWP 15 ke 16 digit": [8],
    "Error dan kegagalan sistem (aplikasi/web)": [9, 11, 17],
    "Perintah pembaruan dan pemasangan": [7, 18],
    "Live chat dan koneksi": [10],
    "Kritik umum terhadap pemerintah": [2, 6, 13],
    "Campuran/tidak terklasifikasi": [0, -1],
}
t2t = {t: n for n, ts in TEMA.items() for t in ts}
n = pd.read_csv(H + "negatif_topik.csv")
n["tema"] = n["topik"].map(t2t)
assert n["tema"].notna().all()
ct = pd.crosstab(n["tema"], n["fase"]).reindex(list(TEMA))
pct = (ct / ct.sum() * 100)
ct.to_csv(T + "T_tema_jumlah_per_fase.csv"); pct.round(2).to_csv(T + "T_tema_persen_per_fase.csv")
tot = ct.sum(1); tot_pct = 100 * tot / tot.sum()
print(pd.concat([tot, tot_pct.round(1), pct.round(1)], axis=1).to_string())
ctx = ct.drop("Campuran/tidak terklasifikasi")
c = st.chi2_contingency(ctx.values)
R["chi_tema_fase"] = dict(chi2=c[0], dof=c[2], p=c[1], V=float(np.sqrt(c[0] / (ctx.values.sum() * (min(ctx.shape) - 1)))), n=int(ctx.values.sum()))
res = pd.DataFrame((ctx.values - c[3]) / np.sqrt(c[3]), index=ctx.index, columns=ctx.columns).round(1)
res.to_csv(T + "T_tema_residual_terstandar.csv"); print(res.to_string()); print(R["chi_tema_fase"])
R["n_negatif_topik"] = int(len(n)); R["n_tak_terklasifikasi"] = int((n["topik"] == -1).sum())

# ---------- 5. validasi kamus kata kunci (tanpa BERTopic)
neg = k[k["pred_indobert"] == "negatif"].copy(); neg["t"] = neg["content"].fillna("").str.lower()
RX = {
    "OTP/verifikasi kode": r"\botp\b|kode verif|verifikasi|verif\b|kode.{0,20}(email|sms)",
    "login/kata sandi": r"login|log in|\blogin|password|kata sandi|\bsandi",
    "NPWP 15/16 digit": r"1[56] ?(digit|angka)|npwp 16",
    "update/pembaruan": r"update|perbarui|pembaruan",
    "wajah/foto": r"wajah|foto|selfie",
    "live chat": r"live ?chat|\bchat\b",
    "error/gagal": r"error|eror|erorr|\bbug\b|crash|force close|gagal",
    "kritik pemerintah": r"pemerintah|negara|anggaran|pejabat|rakyat|korupsi|\bpajak.{0,10}(untuk|buat) apa",
}
kw = {}
for nm, rx in RX.items():
    neg[nm] = neg["t"].str.contains(rx, regex=True)
    kw[nm] = neg.groupby("fase")[nm].mean() * 100
kw = pd.DataFrame(kw).T
kw.round(1).to_csv(T + "T_validasi_kamus_kata_kunci.csv"); print(kw.round(1).to_string())

# ---------- 6. volume bulanan dan musim SPT
neg_all = k.copy(); neg_all["bulan"] = neg_all["at"].dt.to_period("M").astype(str)
vol = neg_all.groupby("bulan").agg(n=("reviewId", "size"), neg=("pred_indobert", lambda s: (s == "negatif").mean() * 100)).round(1)
vol.to_csv(T + "T_bulanan.csv")
R["rata_bulanan_per_fase"] = (k.groupby("fase").size() / np.array([12, 3, 9, 9])).round(1).to_dict()
print(R["rata_bulanan_per_fase"])
R["n_korpus"] = int(len(k)); R["pred"] = k["pred_indobert"].value_counts().to_dict()

# versi aplikasi
print(k["appVersion"].value_counts().head(5))
json.dump(R, open(T + "ringkasan_angka.json", "w"), indent=1, default=float)

# ---------- 7. musim SPT: Januari-Maret vs sisa tahun, per tahun
k["tahun"] = k["at"].dt.year; k["spt"] = k["at"].dt.month <= 3
sp = k.groupby(["tahun", "spt"]).agg(n=("reviewId", "size"), pct_neg=("pred_indobert", lambda s: (s == "negatif").mean() * 100)).round(1)
sp.to_csv(T + "T_musim_SPT.csv"); print(sp.to_string())
