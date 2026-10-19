import os, textwrap
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

O = "hasil/"
T = "hasil/tabel/"
G = "hasil/gambar/"
os.makedirs(G, exist_ok=True)
BLUE, ORANGE, INK, MUTE, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTE, "axes.labelcolor": INK,
                     "xtick.color": MUTE, "ytick.color": MUTE, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300})
BATAS = {"Fase 1": (0, 11), "Fase 2": (12, 14), "Fase 3": (15, 23), "Fase 4": (24, 32)}

# ---------- Gambar 1: volume dan persen negatif per bulan (dua panel, satu sumbu x)
b = pd.read_csv(T + "T_bulanan.csv")
x = np.arange(len(b))
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True, gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12})
a1.bar(x, b["n"], color=BLUE, width=0.72)
a1.set_ylabel("Jumlah ulasan"); a1.set_ylim(0, 640)
a2.plot(x, b["neg"], color=ORANGE, lw=2, marker="o", ms=4)
a2.set_ylabel("Ulasan negatif (%)")
a2.set_ylim(70, 100)
for a in (a1, a2):
    a.grid(axis="y", color=GRID, lw=0.6); a.set_axisbelow(True)
    for edge in (11.5, 14.5, 23.5):
        a.axvline(edge, color=MUTE, lw=0.8, ls="--")
for nm, (s, e) in BATAS.items():
    a1.text((s + e) / 2, a1.get_ylim()[1] * 0.97, nm, ha="center", va="top", color=INK, fontsize=8, fontweight="bold")
a2.set_xticks(x[::3]); a2.set_xticklabels([m[2:] for m in b["bulan"]][::3], rotation=45, ha="right")
a2.set_xlabel("Bulan (tahun-bulan)")
fig.savefig(G + "Gambar1_tren_bulanan.png", bbox_inches="tight"); plt.close(fig)

# ---------- Gambar 2: persen negatif per fase, prediksi vs label manual (dengan CI 95%)
p = pd.read_csv(T + "T_sentimen_per_fase.csv"); m = pd.read_csv(T + "T_negatif_label_manual_sampel_acak.csv")
fig, ax = plt.subplots(figsize=(4.6, 3.2))
xs = np.arange(4)
ax.errorbar(xs - 0.07, p["pct_negatif"], yerr=[p["pct_negatif"] - p["ci_lo"], p["ci_hi"] - p["pct_negatif"]], fmt="o-", color=BLUE, lw=2, capsize=3, ms=6, label="Prediksi IndoBERT (n=5.832)")
ax.errorbar(xs + 0.07, m["pct"], yerr=[m["pct"] - m["ci_lo"], m["ci_hi"] - m["pct"]], fmt="s--", color=ORANGE, lw=2, capsize=3, ms=6, label="Label manual, sampel acak (n=793)")
ax.set_xticks(xs); ax.set_xticklabels(["Fase 1\n2024", "Fase 2\nJan–Mar 2025", "Fase 3\nApr–Des 2025", "Fase 4\nJan–Sep 2026"])
ax.set_ylabel("Ulasan negatif (%)"); ax.set_ylim(70, 100)
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower right", fontsize=7.5)
fig.savefig(G + "Gambar2_negatif_per_fase.png", bbox_inches="tight"); plt.close(fig)

# ---------- Gambar 3: tema keluhan per fase (small multiples)
pc = pd.read_csv(T + "T_tema_persen_per_fase.csv", index_col=0).drop("Campuran/tidak terklasifikasi")
fig, axs = plt.subplots(3, 3, figsize=(7.2, 6.0), sharey=False)
for ax, (nm, r) in zip(axs.ravel(), pc.iterrows()):
    v = r.values
    ax.bar(range(4), v, color=BLUE, width=0.62)
    for i, t in enumerate(v):
        ax.text(i, t + max(v) * 0.03, f"{t:.1f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(range(4)); ax.set_xticklabels(["1", "2", "3", "4"])
    ax.set_ylim(0, max(v) * 1.28)
    ax.set_title(textwrap.fill(nm, 30), fontsize=7.6, loc="left", color=INK)
    ax.tick_params(axis="y", labelsize=7)
fig.text(0.5, 0.005, "Fase", ha="center"); fig.text(0.005, 0.5, "Porsi dari ulasan negatif per fase (%)", va="center", rotation=90)
fig.tight_layout(rect=(0.02, 0.02, 1, 1), h_pad=1.4)
fig.savefig(G + "Gambar3_tema_per_fase.png", bbox_inches="tight"); plt.close(fig)

# ---------- Gambar 4: perbandingan model
mo = pd.read_csv(T + "T_perbandingan_model.csv")
labs = ["Negatif", "Netral", "Positif", "Macro-F1"]
cols = ["f1_negatif", "f1_netral", "f1_positif", "macro_f1"]
fig, ax = plt.subplots(figsize=(4.8, 3.2))
w = 0.36
for i, (nm, c) in enumerate(zip(mo["model"], [ORANGE, BLUE])):
    v = mo.loc[i, cols].values.astype(float)
    xb = np.arange(4) + (i - 0.5) * w
    ax.bar(xb, v, width=w - 0.03, color=c, label=nm)
    for xx, t in zip(xb, v):
        ax.text(xx, t + 0.012, f"{t:.2f}", ha="center", fontsize=7.5, color=INK)
ax.set_xticks(range(4)); ax.set_xticklabels(labs); ax.set_ylim(0, 1.08); ax.set_ylabel("F1-score")
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper right", fontsize=8, ncol=2, bbox_to_anchor=(1, 1.1))
fig.savefig(G + "Gambar4_perbandingan_model.png", bbox_inches="tight"); plt.close(fig)
print(os.listdir(G))
