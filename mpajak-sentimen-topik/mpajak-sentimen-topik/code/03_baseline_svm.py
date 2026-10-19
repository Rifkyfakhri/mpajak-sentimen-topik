"""Baseline TF-IDF + LinearSVC, validasi silang berstrata 5-lipat pada 991 ulasan berlabel."""
import re
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score, confusion_matrix


def bersih(t):
    t = str(t).lower()
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    return re.sub(r"\s+", " ", t).strip()


d = pd.read_csv("data/dataset_berlabel_1000.csv")
d = d[d["dipakai"]].reset_index(drop=True)
x, y = d["ulasan"].map(bersih), d["sentimen_final"].values
oof = np.empty(len(d), dtype=object)
for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(x, y):
    m = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), LinearSVC(C=1.0, class_weight="balanced"))
    oof[te] = m.fit(x.iloc[tr], y[tr]).predict(x.iloc[te])
lab = ["negatif", "netral", "positif"]
print(classification_report(y, oof, labels=lab, digits=3))
print("macro-F1:", round(f1_score(y, oof, average="macro"), 3))
ac = (d["sumber"] == "acak").values
print("macro-F1 sampel acak:", round(f1_score(y[ac], oof[ac], average="macro"), 3))
print(pd.DataFrame(confusion_matrix(y, oof, labels=lab), index=lab, columns=lab))
