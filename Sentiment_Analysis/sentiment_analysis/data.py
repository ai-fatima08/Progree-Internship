import re
import urllib.request
from pathlib import Path

import nltk
import pandas as pd
from sklearn.model_selection import train_test_split

for _p in ["sentence_polarity", "subjectivity"]:
    nltk.download(_p, quiet=True)
from nltk.corpus import sentence_polarity, subjectivity

SEED = 42
TE_DIR = Path(__file__).parent / "extra_data" / "tweeteval"
TE_URL = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/train_text.txt"
TE_LABELS = {"0": "negative", "1": "neutral", "2": "positive"}


def _detok(tokens):
    s = " ".join(tokens)
    s = re.sub(r"\s+([.,!?;:%)])", r"\1", s)
    s = re.sub(r"\(\s+", "(", s)
    return s.replace(" n't", "n't").replace(" 's", "'s")


def load_airline(path):
    df = (
        pd.read_csv(path)[["airline_sentiment", "text"]]
        .dropna()
        .drop_duplicates("text")
    )
    df.columns = ["label", "text"]
    df["domain"] = "airline"
    return df


def load_movie():
    rows = []
    for cat, lab in [("pos", "positive"), ("neg", "negative")]:
        rows += [
            (_detok(s), lab, "movie") for s in sentence_polarity.sents(categories=cat)
        ]
    rows += [
        (_detok(s), "neutral", "movie") for s in subjectivity.sents(categories="obj")
    ]
    return pd.DataFrame(rows, columns=["text", "label", "domain"]).drop_duplicates(
        "text"
    )[["label", "text", "domain"]]


def _te_part(part):
    TE_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for kind in ("text", "labels"):
        f = TE_DIR / f"{part}_{kind}.txt"
        if not f.exists():
            urllib.request.urlretrieve(TE_URL.format(f"{part}_{kind}"), f)
        out[kind] = f.read_text(encoding="utf-8").split("\n")
    n = min(len(out["text"]), len(out["labels"]))
    df = pd.DataFrame(
        {
            "text": out["text"][:n],
            "label": [TE_LABELS.get(l.strip()) for l in out["labels"][:n]],
        }
    )
    df = df.dropna().drop_duplicates("text")
    df["domain"] = "tweeteval"
    return df[["label", "text", "domain"]]


def load_tweeteval(max_train=15000, max_test=3000):
    """Official train/val/test files, sub-sampled (stratified) to keep training fast."""
    tr, va, te = _te_part("train"), _te_part("val"), _te_part("test")
    sub = lambda d, n: (
        d
        if len(d) <= n
        else train_test_split(d, train_size=n, stratify=d.label, random_state=SEED)[0]
    )
    return sub(tr, max_train), va, sub(te, max_test)


def build_splits(airline_csv):
    """Returns train, val, test (each has columns label / text / domain).
    airline + movie: 80% train / 20% test stratified, validation = 15% of train.
    tweeteval: its own official train / val / test files."""
    tr, va, te = [], [], []
    for d, g in pd.concat(
        [load_airline(airline_csv), load_movie()], ignore_index=True
    ).groupby("domain"):
        a, b = train_test_split(g, test_size=0.2, stratify=g.label, random_state=SEED)
        a, v = train_test_split(a, test_size=0.15, stratify=a.label, random_state=SEED)
        tr.append(a)
        va.append(v)
        te.append(b)
    t_tr, t_va, t_te = load_tweeteval()
    tr.append(t_tr)
    va.append(t_va)
    te.append(t_te)
    return pd.concat(tr), pd.concat(va), pd.concat(te)
