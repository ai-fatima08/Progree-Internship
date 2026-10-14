"""
Ablation (every step is trained and tested the same way):
  M0  airline tweets only, TF-IDF + LR                       <- the first version
  M1  + extra domains (movie sentences, general tweets)
  M2  + negation marking ("not bad" != "bad")
  M3  + VADER lexicon & style features
  M4  + character n-grams + extra weight for the neutral class   <- FINAL model
"""

import argparse, itertools, json
import joblib, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from challenge import CHALLENGE
from data import build_splits
from features import VaderStyle, clean_many_neg, clean_many_plain, clean_raw_many

LABELS = ["negative", "neutral", "positive"]
DOMAINS = {
    "airline": "Airline tweets (3 classes)",
    "movie": "Movie sentences (3 classes, neutral = objective plot sentences)",
    "tweeteval": "General tweets (3 classes, human-labelled neutral)",
}

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="Tweets.csv")
args = ap.parse_args()


def build(clean_fn, C=1.0, vader_w=None, char=False, neutral_mult=1.0, y=None):
    """y (training labels) is only needed to compute the class weights when neutral_mult != 1."""
    tfidf = Pipeline(
        [
            ("clean", FunctionTransformer(clean_fn)),
            ("vec", TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2))),
        ]
    )
    blocks = [("tfidf", tfidf)]
    if vader_w:
        blocks.append(("vader", VaderStyle(weight=vader_w)))
    if (
        char
    ):  # character n-grams on lightly cleaned RAW text (catches '?', '!', typos, word forms)
        blocks.append(
            (
                "char",
                Pipeline(
                    [
                        ("raw", FunctionTransformer(clean_raw_many)),
                        (
                            "vec",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(2, 5),
                                sublinear_tf=True,
                                min_df=3,
                                max_features=250000,
                            ),
                        ),
                    ]
                ),
            )
        )
    cw = "balanced"
    if neutral_mult != 1.0:
        base = len(y) / (3 * pd.Series(y).value_counts())
        cw = {k: float(v) for k, v in base.items()}
        cw["neutral"] *= neutral_mult
    return Pipeline(
        [
            ("features", FeatureUnion(blocks)),
            ("clf", LogisticRegression(C=C, max_iter=3000, class_weight=cw)),
        ]
    )


def mf1(y, p):
    return f1_score(y, p, labels=sorted(set(y)), average="macro", zero_division=0)


def domain_scores(model, test):
    out = {}
    for d, g in test.groupby("domain"):
        p = model.predict(g.text)
        out[d] = {
            "n": len(g),
            "macro_f1": mf1(g.label, p),
            "accuracy": accuracy_score(g.label, p),
            "neutral_f1": (
                f1_score(g.label, p, labels=["neutral"], average=None, zero_division=0)[
                    0
                ]
                if "neutral" in set(g.label)
                else None
            ),
        }
    return out


# 1. data 
train, val, test = build_splits(args.data)
print(
    "TRAIN examples per domain / class:\n"
    + train.groupby(["domain", "label"]).size().to_string(),
    "\n",
)
print(f"train {len(train)}  val {len(val)}  test {len(test)}\n")
ch_X, ch_y = [t for t, _ in CHALLENGE], [l for _, l in CHALLENGE]

# 2. choose hyper-parameters for the final model on the VALIDATION set only 
print(
    "Hyper-parameter search for M4 (score = mean macro-F1 over the 3 domains, on validation data)"
)
best, best_s = None, -1
for nm in [1.0, 1.3]:
    m = build(clean_many_neg, 1.0, 1.0, True, nm, train.label).fit(
        train.text, train.label
    )
    s_ = np.mean([v["macro_f1"] for v in domain_scores(m, val).values()])
    print(f"  neutral_weight x{nm}   val score = {s_:.4f}")
    if s_ > best_s:
        best, best_s = nm, s_
NM = best
print(f"-> chosen: neutral weight x{NM}\n")

# 3. ablation: train every variant on train+val, test on the untouched test set 
trval = pd.concat([train, val])
air_trval = trval[trval.domain == "airline"]
variants = [
    ("M0  Airline only  (first version)", lambda: build(clean_many_plain), air_trval),
    ("M1  + more domains", lambda: build(clean_many_plain), trval),
    ("M2  + negation marking", lambda: build(clean_many_neg), trval),
    ("M3  + VADER & style features", lambda: build(clean_many_neg, 1.0, 1.0), trval),
    (
        "M4  + char n-grams, neutral weight",
        lambda: build(clean_many_neg, 1.0, 1.0, True, NM, trval.label),
        trval,
    ),
]
ablation, final = [], None
for name, make, tr_df in variants:
    model = make().fit(tr_df.text, tr_df.label)
    ds = domain_scores(model, test)
    ch_pred = model.predict(ch_X)
    row = {
        "name": name,
        "airline_macro_f1": round(ds["airline"]["macro_f1"], 4),
        "airline_neutral_f1": round(ds["airline"]["neutral_f1"], 4),
        "movie_macro_f1": round(ds["movie"]["macro_f1"], 4),
        "tweeteval_macro_f1": round(ds["tweeteval"]["macro_f1"], 4),
        "challenge_accuracy": round(accuracy_score(ch_y, ch_pred), 4),
    }
    ablation.append(row)
    print(
        f"{name:36s} airline F1 {row['airline_macro_f1']:.3f} (neutral {row['airline_neutral_f1']:.3f}) | "
        f"movie F1 {row['movie_macro_f1']:.3f} | tweets F1 {row['tweeteval_macro_f1']:.3f} | challenge acc {row['challenge_accuracy']:.2f}"
    )
    final = model  # last one is M4 (final)

# 4. final model report on the combined test set 
pred = final.predict(test.text)
y = test.label
rep = classification_report(y, pred, labels=LABELS, output_dict=True, zero_division=0)
cm = confusion_matrix(y, pred, labels=LABELS)
text_rep = classification_report(y, pred, labels=LABELS, digits=4)
ds = domain_scores(final, test)
air = test[test.domain == "airline"]
air_rep = classification_report(
    air.label, final.predict(air.text), labels=LABELS, output_dict=True, zero_division=0
)
ch_pred = final.predict(ch_X)
print("\nFINAL MODEL (combined test set)\n" + text_rep + "\n" + str(cm))
print("\nChallenge set errors:")
errs = [
    {"text": t, "true": a, "pred": p} for t, a, p in zip(ch_X, ch_y, ch_pred) if a != p
]
for e in errs:
    print(f"  true={e['true']:8s} pred={e['pred']:8s} | {e['text']}")

metrics = {
    "model_name": "Logistic Regression + word/char TF-IDF + VADER",
    "labels": LABELS,
    "accuracy": round(accuracy_score(y, pred), 4),
    "macro_f1": round(f1_score(y, pred, average="macro"), 4),
    "weighted_f1": round(f1_score(y, pred, average="weighted"), 4),
    "per_class": {
        k: {
            "precision": round(rep[k]["precision"], 4),
            "recall": round(rep[k]["recall"], 4),
            "f1": round(rep[k]["f1-score"], 4),
            "support": int(rep[k]["support"]),
        }
        for k in LABELS
    },
    "confusion_matrix": cm.tolist(),
    "train_size": int(len(trval)),
    "test_size": int(len(test)),
    "selected_params": {"C": 1.0, "vader_weight": 1.0, "neutral_weight_multiplier": NM},
    "domain_results": [
        {
            "domain": d,
            "description": DOMAINS[d],
            "n": v["n"],
            "macro_f1": round(v["macro_f1"], 4),
            "accuracy": round(v["accuracy"], 4),
            "neutral_f1": (
                None if v["neutral_f1"] is None else round(v["neutral_f1"], 4)
            ),
        }
        for d, v in ds.items()
    ],
    "airline_per_class": {
        k: {
            "precision": round(air_rep[k]["precision"], 4),
            "recall": round(air_rep[k]["recall"], 4),
            "f1": round(air_rep[k]["f1-score"], 4),
            "support": int(air_rep[k]["support"]),
        }
        for k in LABELS
    },
    "ablation": ablation,
    "challenge": {
        "n": len(CHALLENGE),
        "accuracy": round(accuracy_score(ch_y, ch_pred), 4),
        "errors": errs,
    },
}
json.dump(metrics, open("metrics.json", "w"), indent=2)
open("classification_report.txt", "w").write(
    "FINAL MODEL - combined test set\n\n" + text_rep + "\n" + str(cm) + "\n"
)
joblib.dump(final, "model.joblib")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("FeelScope - combined test set")
    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    fig.tight_layout()
    fig.savefig("confusion_matrix.png", dpi=160)
except Exception as e:
    print("(skipped confusion_matrix.png:", e, ")")
print(
    "\nSaved: model.joblib, metrics.json, classification_report.txt, confusion_matrix.png"
)
