"""
Custom feature blocks used inside the sklearn pipeline
The pipeline receives RAW text; cleaning happens inside it.
"""

from functools import lru_cache

import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.base import BaseEstimator, TransformerMixin

from preprocess import clean_text


@lru_cache(maxsize=None)
def _clean_neg(t):
    return clean_text(t, mark_neg=True)


@lru_cache(maxsize=None)
def _clean_plain(t):
    return clean_text(t, mark_neg=False)


def clean_many_neg(texts):
    return [_clean_neg(str(t)) for t in texts]


def clean_many_plain(texts):
    return [_clean_plain(str(t)) for t in texts]


_SIA = None


@lru_cache(maxsize=None)
def _vader_row(t):
    global _SIA
    if _SIA is None:
        _SIA = SentimentIntensityAnalyzer()
    s = _SIA.polarity_scores(t)
    q, ex = t.count("?"), t.count("!")
    return (s["compound"], s["pos"], s["neg"], s["neu"], min(q, 3) / 3, min(ex, 3) / 3)


class VaderStyle(BaseEstimator, TransformerMixin):
    """6 dense features computed on the RAW text:
    VADER compound / pos / neg / neu  (a general-English sentiment lexicon that also
    understands negation, intensifiers and CAPS)  +  '?' count + '!' count.
    `weight` scales them so they are not drowned by thousands of TF-IDF columns."""

    def __init__(self, weight=2.0):
        self.weight = weight

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([_vader_row(str(t)) for t in X]) * self.weight


_RAW_URL = None


def clean_raw_many(texts):
    """Light cleaning for character n-grams: lowercase, drop urls and @mentions, keep ? ! etc."""
    import re

    out = []
    for t in texts:
        t = re.sub(r"https?://\S+|www\.\S+|@\w+", " ", str(t).lower())
        out.append(re.sub(r"\s+", " ", t).strip())
    return out
