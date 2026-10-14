"""
Text preprocessing shared by training (train.py) and the API (app.py).
Steps: lowercase -> remove urls/mentions/hashtags -> expand contractions
       -> remove stop-words -> lemmatize -> (optional) negation marking

Negation marking: the 3 content words after a negator (not, no, never ...) get a
"not_" prefix, until the clause ends.   "not bad at all"  ->  "not not_bad"
so "not bad" and "bad" become DIFFERENT features for the classifier.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def _ensure_nltk():
    for pkg, path in [
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4"),
        ("vader_lexicon", "sentiment/vader_lexicon.zip"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


_ensure_nltk()

NEGATORS = {"not", "no", "never", "nor", "cannot", "without", "hardly", "neither"}
KEEP = {"no", "not", "nor", "never", "against", "very", "too", "but", "only", "without"}
STOP = set(stopwords.words("english")) - KEEP
LEMMA = WordNetLemmatizer()
LEMMA.lemmatize(
    "warmup"
)  

_CONTRACTIONS = [
    (r"\bwon't\b", "will not"),
    (r"\bcan't\b", "can not"),
    (r"\bcannot\b", "can not"),
    (r"n't\b", " not"),
]
NEG_SCOPE = 3


def tokenize(text: str, mark_neg: bool = True):
    """Return the cleaned, lemmatized tokens of a text."""
    t = str(text).lower()
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)  # urls
    t = re.sub(r"[@#]\w+", " ", t)  
    for pat, rep in _CONTRACTIONS:
        t = re.sub(pat, rep, t)
    t = re.sub(r"[.!?;,:()\n]+", " | ", t)  
    t = re.sub(r"[^a-z\s|]", " ", t)  
    out, left = [], 0
    for w in t.split():
        if w == "|":
            left = 0
            continue
        if w in NEGATORS:
            out.append("not")
            left = NEG_SCOPE if mark_neg else 0
            continue
        if len(w) < 2 or w in STOP:
            continue
        lemma = LEMMA.lemmatize(LEMMA.lemmatize(w, "v"), "n")
        if left > 0:
            lemma = "not_" + lemma
            left -= 1
        out.append(lemma)
    return out


def clean_text(text: str, mark_neg: bool = True) -> str:
    return " ".join(tokenize(text, mark_neg))
