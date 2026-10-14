import json
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

import features  
from preprocess import clean_text

BASE = Path(__file__).parent
MODEL_PATH = BASE / "model.joblib"
METRICS_PATH = BASE / "metrics.json"

app = Flask(__name__)
CORS(app)  

model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def _probs(cleaned_list):
    """Return an (n, classes) probability array for any sklearn classifier."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(cleaned_list)
    scores = model.decision_function(cleaned_list)  
    if scores.ndim == 1:
        scores = np.vstack([-scores, scores]).T
    e = np.exp(scores - scores.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _explain(cleaned):
    """Per-word push towards positive (+) or negative (-), scaled to [-1, 1].
    Uses the word-level TF-IDF block of the linear model (first block of the FeatureUnion).
    """
    try:
        feats = dict(model.named_steps["features"].transformer_list)
        vocab = feats["tfidf"].named_steps["vec"].vocabulary_
        clf = model.named_steps["clf"]
        classes = [str(c).lower() for c in clf.classes_]
        diff = (
            clf.coef_[classes.index("positive")] - clf.coef_[classes.index("negative")]
        )
        out, seen = [], set()
        for w in cleaned.split():
            if w in seen:
                continue
            seen.add(w)
            j = vocab.get(w)
            out.append(
                {
                    "w": w.replace("not_", "not "),
                    "s": float(diff[j]) if j is not None else 0.0,
                }
            )
        m = max([abs(x["s"]) for x in out] or [1]) or 1
        for x in out:
            x["s"] = round(x["s"] / m, 3)
        return out
    except Exception:
        return []


def _predict(texts):
    P = _probs(texts)  
    classes = [str(c).lower() for c in model.classes_]
    out = []
    for text, p in zip(texts, P):
        probs = {k: round(float(v), 4) for k, v in zip(classes, p)}
        for k in ("negative", "neutral", "positive"):
            probs.setdefault(k, 0.0)
        out.append(
            {
                "text": text,
                "cleaned": clean_text(text).replace("not_", "not "),
                "label": classes[int(np.argmax(p))],
                "probs": probs,
            }
        )
    return out


@app.get("/")
def home():
    return send_from_directory(BASE, "index.html")


@app.get("/api/health")
def health():
    return jsonify(status="ok", model_loaded=model is not None)


@app.post("/api/predict")
def predict():
    if model is None:
        return jsonify(error="model_not_trained"), 503
    text = (request.get_json(silent=True) or {}).get("text", "").strip()
    if not text:
        return jsonify(error="empty_text"), 400
    result = _predict([text])[0]
    result["words"] = _explain(clean_text(text))
    return jsonify(result)


@app.post("/api/predict-batch")
def predict_batch():
    if model is None:
        return jsonify(error="model_not_trained"), 503
    texts = (request.get_json(silent=True) or {}).get("texts", [])
    texts = [t for t in texts if str(t).strip()][:2000]
    if not texts:
        return jsonify(error="empty_text"), 400
    return jsonify(results=_predict(texts))


@app.get("/api/metrics")
def metrics():
    if not METRICS_PATH.exists():
        return jsonify(error="metrics_not_found"), 404
    return jsonify(json.loads(METRICS_PATH.read_text(encoding="utf-8")))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
