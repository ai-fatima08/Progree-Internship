# Moodora: Multi-Class Sentiment Classifier

Moodora classifies a piece of text as **negative**, **neutral** or **positive**. It is built with Python and Scikit-Learn, trained on three different domains, and ships with a Flask API and a single-file web interface.

Built during the **Progree Artificial Intelligence Internship** (Task 2).

| Metric | Result |
| --- | --- |
| Macro-F1 | **0.718** |
| Accuracy | **72.4 %** |
| Weighted-F1 | 0.725 |
| Test set | 9,019 unseen texts from 3 domains |

---

## Features

- Single scikit-learn `Pipeline`: raw text in, label out (training and prediction use exactly the same processing)
- Stop-word removal that **keeps negations** (not, no, never, nor, without), because they flip sentiment
- WordNet lemmatisation
- Negation marking: the three content words after a negator get a `not_` prefix, so "not bad" and "bad" become different features
- Word 1-2-gram TF-IDF + character 2-5-gram TF-IDF (sub-linear term frequency)
- Six dense features: VADER compound / positive / negative / neutral scores, plus counts of `?` and `!`
- Logistic Regression (C = 1.0) with balanced class weights and an extra x1.3 weight on the neutral class
- Flask REST API and a web UI with a live classifier and a model-report tab (F1 scores and confusion matrix)

## Requirements covered

| Requirement | Implementation |
| --- | --- |
| Python with Scikit-Learn | One Pipeline (`preprocess.py`, `features.py`, `train.py`) |
| Remove stop-words | NLTK English stop-words, except negations |
| Lemmatization | NLTK WordNet lemmatizer |
| TF-IDF representation | Word and character TF-IDF |
| F1-score evaluation | Per-class P / R / F1, macro and weighted F1, confusion matrix, per-domain scores |

## Data

Three public sources are combined so the model learns sentiment rather than one dataset's vocabulary. Each domain keeps its own held-out test set, and hyper-parameters were chosen on a separate validation split. The training set has 41,070 texts.

| Source | Contribution | Test texts |
| --- | --- | --- |
| Twitter US Airline Sentiment (Kaggle) | Airline customer tweets | 2,886 |
| NLTK `sentence_polarity` + `subjectivity` | Movie sentences (objective plot sentences used as neutral) | 3,133 |
| TweetEval sentiment (SemEval-2017) | General tweets with human-labelled neutral | 3,000 |

## Results

**Per class (combined test set)**

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Negative | 0.780 | 0.741 | 0.760 | 3,853 |
| Neutral | 0.683 | 0.733 | 0.707 | 3,062 |
| Positive | 0.692 | 0.682 | 0.687 | 2,104 |

**Per domain**

| Domain | Texts | Accuracy | Macro-F1 | Neutral F1 |
| --- | --- | --- | --- | --- |
| Airline tweets | 2,886 | 0.800 | 0.743 | 0.612 |
| Movie sentences | 3,133 | 0.765 | 0.766 | 0.841 |
| General tweets | 3,000 | 0.609 | 0.589 | 0.657 |

### Ablation study

Each version adds one idea; all are scored on the same held-out sets.

| Version | Airline F1 | Movie F1 | General tweets F1 | Challenge set |
| --- | --- | --- | --- | --- |
| M0 Airline data only | 0.732 | 0.431 | 0.494 | 21 / 30 |
| M1 + more domains | 0.701 | 0.740 | 0.538 | 22 / 30 |
| M2 + negation marking | 0.711 | 0.743 | 0.544 | 23 / 30 |
| M3 + VADER and style features | 0.722 | 0.737 | 0.578 | 22 / 30 |
| **M4 + char n-grams, neutral weight (final)** | **0.743** | **0.766** | **0.589** | **25 / 30** |

**Takeaway:** data diversity mattered more than any single modelling trick. Moving from one domain to three lifted movie F1 from 0.43 to 0.74.

### Challenge set

30 hand-written sentences, never used for training or tuning. The final model gets **25 / 30 (83.3 %)** right. Three of the five errors are positive sentences that look negative on the surface ("I couldn't be happier...", "No complaints at all...", "...far better than I expected").


![Confusion matrix](sentiment_analysis/Moodora_Confusion_Matrix.png)
![Web UI](sentiment_analysis/ui.png)
-->

## Project structure

```
sentiment_analysis/
├── app.py          # Flask API
├── train.py        # training + evaluation
├── preprocess.py   # cleaning, stop-words, lemmas, negation marking
├── features.py     # TF-IDF + VADER feature union
├── data.py         # dataset loading
├── challenge.py    # 30-sentence challenge set
├── index.html      # single-file front end
└── requirements.txt
```

## Getting started

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>/sentiment_analysis
pip install -r requirements.txt
python app.py
```

Then open <http://127.0.0.1:5000>.

**Re-train the model**

```bash
python train.py --data Tweets.csv
```

## API

| Endpoint | Description |
| --- | --- |
| `/api/predict` | Classify a single text |
| `/api/predict-batch` | Classify multiple texts |
| `/api/metrics` | Model report (F1 scores, confusion matrix) |
| `/api/health` | Health check |

## Limitations

- Neutral is the hardest class (airline neutral F1 about 0.61); a bag-of-words model has a ceiling there.
- General tweets reach only macro-F1 0.589. Fine-tuned transformers (BERT / RoBERTa) usually score higher, but were left out because the task asks for Scikit-Learn and TF-IDF.
- "Neutral" for movie sentences means objective plot sentences, only an approximation of neutral opinion.
- Sarcasm, idioms and mixed feelings are not handled; the challenge set is a sanity check, not a benchmark.

## Future work

- Compare against a fine-tuned transformer
- Handle idioms and sarcasm
- Collect more neutral examples

## Acknowledgements

- Twitter US Airline Sentiment (Kaggle)
- NLTK `sentence_polarity` / `subjectivity`
- TweetEval / SemEval-2017 Task 4 (Rosenthal, Farra and Nakov, 2017)
- VADER (Hutto and Gilbert, 2014)
- scikit-learn (Pedregosa et al., 2011)

## Author

**S Fatima Manzoor** (Progree AI Internship, 20 September - 20 October 2026)