# News Text Classification Case Study

Supervised binary classification of news articles (**true** vs **fake**) using classical NLP vectorization and standard ML models. Authored by [Nikos Mavrapidis](https://github.com/NikosMav) (2023); cleaned so a hiring manager can clone, run, and read it as a coherent classification case study rather than leftover coursework.

This is **not** a production fake-news product, and it is **not** a TESSI / retrieval / RAG project. It is early supervised text classification work — adjacent to retrieval engineering (tokenization, vectors, evaluation) but a different problem.

## What this repo contains

| Piece | Role |
| --- | --- |
| `fake_news_classification.ipynb` | Full case study: data → preprocessing → EDA → vectorization → models → comparison → error analysis → limitations |
| `scripts/download_data.py` | Downloads `True.csv` / `Fake.csv` into `./data/` |
| `requirements.txt` | Pinned-enough dependency set for a clean checkout |
| `results_summary.csv` | Metrics from a reproduced local run of the notebook |

## Dataset

[ISOT Fake News Dataset](https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/) (also mirrored on Kaggle as *Fake and Real News Dataset* by Clément Bisaillon):

- `True.csv` — primarily Reuters articles (~21k)
- `Fake.csv` — articles from outlets flagged as unreliable (~23k)
- Columns: `title`, `text`, `subject`, `date`

Labels are corpus-defined (source bucket), not claim-level fact checks.

```bash
python scripts/download_data.py
```

## Method (short)

1. Clean title/body text (Gensim tokenization, stopword removal)
2. Exploratory analysis (subjects, word clouds, length distributions, bigrams)
3. 80/20 train/test split on **processed article body** (`random_state=7`, no replacement)
4. Vectorize with Count (BoW), TF-IDF, and averaged Word2Vec
5. Train Logistic Regression, Multinomial Naive Bayes, linear SVM, Random Forest
6. Compare metrics; inspect errors from the strongest model

Sparse vectorizers use unigram+bigram features with `min_df=2` so the notebook stays runnable on a normal machine. The original Colab homework kept hapaxes (multi-million-dimensional matrices).

## How to run

```bash
git clone https://github.com/NikosMav/FakeNews-Classification.git
cd FakeNews-Classification

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/download_data.py
jupyter notebook fake_news_classification.ipynb
```

A full execute (including EDA plots and the 150-tree Random Forest) typically finishes in a few minutes on a modern laptop/CPU.

## Results (reproduced locally)

Test set: **8,980** articles (train **35,918**). Metrics below are from running this cleaned notebook end-to-end; see `results_summary.csv`.

| Model | Vectorizer | Accuracy | F1 | ROC-AUC |
| --- | --- | ---: | ---: | ---: |
| Linear SVM | Count | 0.9963 | 0.9962 | 0.9995 |
| Logistic Regression | Count | 0.9952 | 0.9950 | 0.9996 |
| Linear SVM | TF-IDF | 0.9931 | 0.9928 | 0.9995 |
| Random Forest (150 trees) | TF-IDF | 0.9869 | 0.9863 | 0.9991 |
| Logistic Regression | TF-IDF | 0.9855 | 0.9849 | 0.9985 |
| Naive Bayes | Count | 0.9698 | 0.9687 | 0.9844 |
| Random Forest (10 trees) | Count | 0.9666 | 0.9646 | 0.9944 |
| Linear SVM | Word2Vec | 0.9665 | 0.9650 | 0.9942 |
| Logistic Regression | Word2Vec | 0.9663 | 0.9647 | 0.9944 |
| Random Forest (10 trees) | TF-IDF | 0.9608 | 0.9583 | 0.9937 |
| Naive Bayes | TF-IDF | 0.9598 | 0.9583 | 0.9915 |
| Random Forest (10 trees) | Word2Vec | 0.9569 | 0.9542 | 0.9910 |
| Naive Bayes | Word2Vec | 0.8822 | 0.8684 | 0.9631 |

Sparse lexical models (Count / TF-IDF + linear classifiers) dominate averaged Word2Vec here. That is expected for this bag-of-words-friendly setup.

**Do not over-read the 99% numbers.** On ISOT, “true” and “fake” come from stylistically different sources (e.g. Reuters datelines vs. blog/propaganda diction). Models can exploit **source/style cues** rather than “truthfulness.” Error analysis in the notebook is more informative than the headline accuracy.

### Notes vs. the original Colab homework

- Removed Google Drive / Colab-only paths; data loads from `./data/`
- Fixed train/test sampling to **without** replacement (the old `replace=True` could leak examples across splits)
- Fixed Random Forest + TF-IDF evaluating with the Count test matrix
- Updated Gensim 4.x Word2Vec APIs; scaled NB Word2Vec features using **train-only** statistics
- Relabeled metrics correctly (the old notebook printed `roc_auc_score` as “Accuracy”)
- Used `LinearSVC` instead of `SVC(kernel="linear")` for the same linear SVM family at practical speed
- Added comparison table, error analysis, and explicit limitations

## Limitations

1. Dataset construction couples label with outlet/style — high accuracy ≠ solved fake-news detection.
2. No transformer baselines, calibration, or temporal/domain-shift evaluation.
3. Word2Vec here is corpus-trained and mean-pooled; weak compared with sparse linear models on this task.
4. Binary corpus labels are not the same as verifying an arbitrary claim on the open web.

## Relation to retrieval / RAG

| This project | Retrieval / RAG |
| --- | --- |
| Labeled documents → class prediction | Query → relevant passages (+ optional generation) |
| Metrics: accuracy, F1, ROC-AUC | Metrics: recall@k, nDCG, grounded answer quality |
| Needs class labels | Needs a corpus index (and usually judgments or downstream eval) |

Shared foundations: text preprocessing, vector representations, careful evaluation. Classification score on ISOT is **not** evidence of retrieval or RAG skill.

## License

MIT — see [LICENSE.md](LICENSE.md).
