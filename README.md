# News Text Classification + Evidence Retrieval

Supervised binary classification of news articles (**true** vs **fake**), plus a follow-on
**evidence-retrieval** chapter over the same ISOT corpus. Authored by
[Nikos Mavrapidis](https://github.com/NikosMav) (2023 classification work; recently made
runnable; retrieval added as the next chapter).

**Brand direction:** AI / retrieval engineer portfolio piece — Chapter 1 is classical NLP
classification groundwork; Chapter 2 is nearest-neighbor passage retrieval (the **R** in
RAG). Neither chapter is a production fake-news product.

| Chapter | Problem | Artifact |
| --- | --- | --- |
| 1 — Classification | Document → source-bucket label | `fake_news_classification.ipynb` |
| 2 — Evidence retrieval | Query / claim / article → nearest passages + metadata | `evidence_retrieval.ipynb` |

## What this repo contains

| Piece | Role |
| --- | --- |
| `fake_news_classification.ipynb` | Chapter 1: data → preprocessing → EDA → vectorization → models → comparison → error analysis |
| `evidence_retrieval.ipynb` | Chapter 2: chunk → embed (local MiniLM) → retrieve passages with title / label / score |
| `scripts/download_data.py` | Downloads `True.csv` / `Fake.csv` into `./data/` |
| `scripts/build_notebook.py` | Regenerates the classification notebook |
| `scripts/build_retrieval_notebook.py` | Regenerates the retrieval notebook |
| `requirements.txt` | Dependencies for both chapters |
| `results_summary.csv` | Classification metrics from a reproduced local run |

## Dataset

[ISOT Fake News Dataset](https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/)
(also mirrored on Kaggle as *Fake and Real News Dataset* by Clément Bisaillon):

- `True.csv` — primarily Reuters articles (~21k)
- `Fake.csv` — articles from outlets flagged as unreliable (~23k)
- Columns: `title`, `text`, `subject`, `date`

Labels are corpus-defined **source buckets**, not claim-level fact checks.

```bash
python scripts/download_data.py
```

## How to run

```bash
git clone https://github.com/NikosMav/FakeNews-Classification.git
cd FakeNews-Classification

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/download_data.py

# Chapter 1 — supervised classification
jupyter notebook fake_news_classification.ipynb

# Chapter 2 — evidence retrieval
jupyter notebook evidence_retrieval.ipynb
```

Chapter 1 typically finishes in a few minutes on a modern CPU. Chapter 2 downloads the
MiniLM weights once, embeds a stratified sample of passages (cached under
`data/retrieval_index/`), then runs interactive retrieval demos.

---

## Chapter 1 — Classification (summary)

1. Clean title/body text (Gensim tokenization, stopword removal)
2. Exploratory analysis (subjects, word clouds, length distributions, bigrams)
3. 80/20 train/test split on **processed article body** (`random_state=7`, no replacement)
4. Vectorize with Count (BoW), TF-IDF, and averaged Word2Vec
5. Train Logistic Regression, Multinomial Naive Bayes, linear SVM, Random Forest
6. Compare metrics; inspect errors from the strongest model

Sparse vectorizers use unigram+bigram features with `min_df=2` so the notebook stays runnable on a normal machine.

### Results (reproduced locally)

Test set: **8,980** articles (train **35,918**). See `results_summary.csv`.

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

**Do not over-read the 99% numbers.** On ISOT, “true” and “fake” come from stylistically different sources. Models can exploit **source/style cues** rather than “truthfulness.”

---

## Chapter 2 — Evidence retrieval

Given a query, claim, or article text, return the **nearest passages** from a chunked ISOT
index with source metadata:

- `title`, `label` / `label_name` (source bucket), `subject`, `date`
- cosine similarity `score`
- passage text (extractive quote)

**Default stack (no paid API):**

- Chunking: overlapping word windows (~120 words)
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` on CPU
- Index: `sklearn.neighbors.NearestNeighbors` (cosine)
- Corpus size: stratified sample of articles (configurable `N_ARTICLES`, default 4000)

**Optional demo in the notebook:** *classify then retrieve neighbors* vs *retrieve first*
(neighborhood source-bucket vote) on the same documents — contrast, not a production
pipeline.

### Honest limitations (retrieval)

1. This is **nearest-neighbor evidence retrieval over ISOT**, not a production fact-checker
   and **not RAG-over-the-web**.
2. Retrieved “fake” neighbors do **not** prove a claim is false; “true” neighbors do **not**
   prove it is true. Labels are source buckets.
3. No cross-encoder re-ranker, hybrid BM25, or labeled retrieval metrics (nDCG / recall@k).
4. **Generation is out of scope** unless you count quoting the top passage as a trivial
   extractive display (included as a tiny demo).

---

## Relation to RAG

| Classification (Ch. 1) | Retrieval (Ch. 2) | Full RAG |
| --- | --- | --- |
| Labeled documents → class prediction | Query → relevant passages (+ metadata) | Retrieve → condition a generator |
| Metrics: accuracy, F1, ROC-AUC | Metrics (production): recall@k, nDCG | + groundedness / answer quality |
| Needs class labels | Needs a corpus index | Needs index + generator (+ eval) |

Shared foundations: text preprocessing, vector representations, careful evaluation.
Chapter 2 implements the retrieval step; generative answering is optional / out of scope
here.

## Notes vs. the original Colab homework (Chapter 1)

- Removed Google Drive / Colab-only paths; data loads from `./data/`
- Fixed train/test sampling to **without** replacement
- Fixed Random Forest + TF-IDF evaluating with the Count test matrix
- Updated Gensim 4.x Word2Vec APIs; scaled NB Word2Vec features using **train-only** statistics
- Relabeled metrics correctly (the old notebook printed `roc_auc_score` as “Accuracy”)
- Used `LinearSVC` instead of `SVC(kernel="linear")` for practical speed
- Added comparison table, error analysis, and explicit limitations

## License

MIT — see [LICENSE.md](LICENSE.md).
