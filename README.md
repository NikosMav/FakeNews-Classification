# News Evidence Retrieval

Passage retrieval over the ISOT news corpus with TF-IDF, MiniLM embeddings, and
hybrid reciprocal rank fusion (RRF). The repository also preserves the original
supervised news-classification study as a documented baseline.

This project retrieves related evidence from a fixed corpus. It is **not a
fact-checker**: ISOT labels describe source buckets, and a retrieved neighbor does
not establish whether a claim is true or false.

## What this project demonstrates

- Reproducible ingestion and stratified sampling of ISOT articles
- Overlapping passage chunking with article and source metadata
- Sparse TF-IDF, dense MiniLM, and hybrid RRF retrieval
- Saved, reloadable indexes and a command-line interface
- Retrieval evaluation, ablations, paraphrase stress tests, and failure analysis
- Lightweight unit tests and CI without dataset or model downloads

## Retrieval pipeline

```text
ISOT CSVs -> article sample -> passages -> TF-IDF + MiniLM indexes
                                             |
query ---------------------------------------+-> RRF -> ranked passages
```

The default index contains body passages of about 120 words with 20-word overlap.
Index artifacts store passage metadata, sparse vectors, dense vectors, and the
configuration required to reload the same index.

## Results

The main evaluation uses 4,000 sampled articles and 300 title-to-body queries.
Gold passages come from the query article, so this is a pipeline-quality proxy,
not human relevance judgment or claim verification.

<!-- METRICS_TABLE_START -->
| Method | Article Hit@1 | Article Hit@5 | Article Hit@10 | Passage Recall@5 | nDCG@5 | nDCG@10 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tfidf | 0.6533 | 0.8267 | 0.8700 | 0.4675 | 0.5181 | 0.5321 | 0.7290 |
| dense | 0.7767 | 0.8867 | 0.9033 | 0.5528 | 0.6163 | 0.6324 | 0.8205 |
| hybrid | 0.7833 | 0.9100 | 0.9400 | 0.5531 | 0.6258 | 0.6397 | 0.8415 |
<!-- METRICS_TABLE_END -->

The deterministic paraphrase stress test keeps the same articles and judgments
while reducing title overlap; hybrid MRR is `0.8291`. Detailed protocols,
ablations, per-query outputs, and sampled failures are in [`results/`](results/).

## Quick start

The test suite uses synthetic data and mocked dense encoders, so it does not
download ISOT or MiniLM:

```bash
git clone https://github.com/NikosMav/news-evidence-retrieval.git
cd news-evidence-retrieval
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
python -m evidence_retrieval -h
```

Build and query a real local index:

```bash
python scripts/download_data.py
python -m evidence_retrieval build
python -m evidence_retrieval query "Federal Reserve raises interest rates" --top-k 5
```

The first build downloads `sentence-transformers/all-MiniLM-L6-v2` and writes the
index under `data/retrieval_index/default/`.

Regenerate the committed retrieval evidence:

```bash
python -m evidence_retrieval eval
python -m evidence_retrieval eval --paraphrase-only
```

## Repository map

| Path | Purpose |
| --- | --- |
| `evidence_retrieval/` | Chunking, encoders, index, evaluation, and CLI |
| `tests/` | Unit and in-memory integration tests |
| `results/` | Metrics, protocol metadata, ablations, and failures |
| `scripts/` | Data download, evaluation, and notebook helpers |
| `evidence_retrieval.ipynb` | Short retrieval walkthrough |
| `fake_news_classification.ipynb` | Original classification case study |

## Classification baseline

The classification notebook compares Count, TF-IDF, and Word2Vec features across
logistic regression, Naive Bayes, linear SVM, and random forest models. Its best
committed result is Count + linear SVM at `0.9963` test accuracy.

That number should not be interpreted as real-world fact-check accuracy. A random
ISOT article split can expose source and writing-style cues shared between train and
test data. The retrieval project therefore treats classification as historical
context rather than a truth-verification system.

To run the notebook stack:

```bash
pip install -r requirements.txt
jupyter notebook fake_news_classification.ipynb
```

## Limitations

- The corpus is closed and historical; no web evidence is fetched.
- Retrieval judgments are synthetic same-article proxies, not independent qrels.
- Paraphrases are deterministic transformations rather than human-written claims.
- The current dense index is an in-memory, CPU-oriented demonstration.
- Source-bucket labels can encode outlet and style artifacts.

## Data and license

`scripts/download_data.py` downloads the
[ISOT Fake News Dataset](https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/).
The large CSVs and generated indexes are intentionally excluded from Git.

Code is released under the [MIT License](LICENSE.md). Dataset use remains subject
to the source dataset's terms.
