# Dataset directory

Place `True.csv` and `Fake.csv` here (ISOT Fake News Dataset).

```bash
python scripts/download_data.py
```

These CSV files are intentionally not committed (large binary-ish text dumps).

The evidence-retrieval notebook may also create `retrieval_index/` here (cached
embeddings + chunk metadata). That cache is gitignored and safe to delete; the notebook
will rebuild it.
