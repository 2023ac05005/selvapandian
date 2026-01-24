
# Hybrid RAG (Dense + BM25 + RRF) over Wikipedia

This repository provides a complete, **ready-to-deploy** Hybrid Retrieval-Augmented Generation (RAG) system:

- **Dense retrieval** with Sentence-Transformers + **FAISS**
- **Sparse retrieval** with **BM25**
- **Reciprocal Rank Fusion (RRF)** to combine dense & sparse (k=60)
- **Open-source generator** (Flan-T5-base)
- **Streamlit UI** with per-chunk ranks & sources
- **Automated evaluation** over **100 generated questions** (ROUGE-L, BERTScore, retrieval metrics, and innovative **Attribution** & **Stability** scores)

## Quickstart

> Python 3.10 recommended

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 1) Create your **fixed** 200 Wikipedia URLs (unique per group)

Edit the call in `src/collect_urls.py` and run:

```bash
python -m src.collect_urls --init-fixed --group-id G23-AI-07 --roll 21IT001 21IT017 21IT042
```

This writes `data/fixed_urls.json`. Commit it to the repo.

### 2) Generate a run (adds 300 random URLs, total 500)

```bash
python -m src.collect_urls --sample-run
```

This writes `data/sampled_urls_run_<stamp>.json`.

### 3) Scrape → clean → chunk (200–400 tokens, 50 overlap)

```bash
python -m src.scrape_and_chunk --in data/sampled_urls_run_<stamp>.json --out data/chunks.parquet
```

### 4) Build indices (FAISS + BM25)

```bash
python -m src.build_indices --chunks data/chunks.parquet --out data/indices
```

### 5) Run UI (Streamlit)

```bash
streamlit run app/ui_streamlit.py
```

### 6) Evaluate on 100 generated questions

```bash
python -m src.evaluate --chunks data/chunks.parquet --eval-out data/metrics.json
```

### 7) Stability experiment (optional, for the report)

Run steps 2–6 again with a new run. Then:

```bash
python -m src.evaluate --compare data/metrics.json data/metrics_run2.json --stability-out data/stability.json
```

## Deployment options

### Streamlit Community Cloud
- Push this repo to GitHub
- Create new app → `app/ui_streamlit.py`
- Set Python version to 3.10 and supply `requirements.txt`

### Docker (local or server)
```bash
# Build
docker build -t hybrid-rag:latest .
# Run UI
docker run -it --rm -p 8501:8501 -v $PWD/data:/app/data hybrid-rag:latest
```

## Repo layout
```
hybrid-rag/
  ├─ app/ui_streamlit.py
  ├─ src/
  │   ├─ collect_urls.py
  │   ├─ scrape_and_chunk.py
  │   ├─ build_indices.py
  │   ├─ retrieve.py
  │   ├─ rrf.py
  │   ├─ generator.py
  │   └─ evaluate.py
  ├─ data/
  ├─ .streamlit/config.toml
  ├─ requirements.txt
  ├─ Dockerfile
  ├─ Makefile
  └─ README.md
```

## Notes
- Ensure your **200 fixed URLs** are unique across groups. The script derives a seed from your group ID + roll numbers to make your set reproducible.
- The **300 random URLs** change on each indexing run, enabling the **stability** metric.
- All code uses **open-source** models/libraries only.

