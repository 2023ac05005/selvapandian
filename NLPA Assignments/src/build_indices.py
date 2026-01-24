
import json, pickle, argparse
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

EMB_NAME = "sentence-transformers/all-mpnet-base-v2"

def build_dense(df):
    model = SentenceTransformer(EMB_NAME)
    embs = model.encode(df["text"].tolist(), normalize_embeddings=True, show_progress_bar=True)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs.astype(np.float32))
    return index

def build_bm25(df):
    corpus = [doc.split() for doc in df["text"].tolist()]
    bm25 = BM25Okapi(corpus)
    return bm25

def persist(index, bm25, df, out_dir):
    import os
    os.makedirs(out_dir, exist_ok=True)
    faiss.write_index(index, f"{out_dir}/faiss.index")
    with open(f"{out_dir}/bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    meta = df[["chunk_id","title","url","chunk_index"]].to_dict(orient="records")
    with open(f"{out_dir}/meta.json","w") as f:
        json.dump(meta, f)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--chunks', type=str, required=True)
    ap.add_argument('--out', type=str, default='data/indices')
    args = ap.parse_args()
    df = pd.read_parquet(args.chunks)
    index = build_dense(df)
    bm25 = build_bm25(df)
    persist(index, bm25, df, args.out)
    print('Indices built and saved to', args.out)
