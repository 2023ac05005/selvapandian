
import json, pickle
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from .rrf import rrf_fuse

class HybridRetriever:
    def __init__(self, data_dir="data", emb_name="sentence-transformers/all-mpnet-base-v2"):
        self.model = SentenceTransformer(emb_name)
        self.index = faiss.read_index(f"{data_dir}/indices/faiss.index")
        with open(f"{data_dir}/indices/meta.json") as f:
            self.meta = {m["chunk_id"]: m for m in json.load(f)}
        self.df = pd.read_parquet(f"{data_dir}/chunks.parquet")
        with open(f"{data_dir}/indices/bm25.pkl","rb") as f:
            self.bm25: BM25Okapi = pickle.load(f)

    def dense_search(self, q, top_k=20):
        qv = self.model.encode([q], normalize_embeddings=True).astype(np.float32)
        D, I = self.index.search(qv, top_k)
        rows = self.df.iloc[I[0]]
        return rows["chunk_id"].tolist(), D[0].tolist()

    def bm25_search(self, q, top_k=20):
        tokens = q.split()
        scores = self.bm25.get_scores(tokens)
        top = np.argsort(scores)[::-1][:top_k]
        rows = self.df.iloc[top]
        return rows["chunk_id"].tolist(), scores[top].tolist()

    def hybrid(self, q, k_dense=20, k_sparse=20, k_rrf=60, top_n=8):
        d_ids, _ = self.dense_search(q, k_dense)
        s_ids, _ = self.bm25_search(q, k_sparse)
        fused = rrf_fuse(d_ids, s_ids, k=k_rrf)
        top = [cid for cid, _ in fused[:top_n]]
        out = []
        for cid in top:
            row = self.df[self.df["chunk_id"] == cid].iloc[0]
            out.append({
                "chunk_id": cid,
                "title": row["title"],
                "url": row["url"],
                "text": row["text"],
                "dense_rank": (d_ids.index(cid)+1) if cid in d_ids else None,
                "sparse_rank": (s_ids.index(cid)+1) if cid in s_ids else None,
            })
        return out
