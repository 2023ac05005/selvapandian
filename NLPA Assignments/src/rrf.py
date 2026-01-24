
from typing import List, Tuple

def rrf_fuse(dense_order: List[str], sparse_order: List[str], k: int = 60) -> List[Tuple[str, float]]:
    """Reciprocal Rank Fusion.
    Returns list of (chunk_id, score) sorted descending.
    """
    scores = {}
    for rank, cid in enumerate(dense_order, 1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    for rank, cid in enumerate(sparse_order, 1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
