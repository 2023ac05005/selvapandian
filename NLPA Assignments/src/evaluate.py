
import argparse, json, re, numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from bert_score import score as bertscore
from .retrieve import HybridRetriever
from .generator import Generator

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def tokenize_simple(x):
    import re
    return re.findall(r"\w+", x.lower())

def lexical_overlap(a, b):
    A, B = set(tokenize_simple(a)), set(tokenize_simple(b))
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def generate_questions(df, n=100, seed=123):
    import random
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    QG_MODEL = "google/flan-t5-base"
    random.seed(seed)
    # one chunk per title to diversify
    sample = df.groupby("title", group_keys=False).apply(lambda x: x.sample(1, random_state=seed))
    sample = sample.sample(min(len(sample), max(n*2, 120)), random_state=seed)
    tok = AutoTokenizer.from_pretrained(QG_MODEL)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(QG_MODEL)
    qs = []
    for _, row in sample.iterrows():
        prompt = f"Generate a clear factual question answerable from the passage.
Passage: {row['text']}
Question:"
        inps = tok(prompt, return_tensors="pt", truncation=True)
        out = mdl.generate(**inps, max_new_tokens=64)
        q = tok.decode(out[0], skip_special_tokens=True).strip().rstrip("?") + "?"
        qs.append({"question": q, "chunk_id": row["chunk_id"], "title": row["title"], "url": row["url"]})
        if len(qs) >= n*3:
            break
    # de-duplicate with TF-IDF cosine
    vec = TfidfVectorizer().fit_transform([x["question"] for x in qs])
    keep, used = [], set()
    sims = cosine_similarity(vec)
    for i in range(len(qs)):
        if i in used:
            continue
        keep.append(qs[i])
        for j in range(i+1, len(qs)):
            if sims[i, j] > 0.8:
                used.add(j)
        if len(keep) >= n:
            break
    return keep

def eval_retrieval(questions, retriever, df, k=10):
    rec, mrr, ndcgs = [], [], []
    for q in questions:
        fused = retriever.hybrid(q["question"], top_n=k*2)
        ids = [c["chunk_id"] for c in fused][:k]
        gt_chunk = df[df["chunk_id"] == q["chunk_id"]].iloc[0]
        relevant = [1 if df[df["chunk_id"] == cid].iloc[0]["url"] == q["url"] else 0 for cid in ids]
        rec.append(1 if any(relevant) else 0)
        rank = next((i+1 for i, r in enumerate(relevant) if r == 1), None)
        mrr.append(1.0/rank if rank else 0.0)
        gains = [lexical_overlap(df[df["chunk_id"] == cid].iloc[0]["text"], gt_chunk["text"]) for cid in ids]
        ndcgs.append(ndcg_score([gains], [list(range(k,0,-1))]))
    return {"Recall@k": float(np.mean(rec)), "MRR@k": float(np.mean(mrr)), "nDCG@k": float(np.mean(ndcgs))}

def attribution_metrics(answer, contexts, cited_indices):
    ans_toks = set(tokenize_simple(answer))
    cited_text = " ".join([contexts[i]["text"] for i in cited_indices])
    cited_toks = set(tokenize_simple(cited_text))
    ap = len(ans_toks & cited_toks) / max(1, len(ans_toks))
    ar = len(ans_toks & cited_toks) / max(1, len(cited_toks))
    return ap, ar

def eval_generation(questions, retriever, generator, df, k=8):
    rouge_ls, bert_f1s, aps, ars = [], [], [], []
    for q in questions:
        ctx = retriever.hybrid(q["question"], top_n=k)
        answer = generator.generate(q["question"], ctx)
        cited = sorted(set(int(x)-1 for x in re.findall(r"\[(\d+)\]", answer) if 1 <= int(x) <= len(ctx)))
        ref = ctx[0]["text"] if ctx else ""
        R = scorer.score(ref, answer)["rougeL"].fmeasure if ref else 0.0
        if ref:
            P, Rb, F1 = bertscore([answer], [ref], lang="en", rescale_with_baseline=True)
            bert = float(F1[0])
        else:
            bert = 0.0
        ap, ar = attribution_metrics(answer, ctx, cited if cited else [0]) if ctx else (0.0, 0.0)
        rouge_ls.append(R); bert_f1s.append(bert); aps.append(ap); ars.append(ar)
    return {
        "ROUGE-L": float(np.mean(rouge_ls)),
        "BERTScore-F1": float(np.mean(bert_f1s)),
        "AttrPrecision": float(np.mean(aps)),
        "AttrRecall": float(np.mean(ars)),
    }

def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b: return 1.0
    return len(a & b) / max(1, len(a | b))

def compare_runs_metrics(m1, m2):
    diff = {}
    keys = set(m1.keys()) & set(m2.keys())
    for k in keys:
        diff[k] = float(m2[k] - m1[k])
    return diff

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chunks', type=str, help='data/chunks.parquet')
    ap.add_argument('--eval-out', type=str, default='data/metrics.json')
    ap.add_argument('--compare', nargs=2, help='metricsA.json metricsB.json')
    ap.add_argument('--stability-out', type=str, default='data/stability.json')
    args = ap.parse_args()

    if args.compare:
        with open(args.compare[0]) as f: m1 = json.load(f)
        with open(args.compare[1]) as f: m2 = json.load(f)
        delta = compare_runs_metrics(m1["generation"], m2["generation"])  # compare BERT/ROUGE/AP/AR
        out = {"generation_delta": delta}
        with open(args.stability_out, 'w') as f: json.dump(out, f, indent=2)
        print('Stability deltas written to', args.stability_out)
        return

    # Regular evaluation path
    if not args.chunks:
        raise SystemExit('--chunks is required for evaluation')
    df = pd.read_parquet(args.chunks)
    retriever = HybridRetriever()
    generator = Generator()

    questions = generate_questions(df, n=100)
    ret = eval_retrieval(questions, retriever, df, k=10)
    gen = eval_generation(questions, retriever, generator, df, k=8)
    metrics = {"retrieval": ret, "generation": gen}
    with open(args.eval_out, 'w') as f:
        json.dump(metrics, f, indent=2)
    print('Evaluation written to', args.eval_out)

if __name__ == '__main__':
    main()
