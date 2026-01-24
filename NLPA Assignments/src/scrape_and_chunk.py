
import json, re, uuid, argparse
import pandas as pd
import wikipediaapi
from transformers import AutoTokenizer

TOKENIZER_NAME = "sentence-transformers/all-mpnet-base-v2"

def clean_text(t):
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def chunk_tokens(tokens, min_tok=200, max_tok=400, overlap=50):
    i, n = 0, len(tokens)
    while i < n:
        j = min(i + max_tok, n)
        if j - i < min_tok and j != n:
            j = min(n, i + min_tok)
        yield tokens[i:j]
        if j == n:
            break
        i = max(0, j - overlap)

def fetch_page(wiki, title):
    p = wiki.page(title)
    if p.exists():
        return p.text
    return None

def scrape_to_chunks(sampled_urls_json, out_parquet):
    with open(sampled_urls_json) as f:
        urls = json.load(f)["urls"]
    wiki = wikipediaapi.Wikipedia(language="en", user_agent="hybrid-rag/1.0")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    rows = []
    for it in urls:
        title, url = it["title"], it["url"]
        text = fetch_page(wiki, title)
        if not text or len(text.split()) < 200:
            continue
        text = clean_text(text)
        ids = tokenizer.encode(text, add_special_tokens=False)
        for idx, tok_span in enumerate(chunk_tokens(ids, 200, 400, 50)):
            chunk_text = tokenizer.decode(tok_span, skip_special_tokens=True)
            rows.append({
                "chunk_id": str(uuid.uuid4()),
                "title": title,
                "url": url,
                "chunk_index": idx,
                "text": chunk_text
            })
    df = pd.DataFrame(rows)
    df.to_parquet(out_parquet, index=False)
    print(f"Chunks saved: {len(df)} -> {out_parquet}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='in_file', required=True)
    ap.add_argument('--out', dest='out_file', default='data/chunks.parquet')
    args = ap.parse_args()
    scrape_to_chunks(args.in_file, args.out_file)
