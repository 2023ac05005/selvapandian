
import json, random, time, hashlib, argparse
import wikipediaapi

MIN_WORDS = 200

def page_word_count(page):
    text = page.text or ""
    return len(text.split())

def random_wiki_pages(n, min_words=MIN_WORDS, lang="en", seed=42, exclude=set()):
    random.seed(seed)
    wiki = wikipediaapi.Wikipedia(language=lang, user_agent="hybrid-rag/1.0 (academics)")
    CATS = [
        "Category:Science", "Category:Technology", "Category:History",
        "Category:Geography", "Category:Culture", "Category:Mathematics",
        "Category:Biology", "Category:Physics", "Category:Chemistry",
        "Category:Music", "Category:Sports", "Category:Politics"
    ]
    collected = []
    tried = set()
    attempts = 0
    while len(collected) < n and attempts < 30000:
        attempts += 1
        cat = wiki.page(random.choice(CATS))
        if not cat.exists():
            continue
        members = list(cat.categorymembers.keys())
        if not members:
            continue
        title = random.choice(members)
        if title in tried or title in exclude:
            continue
        tried.add(title)
        p = wiki.page(title)
        if not p.exists():
            continue
        if page_word_count(p) < min_words:
            continue
        url = p.fullurl
        collected.append({"title": p.title, "url": url})
    return collected

def save_fixed_urls(path, group_id, roll_numbers):
    seed = int(hashlib.sha256((group_id + "|" + "|".join(sorted(roll_numbers))).encode()).hexdigest(), 16) % (2**32)
    fixed = random_wiki_pages(200, seed=seed)
    with open(path, "w") as f:
        json.dump({"group_id": group_id, "roll_numbers": roll_numbers, "urls": fixed}, f, indent=2)
    print(f"Saved {len(fixed)} fixed URLs -> {path}")

def sample_run_urls(fixed_urls_path, out_path, run_seed=None):
    with open(fixed_urls_path) as f:
        fixed = json.load(f)["urls"]
    exclude_titles = set([x["title"] for x in fixed])
    if run_seed is None:
        run_seed = int(time.time())
    rand = random_wiki_pages(300, seed=run_seed, exclude=exclude_titles)
    with open(out_path, "w") as f:
        json.dump({"run_seed": run_seed, "urls": fixed + rand}, f, indent=2)
    print(f"Saved run URLs (500 total) -> {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--init-fixed', action='store_true', help='Create fixed_urls.json')
    ap.add_argument('--group-id', type=str, default='GROUP-ID')
    ap.add_argument('--roll', nargs='*', default=['ROLL1','ROLL2'])
    ap.add_argument('--fixed-out', type=str, default='data/fixed_urls.json')
    ap.add_argument('--sample-run', action='store_true', help='Generate a run with 300 random URLs')
    ap.add_argument('--run-out', type=str, default=None)
    args = ap.parse_args()

    if args.init_fixed:
        save_fixed_urls(args.fixed_out, args.group_id, args.roll)

    if args.sample_run:
        import os, time
        out = args.run_out or f"data/sampled_urls_run_{int(time.time())}.json"
        if not os.path.exists('data/fixed_urls.json'):
            raise SystemExit('data/fixed_urls.json not found. Run with --init-fixed first.')
        sample_run_urls('data/fixed_urls.json', out)
