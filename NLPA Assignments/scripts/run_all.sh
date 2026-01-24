
#!/usr/bin/env bash
set -euo pipefail

python -m src.collect_urls --sample-run
LATEST=$(ls -t data/sampled_urls_run_*.json | head -n1)
python -m src.scrape_and_chunk --in "$LATEST" --out data/chunks.parquet
python -m src.build_indices --chunks data/chunks.parquet --out data/indices
python -m src.evaluate --chunks data/chunks.parquet --eval-out data/metrics.json
