
.PHONY: init sample scrape index ui eval docker

init:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

sample:
	python -m src.collect_urls --sample-run

scrape:
	python -m src.scrape_and_chunk --in data/sampled_urls_run_*.json --out data/chunks.parquet

index:
	python -m src.build_indices --chunks data/chunks.parquet --out data/indices

ui:
	streamlit run app/ui_streamlit.py

eval:
	python -m src.evaluate --chunks data/chunks.parquet --eval-out data/metrics.json

docker:
	docker build -t hybrid-rag:latest .
