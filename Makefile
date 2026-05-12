.PHONY: install endmembers preprocess train ablation test lint format

install:
	pip install -e ".[dev]"

endmembers:
	python scripts/prepare_endmembers.py

train:
	python scripts/train.py --config configs/gcsuae_default.yaml

ablation:
	python scripts/run_ablation.py --config configs/ablation.yaml

test:
	pytest tests/ -v

lint:
	ruff check .

format:
	ruff format .
