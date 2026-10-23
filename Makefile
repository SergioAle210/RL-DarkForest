.PHONY: install test lint format sanity sweep-coarse sweep

install:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

sanity:
	uv run python -m train.ippo --config configs/h1_sanity.yaml

sweep-coarse:
	uv run python -m train.sweep --config configs/sweep.yaml --grid coarse_grid

sweep:
	uv run python -m train.sweep --config configs/sweep.yaml --grid grid
