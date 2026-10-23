"""Lanza el barrido de parámetros: producto cartesiano de `grid` × seeds.

Uso: python -m train.sweep --config configs/sweep.yaml --grid coarse_grid
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

import yaml


def expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """{'a': [1, 2], 'b': [3]} -> [{'a': 1, 'b': 3}, {'a': 2, 'b': 3}]."""
    keys = list(grid)
    return [
        dict(zip(keys, vals, strict=True)) for vals in itertools.product(*(grid[k] for k in keys))
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--grid", default="grid")
    p.add_argument("--base", default="configs/base.yaml")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    spec = yaml.safe_load(Path(args.config).read_text())
    cells = expand_grid(spec[args.grid])
    for cell in cells:
        for seed in spec["seeds"]:
            overrides = {**cell, "train.seed": seed}
            if args.dry_run:
                print(overrides)
                continue
            raise NotImplementedError("llamar a train.ippo.train y guardar resultados")


if __name__ == "__main__":
    main()
