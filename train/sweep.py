"""Lanza el barrido de parámetros: producto cartesiano de `grid` × seeds.

Uso: python -m train.sweep --config configs/sweep.yaml --grid coarse_grid
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

import yaml

from darkforest.config import load_config


def expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """{'a': [1, 2], 'b': [3]} -> [{'a': 1, 'b': 3}, {'a': 2, 'b': 3}]."""
    keys = list(grid)
    return [
        dict(zip(keys, vals, strict=True)) for vals in itertools.product(*(grid[k] for k in keys))
    ]


def validate_cell(base: Path, overrides: dict[str, Any]) -> str | None:
    """Devuelve el motivo por el que la corrida es inválida, o None si se puede construir."""
    try:
        load_config(base, overrides)
    except (KeyError, TypeError, ValueError) as error:
        return str(error)
    return None


def main() -> None:
    """Valida todas las corridas del barrido antes de ejecutar o simular su lanzamiento."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--grid", default="grid")
    p.add_argument("--base", default="configs/base.yaml")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    spec = yaml.safe_load(Path(args.config).read_text())
    runs = [
        {**cell, "train.seed": seed}
        for cell in expand_grid(spec[args.grid])
        for seed in spec["seeds"]
    ]
    invalid = 0
    for overrides in runs:
        reason = validate_cell(Path(args.base), overrides)
        invalid += reason is not None
        if args.dry_run:
            print(overrides, "valida" if reason is None else f"invalida: {reason}")
    if args.dry_run:
        print(f"Corridas: {len(runs)}, inválidas: {invalid}")
    if invalid:
        raise SystemExit(f"{invalid} corridas con configuración inválida; no se entrena nada")
    if not args.dry_run:
        raise NotImplementedError("llamar a train.ippo.train y guardar resultados")


if __name__ == "__main__":
    main()
