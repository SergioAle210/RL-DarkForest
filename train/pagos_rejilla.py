"""Mide una matriz de pagos por celda (λ, c) y las guarda en disco para el replicador.

Cada celda produce un JSON con la matriz, su error estándar y la supervivencia. Las celdas ya
medidas se saltan, así que la corrida se puede interrumpir y reanudar. analysis.phase_diagram lee
estos archivos y construye el mapa de regímenes. Lectura en docs/DIAGNOSTICO_DISENO.md.

Uso: python -m train.pagos_rejilla --out-dir runs/pagos --set env.detection_growth=0.05
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from train.matriz_pagos import payoff_matrix

MODES = ("oculta", "solo", "coop_real", "ataque_solo")


def cell_path(out_dir: Path, lethality: float, speed: float) -> Path:
    """Archivo de la celda (λ, c)."""
    return out_dir / f"pagos_lam{lethality:g}_c{speed:g}.json"


def main() -> None:
    """Recorre la rejilla λ por c y guarda una matriz de pagos por celda."""
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--lethality", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    p.add_argument("--signal-speed", type=float, nargs="+", default=[0.05, 0.22, 1.0])
    p.add_argument("--modes", nargs="+", default=list(MODES))
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--set", action="append", default=[], metavar="CLAVE=VALOR", dest="extra")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extra: dict[str, Any] = {k: yaml.safe_load(v) for k, v in (e.split("=", 1) for e in args.extra)}
    for lethality in args.lethality:
        for speed in args.signal_speed:
            path = cell_path(out_dir, lethality, speed)
            if path.exists():
                print(f"lambda={lethality:g} c={speed:g}: ya medida", flush=True)
                continue
            print(f"lambda={lethality:g} c={speed:g}", flush=True)
            result = payoff_matrix(
                lethality,
                args.modes,
                args.seeds,
                args.episodes,
                args.steps,
                {**extra, "env.signal_speed": speed},
            )
            result["signal_speed"] = speed
            path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
