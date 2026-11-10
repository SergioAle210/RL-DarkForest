"""Agrega los JSON de train.resumen_corrida por trabajo: media y rango entre seeds.

Uso: python -m analysis.resumen_lote kout3/resumenes
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def load(directory: Path) -> dict[str, list[dict]]:
    """Resúmenes finales (seed<N>.json) agrupados por trabajo."""
    jobs = {}
    for folder in sorted(Path(directory).iterdir()):
        files = sorted(f for f in folder.glob("seed*.json") if "_iter_" not in f.name)
        jobs[folder.name] = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    return jobs


def stat(values) -> str:
    """Media y rango de una lista."""
    v = np.asarray(values, dtype=float)
    return f"{v.mean():+.2f} [{v.min():+.2f}, {v.max():+.2f}]"


def main() -> None:
    """Imprime una tabla por modo (muestreo y greedy)."""
    jobs = load(Path(sys.argv[1]))
    for mode in ("muestreo", "greedy"):
        print(f"\n== {mode} ==")
        print(
            f"{'trabajo':15s} n  retorno | canal apagado | barajado | "
            "P(ataque|contacto) | tx+compartir"
        )
        for name, runs in jobs.items():
            if not runs:
                continue
            ev = [r[mode]["evaluacion"] for r in runs]
            off = [r[mode]["canal_apagado"]["mean_return"] for r in runs]
            shuf = [r[mode]["mensajes_barajados"]["mean_return"] for r in runs]
            attack = [r[mode]["diagnostico"]["p_any_attack_given_contact"] for r in runs]
            talk = [e["action_transmitir"] + e["action_compartir"] for e in ev]
            print(
                f"{name:15s} {len(runs)}  {stat([e['mean_return'] for e in ev])} | "
                f"{stat(off)} | {stat(shuf)} | {np.mean(attack):.2f} | {np.mean(talk) * 100:.1f}%"
            )


if __name__ == "__main__":
    main()
