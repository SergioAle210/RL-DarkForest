"""Ablación del canal con políticas fijas, sin entrenar.

Compara ignorar, solo y coop_real con el canal encendido, apagado (env.channel_enabled) y con
los mensajes barajados (env.shuffle_messages), por letalidad y radio de la zona. Sirve para
tres lecturas, todas en docs/DIAGNOSTICO_DISENO.md: H1, con λ = 0 apagar el canal no debe
cambiar a quien no lo usa (si cambia hay una fuga de información); con λ > 0, cuánto del
retorno de coop_real depende del canal; y si importa que el mensaje sea verdadero o solo que
llegue. Cada celda (λ, radio) se guarda en un JSON; las ya medidas solo agregan lo que falta.

Uso: python -m train.ablacion_canal --out-dir runs/ablacion_canal
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train.valor_cooperacion import PERIODO_ALEATORIO, mean_se, run

MODES = ("ignorar", "solo", "coop_real")


def cell_path(out_dir: Path, lethality: float, radius: float) -> Path:
    """Archivo de la celda (λ, radio)."""
    return out_dir / f"ablacion_lam{lethality:g}_r{radius:g}.json"


VARIANTS = {
    "on": {},
    "off": {"env.channel_enabled": False},
    "shuffle": {"env.shuffle_messages": True},
}


def measure_cell(
    lethality: float,
    radius: float,
    seeds: int,
    episodes: int,
    steps: int,
    margin: float,
    cell: dict | None = None,
) -> dict:
    """Retorno y supervivencia por seed de cada política en cada variante del canal.

    Con una celda ya medida solo calcula las combinaciones que le faltan.
    """
    cell = cell or {"lethality": lethality, "radius": radius, "margin": margin, "seeds": seeds}
    for variant, extra in VARIANTS.items():
        over = {"env.anomaly.radius": radius, "env.signal_speed": 1.0, **extra}
        for mode in MODES:
            if f"{mode}_{variant}" in cell:
                continue
            runs = [
                run(
                    lethality,
                    PERIODO_ALEATORIO,
                    mode,
                    margin if mode == "coop_real" else 0.0,
                    s,
                    episodes=episodes,
                    steps=steps,
                    env_over=over,
                )
                for s in range(seeds)
            ]
            cell[f"{mode}_{variant}"] = {
                "return": [r[0] for r in runs],
                "survival": [r[1] for r in runs],
            }
    return cell


def summarize(cell: dict) -> str:
    """Una línea por política y las diferencias pareadas por seed que sostienen cada lectura."""
    lines = [f"lambda={cell['lethality']:g} radio={cell['radius']:g}"]
    for mode in MODES:
        on, off, mixed = (np.array(cell[f"{mode}_{k}"]["return"]) for k in VARIANTS)
        diff, diff_mixed = mean_se(on - off), mean_se(on - mixed)
        lines.append(
            f"  {mode:10s} canal {np.mean(on):+.3f}  sin canal {np.mean(off):+.3f}  "
            f"barajado {np.mean(mixed):+.3f}  apagar {diff[0]:+.3f} ± {diff[1]:.3f}  "
            f"barajar {diff_mixed[0]:+.3f} ± {diff_mixed[1]:.3f}"
        )
    on = np.array(cell["coop_real_on"]["return"])
    best = np.maximum(np.array(cell["ignorar_on"]["return"]), np.array(cell["solo_on"]["return"]))
    gain = mean_se(on - best)
    lines.append(f"  coop_real con canal menos la mejor sin canal: {gain[0]:+.3f} ± {gain[1]:.3f}")
    return "\n".join(lines)


def main() -> None:
    """Recorre λ por radio, guarda una celda por JSON e imprime su resumen."""
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--lethality", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    p.add_argument("--radius", type=float, nargs="+", default=[0.25, 0.45])
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--margin", type=float, default=0.2)
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for lethality in args.lethality:
        for radius in args.radius:
            path = cell_path(out_dir, lethality, radius)
            cell = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
            cell = measure_cell(
                lethality, radius, args.seeds, args.episodes, args.steps, args.margin, cell
            )
            path.write_text(json.dumps(cell, indent=2), encoding="utf-8")
            print(summarize(cell), flush=True)


if __name__ == "__main__":
    main()
