"""Coaliciones por tamaño de población a partir de episodios guardados (H4).

Lee las carpetas N<n>/ de train.coaliciones y, por tamaño, promedia supervivencia, mensajes por
episodio, número de comunidades y fracción de mensajes dentro de la comunidad.

Uso: python -m analysis.coaliciones salida
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from analysis.metrics import survival_rate, transmissions_per_episode
from analysis.phase_diagram import coalition_graph
from darkforest.trajectories import load_episode


def summarize_size(folder: Path, fuente: str = "mensajes") -> dict:
    """Promedios sobre los episodios de una carpeta N<n>/0/trajectories."""
    rows = []
    for path in sorted((folder / "0" / "trajectories").glob("episode_*.npz")):
        traj, _ = load_episode(path)
        graph = coalition_graph(traj, fuente=fuente)
        tx = transmissions_per_episode(traj)
        rows.append(
            {
                "survival": survival_rate(traj)["survival"],
                "messages": tx["symbols"] + tx["shared_measurements"],
                "communities": len(graph["communities"]),
                "inside": graph["inside_fraction"],
                "largest": max(len(c) for c in graph["communities"]),
            }
        )
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]} | {"n": len(rows)}


def main() -> None:
    """Imprime una fila por tamaño de población."""
    root = Path(sys.argv[1])
    fuente = sys.argv[2] if len(sys.argv) > 2 else "mensajes"
    print(f"fuente de los vínculos: {fuente}")
    print("N   episodios  superv.  mensajes  comunidades  mayor  dentro")
    for folder in sorted(root.glob("N*"), key=lambda p: int(p.name[1:])):
        r = summarize_size(folder, fuente)
        print(
            f"{folder.name[1:]:>2}  {r['n']:>8}  {r['survival']:.2f}  {r['messages']:>8.1f}  "
            f"{r['communities']:>11.1f}  {r['largest']:>5.1f}  {r['inside']:.2f}"
        )


if __name__ == "__main__":
    main()
