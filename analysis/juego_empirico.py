"""Juego empírico: dinámica del replicador sobre la matriz de cruce de políticas entrenadas.

Lee el JSON de train.cruce (estrategias fijas y redes, retorno de fila contra columna) y aplica
la misma dinámica del replicador que se usó con las estrategias fijas, pero con las redes como
estrategias más. Reporta qué políticas son estables y qué fracción de poblaciones iniciales
termina en cada una. No ejecuta el simulador.

Uso: python -m analysis.juego_empirico cruce.json salida.png
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

from analysis.phase_diagram import regime_cell


def policy_name(spec: str) -> str:
    """Nombre corto: 'fijo:solo' es 'solo'; una red toma el número de seed de su ruta."""
    if spec.startswith("fijo:"):
        return spec.split(":", 1)[1]
    seed = re.search(r"[/\\](\d+)[/\\]checkpoints", spec)
    return f"red_s{seed.group(1)}" if seed else "red"


def load_cross(path: Path) -> tuple[np.ndarray, list[str]]:
    """Matriz de retornos y nombres de las políticas de un cruce guardado."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return np.asarray(data["retorno"], dtype=float), [policy_name(s) for s in data["politicas"]]


def analyze(path: Path) -> dict:
    """Régimen del replicador sobre el juego empírico de un cruce."""
    matrix, names = load_cross(path)
    cell = regime_cell({"matrix": matrix.tolist(), "modes": names})
    return {"matrix": matrix, "names": names, **cell}


def plot_game(result: dict, out: Path) -> None:
    """Matriz de pagos a la izquierda y fracción de poblaciones que termina en cada política."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names, matrix = result["names"], result["matrix"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 2]})
    image = left.imshow(matrix, cmap="viridis")
    for i in range(len(names)):
        for j in range(len(names)):
            left.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=6, color="w")
    left.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=7)
    left.set_yticks(range(len(names)), names, fontsize=7)
    left.set_title("retorno de la fila contra la columna", fontsize=9)
    fig.colorbar(image, ax=left, shrink=0.8)
    right.barh(names, result["shares"])
    stable = set(result["stable_vertices"])
    for tick in right.get_yticklabels():
        tick.set_fontweight("bold" if tick.get_text() in stable else "normal")
    right.set_xlabel("fracción de poblaciones iniciales")
    right.set_title("cuenca de atracción (negrita: estable)", fontsize=9)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    """Imprime el resumen del juego empírico y, si se pide, lo dibuja."""
    result = analyze(Path(sys.argv[1]))
    print("estables:", ", ".join(result["stable_vertices"]) or "ninguna")
    for name, share in sorted(
        zip(result["names"], result["shares"], strict=True), key=lambda p: -p[1]
    ):
        print(f"  {name:12s} {share:.2f}")
    if len(sys.argv) > 2:
        plot_game(result, Path(sys.argv[2]))


if __name__ == "__main__":
    main()
