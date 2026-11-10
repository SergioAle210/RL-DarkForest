"""Barrido λ por c con redes entrenadas: retorno, dependencia del canal y variación entre seeds.

Lee los JSON de train.resumen_corrida (ft_<celda>/seed<N>.json) y los de train.cruce
(cruce_ft_<celda>.json) de la corrida final y dibuja tres mapas: retorno greedy de la red, caída
de retorno al apagar el canal y retorno de la red como fracción de la pareja coop_real.

Uso: python -m analysis.barrido_entrenado kout8/resumenes docs/assets/barrido_entrenado.png
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np


def load_cells(directory: Path) -> dict[tuple[float, float], dict]:
    """Por celda (λ, c): retornos por seed, caída por apagar el canal y retornos del cruce."""
    cells = {}
    for folder in sorted(Path(directory).glob("ft_*")):
        match = re.fullmatch(r"ft_l(.+)_c(.+)", folder.name)
        lam, speed = (float(g.replace("p", ".")) for g in match.groups())
        ret, drop = [], []
        for path in sorted(folder.glob("seed*.json")):
            greedy = json.loads(path.read_text(encoding="utf-8"))["greedy"]
            ret.append(greedy["evaluacion"]["mean_return"])
            drop.append(
                greedy["evaluacion"]["mean_return"] - greedy["canal_apagado"]["mean_return"]
            )
        cross = json.loads((Path(directory) / f"cruce_{folder.name}.json").read_text("utf-8"))
        matrix = np.asarray(cross["retorno"])
        nets = matrix[5:, 5:]
        cells[(lam, speed)] = {
            "return": np.array(ret),
            "drop": np.array(drop),
            "net_self": float(np.mean([nets[i, i] for i in range(len(nets))])),
            "coop_pair": float(matrix[0, 0]),
            "solo_pair": float(matrix[2, 2]),
        }
    return cells


def plot_sweep(cells: dict, out: Path) -> None:
    """Tres mapas con la media entre seeds y, entre corchetes, el rango."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lams = sorted({k[0] for k in cells})
    speeds = sorted({k[1] for k in cells})
    panels = [
        ("retorno greedy de la red", lambda c: c["return"], "viridis"),
        ("caída al apagar el canal", lambda c: c["drop"], "magma"),
        ("red / pareja coop_real", lambda c: np.array([c["net_self"] / c["coop_pair"]]), "cividis"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (title, getter, cmap) in zip(axes, panels, strict=True):
        grid = np.zeros((len(lams), len(speeds)))
        labels = {}
        for (lam, speed), cell in cells.items():
            values = getter(cell)
            y, x = lams.index(lam), speeds.index(speed)
            grid[y, x] = values.mean()
            spread = f"{chr(10)}[{values.min():.2f}, {values.max():.2f}]" if values.size > 1 else ""
            labels[(y, x)] = f"{values.mean():.2f}{spread}"
        low, high = grid.min(), grid.max()
        for (y, x), text in labels.items():
            bright = (grid[y, x] - low) / max(high - low, 1e-9) > 0.6
            ax.text(x, y, text, ha="center", va="center", fontsize=7, color="k" if bright else "w")
        image = ax.imshow(grid, origin="lower", cmap=cmap)
        ax.set_xticks(range(len(speeds)), [f"{v:g}" for v in speeds])
        ax.set_yticks(range(len(lams)), [f"{v:g}" for v in lams])
        ax.set_xlabel("velocidad de la señal c")
        ax.set_ylabel("letalidad λ")
        ax.set_title(title, fontsize=9)
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    """Dibuja el barrido y imprime la variación entre seeds de cada celda."""
    cells = load_cells(Path(sys.argv[1]))
    for (lam, speed), cell in sorted(cells.items()):
        print(
            f"lam={lam:g} c={speed:g}: caída por seed "
            + " ".join(f"{v:+.2f}" for v in cell["drop"])
            + f" | red {cell['net_self']:+.2f} contra coop_real {cell['coop_pair']:+.2f}"
        )
    if len(sys.argv) > 2:
        plot_sweep(cells, Path(sys.argv[2]))


if __name__ == "__main__":
    main()
