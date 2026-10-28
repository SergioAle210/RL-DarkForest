"""Diagrama de fases λ × c a partir de resultados del barrido."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analysis.replicator import (
    basin_shares,
    critical_mass_interval,
    fixed_points,
    is_stable,
)


def load_payoff_grid(directory: Path) -> dict[tuple[float, float], dict]:
    """Lee los archivos de train.pagos_rejilla; la clave de cada celda es (λ, c)."""
    grid = {}
    for path in sorted(Path(directory).glob("pagos_lam*_c*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        grid[(float(data["lethality"]), float(data["signal_speed"]))] = data
    return grid


def regime_cell(data: dict, seed: int = 0) -> dict:
    """Régimen de una celda: cuenca dominante del replicador, estabilidad y biestabilidad."""
    A, modes = np.asarray(data["matrix"], dtype=float), list(data["modes"])
    shares, mixture = basin_shares(A, seed=seed)
    identity = np.eye(len(modes))
    stable = [modes[i] for i in range(len(modes)) if is_stable(A, identity[i])]
    interior = [
        point for point in fixed_points(A) if (point > 1e-9).sum() > 1 and is_stable(A, point)
    ]
    return {
        "modes": modes,
        "shares": shares,
        "mixture": mixture,
        "dominant": modes[int(np.argmax(shares))],
        "stable_vertices": stable,
        "stable_mixtures": len(interior),
        "bistable": len(stable) + len(interior) >= 2,
    }


def regime_map(grid: dict[tuple[float, float], dict]) -> dict[tuple[float, float], dict]:
    """Régimen de cada celda (λ, c) de una rejilla de matrices de pagos."""
    return {cell: regime_cell(data) for cell, data in sorted(grid.items())}


# pares (invasor, residente) cuya masa crítica se reporta, con su etiqueta
CRITICAL_PAIRS = {
    "cooperadores necesarios frente a solos": ("coop_real", "solo"),
    "atacantes necesarios frente a solos": ("ataque_solo", "solo"),
    "atacantes necesarios frente a cooperadores": ("ataque_solo", "coop_real"),
}


def critical_mass_table(
    grid: dict[tuple[float, float], dict],
) -> dict[tuple[float, float], dict[str, tuple[float, float, float]]]:
    """Masas críticas (mediana, p5, p95) de cada par en cada celda (λ, c)."""
    table = {}
    for cell, data in sorted(grid.items()):
        A, se, modes = np.asarray(data["matrix"]), np.asarray(data["se"]), list(data["modes"])
        table[cell] = {
            label: critical_mass_interval(A, se, modes.index(invader), modes.index(resident))
            for label, (invader, resident) in CRITICAL_PAIRS.items()
            if invader in modes and resident in modes
        }
    return table


def plot_critical_masses(table: dict, out: Path) -> None:
    """Tres mapas λ por c con la masa crítica de cada par y su rango del 5% al 95%."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lethalities = sorted({lam for lam, _ in table})
    speeds = sorted({speed for _, speed in table})
    labels = [label for label in CRITICAL_PAIRS if all(label in cell for cell in table.values())]
    fig, axes = plt.subplots(1, len(labels), figsize=(4.6 * len(labels), 3.6), squeeze=False)
    axes = axes[0]
    for ax, label in zip(axes, labels, strict=True):
        values = np.full((len(lethalities), len(speeds)), np.nan)
        for (lam, speed), cell in table.items():
            median, low, high = cell[label]
            y, x = lethalities.index(lam), speeds.index(speed)
            values[y, x] = median
            ax.text(
                x,
                y,
                f"{median:.2f}" + chr(10) + f"[{low:.2f}, {high:.2f}]",
                ha="center",
                va="center",
                fontsize=7,
            )
        image = ax.imshow(values, origin="lower", vmin=0, vmax=1, cmap="viridis_r")
        ax.set_xticks(range(len(speeds)), [f"{v:g}" for v in speeds])
        ax.set_yticks(range(len(lethalities)), [f"{v:g}" for v in lethalities])
        ax.set_xlabel("velocidad de la señal c")
        ax.set_ylabel("letalidad λ")
        ax.set_title(label, fontsize=9)
    fig.colorbar(image, ax=axes, shrink=0.8, label="fracción de la población")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def aggregate(results_dir: Path) -> dict:
    """Lee runs/*/*/results.json y agrega por celda (λ, c): régimen modal, dispersión entre
    seeds (para H5)."""
    raise NotImplementedError


def critical_lambda(grid: dict) -> np.ndarray:
    """λ*(c): frontera donde emerge la comunicación (H2)."""
    raise NotImplementedError


def plot_phase_diagram(grid: dict, out: Path) -> None:
    raise NotImplementedError


def plot_bistability(grid: dict, out: Path) -> None:
    """Distribución de resultados por seed cerca de la frontera (H5)."""
    raise NotImplementedError


def coalition_graph(traj) -> object:
    """Grafo de quién responde a quién + detección de comunidades (H4)."""
    raise NotImplementedError
