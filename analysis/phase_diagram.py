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
    """Mapa λ por c del régimen dominante del replicador y de los vértices estables.

    El color es la estrategia cuya cuenca de atracción es mayor con mezclas iniciales
    aleatorias; el texto indica cuántas cuentas hay entre los vértices estables, porque más de
    uno significa biestabilidad. Con estrategias fijas, no con redes entrenadas.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    regimes = regime_map(grid)
    lethalities = sorted({lam for lam, _ in regimes})
    speeds = sorted({speed for _, speed in regimes})
    modes = sorted({cell["dominant"] for cell in regimes.values()})
    palette = dict(zip(modes, plt.get_cmap("Set2").colors, strict=False))
    fig, ax = plt.subplots(figsize=(1.9 * len(speeds) + 2.5, 1.5 * len(lethalities) + 1.5))
    for (lam, speed), cell in regimes.items():
        y, x = lethalities.index(lam), speeds.index(speed)
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color=palette[cell["dominant"]]))
        stable = ", ".join(cell["stable_vertices"]) or "ninguno"
        mark = "biestable" if cell["bistable"] else "un equilibrio"
        ax.text(x, y, f"{stable}" + chr(10) + mark, ha="center", va="center", fontsize=7)
    ax.set_xlim(-0.5, len(speeds) - 0.5)
    ax.set_ylim(-0.5, len(lethalities) - 0.5)
    ax.set_xticks(range(len(speeds)), [f"{v:g}" for v in speeds])
    ax.set_yticks(range(len(lethalities)), [f"{v:g}" for v in lethalities])
    ax.set_xlabel("velocidad de la señal c")
    ax.set_ylabel("letalidad λ")
    ax.set_title("Régimen dominante (color) y vértices estables (texto)", fontsize=9)
    ax.legend(
        handles=[Patch(color=palette[m], label=m) for m in modes],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_bistability(grid: dict, out: Path) -> None:
    """Distribución de resultados por seed cerca de la frontera (H5)."""
    raise NotImplementedError


def coalition_graph(traj, min_events: int = 3, fuente: str = "mensajes") -> dict:
    """Grafo de quién recibe mensajes de quién y comunidades entre civilizaciones (H4).

    Con fuente "mensajes" cuenta los mensajes explícitos recibidos (compartir o símbolo): cada vez
    que sube el sello del último mensaje de un emisor a un receptor en message_t. Con fuente
    "senales" cuenta cualquier señal recibida (info_receptions), que incluye las emisiones
    ambientales y por eso une a casi todas. Dos civilizaciones quedan unidas si se recibieron en
    ambos sentidos y suman al menos min_events; las comunidades son los componentes conexos.
    """
    import numpy as np

    if fuente == "mensajes":
        sello = np.asarray(traj["message_t"])
        counts = (sello[1:] > sello[:-1]).sum(axis=0).astype(int)
    else:
        counts = np.asarray(traj["info_receptions"]).sum(axis=0).astype(int)
    n = counts.shape[0]
    linked = (counts > 0) & (counts.T > 0) & ((counts + counts.T) >= min_events)
    linked &= ~np.eye(n, dtype=bool)
    seen, communities = set(), []
    for start in range(n):
        if start in seen:
            continue
        group, stack = set(), [start]
        while stack:
            node = stack.pop()
            if node in group:
                continue
            group.add(node)
            stack += [int(j) for j in np.flatnonzero(linked[node]) if j not in group]
        seen |= group
        communities.append(sorted(group))
    inside = sum(counts[i, j] for g in communities for i in g for j in g if i != j)
    total = int(counts.sum() - np.trace(counts))
    return {
        "counts": counts,
        "communities": communities,
        "inside_fraction": float(inside / total) if total else 0.0,
    }
