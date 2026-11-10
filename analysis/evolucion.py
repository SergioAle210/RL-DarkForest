"""Gráfica de cómo evoluciona una red durante el entrenamiento, a partir de sus snapshots.

Lee los JSON de train.resumen_corrida de una seed (seed0_iter_*.json) y dibuja el retorno
muestreado y en greedy, la probabilidad de algún ataque tras el contacto y la fracción de
acciones de transmitir o compartir.

Uso: python -m analysis.evolucion kout3/resumenes/evolucion docs/assets/evolucion_aprendizaje.png
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def load_series(directory: Path) -> list[dict]:
    """Una fila por snapshot, ordenada por iteración."""
    rows = []
    for path in Path(directory).glob("seed0_iter_*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        sampled, greedy = data["muestreo"], data["greedy"]
        rows.append(
            {
                "iteration": int(re.search(r"iter_(\d+)", path.name).group(1)),
                "sampled": sampled["evaluacion"]["mean_return"],
                "greedy": greedy["evaluacion"]["mean_return"],
                "attack": sampled["diagnostico"]["p_any_attack_given_contact"],
                "talk": sampled["evaluacion"]["action_transmitir"]
                + sampled["evaluacion"]["action_compartir"],
            }
        )
    return sorted(rows, key=lambda row: row["iteration"])


def plot_evolution(rows: list[dict], out: Path, references: dict[str, float] | None = None) -> None:
    """Dos paneles: retorno con referencias y conducta de ataque y comunicación."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [row["iteration"] for row in rows]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.6))
    left.plot(x, [row["greedy"] for row in rows], marker="o", label="red greedy")
    left.plot(x, [row["sampled"] for row in rows], marker="o", label="red muestreada")
    for name, value in (references or {}).items():
        left.axhline(value, ls="--", lw=0.8, color="gray")
        left.text(x[0], value + 0.1, name, fontsize=7, color="gray")
    left.set_xlabel("iteración de entrenamiento")
    left.set_ylabel("retorno por civilización")
    left.legend(fontsize=8)
    right.plot(x, [row["attack"] for row in rows], marker="o", color="tab:red")
    right.set_ylabel("P(algún ataque | contacto), muestreada", color="tab:red")
    right.set_xlabel("iteración de entrenamiento")
    twin = right.twinx()
    twin.plot(x, [100 * row["talk"] for row in rows], marker="s", color="tab:blue")
    twin.set_ylabel("% de acciones transmitir o compartir", color="tab:blue")
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    """Dibuja la evolución de la seed 0 de una corrida con snapshots."""
    rows = load_series(Path(sys.argv[1]))
    references = {"ignorar": 2.77, "solo": 3.71, "coop_real": 6.94, "oráculo": 8.71}
    plot_evolution(rows, Path(sys.argv[2]), references)


if __name__ == "__main__":
    main()
