"""Diagrama de fases λ × c a partir de resultados del barrido."""

from __future__ import annotations

from pathlib import Path

import numpy as np


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
