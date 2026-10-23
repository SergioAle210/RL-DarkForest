"""Línea base analítica: dinámica del replicador sobre cuatro arquetipos."""

from __future__ import annotations

import numpy as np

from darkforest.archetypes import ARCHETYPES


def payoff_matrix(env_cfg, n_episodes: int, seed: int) -> np.ndarray:
    """Matriz de pagos A[i, j] medida en el mismo simulador (recompensa media del arquetipo i
    contra j)."""
    raise NotImplementedError


def replicator_flow(A: np.ndarray, x0: np.ndarray, steps: int, dt: float) -> np.ndarray:
    """Integra dx_i/dt = x_i ((Ax)_i - x·Ax). Devuelve (steps, K)."""
    raise NotImplementedError


def fixed_points(A: np.ndarray) -> np.ndarray:
    raise NotImplementedError


ARCHETYPE_NAMES = tuple(ARCHETYPES)
