"""Población de políticas pasadas para autojuego ficticio."""

from __future__ import annotations

from typing import Any

import jax

Params = Any


class Population:
    """Guarda copias congeladas de parámetros y asigna políticas a civilizaciones."""

    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.snapshots: list[Params] = []

    def add(self, params: Params) -> None:
        raise NotImplementedError

    def assign(self, key: jax.Array, n_envs: int, n_agents: int, rho: float) -> jax.Array:
        """(n_envs, n_agents) índice de política: -1 = actual, >=0 = snapshot.
        Cada civilización toma un snapshot con prob. rho."""
        raise NotImplementedError
