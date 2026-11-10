"""Población de políticas pasadas para autojuego ficticio."""

from __future__ import annotations

from typing import Any

import jax

Params = Any


class Population:
    """Guarda copias congeladas de parámetros; al llenarse descarta la más antigua."""

    def __init__(self, max_size: int = 20):
        self.max_size = max_size
        self.snapshots: list[Params] = []

    def add(self, params: Params) -> None:
        """Agrega una copia de los parámetros actuales."""
        self.snapshots.append(jax.tree.map(lambda x: x, params))
        if len(self.snapshots) > self.max_size:
            self.snapshots.pop(0)

    def sample(self, key: jax.Array) -> Params:
        """Elige un snapshot uniformemente al azar."""
        index = int(jax.random.randint(key, (), 0, len(self.snapshots)))
        return self.snapshots[index]
