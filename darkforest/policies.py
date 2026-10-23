"""Interfaz común de política. Redes, arquetipos y agente aleatorio la implementan.

Esto desacopla el entorno, el entrenamiento y el análisis: cualquier agente
(aprendido, heurístico, congelado de la población o externo) es intercambiable.
"""

from __future__ import annotations

from typing import Any, Protocol

import jax

from darkforest.types import Obs

Carry = Any  # estado recurrente (GRU) o None para políticas sin memoria


class Policy(Protocol):
    """Política que actúa por lotes sobre las N civilizaciones de un entorno."""

    def init_carry(self, n_agents: int) -> Carry: ...

    def act(self, key: jax.Array, carry: Carry, obs: Obs) -> tuple[Carry, jax.Array]:
        """Devuelve (nuevo carry, acciones (N,) enteras)."""
        ...


class RandomPolicy:
    """Modelo nulo: acción uniforme sobre el espacio discreto."""

    def __init__(self, n_discrete: int):
        self.n_discrete = n_discrete

    def init_carry(self, n_agents: int) -> Carry:
        return None

    def act(self, key: jax.Array, carry: Carry, obs: Obs) -> tuple[Carry, jax.Array]:
        raise NotImplementedError
