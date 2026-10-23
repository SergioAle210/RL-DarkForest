"""Cuatro arquetipos heurísticos. Sirven de modelo nulo, de oponentes de evaluación
y de estrategias para la dinámica del replicador."""

from __future__ import annotations

import jax

from darkforest.policies import Carry, Policy
from darkforest.types import Obs


class AlwaysHidden:
    """Siempre oculto: crece solo cuando es seguro, nunca transmite."""

    def init_carry(self, n_agents: int) -> Carry:
        return None

    def act(self, key: jax.Array, carry: Carry, obs: Obs) -> tuple[Carry, jax.Array]:
        raise NotImplementedError


class AlwaysTransmit:
    """Siempre transmite y comparte mediciones."""

    def init_carry(self, n_agents: int) -> Carry:
        return None

    def act(self, key: jax.Array, carry: Carry, obs: Obs) -> tuple[Carry, jax.Array]:
        raise NotImplementedError


class Reciprocal:
    """Responde con una transmisión a quien lo detectó con una; si no, se oculta."""

    def init_carry(self, n_agents: int) -> Carry:
        raise NotImplementedError

    def act(self, key: jax.Array, carry: Carry, obs: Obs) -> tuple[Carry, jax.Array]:
        raise NotImplementedError


class PreemptiveAttacker:
    """Ataca a toda civilización detectada."""

    def init_carry(self, n_agents: int) -> Carry:
        return None

    def act(self, key: jax.Array, carry: Carry, obs: Obs) -> tuple[Carry, jax.Array]:
        raise NotImplementedError


ARCHETYPES: dict[str, type[Policy]] = {
    "hidden": AlwaysHidden,
    "transmit": AlwaysTransmit,
    "reciprocal": Reciprocal,
    "attacker": PreemptiveAttacker,
}
