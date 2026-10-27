"""Cuatro arquetipos heurísticos. Sirven de modelo nulo, de oponentes de evaluación
y de estrategias para la dinámica del replicador."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from darkforest.config import EnvConfig
from darkforest.policies import Carry, Policy
from darkforest.types import Obs


class _Archetype:
    """Configuración común de las heurísticas."""

    def __init__(self, cfg: EnvConfig | None = None) -> None:
        """Conserva las dimensiones del catálogo sin acceder al estado oculto."""
        self.cfg = cfg if cfg is not None else EnvConfig()


class AlwaysHidden(_Archetype):
    """Siempre oculto: crece solo cuando es seguro, nunca transmite."""

    def init_carry(self, n_agents: int) -> Carry:
        """Esta heurística no necesita memoria."""
        return None

    def act(self, key: jax.Array, carry: Carry, obs: Obs) -> tuple[Carry, jax.Array]:
        """Crece sin contactos conocidos y se oculta cuando detecta a alguien."""
        return carry, jnp.where(obs.alive & ~jnp.any(obs.det_mask, axis=-1), 0, 1)


class AlwaysTransmit(_Archetype):
    """Siempre transmite y comparte mediciones."""

    def init_carry(self, n_agents: int) -> Carry:
        """Inicia el ciclo observar, compartir y transmitir."""
        return jnp.zeros(n_agents, dtype=jnp.int32)

    def act(self, key: jax.Array, carry: Carry, obs: Obs) -> tuple[Carry, jax.Array]:
        """Alterna mediciones y comunicación sin usar información oculta."""
        phase = carry % 3
        actions = jnp.where(phase == 0, 2, jnp.where(phase == 1, 3 + self.cfg.n_symbols, 3))
        return carry + obs.alive.astype(jnp.int32), jnp.where(obs.alive, actions, 1)


class Reciprocal(_Archetype):
    """Responde con una transmisión a quien lo detectó con una; si no, se oculta."""

    def init_carry(self, n_agents: int) -> Carry:
        """Esta heurística usa los mensajes persistentes de la observación."""
        return None

    def act(self, key: jax.Array, carry: Carry, obs: Obs) -> tuple[Carry, jax.Array]:
        """Responde al observar un mensaje y se oculta en ausencia de mensajes."""
        received = jnp.any(obs.det_mask & (obs.detections[..., 3] >= 0), axis=-1)
        return carry, jnp.where(obs.alive & received, 3, 1)


class PreemptiveAttacker(_Archetype):
    """Ataca a toda civilización detectada."""

    def init_carry(self, n_agents: int) -> Carry:
        """Esta heurística no necesita memoria."""
        return None

    def act(self, key: jax.Array, carry: Carry, obs: Obs) -> tuple[Carry, jax.Array]:
        """Ataca el primer contacto disponible y crece mientras no detecta otros."""
        target = jnp.argmax(obs.det_mask, axis=-1)
        actions = jnp.where(jnp.any(obs.det_mask, axis=-1), 4 + self.cfg.n_symbols + target, 0)
        return carry, jnp.where(obs.alive, actions, 1)


ARCHETYPES: dict[str, type[Policy]] = {
    "hidden": AlwaysHidden,
    "transmit": AlwaysTransmit,
    "reciprocal": Reciprocal,
    "attacker": PreemptiveAttacker,
}
