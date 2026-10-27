"""POSG Dark Forest en JAX. API funcional al estilo JaxMARL: reset/step puros.

Todas las funciones son jit/vmap-compatibles: sin estado global, formas estáticas.
"""

from __future__ import annotations

from numbers import Integral
from typing import NamedTuple

import jax
import jax.numpy as jnp

from darkforest.config import ACTIONS, EnvConfig
from darkforest.types import EnvState, Obs, StepInfo


class StepOut(NamedTuple):
    obs: Obs
    state: EnvState
    reward: jax.Array  # (N,)
    done: jax.Array  # (N,) eliminada o fin del episodio
    info: StepInfo


class DarkForestEnv:
    """Entorno multiagente. Los agentes comparten espacio de acciones y de observación."""

    def __init__(self, cfg: EnvConfig) -> None:
        """Conserva la configuración y las dimensiones del espacio de acciones."""
        self.cfg = cfg
        self.n_agents = cfg.n_civs
        self.n_actions = len(ACTIONS)
        self.n_discrete = cfg.n_discrete

    def reset(self, key: jax.Array) -> tuple[Obs, EnvState]:
        """Posiciones aleatorias en el toro, θ aleatorio, colas vacías."""
        raise NotImplementedError

    def step(self, key: jax.Array, state: EnvState, actions: jax.Array) -> StepOut:
        """actions: (N,) int en [0, n_discrete). Orden: acciones → señales/proyectiles
        → detección → anomalía → recompensa."""
        raise NotImplementedError

    def get_obs(self, key: jax.Array, state: EnvState) -> Obs:
        """Detección: p_det(e_j, d_ij) para anillos que alcanzaron a i."""
        raise NotImplementedError

    def decode_action(self, action: int | jax.Array) -> tuple[jax.Array, jax.Array]:
        """Decodifica un entero o array JAX; una acción inválida produce (-1, -1)."""
        if isinstance(action, Integral) and not isinstance(action, bool):
            if not 0 <= action < self.n_discrete:
                action = -1
        try:
            action = jnp.asarray(action)
        except (TypeError, ValueError) as error:
            raise TypeError("La acción debe tener un tipo entero.") from error
        if not jnp.issubdtype(action.dtype, jnp.integer):
            raise TypeError("La acción debe tener un tipo entero.")

        valid = action >= 0
        if self.n_discrete <= jnp.iinfo(action.dtype).max:
            valid = valid & (action < self.n_discrete)
        # Se valida antes de convertir para conservar enteros sin signo fuera de rango.
        integer_dtype = jnp.int64 if jax.config.x64_enabled else jnp.int32
        action = jnp.where(valid, action, 0).astype(integer_dtype)
        share = 3 + self.cfg.n_symbols
        attack = share + 1
        refuge = self.n_discrete - 1

        kind = jnp.select(
            [action < 3, action < share, action == share, action < refuge],
            [action, 3, 4, 5],
            default=6,
        )
        argument = jnp.where(
            (action >= 3) & (action < share),
            action - 3,
            jnp.where((action >= attack) & (action < refuge), action - attack, -1),
        )
        return jnp.where(valid, kind, -1), jnp.where(valid, argument, -1)


def reward_fn(resources_t: jax.Array, resources_t1: jax.Array, alive_t: jax.Array) -> jax.Array:
    """Única recompensa: log(1 + r_{t+1}) - log(1 + r_t); 0 si ya estaba eliminada."""
    raise NotImplementedError
