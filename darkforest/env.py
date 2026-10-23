"""POSG Dark Forest en JAX. API funcional al estilo JaxMARL: reset/step puros.

Todas las funciones son jit/vmap-compatibles: sin estado global, formas estáticas.
"""

from __future__ import annotations

from typing import NamedTuple

import jax

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

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.n_agents = cfg.n_civs
        self.n_actions = len(ACTIONS)
        # `transmitir(k)` y `atacar(j)` son parametrizadas: se aplanan a acciones discretas.
        self.n_discrete = (
            2  # crecer, ocultarse
            + 1  # observar
            + cfg.n_symbols  # transmitir(k)
            + 1  # compartir
            + cfg.max_detections  # atacar(j)
            + 1  # refugiarse
        )

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

    def decode_action(self, action: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Devuelve (tipo de acción en ACTIONS, argumento k o j)."""
        raise NotImplementedError


def reward_fn(resources_t: jax.Array, resources_t1: jax.Array, alive_t: jax.Array) -> jax.Array:
    """Única recompensa: log(1 + r_{t+1}) - log(1 + r_t); 0 si ya estaba eliminada."""
    raise NotImplementedError
