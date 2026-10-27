"""Actor-crítico recurrente (GRU) con parámetros compartidos e identificador de agente."""

from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from darkforest.types import Obs


class ActorCriticGRU(nn.Module):
    n_actions: int
    hidden: int = 128

    @nn.compact
    def __call__(
        self, carry: jax.Array, obs_flat: jax.Array, agent_id: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Devuelve (nuevo carry, logits (n_actions,), valor ())."""
        identity = jax.nn.one_hot(agent_id, 8)
        features = jnp.concatenate([obs_flat, identity], axis=-1)
        features = jnp.tanh(nn.Dense(self.hidden)(features))
        carry, features = nn.GRUCell(features=self.hidden)(carry, features)
        return carry, nn.Dense(self.n_actions)(features), nn.Dense(1)(features)[..., 0]

    @staticmethod
    def initial_carry(batch: int, hidden: int) -> jax.Array:
        """Crea memoria nula para un lote de civilizaciones."""
        return jnp.zeros((batch, hidden))


def flatten_obs(obs: Obs) -> jax.Array:
    """Aplana `Obs` a un vector por agente."""
    prefix = obs.alive.shape
    detections = jnp.where(obs.det_mask[..., None], obs.detections, 0)
    values = jnp.concatenate(
        [
            obs.own,
            detections.reshape((*prefix, -1)),
            obs.det_mask.astype(float),
            obs.own_meas,
            obs.since_burst[..., None],
            obs.alive[..., None],
        ],
        axis=-1,
    )
    return jnp.sign(values) * jnp.log1p(jnp.abs(values))


class NetworkPolicy:
    """Adaptador de la GRU al protocolo compartido de políticas."""

    def __init__(self, model: ActorCriticGRU, params: Any) -> None:
        """Conserva la arquitectura y sus parámetros aprendidos."""
        self.model = model
        self.params = params

    def init_carry(self, n_agents: int) -> jax.Array:
        """Reinicia la memoria al comenzar un episodio."""
        return self.model.initial_carry(n_agents, self.model.hidden)

    def act(self, key: jax.Array, carry: jax.Array, obs: Obs) -> tuple[jax.Array, jax.Array]:
        """Muestrea desde el actor y anula la memoria de agentes eliminados."""
        carry = jnp.where(obs.alive[..., None], carry, 0)
        carry, logits, _ = self.model.apply(self.params, carry, flatten_obs(obs), obs.agent_id)
        actions = jax.random.categorical(key, logits)
        return jnp.where(obs.alive[..., None], carry, 0), jnp.where(obs.alive, actions, 1)
