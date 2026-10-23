"""Actor-crítico recurrente (GRU) con parámetros compartidos e identificador de agente."""

from __future__ import annotations

import flax.linen as nn
import jax


class ActorCriticGRU(nn.Module):
    n_actions: int
    hidden: int = 128

    @nn.compact
    def __call__(self, carry: jax.Array, obs_flat: jax.Array, agent_id: jax.Array):
        """Devuelve (nuevo carry, logits (n_actions,), valor ())."""
        raise NotImplementedError

    @staticmethod
    def initial_carry(batch: int, hidden: int) -> jax.Array:
        raise NotImplementedError


def flatten_obs(obs) -> jax.Array:
    """Aplana `Obs` a un vector por agente."""
    raise NotImplementedError
