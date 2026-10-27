"""Formas, memoria y aislamiento de observaciones de la GRU."""

import jax
import jax.numpy as jnp
import numpy as np

from darkforest.config import EnvConfig
from darkforest.env import DarkForestEnv
from darkforest.networks import ActorCriticGRU, NetworkPolicy, flatten_obs


def test_shared_model_and_masked_observations():
    """La misma red admite dos y cuatro agentes sin leer el estado completo."""
    key = jax.random.PRNGKey(0)
    model = ActorCriticGRU(13, 8)
    obs, _ = DarkForestEnv(EnvConfig(n_civs=2)).reset(key)
    masked = obs.replace(detections=jnp.ones_like(obs.detections) * 999)
    np.testing.assert_array_equal(flatten_obs(obs), flatten_obs(masked))
    params = model.init(key, model.initial_carry(2, 8), flatten_obs(obs), obs.agent_id)
    policy = NetworkPolicy(model, params)
    for n in (2, 4):
        observation, _ = DarkForestEnv(EnvConfig(n_civs=n)).reset(key)
        carry, actions = jax.jit(policy.act)(key, policy.init_carry(n), observation)
        assert carry.shape == (n, 8)
        assert actions.shape == (n,)
        assert np.all((actions >= 0) & (actions < 13))
        dead = observation.replace(alive=jnp.zeros(n, dtype=bool))
        carry, actions = policy.act(key, carry, dead)
        assert not carry.any()
        np.testing.assert_array_equal(actions, np.ones(n))
