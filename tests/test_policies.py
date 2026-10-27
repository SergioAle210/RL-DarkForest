import jax
import jax.numpy as jnp
import numpy as np
import pytest

from darkforest.archetypes import ARCHETYPES
from darkforest.config import EnvConfig
from darkforest.env import DarkForestEnv
from darkforest.policies import RandomPolicy


def test_archetypes_expose_policy_interface():
    """Mantiene el protocolo común declarado en el esqueleto original."""
    for name, cls in ARCHETYPES.items():
        assert hasattr(cls, "init_carry"), name
        assert hasattr(cls, "act"), name


@pytest.mark.parametrize("name", ["random", *ARCHETYPES])
def test_policies_work_with_configured_catalog_and_dead_agents(name):
    """Cada política produce acciones válidas bajo jit con un catálogo diferente."""
    cfg = EnvConfig(n_civs=4, n_symbols=2)
    key = jax.random.PRNGKey(0)
    obs, _ = DarkForestEnv(cfg).reset(key)
    policy = RandomPolicy(cfg.n_discrete) if name == "random" else ARCHETYPES[name](cfg)
    carry = policy.init_carry(4)
    act = jax.jit(policy.act)
    for _ in range(4):
        key, action_key = jax.random.split(key)
        carry, actions = act(action_key, carry, obs)
        assert actions.shape == (4,)
        assert np.all((actions >= 0) & (actions < cfg.n_discrete))
    obs = obs.replace(alive=jnp.zeros(4, dtype=bool))
    _, actions = act(key, carry, obs)
    np.testing.assert_array_equal(actions, [1, 1, 1, 1])
