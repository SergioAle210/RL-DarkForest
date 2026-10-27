"""Invariantes del entorno. Se activan al implementar env.py (quitar xfail)."""

import pytest

jax = pytest.importorskip("jax")

from darkforest.config import EnvConfig  # noqa: E402
from darkforest.env import DarkForestEnv  # noqa: E402

pytestmark = pytest.mark.xfail(raises=NotImplementedError, reason="esqueleto", strict=False)


@pytest.fixture
def env():
    return DarkForestEnv(EnvConfig(n_civs=2))


def test_reset_shapes(env):
    obs, state = env.reset(jax.random.PRNGKey(0))
    assert obs.own.shape[0] == 2


def test_step_is_jittable(env):
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    jax.jit(env.step)(key, state, jax.numpy.zeros(2, dtype=int))


def test_dead_agents_get_no_reward(env):
    """Reserva la prueba de agentes eliminados hasta implementar las transiciones."""
    pytest.skip("Pendiente: transiciones y recompensa de agentes eliminados")


def test_resources_nonnegative(env):
    """Reserva el invariante de recursos hasta implementar las transiciones."""
    pytest.skip("Pendiente: consumo y actualización de recursos")


def test_signals_travel_at_c(env):
    """Reserva la comprobación física hasta implementar la propagación."""
    pytest.skip("Pendiente: propagación de señales")


def test_reward_is_only_log_growth():
    """Auditoría de no imposición: reward == log(1+r') - log(1+r)."""
    from darkforest.env import reward_fn

    r = jax.numpy.array([1.0, 2.0])
    out = reward_fn(r, r * 1.1, jax.numpy.array([True, True]))
    assert jax.numpy.allclose(out, jax.numpy.log1p(r * 1.1) - jax.numpy.log1p(r))


def test_h1_sanity_placeholder():
    """λ = 0: silencio o ataque preventivo dominan. Implementar con archetypes."""
    raise NotImplementedError
