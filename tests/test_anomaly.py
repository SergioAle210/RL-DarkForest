"""Identificabilidad y tiempo de la anomalía."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from darkforest.anomaly import burst_due, measure, projection, sample_theta
from darkforest.config import AnomalyConfig, EnvConfig
from darkforest.env import DarkForestEnv


def test_two_projections_reconstruct_but_one_does_not():
    """Dos direcciones independientes recuperan fase, periodo y centro sin ruido."""
    theta = jnp.array([3.0, 20.0, 0.3, 0.7, 0.25])
    changed = theta.at[1].set(30).at[3].set(0.9)
    np.testing.assert_allclose(projection(theta, 0.0), projection(changed, 0.0))
    angles = jnp.array([0.0, 1.2])
    values = projection(theta, angles)
    matrix = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
    np.testing.assert_allclose(jnp.linalg.solve(matrix, values[:, 0]), theta[:2], atol=1e-5)
    np.testing.assert_allclose(jnp.linalg.solve(matrix, values[:, 1]), theta[2:4], atol=1e-5)


def test_phase_period_and_reproducible_measurement():
    """Comprueba la periodicidad y la reproducción del ruido con una clave fija."""
    key = jax.random.PRNGKey(0)
    theta = sample_theta(key, AnomalyConfig(period=7))
    due = jax.jit(jax.vmap(burst_due, (0, None)))(jnp.arange(1, 29), theta)
    assert int(due.sum()) == 4
    np.testing.assert_array_equal(measure(key, theta, 0.3, 0.1), measure(key, theta, 0.3, 0.1))


def test_fixed_period_keeps_previous_sampling():
    """Con period_spread = 0 la muestra coincide con la del muestreo original."""
    key = jax.random.PRNGKey(3)
    phase_key, center_key = jax.random.split(key)
    expected_phase = jax.random.randint(phase_key, (), 0, 20)
    expected_center = jax.random.uniform(center_key, (2,))
    theta = sample_theta(key, AnomalyConfig())
    np.testing.assert_allclose(theta[:2], [expected_phase, 20])
    np.testing.assert_allclose(theta[2:4], expected_center)


def test_random_period_varies_per_episode_within_range():
    """El periodo se sortea en [period - k, period + k] y la fase cabe en él."""
    cfg = AnomalyConfig(period=20, period_spread=8)
    thetas = jax.vmap(lambda key: sample_theta(key, cfg))(
        jax.random.split(jax.random.PRNGKey(0), 500)
    )
    periods, phases = np.asarray(thetas[:, 1]), np.asarray(thetas[:, 0])
    assert periods.min() == 12 and periods.max() == 28
    assert len(np.unique(periods)) == 17
    assert (phases >= 0).all() and (phases < periods).all()
    fixed = jax.vmap(lambda key: sample_theta(key, AnomalyConfig()))(
        jax.random.split(jax.random.PRNGKey(0), 50)
    )
    np.testing.assert_array_equal(fixed[:, 1], 20)


def test_random_period_bursts_follow_each_episode_period():
    """Cada episodio recibe estallidos separados exactamente por su propio periodo."""
    env = DarkForestEnv(EnvConfig(n_civs=2, anomaly=AnomalyConfig(period_spread=8)))
    _, states = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(1), 6))
    steps = jnp.arange(1, 121)
    due = jax.vmap(lambda theta: jax.vmap(burst_due, (0, None))(steps, theta))(states.theta)
    for episode in range(6):
        times = np.flatnonzero(np.asarray(due[episode])) + 1
        assert (np.diff(times) == int(states.theta[episode, 1])).all()
    assert len({int(period) for period in states.theta[:, 1]}) > 1


@pytest.mark.parametrize("spread", (-1, 20, 25))
def test_period_spread_must_be_below_period(spread):
    """El rango debe dejar periodos positivos."""
    with pytest.raises(ValueError):
        AnomalyConfig(period=20, period_spread=spread)


@pytest.mark.parametrize("spread", (True, 1.5, "2"))
def test_period_spread_requires_integer(spread):
    """Un booleano o un decimal no es un rango válido."""
    with pytest.raises(TypeError):
        AnomalyConfig(period_spread=spread)
