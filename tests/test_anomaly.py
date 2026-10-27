"""Identificabilidad y tiempo de la anomalía."""

import jax
import jax.numpy as jnp
import numpy as np

from darkforest.anomaly import burst_due, measure, projection, sample_theta
from darkforest.config import AnomalyConfig


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
