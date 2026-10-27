"""Anomalía periódica con parámetros ocultos θ y proyecciones ruidosas h_i(θ)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from darkforest.config import AnomalyConfig
from darkforest.geometry import distance


def sample_theta(key: jax.Array, cfg: AnomalyConfig) -> jax.Array:
    """Muestrea fase y centro; conserva periodo y radio de la configuración."""
    phase_key, center_key = jax.random.split(key)
    phase = jax.random.randint(phase_key, (), 0, cfg.period)
    center = jax.random.uniform(center_key, (2,))
    return jnp.array([phase, cfg.period, center[0], center[1], cfg.radius])


def projection(theta: jax.Array, view_angle: jax.Array) -> jax.Array:
    """h_i(θ): proyección determinista según el ángulo de observación."""
    direction = jnp.stack([jnp.cos(view_angle), jnp.sin(view_angle)], axis=-1)
    return jnp.stack([direction @ theta[:2], direction @ theta[2:4]], axis=-1)


def measure(key: jax.Array, theta: jax.Array, view_angle: jax.Array, noise_std: float) -> jax.Array:
    """z_i = h_i(θ) + ε."""
    value = projection(theta, view_angle)
    return value + noise_std * jax.random.normal(key, value.shape)


def burst_due(t: jax.Array, theta: jax.Array) -> jax.Array:
    """True si en el paso t ocurre un estallido."""
    return (t > 0) & (((t - theta[0].astype(jnp.int32)) % theta[1].astype(jnp.int32)) == 0)


def burst_victims(
    key: jax.Array,
    pos: jax.Array,
    theta: jax.Array,
    refuge: jax.Array,
    lethality: float,
) -> jax.Array:
    """(N,) bool: dentro de la zona afectada, sin refugio, y muere con prob. λ."""
    exposed = (distance(pos, theta[2:4]) <= theta[4]) & ~refuge
    return exposed & jax.random.bernoulli(key, jnp.asarray(lethality, dtype=float), pos.shape[:-1])
