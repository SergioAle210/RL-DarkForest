"""Anomalía periódica con parámetros ocultos θ y proyecciones ruidosas h_i(θ)."""

from __future__ import annotations

import jax

from darkforest.config import AnomalyConfig


def sample_theta(key: jax.Array, cfg: AnomalyConfig) -> jax.Array:
    """(fase, periodo, cx, cy, ...). Debe requerir >= 2 proyecciones para identificarse."""
    raise NotImplementedError


def projection(theta: jax.Array, view_angle: jax.Array) -> jax.Array:
    """h_i(θ): proyección determinista según el ángulo de observación."""
    raise NotImplementedError


def measure(key: jax.Array, theta: jax.Array, view_angle: jax.Array, noise_std: float) -> jax.Array:
    """z_i = h_i(θ) + ε."""
    raise NotImplementedError


def burst_due(t: jax.Array, theta: jax.Array) -> jax.Array:
    """True si en el paso t ocurre un estallido."""
    raise NotImplementedError


def burst_victims(
    key: jax.Array,
    pos: jax.Array,
    theta: jax.Array,
    refuge: jax.Array,
    lethality: float,
) -> jax.Array:
    """(N,) bool: dentro de la zona afectada, sin refugio, y muere con prob. λ."""
    raise NotImplementedError
