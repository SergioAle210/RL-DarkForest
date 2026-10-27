"""Distancias y desplazamientos en el toro unitario."""

import jax
import jax.numpy as jnp


def displacement(origin: jax.Array, target: jax.Array) -> jax.Array:
    """Devuelve el desplazamiento mínimo entre dos posiciones."""
    return (target - origin + 0.5) % 1.0 - 0.5


def distance(origin: jax.Array, target: jax.Array) -> jax.Array:
    """Calcula la distancia euclidiana del desplazamiento mínimo."""
    return jnp.linalg.norm(displacement(origin, target), axis=-1)
