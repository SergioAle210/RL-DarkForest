"""Socios de entrenamiento: estrategias fijas y asignación de quién juega con quién.

Las estrategias son las de train.valor_cooperacion y train.matriz_pagos (coop_real, solo,
ataque_solo e ignorar), vectorizadas sobre el lote de entornos para usarlas dentro del rollout
de IPPO. Cada civilización tiene un código de política: -1 es la red que aprende, -2 un snapshot
congelado y 0..6 una estrategia fija (índice en FIJOS). Valen para dos o más civilizaciones.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

from darkforest.config import Config
from darkforest.types import EnvState, Obs
from train.valor_cooperacion import PERIODO_ALEATORIO, PERIODO_FIJO, decide, update_bursts

APRENDIZ, SNAPSHOT = -1, -2
FIJOS = ("coop_real", "solo", "ataque_solo", "ignorar", "recip", "prudente", "mentiroso")
MARGEN = 0.2  # el de la rejilla de pagos de docs/DIAGNOSTICO_DISENO.md


@struct.dataclass
class Auxiliar:
    """Memoria de las estrategias fijas: primeros estallidos vistos y si ya dispararon."""

    first: jax.Array  # (B,) turno del primer estallido observado
    second: jax.Array  # (B,) turno del segundo
    fired: jax.Array  # (B, N) bool, ya lanzó un ataque


def initial_auxiliar(batch: int, n_agents: int) -> Auxiliar:
    """Memoria vacía para un lote de entornos."""
    zeros = jnp.zeros(batch, dtype=jnp.int32)
    return Auxiliar(zeros, zeros, jnp.zeros((batch, n_agents), dtype=bool))


def fixed_actions(cfg: Config, state: EnvState, obs: Obs, aux: Auxiliar) -> jax.Array:
    """Acción de cada estrategia fija, forma (len(FIJOS), B, N)."""
    env = cfg.env
    world = PERIODO_ALEATORIO if env.anomaly.period_spread else PERIODO_FIJO
    shelter, share, k = env.n_discrete - 1, 3 + env.n_symbols, env.n_symbols
    first, second = aux.first, aux.second
    coop = decide(state, first, second, MARGEN, "coop_real", world, shelter, share)
    solo = decide(state, first, second, MARGEN, "solo", world, shelter, share)
    target = jnp.argmax(obs.det_mask, axis=-1)
    fire = jnp.any(obs.det_mask, axis=-1) & ~aux.fired & (solo != shelter)
    attack = jnp.where(fire, 4 + k + target, solo)
    ignore = jnp.zeros_like(solo)
    recip = decide(state, first, second, MARGEN, "recip", world, shelter, share)
    prudent = decide(state, first, second, MARGEN, "prudente", world, shelter, share)
    liar = decide(state, first, second, MARGEN, "mentiroso", world, shelter, share)
    return jnp.stack([coop, solo, attack, ignore, recip, prudent, liar]).astype(jnp.int32)


def update_auxiliar(
    cfg: Config, aux: Auxiliar, old: EnvState, new: EnvState, actions: jax.Array
) -> Auxiliar:
    """Registra estallidos y ataques lanzados tras un paso del entorno."""
    first, second = update_bursts(new.last_burst, old.last_burst, aux.first, aux.second)
    shelter, k = cfg.env.n_discrete - 1, cfg.env.n_symbols
    shot = (actions >= 4 + k) & (actions < shelter)
    return Auxiliar(first, second, aux.fired | shot)


def draw_assignment(key: jax.Array, cfg: Config, batch: int) -> jax.Array:
    """Sortea los códigos de política de cada civilización, forma (B, N).

    Con probabilidad 1 - rho todas son la red que aprende; si no, una civilización al azar juega
    con un socio sorteado según los pesos y las demás son la red que aprende.
    """
    play = cfg.train.self_play
    n = cfg.env.n_civs
    if play.rho == 0:
        return jnp.full((batch, n), APRENDIZ, dtype=jnp.int32)
    weights = play.partner_weights()
    probs = jnp.asarray(weights / weights.sum())
    codes = jnp.array([SNAPSHOT, 0, 1, 2, 3, 4, 5, 6], dtype=jnp.int32)
    k_play, k_slot, k_partner = jax.random.split(key, 3)
    plays = jax.random.bernoulli(k_play, play.rho, (batch,))
    slot = jax.random.randint(k_slot, (batch,), 0, n)
    partner = codes[jax.random.categorical(k_partner, jnp.log(probs + 1e-30), shape=(batch,))]
    mine = jnp.arange(n)[None, :] == slot[:, None]
    return jnp.where(plays[:, None] & mine, partner[:, None], APRENDIZ).astype(jnp.int32)


def choose_actions(
    assignment: jax.Array, learner: jax.Array, snapshot: jax.Array | None, fixed: jax.Array
) -> jax.Array:
    """Combina las acciones de la red, del snapshot y de las estrategias fijas por código."""
    chosen = learner
    if snapshot is not None:
        chosen = jnp.where(assignment == SNAPSHOT, snapshot, chosen)
    index = jnp.clip(assignment, 0, len(FIJOS) - 1)
    scripted = jnp.take_along_axis(fixed, index[None], axis=0)[0]
    return jnp.where(assignment >= 0, scripted, chosen)
