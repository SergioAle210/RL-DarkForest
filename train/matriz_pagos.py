"""Matriz de pagos de dos civilizaciones con estrategias fijas, sin entrenar.

Diagnóstico previo al entrenamiento: muestra si alguna estrategia gana siempre con las reglas
actuales, es decir, si el resultado ya está en las reglas. Lectura en docs/DIAGNOSTICO_DISENO.md.

crecer: siempre crece. silencio: siempre se oculta. ataque_unico: dispara una vez al primer
contacto detectado y luego sigue creciendo. Cualquier modo de train.valor_cooperacion también
sirve (solo, oculta, coop_real...).

Uso: python -m train.matriz_pagos --lethality 0 --modes crecer silencio ataque_unico
"""

from __future__ import annotations

import argparse
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from train.valor_cooperacion import PERIODO_ALEATORIO, build_env, decide, mean_se, update_bursts

MODES = ("crecer", "silencio", "ataque_unico")
MARGIN = 0.2


def run_pair(
    lethality: float,
    mode0: str,
    mode1: str,
    seed: int,
    *,
    episodes: int = 400,
    steps: int = 200,
    env_over: dict[str, Any] | None = None,
) -> tuple[float, float]:
    """Retorno medio y supervivencia de la civilización 0 jugando mode0 contra mode1."""
    env = build_env(lethality, PERIODO_ALEATORIO, steps, env_over)
    reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
    shelter, share, k = env.n_discrete - 1, 3 + env.cfg.n_symbols, env.cfg.n_symbols
    key = jax.random.PRNGKey(seed)
    obs, state = reset(jax.random.split(key, episodes))
    first = second = jnp.zeros(episodes, dtype=jnp.int32)
    fired = jnp.zeros((episodes, 2), dtype=bool)
    total = jnp.zeros((episodes, 2))

    def actions_for(mode: str, idx: int) -> jax.Array:
        """Acción de la civilización idx para la estrategia indicada."""
        if mode == "crecer":
            return jnp.zeros(episodes, dtype=jnp.int32)
        if mode == "silencio":
            return jnp.ones(episodes, dtype=jnp.int32)
        if mode == "ataque_unico":
            mask = obs.det_mask[:, idx]
            target = jnp.argmax(mask, axis=-1)
            act = jnp.where(jnp.any(mask, axis=-1) & ~fired[:, idx], 4 + k + target, 0)
            return jnp.where(obs.alive[:, idx], act, 1).astype(jnp.int32)
        acts = decide(state, first, second, MARGIN, mode, PERIODO_ALEATORIO, shelter, share)
        return acts[:, idx]

    for _ in range(steps):
        key, step_key = jax.random.split(key)
        actions = jnp.stack([actions_for(mode0, 0), actions_for(mode1, 1)], axis=1)
        fired |= (actions >= 4 + k) & (actions < shelter)
        out = step(jax.random.split(step_key, episodes), state, actions)
        first, second = update_bursts(out.state.last_burst, state.last_burst, first, second)
        total += out.reward
        state, obs = out.state, out.obs
    return float(total[:, 0].mean()), float(state.alive[:, 0].mean())


def main() -> None:
    """Imprime la matriz de pagos y la mejor respuesta contra cada columna."""
    p = argparse.ArgumentParser()
    p.add_argument("--lethality", type=float, default=0.0)
    p.add_argument("--modes", nargs="+", default=list(MODES))
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--p-kill", type=float)
    p.add_argument("--cancel", action="store_true", help="env.projectiles_cancel_on_death")
    args = p.parse_args()
    over: dict[str, Any] = {}
    if args.cancel:
        over["env.projectiles_cancel_on_death"] = True
    if args.p_kill is not None:
        over["env.p_kill"] = args.p_kill
    print(
        f"lambda={args.lethality}, retorno de la fila contra la columna, {args.seeds} seeds, {over}"
    )
    matrix = np.zeros((len(args.modes), len(args.modes)))
    for i, row in enumerate(args.modes):
        for j, col in enumerate(args.modes):
            runs = [
                run_pair(
                    args.lethality,
                    row,
                    col,
                    s,
                    episodes=args.episodes,
                    steps=args.steps,
                    env_over=over,
                )
                for s in range(args.seeds)
            ]
            matrix[i, j], se = mean_se([r[0] for r in runs])
            survival = np.mean([r[1] for r in runs])
            print(
                f"  {row:12s} contra {col:12s} {matrix[i, j]:+.3f} +- {se:.3f} "
                f"superv={survival:.3f}"
            )
    for j, col in enumerate(args.modes):
        print(f"  mejor respuesta contra {col}: {args.modes[int(np.argmax(matrix[:, j]))]}")


if __name__ == "__main__":
    main()
