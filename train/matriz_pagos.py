"""Matriz de pagos de dos civilizaciones con estrategias fijas, sin entrenar.

Diagnóstico previo al entrenamiento: muestra si alguna estrategia gana siempre con las reglas
actuales, es decir, si el resultado ya está en las reglas. Lectura en docs/DIAGNOSTICO_DISENO.md.

crecer: siempre crece. silencio: siempre se oculta. ataque_unico: dispara una vez al primer
contacto detectado y luego sigue creciendo. ataque_solo: como solo (se protege de la anomalía)
pero dispara una vez al primer contacto. Cualquier modo de train.valor_cooperacion también
sirve (solo, oculta, coop_real...). Con --set se cambia cualquier parámetro del entorno.

Uso: python -m train.matriz_pagos --lethality 0 --modes crecer silencio ataque_unico
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import yaml

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
    full: bool = False,
) -> tuple[float, ...]:
    """Retorno medio y supervivencia de la civilización 0 jugando mode0 contra mode1.

    Con full=True devuelve además retorno, supervivencia y disparos por episodio de ambas.
    """
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
        if mode == "ataque_solo":
            base = decide(state, first, second, MARGIN, "solo", PERIODO_ALEATORIO, shelter, share)
            base = base[:, idx]
            mask = obs.det_mask[:, idx]
            fire = jnp.any(mask, axis=-1) & ~fired[:, idx] & (base != shelter)
            return jnp.where(fire, 4 + k + jnp.argmax(mask, axis=-1), base).astype(jnp.int32)
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
    if full:
        shots = fired.mean(axis=0)
        return (
            float(total[:, 0].mean()),
            float(state.alive[:, 0].mean()),
            float(total[:, 1].mean()),
            float(state.alive[:, 1].mean()),
            float(shots[0]),
            float(shots[1]),
        )
    return float(total[:, 0].mean()), float(state.alive[:, 0].mean())


def payoff_matrix(
    lethality: float,
    modes: list[str],
    seeds: int,
    episodes: int,
    steps: int,
    env_over: dict[str, Any] | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Matriz A[i, j] con el retorno medio de la estrategia i contra la j, lista para guardar."""
    size = len(modes)
    matrix, errors, survival = (np.zeros((size, size)) for _ in range(3))
    for i, row in enumerate(modes):
        for j, col in enumerate(modes):
            runs = [
                run_pair(lethality, row, col, s, episodes=episodes, steps=steps, env_over=env_over)
                for s in range(seeds)
            ]
            matrix[i, j], errors[i, j] = mean_se([r[0] for r in runs])
            survival[i, j] = np.mean([r[1] for r in runs])
            if verbose:
                print(
                    f"  {row:12s} contra {col:12s} {matrix[i, j]:+.3f} +- {errors[i, j]:.3f} "
                    f"superv={survival[i, j]:.3f}",
                    flush=True,
                )
    return {
        "modes": list(modes),
        "lethality": lethality,
        "overrides": env_over or {},
        "seeds": seeds,
        "episodes": episodes,
        "steps": steps,
        "matrix": matrix.tolist(),
        "se": errors.tolist(),
        "survival": survival.tolist(),
    }


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
    p.add_argument("--set", action="append", default=[], metavar="CLAVE=VALOR", dest="extra")
    p.add_argument("--out", help="guarda la matriz y sus metadatos en este archivo JSON")
    args = p.parse_args()
    over: dict[str, Any] = {k: yaml.safe_load(v) for k, v in (e.split("=", 1) for e in args.extra)}
    if args.cancel:
        over["env.projectiles_cancel_on_death"] = True
    if args.p_kill is not None:
        over["env.p_kill"] = args.p_kill
    print(
        f"lambda={args.lethality}, retorno de la fila contra la columna, {args.seeds} seeds, {over}"
    )
    result = payoff_matrix(
        args.lethality, args.modes, args.seeds, args.episodes, args.steps, over, verbose=True
    )
    matrix = np.asarray(result["matrix"])
    for j, col in enumerate(args.modes):
        print(f"  mejor respuesta contra {col}: {args.modes[int(np.argmax(matrix[:, j]))]}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
