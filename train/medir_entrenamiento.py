"""Mide la velocidad, la memoria y la estabilidad del entrenamiento IPPO en esta máquina.

Separa la primera iteración (que compila) de la ejecución estable, para estimar cuántos pasos de
entorno caben en un presupuesto de tiempo. Ejecuta exactamente las iteraciones del entrenamiento
real (`train.ippo.run_iteration`), sin guardar corridas ni episodios.

Uso: python -m train.medir_entrenamiento --config configs/base.yaml --n-envs 64 256 [clave=valor]
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from darkforest.config import Config, load_config
from darkforest.env import DarkForestEnv
from train.ippo import initialize, make_rollout, make_update, run_iteration


def process_memory_mb() -> float | None:
    """Memoria residente del proceso en MB, o None si psutil no está disponible."""
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process().memory_info().rss / 2**20


def measure(cfg: Config, iterations: int = 3) -> dict[str, Any]:
    """Tiempo de compilación y de iteraciones estables, pasos por segundo y estabilidad."""
    env = DarkForestEnv(cfg.env)
    rollout = jax.jit(make_rollout(env, cfg))
    update = make_update(cfg)
    key, init_key = jax.random.split(jax.random.PRNGKey(cfg.train.seed))
    state, runner = initialize(cfg, init_key)
    assignment = jnp.full((cfg.train.n_envs, cfg.env.n_civs), -1)
    durations, finite = [], True
    for _ in range(iterations + 1):
        start = time.perf_counter()
        state, runner, key, metrics = run_iteration(
            cfg, env, rollout, update, state, runner, key, assignment
        )
        jax.block_until_ready((state.params, metrics))
        durations.append(time.perf_counter() - start)
        finite &= all(bool(np.isfinite(float(value))) for value in metrics.values())
    steady = float(np.mean(durations[1:]))
    chunk = cfg.train.n_envs * cfg.train.rollout_len
    return {
        "n_envs": cfg.train.n_envs,
        "rollout_len": cfg.train.rollout_len,
        "first_s": durations[0],
        "steady_s": steady,
        "steps_per_s": chunk / steady,
        "memory_mb": process_memory_mb(),
        "finite": finite,
    }


def main() -> None:
    """Imprime una fila por tamaño de lote con los tiempos medidos."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--n-envs", type=int, nargs="+", default=[64, 256, 1024])
    p.add_argument("--rollout-len", type=int)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("overrides", nargs="*", help="clave.punteada=valor")
    args = p.parse_args()
    extra = {k: yaml.safe_load(v) for k, v in (o.split("=", 1) for o in args.overrides)}
    print(f"{jax.devices()}, configuración {args.config}, {args.iterations} iteraciones estables")
    for n_envs in args.n_envs:
        overrides = {**extra, "train.n_envs": n_envs}
        if args.rollout_len:
            overrides["train.rollout_len"] = args.rollout_len
        result = measure(load_config(args.config, overrides), args.iterations)
        memory = "n/d" if result["memory_mb"] is None else f"{result['memory_mb']:.0f} MB"
        per_million = 1e6 / result["steps_per_s"] / 60
        print(
            f"n_envs={result['n_envs']} rollout={result['rollout_len']}: "
            f"primera iteración {result['first_s']:.1f} s, estable {result['steady_s']:.2f} s, "
            f"{result['steps_per_s']:.0f} pasos/s, un millón de pasos en {per_million:.1f} min, "
            f"memoria {memory}, métricas finitas {result['finite']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
