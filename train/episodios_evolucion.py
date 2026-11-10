"""Guarda un episodio por snapshot de una corrida para animar cómo cambia la conducta.

Uso: python -m train.episodios_evolucion --run runs/<corrida>/0 --out salida --iters 10 80 160
Luego: python -m viz.galaxy --episode salida/trajectories/episode_000001.npz
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import jax

from darkforest.env import DarkForestEnv
from train.episodes import checkpoint_policy, collect_episode, save_episode


def main() -> None:
    """Un episodio sampled por cada iteración pedida, con la misma semilla de mundo."""
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--iters", type=int, nargs="+", required=True)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--greedy", action="store_true")
    args = p.parse_args()
    folder = args.out / "evolucion" / "0"
    folder.mkdir(parents=True, exist_ok=True)
    for index, iteration in enumerate(args.iters, start=1):
        path = args.run / "snapshots" / f"iter_{iteration:05d}.msgpack"
        cfg, policy = checkpoint_policy(path, greedy=args.greedy)
        cfg = replace(cfg, logging=replace(cfg.logging, out_dir=str(args.out)))
        trajectory = collect_episode(DarkForestEnv(cfg.env), policy, jax.random.PRNGKey(args.seed))
        save_episode(
            folder,
            trajectory,
            cfg=cfg,
            policy_name=f"snapshot_{iteration}",
            episode_id=index,
        )
        print(f"iteración {iteration}: episodio {index}", flush=True)


if __name__ == "__main__":
    main()
