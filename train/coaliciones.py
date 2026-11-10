"""Genera episodios de una red entrenada con 2 civilizaciones jugando con N = 2, 4, 6 y 8.

La red no depende de N (la identidad entra como one-hot de 8 y las detecciones son M ranuras
fijas), así que sus pesos se evalúan sin reentrenar con más civilizaciones. Guarda los episodios
en <out>/N<n>/trajectories para analysis.coaliciones (H4: grafo de quién recibe de quién).

Uso: python -m train.coaliciones --checkpoint <ruta> --out salida --episodes 12
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import jax

from darkforest.env import DarkForestEnv
from train.episodes import checkpoint_policy, collect_episode, save_episode


def main() -> None:
    """Un conjunto de episodios por tamaño de población, con la misma red y semillas distintas."""
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--sizes", type=int, nargs="+", default=[2, 4, 6, 8])
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--greedy", action="store_true")
    args = p.parse_args()
    for size in args.sizes:
        cfg, policy = checkpoint_policy(args.checkpoint, greedy=args.greedy)
        cfg = replace(cfg, env=replace(cfg.env, n_civs=size))
        folder = args.out / f"N{size}" / "0"
        folder.mkdir(parents=True, exist_ok=True)
        env = DarkForestEnv(cfg.env)
        for index in range(args.episodes):
            trajectory = collect_episode(env, policy, jax.random.PRNGKey(1000 + index))
            save_episode(folder, trajectory, cfg=cfg, policy_name=f"N{size}", episode_id=index + 1)
        print(f"N={size}: {args.episodes} episodios", flush=True)


if __name__ == "__main__":
    main()
