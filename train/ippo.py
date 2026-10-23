"""IPPO con parámetros compartidos, GRU y autojuego ficticio.

Uso: python -m train.ippo --config configs/base.yaml [clave=valor ...]
"""

from __future__ import annotations

import argparse
from typing import Any

import jax

from darkforest.config import Config, load_config
from darkforest.env import DarkForestEnv

TrainState = Any


def make_rollout(env: DarkForestEnv, cfg: Config):
    """Devuelve rollout(key, params, pop_params, assignment, env_state) -> trayectorias.
    vmap sobre n_envs, scan sobre rollout_len."""
    raise NotImplementedError


def compute_gae(rewards, values, dones, last_value, gamma: float, lam: float):
    """GAE; la máscara de civilizaciones controladas por π_θ se aplica en la pérdida."""
    raise NotImplementedError


def ppo_loss(params, batch, cfg: Config):
    """Pérdida PPO: clip + valor + entropía, solo sobre civilizaciones controladas por π_θ."""
    raise NotImplementedError


def train(cfg: Config, name: str) -> TrainState:
    """Ciclo completo (sección 6.2): rollout → GAE → PPO → snapshot cada K iteraciones."""
    raise NotImplementedError


def evaluate(cfg: Config, params: Any, key: jax.Array) -> dict[str, float]:
    """π_θ contra sí misma y contra arquetipos; devuelve métricas y régimen."""
    raise NotImplementedError


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("overrides", nargs="*", help="clave.punteada=valor")
    args = p.parse_args()
    overrides = {k: _parse(v) for k, v in (o.split("=", 1) for o in args.overrides)}
    cfg = load_config(args.config, overrides)
    train(cfg, name=args.config)


def _parse(v: str):
    import yaml

    return yaml.safe_load(v)


if __name__ == "__main__":
    main()
