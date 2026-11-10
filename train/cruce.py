"""Matriz de cruce: cada política contra cada otra, con 2 civilizaciones.

Una política es una red (checkpoint o snapshot) o una estrategia fija de train.socios. Cada
pareja juega las dos asignaciones de asiento y se reporta el retorno medio de la fila contra la
columna, la supervivencia y los disparos por episodio. Sirve para ver si una red coopera con
otras redes, con sus versiones anteriores y con las estrategias fijas, no solo consigo misma.

Uso: python -m train.cruce --config configs/costo_ignorar_optim.yaml --out cruce.json \
         fijo:coop_real fijo:solo red:runs/<corrida>/0/checkpoints/final.msgpack red:<ruta>:greedy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from darkforest.config import Config, load_config
from darkforest.env import DarkForestEnv
from train import socios
from train.episodes import checkpoint_policy


def load_player(spec: str, cfg: Config):
    """Interpreta 'fijo:<nombre>' o 'red:<ruta>[:greedy]' y devuelve (nombre, función de acción)."""
    kind, _, rest = spec.partition(":")
    if kind == "fijo":
        code = socios.FIJOS.index(rest)

        def act_fixed(key, carry, state, obs, aux):
            """Estrategia fija vectorizada sobre el lote de episodios."""
            return carry, socios.fixed_actions(cfg, state, obs, aux)[code]

        return spec, act_fixed, lambda episodes: None
    path, _, flag = rest.rpartition(":") if rest.endswith(":greedy") else (rest, "", "")
    _, policy = checkpoint_policy(path, greedy=bool(flag))
    act = jax.jit(jax.vmap(policy.act))
    n = cfg.env.n_civs

    def act_net(key, carry, state, obs, aux):
        """Red recurrente que actúa con sus propios pesos en ambos asientos."""
        return act(jax.random.split(key, obs.alive.shape[0]), carry, obs)

    def init(episodes):
        """Memoria inicial de la red para todos los episodios."""
        return jax.tree.map(
            lambda c: jnp.broadcast_to(c, (episodes, *c.shape)), policy.init_carry(n)
        )

    return spec, act_net, init


def play(cfg: Config, a, b, episodes: int, seed: int) -> dict[str, float]:
    """Juega a como civilización 0 y b como civilización 1; devuelve métricas por asiento."""
    env = DarkForestEnv(cfg.env)
    reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
    key, reset_key = jax.random.split(jax.random.PRNGKey(seed))
    obs, state = reset(jax.random.split(reset_key, episodes))
    aux = socios.initial_auxiliar(episodes, 2)
    carry_a, carry_b = a[2](episodes), b[2](episodes)
    total = jnp.zeros((episodes, 2))
    shelter, k = env.n_discrete - 1, env.cfg.n_symbols
    shots = jnp.zeros((episodes, 2), dtype=bool)
    for _ in range(cfg.env.max_steps):
        key, key_a, key_b, step_key = jax.random.split(key, 4)
        carry_a, act_a = a[1](key_a, carry_a, state, obs, aux)
        carry_b, act_b = b[1](key_b, carry_b, state, obs, aux)
        actions = jnp.stack([act_a[:, 0], act_b[:, 1]], axis=1).astype(jnp.int32)
        out = step(jax.random.split(step_key, episodes), state, actions)
        aux = socios.update_auxiliar(cfg, aux, state, out.state, actions)
        shots |= (actions >= 4 + k) & (actions < shelter)
        total += out.reward
        state, obs = out.state, out.obs
    alive = np.asarray(state.alive)
    return {
        "retorno": [float(total[:, i].mean()) for i in range(2)],
        "supervivencia": [float(alive[:, i].mean()) for i in range(2)],
        "disparos": [float(np.asarray(shots)[:, i].mean()) for i in range(2)],
    }


def cross_play(cfg: Config, specs: list[str], episodes: int = 200, seed: int = 7) -> dict:
    """Matriz con el retorno de la fila contra la columna, promediando los dos asientos."""
    players = [load_player(spec, cfg) for spec in specs]
    size = len(players)
    ret, surv, shots = (np.zeros((size, size)) for _ in range(3))
    for i in range(size):
        for j in range(i, size):
            ij = play(cfg, players[i], players[j], episodes, seed)
            ji = play(cfg, players[j], players[i], episodes, seed + 1)
            ret[i, j] = (ij["retorno"][0] + ji["retorno"][1]) / 2
            ret[j, i] = (ij["retorno"][1] + ji["retorno"][0]) / 2
            surv[i, j] = (ij["supervivencia"][0] + ji["supervivencia"][1]) / 2
            surv[j, i] = (ij["supervivencia"][1] + ji["supervivencia"][0]) / 2
            shots[i, j] = (ij["disparos"][0] + ji["disparos"][1]) / 2
            shots[j, i] = (ij["disparos"][1] + ji["disparos"][0]) / 2
    return {
        "politicas": specs,
        "retorno": ret.tolist(),
        "supervivencia": surv.tolist(),
        "disparos": shots.tolist(),
        "episodios": episodes,
    }


def main() -> None:
    """Calcula la matriz de cruce y la guarda o la imprime."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--out")
    p.add_argument("--set", action="append", default=[], metavar="CLAVE=VALOR", dest="extra")
    p.add_argument("policies", nargs="+")
    args = p.parse_args()
    overrides = {k: yaml.safe_load(v) for k, v in (e.split("=", 1) for e in args.extra)}
    cfg = load_config(args.config, overrides)
    result = cross_play(cfg, args.policies, args.episodes)
    for name, row in zip(args.policies, result["retorno"], strict=True):
        print(f"{name[:34]:34s} " + " ".join(f"{v:+6.2f}" for v in row))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
