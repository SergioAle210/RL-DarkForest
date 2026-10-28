"""Evalúa una red entrenada en cientos de episodios y la compara con las políticas fijas.

El entrenamiento solo guarda el retorno de un episodio antes y después. Este módulo promedia
muchos episodios con las mismas reglas del entrenamiento y reporta retorno, supervivencia, causas
de muerte y uso de cada familia de acciones. Con --baselines compara contra ignorar, solo,
coop_real y oraculo de train.valor_cooperacion, bajo la misma configuración del entorno.

Uso: python -m train.evaluar_checkpoint --checkpoint runs/<corrida>/0/checkpoints/final.msgpack
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from darkforest.config import ACTIONS, Config
from darkforest.env import DarkForestEnv
from darkforest.policies import Policy
from train.episodes import checkpoint_policy
from train.valor_cooperacion import PERIODO_ALEATORIO, PERIODO_FIJO, run

# márgenes elegidos en docs/DIAGNOSTICO_DISENO.md para el mundo de periodo aleatorio
BASELINES = {"ignorar": 0.0, "solo": 0.0, "coop_real": 0.2, "oraculo": 0.0}


def evaluate(cfg: Config, policy: Policy, seed: int = 100, episodes: int = 400) -> dict[str, float]:
    """Promedia episodios completos de la política en todas las civilizaciones."""
    env = DarkForestEnv(cfg.env)
    n = env.n_agents
    reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
    act = jax.jit(jax.vmap(policy.act))
    key, reset_key = jax.random.split(jax.random.PRNGKey(seed))
    obs, state = reset(jax.random.split(reset_key, episodes))
    carry = jnp.broadcast_to(policy.init_carry(n), (episodes, *policy.init_carry(n).shape))
    total = jnp.zeros((episodes, n))
    kinds = np.zeros(len(ACTIONS))
    anomaly_deaths = attack_deaths = launches = 0.0
    for _ in range(cfg.env.max_steps):
        key, act_key, step_key = jax.random.split(key, 3)
        carry, actions = act(jax.random.split(act_key, episodes), carry, obs)
        out = step(jax.random.split(step_key, episodes), state, actions)
        executed = np.asarray(out.info.executed_kind)
        alive = np.asarray(state.alive)
        for index in range(len(ACTIONS)):
            kinds[index] += ((executed == index) & alive).sum()
        anomaly_deaths += float((out.info.death_cause == 1).sum())
        attack_deaths += float((out.info.death_cause == 2).sum())
        launches += float(out.info.launches.sum())
        total += out.reward
        state, obs = out.state, out.obs
    civilizations = episodes * n
    result = {
        "mean_return": float(total.mean()),
        "survival": float(state.alive.mean()),
        "deaths_anomaly": anomaly_deaths / civilizations,
        "deaths_attack": attack_deaths / civilizations,
        "attacks": launches / civilizations,
    }
    result.update(
        {f"action_{name}": float(kinds[i] / max(kinds.sum(), 1)) for i, name in enumerate(ACTIONS)}
    )
    return result


def baseline_returns(cfg: Config, seeds: int, episodes: int) -> dict[str, tuple[float, float]]:
    """Retorno y supervivencia de las políticas fijas con las mismas reglas del checkpoint."""
    world = PERIODO_ALEATORIO if cfg.env.anomaly.period_spread else PERIODO_FIJO
    over = {
        "env.detection_low": cfg.env.detection_low,
        "env.detection_high": cfg.env.detection_high,
        "env.signal_speed": cfg.env.signal_speed,
        "env.cost_tx": cfg.env.cost_tx,
        "env.p_kill": cfg.env.p_kill,
        "env.projectiles_cancel_on_death": cfg.env.projectiles_cancel_on_death,
    }
    if cfg.env.detection_growth is not None:
        over["env.detection_growth"] = cfg.env.detection_growth
    results = {}
    for mode, margin in BASELINES.items():
        runs = [
            run(
                cfg.env.anomaly.lethality,
                world,
                mode,
                margin,
                s,
                episodes=episodes,
                steps=cfg.env.max_steps,
                env_over=over,
            )
            for s in range(seeds)
        ]
        results[mode] = (float(np.mean([r[0] for r in runs])), float(np.mean([r[1] for r in runs])))
    return results


def main() -> None:
    """Imprime la evaluación de la red y, si se pide, las políticas fijas de referencia."""
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--baselines", action="store_true")
    args = p.parse_args()
    cfg, policy = checkpoint_policy(args.checkpoint)
    result = evaluate(cfg, policy, args.seed, args.episodes)
    print(
        f"Red entrenada, {args.episodes} episodios de {cfg.env.max_steps} pasos, seed {args.seed}"
    )
    print(
        f"  retorno por civilización {result['mean_return']:+.3f}, "
        f"supervivencia {result['survival']:.3f}"
    )
    print(
        f"  muertes por anomalía {result['deaths_anomaly']:.3f} y por ataque "
        f"{result['deaths_attack']:.3f} por civilización; ataques lanzados {result['attacks']:.2f}"
    )
    print("  acciones: " + " ".join(f"{n}={result[f'action_{n}']:.3f}" for n in ACTIONS))
    if args.baselines:
        print()
        print("Políticas fijas con las mismas reglas (3 seeds)")
        for mode, (ret, surv) in baseline_returns(cfg, 3, args.episodes).items():
            print(f"  {mode:10s} retorno {ret:+.3f}, supervivencia {surv:.3f}")


if __name__ == "__main__":
    main()
