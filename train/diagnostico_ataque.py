"""Diagnostica por qué dos civilizaciones que se conocen se matan en vez de coordinarse.

Corre muchos episodios con una política y separa los que tuvieron contacto de los que no.
Reporta cuánto se ataca antes y después del contacto, cuántos ataques son mutuos, quién dispara
primero, si hubo transmitir o compartir entre el contacto y el primer disparo, y el costo en
retorno de haber tenido contacto. Solo observa: no cambia el entorno ni la recompensa.

Uso: python -m train.diagnostico_ataque --checkpoint runs/<corrida>/0/checkpoints/final.msgpack
     python -m train.diagnostico_ataque --config configs/costo_ignorar.yaml --policy random
"""

from __future__ import annotations

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np

from darkforest.config import ACTIONS, Config, load_config
from darkforest.env import DarkForestEnv
from darkforest.policies import Policy, RandomPolicy
from train.episodes import checkpoint_policy

ATTACK, TRANSMIT, SHARE = (ACTIONS.index(name) for name in ("atacar", "transmitir", "compartir"))


def diagnose(cfg: Config, policy: Policy, seed: int = 100, episodes: int = 400) -> dict:
    """Mide la secuencia contacto, mensajes y ataques en episodios completos."""
    env = DarkForestEnv(cfg.env)
    n = env.n_agents
    reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
    act = jax.jit(jax.vmap(policy.act))
    key, reset_key = jax.random.split(jax.random.PRNGKey(seed))
    obs, state = reset(jax.random.split(reset_key, episodes))
    carry = jax.tree.map(lambda c: jnp.broadcast_to(c, (episodes, *c.shape)), policy.init_carry(n))
    total = np.zeros((episodes, n))
    first_contact = np.full(episodes, -1)
    first_attack = np.full(episodes, -1)
    first_message = np.full(episodes, -1)
    attackers = np.zeros((episodes, n), bool)
    attack_deaths = np.zeros(episodes)
    steps_before = steps_after = attacks_before = attacks_after = 0.0
    for t in range(cfg.env.max_steps):
        key, act_key, step_key = jax.random.split(key, 3)
        carry, actions = act(jax.random.split(act_key, episodes), carry, obs)
        out = step(jax.random.split(step_key, episodes), state, actions)
        kind = np.asarray(out.info.executed_kind)
        alive = np.asarray(state.alive)
        contact = np.asarray((state.contact_t >= 0).any(axis=(1, 2)))
        attacked = (kind == ATTACK) & alive
        message = ((kind == TRANSMIT) | (kind == SHARE)) & alive
        first_contact = np.where((first_contact < 0) & contact, t, first_contact)
        first_attack = np.where((first_attack < 0) & attacked.any(1), t, first_attack)
        late = contact & (first_attack < 0)
        first_message = np.where((first_message < 0) & message.any(1) & late, t, first_message)
        attackers |= attacked
        steps_before += float((alive & ~contact[:, None]).sum())
        steps_after += float((alive & contact[:, None]).sum())
        attacks_before += float((attacked & ~contact[:, None]).sum())
        attacks_after += float((attacked & contact[:, None]).sum())
        attack_deaths += np.asarray(out.info.death_cause == 2).sum(axis=1)
        total += np.asarray(out.reward)
        state, obs = out.state, out.obs
    met = first_contact >= 0
    shot = met & (first_attack >= 0)
    both = attackers.all(axis=1) if n == 2 else attackers.sum(axis=1) >= 2
    talked_first = (
        met & (first_message >= 0) & ((first_attack < 0) | (first_message < first_attack))
    )

    def rate(num, den):
        """Cociente seguro para tasas sobre conjuntos que pueden quedar vacíos."""
        return float(num / den) if den else float("nan")

    ret = total.mean(axis=1)
    return {
        "episodes": episodes,
        "contact_rate": rate(met.sum(), episodes),
        "mean_first_contact_step": float(first_contact[met].mean()) if met.any() else None,
        "attack_rate_per_step_before_contact": rate(attacks_before, steps_before),
        "attack_rate_per_step_after_contact": rate(attacks_after, steps_after),
        "p_any_attack_given_contact": rate(shot.sum(), met.sum()),
        "p_mutual_attack_given_contact": rate((met & both).sum(), met.sum()),
        "p_message_before_first_attack_given_contact": rate(talked_first.sum(), met.sum()),
        "mean_steps_contact_to_first_attack": (
            float((first_attack[shot] - first_contact[shot]).mean()) if shot.any() else None
        ),
        "attack_deaths_per_civ_given_contact": rate(attack_deaths[met].sum(), met.sum() * n),
        "return_given_contact": float(ret[met].mean()) if met.any() else None,
        "return_given_no_contact": float(ret[~met].mean()) if (~met).any() else None,
        "return_given_contact_and_no_attack": (
            float(ret[met & ~shot].mean()) if (met & ~shot).any() else None
        ),
        "return_given_contact_and_attack": (float(ret[shot].mean()) if shot.any() else None),
    }


def main() -> None:
    """Imprime el diagnóstico de un checkpoint o de la política aleatoria."""
    p = argparse.ArgumentParser()
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--config")
    p.add_argument("--policy", choices=("random",), default="random")
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--greedy", action="store_true")
    args = p.parse_args()
    if args.checkpoint:
        cfg, policy = checkpoint_policy(args.checkpoint, greedy=args.greedy)
        label = args.checkpoint
    else:
        cfg = load_config(args.config)
        policy, label = RandomPolicy(cfg.env.n_discrete), "aleatoria"
    result = diagnose(cfg, policy, args.seed, args.episodes)
    print(json.dumps({"fuente": label, "greedy": args.greedy, **result}, indent=2))


if __name__ == "__main__":
    main()
