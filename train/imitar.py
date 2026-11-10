"""Imitación previa: clona a las estrategias cooperativas y deja pesos listos para afinar con RL.

Los expertos son las estrategias fijas de train.socios (coop_real y recip) jugando en pareja. Se
registran sus observaciones y acciones en episodios completos y se entrena la red recurrente por
entropía cruzada sobre las acciones y regresión del valor sobre el retorno descontado (con la
misma escala que usa PPO, para que el crítico llegue razonable al ajuste fino). No cambia el
entorno ni la recompensa: solo da un punto de partida a la política.

Uso: python -m train.imitar --config configs/pob_socios.yaml --set env.cost_ref=0.25 \
         --out runs/imitacion/costoso --steps 600
Los pesos quedan en <out>/checkpoints/final.msgpack con la config resuelta en <out>/config.yaml,
así que sirven como train.init_from y para train.cruce o train.resumen_corrida.
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from flax import serialization

from darkforest.config import Config, load_config, to_dict
from darkforest.env import DarkForestEnv
from darkforest.networks import ActorCriticGRU, flatten_obs
from darkforest.types import Obs
from train import socios

EXPERTOS = ("coop_real", "recip")


def collect(cfg: Config, episodes: int, seed: int = 0) -> dict:
    """Episodios de las parejas de expertos: obs (T, B, N, ...), acciones, recompensas."""
    env = DarkForestEnv(cfg.env)
    n = env.n_agents
    if n != 2:
        raise ValueError("La imitación usa estrategias fijas de dos civilizaciones")
    reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
    pairs = list(product(EXPERTOS, repeat=2))
    key = jax.random.PRNGKey(seed)
    chunks = []
    for pair in pairs:
        codes = [socios.FIJOS.index(name) for name in pair]
        key, reset_key = jax.random.split(key)
        obs, state = reset(jax.random.split(reset_key, episodes // len(pairs)))
        aux = socios.initial_auxiliar(episodes // len(pairs), n)
        record = {"obs": [], "actions": [], "rewards": []}
        for _ in range(cfg.env.max_steps):
            key, step_key = jax.random.split(key)
            fixed = socios.fixed_actions(cfg, state, obs, aux)
            actions = jnp.stack([fixed[codes[0]][:, 0], fixed[codes[1]][:, 1]], axis=1)
            out = step(jax.random.split(step_key, actions.shape[0]), state, actions)
            record["obs"].append(obs)
            record["actions"].append(actions)
            record["rewards"].append(out.reward)
            aux = socios.update_auxiliar(cfg, aux, state, out.state, actions)
            state, obs = out.state, out.obs
        chunks.append(
            {
                "obs": jax.tree.map(lambda *xs: jnp.stack(xs), *record["obs"]),
                "actions": jnp.stack(record["actions"]),
                "rewards": jnp.stack(record["rewards"]),
            }
        )
    return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=1), *chunks)


def collect_student(
    cfg: Config, params, episodes: int, seed: int = 0, expert: str = "coop_real"
) -> dict:
    """Episodios jugados por la red en ambos asientos, etiquetados con la acción del experto.

    Es el paso de DAgger: la red visita sus propios estados y el experto dice qué habría hecho en
    cada uno, de modo que los errores de imitación se corrigen donde ocurren.
    """
    env = DarkForestEnv(cfg.env)
    model = ActorCriticGRU(cfg.env.n_discrete, cfg.train.gru_hidden)
    code = socios.FIJOS.index(expert)
    reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))

    @jax.jit
    def act(key, carry, resets, obs):
        """Acción muestreada de la red con la memoria reiniciada donde corresponde."""
        carry = jnp.where(resets[..., None], 0, carry)
        carry, logits, _ = model.apply(params, carry, flatten_obs(obs), obs.agent_id)
        return carry, jax.random.categorical(key, logits)

    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    obs, state = reset(jax.random.split(reset_key, episodes))
    aux = socios.initial_auxiliar(episodes, env.n_agents)
    carry = jnp.zeros((episodes, env.n_agents, cfg.train.gru_hidden))
    resets = jnp.ones((episodes, env.n_agents), dtype=bool)
    record = {"obs": [], "actions": [], "rewards": []}
    for _ in range(cfg.env.max_steps):
        key, act_key, step_key = jax.random.split(key, 3)
        labels = socios.fixed_actions(cfg, state, obs, aux)[code]
        carry, actions = act(act_key, carry, resets | ~obs.alive, obs)
        out = step(jax.random.split(step_key, episodes), state, actions)
        record["obs"].append(obs)
        record["actions"].append(labels)
        record["rewards"].append(out.reward)
        aux = socios.update_auxiliar(cfg, aux, state, out.state, actions)
        resets = jnp.zeros_like(resets)
        state, obs = out.state, out.obs
    return {
        "obs": jax.tree.map(lambda *xs: jnp.stack(xs), *record["obs"]),
        "actions": jnp.stack(record["actions"]),
        "rewards": jnp.stack(record["rewards"]),
    }


def dagger(
    cfg: Config,
    data: dict,
    rounds: int,
    steps: int,
    episodes: int,
    batch_size: int,
    seed: int,
    rare_power: float = 0.0,
):
    """Entrena, deja jugar a la red, agrega los estados corregidos y vuelve a entrenar."""
    params, _ = train_imitation(cfg, data, steps, batch_size, seed=seed, rare_power=rare_power)
    for index in range(rounds):
        extra = collect_student(cfg, params, episodes, seed + 100 + index)
        data = jax.tree.map(lambda a, b: jnp.concatenate([a, b], axis=1), data, extra)
        print(f"ronda {index + 1}: {data['actions'].shape[1]} episodios", flush=True)
        params, _ = train_imitation(
            cfg, data, steps, batch_size, seed=seed + index, params0=params, rare_power=rare_power
        )
    return params


def discounted_returns(rewards: jax.Array, gamma: float, scale: float) -> jax.Array:
    """Retorno descontado hacia atrás sobre el eje del tiempo, ya escalado."""

    def back(carry, reward):
        """Acumula el retorno del paso siguiente."""
        carry = reward * scale + gamma * carry
        return carry, carry

    _, out = jax.lax.scan(back, jnp.zeros_like(rewards[0]), rewards, reverse=True)
    return out


def loss_fn(
    params, model: ActorCriticGRU, batch: dict, hidden: int, vf_coef: float, class_weights=None
):
    """Entropía cruzada sobre las acciones del experto y MSE sobre el retorno."""
    obs: Obs = batch["obs"]
    t_axis = obs.alive.shape[0]
    first = (jnp.arange(t_axis) == 0)[:, None, None]
    resets = first | ~obs.alive

    def sequence(carry, data):
        """Reinicia la memoria como en el entrenamiento: al empezar y con las civs muertas."""
        o, reset = data
        carry = jnp.where(reset[..., None], 0, carry)
        carry, logits, value = model.apply(params, carry, flatten_obs(o), o.agent_id)
        return carry, (logits, value)

    carry0 = jnp.zeros((*obs.alive.shape[1:], hidden))
    _, (logits, values) = jax.lax.scan(sequence, carry0, (obs, resets))
    mask = obs.alive.astype(float)
    count = jnp.maximum(mask.sum(), 1)
    log_probs = jax.nn.log_softmax(logits)
    taken = jnp.take_along_axis(log_probs, batch["actions"][..., None], -1)[..., 0]
    weight = 1.0 if class_weights is None else class_weights[batch["actions"]]
    actor = -(mask * weight * taken).sum() / count
    critic = (mask * (values - batch["returns"]) ** 2).sum() / count
    accuracy = (mask * (jnp.argmax(logits, -1) == batch["actions"])).sum() / count
    return actor + vf_coef * critic, {"actor": actor, "critic": critic, "accuracy": accuracy}


def take_episodes(data: dict, index: jax.Array) -> dict:
    """Selecciona episodios completos sobre el eje del lote."""
    return jax.tree.map(lambda x: x[:, index], data)


def train_imitation(
    cfg: Config,
    data: dict,
    steps: int,
    batch_size: int = 32,
    lr: float = 2e-3,
    seed: int = 0,
    params0=None,
    rare_power: float = 0.0,
):
    """Entrena la red recurrente sobre los episodios de los expertos y devuelve sus pesos."""
    hidden = cfg.train.gru_hidden
    model = ActorCriticGRU(cfg.env.n_discrete, hidden)
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    sample = jax.tree.map(lambda x: x[0], data["obs"])
    carry = jnp.zeros((*sample.alive.shape, hidden))
    params = model.init(init_key, carry, flatten_obs(sample), sample.agent_id)
    params = params if params0 is None else params0
    returns = discounted_returns(data["rewards"], cfg.train.gamma, cfg.train.reward_scale)
    data = {"obs": data["obs"], "actions": data["actions"], "returns": returns}
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    opt_state = optimizer.init(params)
    vf_coef = cfg.train.vf_coef
    class_weights = None
    if rare_power > 0:
        # pesa cada acción por (1 / frecuencia) ** rare_power, normalizado a media 1 y con tope
        alive = np.asarray(data["obs"].alive)
        counts = np.bincount(np.asarray(data["actions"])[alive], minlength=cfg.env.n_discrete)
        weights = (1.0 / np.maximum(counts, 1)) ** rare_power
        weights = np.minimum(weights / (weights * counts).sum() * counts.sum(), 10.0)
        class_weights = jnp.asarray(weights, dtype=jnp.float32)
        print("pesos por acción:", np.round(weights, 2).tolist(), flush=True)

    @jax.jit
    def update(params, opt_state, batch):
        """Un paso de gradiente sobre un lote de episodios completos."""
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, model, batch, hidden, vf_coef, class_weights
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss, metrics

    episodes = data["actions"].shape[1]
    history = []
    for index in range(steps):
        key, pick = jax.random.split(key)
        chosen = jax.random.choice(pick, episodes, (min(batch_size, episodes),), replace=False)
        batch = take_episodes(data, chosen)
        params, opt_state, loss, metrics = update(params, opt_state, batch)
        if index % 25 == 0 or index == steps - 1:
            history.append({"step": index, **{k: float(v) for k, v in metrics.items()}})
            print(history[-1], flush=True)
    return params, history


def save_policy(cfg: Config, params, out: Path) -> Path:
    """Escribe config.yaml y checkpoints/final.msgpack con el formato de checkpoint_policy."""
    out = Path(out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(
        yaml.safe_dump(to_dict(cfg), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    path = out / "checkpoints" / "final.msgpack"
    path.write_bytes(serialization.to_bytes({"train_state": {"params": params}, "iteration": 0}))
    return path


def main() -> None:
    """Clona a los expertos en la config indicada y guarda los pesos."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[], metavar="CLAVE=VALOR", dest="extra")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dagger", type=int, default=0, help="rondas de DAgger tras la imitación")
    p.add_argument("--dagger-episodes", type=int, default=128)
    p.add_argument("--rare-power", type=float, default=0.0, help="peso por rareza de la acción")
    args = p.parse_args()
    overrides = {k: yaml.safe_load(v) for k, v in (e.split("=", 1) for e in args.extra)}
    cfg = load_config(args.config, overrides)
    data = collect(cfg, args.episodes, args.seed)
    print("episodios:", data["actions"].shape, flush=True)
    if args.dagger:
        params = dagger(
            cfg,
            data,
            args.dagger,
            args.steps,
            args.dagger_episodes,
            args.batch,
            args.seed,
            args.rare_power,
        )
    else:
        params, _ = train_imitation(
            cfg, data, args.steps, args.batch, seed=args.seed, rare_power=args.rare_power
        )
    print("guardado:", save_policy(cfg, params, args.out))


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    main()
