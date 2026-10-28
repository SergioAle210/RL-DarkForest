"""IPPO con parámetros compartidos, GRU y autojuego ficticio.

Uso: python -m train.ippo --config configs/base.yaml [clave=valor ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization, struct
from flax.training.train_state import TrainState

from darkforest.config import ACTIONS, Config, load_config
from darkforest.env import DarkForestEnv
from darkforest.networks import ActorCriticGRU, NetworkPolicy, flatten_obs
from darkforest.types import EnvState, Obs
from train.episodes import collect_episode, save_episode
from train.logging_utils import Logger


@struct.dataclass
class RunnerState:
    """Estado del entorno y memoria conservados entre fragmentos de recolección."""

    state: EnvState
    obs: Obs
    carry: jax.Array
    reset: jax.Array


def initialize(cfg: Config, key: jax.Array) -> tuple[TrainState, RunnerState]:
    """Inicializa entornos independientes, parámetros y optimizador."""
    env_key, model_key = jax.random.split(key)
    env = DarkForestEnv(cfg.env)
    obs, state = jax.vmap(env.reset)(jax.random.split(env_key, cfg.train.n_envs))
    model = ActorCriticGRU(cfg.env.n_discrete, cfg.train.gru_hidden)
    carry = jnp.zeros((*obs.alive.shape, cfg.train.gru_hidden))
    params = model.init(model_key, carry, flatten_obs(obs), obs.agent_id)
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.train.max_grad_norm), optax.adam(cfg.train.lr)
    )
    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    return train_state, RunnerState(state, obs, carry, jnp.ones_like(obs.alive))


def make_rollout(env: DarkForestEnv, cfg: Config):
    """Construye rollouts recurrentes; este primer entrenador exige rho=0."""
    if cfg.train.self_play.rho != 0:
        raise NotImplementedError(
            "El autojuego con snapshots está pendiente; usa train.self_play.rho=0"
        )
    model = ActorCriticGRU(cfg.env.n_discrete, cfg.train.gru_hidden)

    def rollout(key, params, pop_params, assignment, env_state):
        """Conserva memoria entre fragmentos y reinicia únicamente episodios terminados."""
        if pop_params is not None:
            raise NotImplementedError("Las poblaciones congeladas se integrarán después")
        initial_carry = env_state.carry

        def transition(runner, random_key):
            """Recoge una transición y su valor antes del reinicio automático."""
            action_key, step_key, reset_key = jax.random.split(random_key, 3)
            resets = runner.reset | ~runner.obs.alive
            memory = jnp.where(resets[..., None], 0, runner.carry)
            carry, logits, values = model.apply(
                params, memory, flatten_obs(runner.obs), runner.obs.agent_id
            )
            actions = jax.random.categorical(action_key, logits)
            log_prob = jnp.take_along_axis(jax.nn.log_softmax(logits), actions[..., None], axis=-1)[
                ..., 0
            ]
            out = jax.vmap(env.step)(
                jax.random.split(step_key, cfg.train.n_envs), runner.state, actions
            )
            carry = jnp.where(out.state.alive[..., None], carry, 0)
            _, _, next_values = model.apply(params, carry, flatten_obs(out.obs), out.obs.agent_id)
            finished = jnp.all(out.done, axis=-1)

            def reset_finished(key, obs, state, done):
                """Genera un mundo nuevo solo al finalizar el episodio anterior."""
                return jax.lax.cond(done, lambda: env.reset(key), lambda: (obs, state))

            obs, state = jax.vmap(reset_finished)(
                jax.random.split(reset_key, cfg.train.n_envs), out.obs, out.state, finished
            )
            carry = jnp.where(finished[..., None, None], 0, carry)
            data = {
                "obs": runner.obs,
                "resets": resets,
                "actions": actions,
                "log_probs": log_prob,
                "values": values,
                "next_values": next_values,
                "rewards": out.reward,
                "dones": out.done,
                "terminated": ~out.state.alive,
                "valid": runner.obs.alive & (assignment == -1),
            }
            return RunnerState(
                state, obs, carry, jnp.broadcast_to(finished[:, None], obs.alive.shape)
            ), data

        runner, batch = jax.lax.scan(
            transition, env_state, jax.random.split(key, cfg.train.rollout_len)
        )
        batch["initial_carry"] = initial_carry
        return runner, batch

    return rollout


def compute_gae(
    rewards,
    values,
    dones,
    last_value,
    gamma: float,
    lam: float,
    *,
    terminated=None,
    next_values=None,
):
    """Calcula GAE con bootstrap en cortes temporales y sin cruzar reinicios."""
    if terminated is None:
        terminated = dones
    if next_values is None:
        next_values = jnp.concatenate([values[1:], last_value[None]], axis=0)
    deltas = rewards + gamma * next_values * (1 - terminated) - values

    def accumulate(carry, values):
        """Acumula ventajas hacia atrás hasta el límite del episodio."""
        delta, done = values
        advantage = delta + gamma * lam * (1 - done) * carry
        return advantage, advantage

    return jax.lax.scan(accumulate, jnp.zeros_like(last_value), (deltas, dones), reverse=True)[1]


def ppo_loss(params, batch, cfg: Config):
    """Recalcula secuencias GRU y aplica PPO solo a agentes vivos de la política actual."""
    model = ActorCriticGRU(cfg.env.n_discrete, cfg.train.gru_hidden)

    def sequence(carry, data):
        """Reinicia la memoria en los mismos límites que durante la recolección."""
        obs, reset = data
        carry = jnp.where(reset[..., None], 0, carry)
        carry, logits, value = model.apply(params, carry, flatten_obs(obs), obs.agent_id)
        return carry, (logits, value)

    _, (logits, values) = jax.lax.scan(
        sequence, batch["initial_carry"], (batch["obs"], batch["resets"])
    )
    mask = batch["valid"].astype(float)
    count = jnp.maximum(mask.sum(), 1)
    log_probs = jax.nn.log_softmax(logits)
    selected = jnp.take_along_axis(log_probs, batch["actions"][..., None], -1)[..., 0]
    ratio = jnp.exp(selected - batch["log_probs"])
    clipped = jnp.clip(ratio, 1 - cfg.train.clip_eps, 1 + cfg.train.clip_eps)
    surrogate = jnp.minimum(ratio * batch["advantages"], clipped * batch["advantages"])
    actor = -(mask * surrogate).sum() / count
    critic = (mask * (values - batch["returns"]) ** 2).sum() / count
    entropy = -(mask * (jnp.exp(log_probs) * log_probs).sum(axis=-1)).sum() / count
    loss = actor + cfg.train.vf_coef * critic - cfg.train.ent_coef * entropy
    return loss, {"loss": loss, "actor_loss": actor, "value_loss": critic, "entropy": entropy}


def rollout_metrics(batch: dict, env: DarkForestEnv) -> dict[str, jax.Array]:
    """Recompensa por paso, muertes y uso de cada familia de acciones en un rollout."""
    valid = batch["valid"]
    count = jnp.maximum(valid.sum(), 1)
    kind, _ = env.decode_action(batch["actions"])
    metrics = {
        "mean_reward": (batch["rewards"] * valid).sum() / count,
        "death_rate": (valid & batch["terminated"]).sum() / count,
    }
    for index, name in enumerate(ACTIONS):
        metrics[f"action_{name}"] = ((kind == index) & valid).sum() / count
    return metrics


def make_update(cfg: Config):
    """Compila una actualización con gradientes recurrentes y recorte de norma."""

    @jax.jit
    def update(state, batch):
        """Actualiza parámetros y devuelve diagnósticos del gradiente."""
        (_, metrics), grads = jax.value_and_grad(ppo_loss, has_aux=True)(state.params, batch, cfg)
        return state.apply_gradients(grads=grads), {**metrics, "grad_norm": optax.tree.norm(grads)}

    return update


def train(cfg: Config, name: str) -> TrainState:
    """Entrena IPPO recurrente y guarda un checkpoint y episodios de comparación."""
    env = DarkForestEnv(cfg.env)
    rollout = jax.jit(make_rollout(env, cfg))
    chunk = cfg.train.n_envs * cfg.train.rollout_len
    if cfg.train.total_steps % chunk:
        raise ValueError("train.total_steps: debe ser múltiplo de n_envs * rollout_len")
    key, init_key, evaluation_key = jax.random.split(jax.random.PRNGKey(cfg.train.seed), 3)
    state, runner = initialize(cfg, init_key)
    initial_params = state.params
    model = ActorCriticGRU(cfg.env.n_discrete, cfg.train.gru_hidden)
    assignment = jnp.full((cfg.train.n_envs, cfg.env.n_civs), -1)
    update = make_update(cfg)
    logger = Logger(cfg, name)
    print(f"Corrida: {logger.dir}", flush=True)
    try:
        before = collect_episode(env, NetworkPolicy(model, state.params), evaluation_key)
        save_episode(logger.dir, before, cfg=cfg, policy_name="red_inicial", episode_id=1)
        for index in range(cfg.train.total_steps // chunk):
            key, rollout_key = jax.random.split(key)
            runner, batch = rollout(rollout_key, state.params, None, assignment, runner)
            advantages = compute_gae(
                batch["rewards"],
                batch["values"],
                batch["dones"],
                batch["next_values"][-1],
                cfg.train.gamma,
                cfg.train.gae_lambda,
                terminated=batch["terminated"],
                next_values=batch["next_values"],
            )
            mask = batch["valid"].astype(float)
            count = jnp.maximum(mask.sum(), 1)
            mean = (advantages * mask).sum() / count
            variance = (((advantages - mean) ** 2) * mask).sum() / count
            batch["returns"] = jax.lax.stop_gradient(advantages + batch["values"])
            batch["advantages"] = jax.lax.stop_gradient(
                (advantages - mean) / jnp.sqrt(variance + 1e-8)
            )
            for _ in range(cfg.train.update_epochs):
                state, metrics = update(state, batch)
            metrics = {**metrics, **rollout_metrics(batch, env)}
            metrics = {name: float(value) for name, value in metrics.items()}
            if not all(np.isfinite(value) for value in metrics.values()):
                raise FloatingPointError("El entrenamiento produjo métricas no finitas")
            logger.log((index + 1) * chunk, metrics)
        final = collect_episode(env, NetworkPolicy(model, state.params), evaluation_key)
        save_episode(logger.dir, final, cfg=cfg, policy_name="red_entrenada", episode_id=2)
        delta = float(
            optax.tree.norm(jax.tree.map(lambda a, b: a - b, state.params, initial_params))
        )
        report = {
            "environment_steps": cfg.train.total_steps,
            "optimizer_steps": int(state.step),
            "parameter_delta_norm": delta,
            "exploratory": True,
            "mean_return_before": float(before["rewards"].sum(axis=0).mean()),
            "mean_return_after": float(final["rewards"].sum(axis=0).mean()),
        }
        (logger.dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        checkpoints = logger.dir / "checkpoints"
        checkpoints.mkdir()
        (checkpoints / "final.msgpack").write_bytes(
            serialization.to_bytes({"train_state": state, "runner": runner, "key": key})
        )
        print(json.dumps(report), flush=True)
    finally:
        logger.close()
    return state


def evaluate(cfg: Config, params: Any, key: jax.Array) -> dict[str, float]:
    """Evalúa la política compartida; torneos y clasificación quedan para otra entrega."""
    model = ActorCriticGRU(cfg.env.n_discrete, cfg.train.gru_hidden)
    trajectory = collect_episode(DarkForestEnv(cfg.env), NetworkPolicy(model, params), key)
    return {
        "mean_return": float(trajectory["rewards"].sum(axis=0).mean()),
        "survival": float(trajectory["alive"][-1].mean()),
    }


def main() -> None:
    """Carga una configuración y ejecuta el presupuesto de entrenamiento indicado."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--name")
    p.add_argument("overrides", nargs="*", help="clave.punteada=valor")
    args = p.parse_args()
    overrides = {k: _parse(v) for k, v in (o.split("=", 1) for o in args.overrides)}
    cfg = load_config(args.config, overrides)
    train(cfg, name=args.name or Path(args.config).stem)


def _parse(v: str):
    """Interpreta el valor de un override con los tipos de YAML."""
    import yaml

    return yaml.safe_load(v)


if __name__ == "__main__":
    main()
