"""IPPO con parámetros compartidos, GRU y autojuego ficticio.

Uso: python -m train.ippo --config configs/base.yaml [clave=valor ...]
     python -m train.ippo --resume runs/<config>/<seed> [train.total_steps=N]
"""

from __future__ import annotations

import argparse
import json
import os
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
from train import socios
from train.episodes import collect_episode, save_episode
from train.logging_utils import Logger, record_resume
from train.selfplay import Population


@struct.dataclass
class RunnerState:
    """Estado del entorno y memoria conservados entre fragmentos de recolección."""

    state: EnvState
    obs: Obs
    carry: jax.Array
    reset: jax.Array
    assign: jax.Array  # (B, N) código de política de cada civilización, ver train.socios
    aux: socios.Auxiliar  # memoria de las estrategias fijas
    pop_carry: jax.Array  # memoria de la red del snapshot que juega como socio


def initialize(cfg: Config, key: jax.Array) -> tuple[TrainState, RunnerState]:
    """Inicializa entornos independientes, parámetros y optimizador."""
    env_key, model_key = jax.random.split(key)
    assign_key = jax.random.fold_in(key, 1)
    env = DarkForestEnv(cfg.env)
    obs, state = jax.vmap(env.reset)(jax.random.split(env_key, cfg.train.n_envs))
    model = ActorCriticGRU(cfg.env.n_discrete, cfg.train.gru_hidden)
    carry = jnp.zeros((*obs.alive.shape, cfg.train.gru_hidden))
    params = model.init(model_key, carry, flatten_obs(obs), obs.agent_id)
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.train.max_grad_norm), optax.adam(cfg.train.lr)
    )
    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    runner = RunnerState(
        state,
        obs,
        carry,
        jnp.ones_like(obs.alive),
        socios.draw_assignment(assign_key, cfg, cfg.train.n_envs),
        socios.initial_auxiliar(cfg.train.n_envs, cfg.env.n_civs),
        jnp.zeros_like(carry),
    )
    return train_state, runner


def make_rollout(env: DarkForestEnv, cfg: Config):
    """Construye rollouts recurrentes con socios opcionales (snapshots y estrategias fijas)."""
    play = cfg.train.self_play
    with_partners = play.rho > 0
    model = ActorCriticGRU(cfg.env.n_discrete, cfg.train.gru_hidden)

    def rollout(key, params, pop_params, assignment, env_state):
        """Conserva memoria entre fragmentos y reinicia únicamente episodios terminados."""
        if with_partners and pop_params is None and play.w_snapshot > 0:
            raise ValueError("Con w_snapshot > 0 el rollout necesita los pesos de un snapshot")
        initial_carry = env_state.carry
        env_state = env_state.replace(assign=assignment)

        def transition(runner, random_key):
            """Recoge una transición y su valor antes del reinicio automático."""
            action_key, step_key, reset_key = jax.random.split(random_key, 3)
            pop_key, draw_key = jax.random.split(jax.random.fold_in(random_key, 7))
            resets = runner.reset | ~runner.obs.alive
            memory = jnp.where(resets[..., None], 0, runner.carry)
            flat = flatten_obs(runner.obs)
            carry, logits, values = model.apply(params, memory, flat, runner.obs.agent_id)
            actions = jax.random.categorical(action_key, logits)
            log_prob = jnp.take_along_axis(jax.nn.log_softmax(logits), actions[..., None], axis=-1)[
                ..., 0
            ]
            pop_carry = runner.pop_carry
            if with_partners:
                snapshot = None
                if pop_params is not None:
                    pop_memory = jnp.where(resets[..., None], 0, runner.pop_carry)
                    pop_carry, pop_logits, _ = model.apply(
                        pop_params, pop_memory, flat, runner.obs.agent_id
                    )
                    snapshot = jax.random.categorical(pop_key, pop_logits)
                fixed = socios.fixed_actions(cfg, runner.state, runner.obs, runner.aux)
                actions = socios.choose_actions(runner.assign, actions, snapshot, fixed)
            out = jax.vmap(env.step)(
                jax.random.split(step_key, cfg.train.n_envs), runner.state, actions
            )
            aux = runner.aux
            if with_partners:
                aux = socios.update_auxiliar(cfg, aux, runner.state, out.state, actions)
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
            assign = runner.assign
            if with_partners:
                fresh = socios.draw_assignment(draw_key, cfg, cfg.train.n_envs)
                assign = jnp.where(finished[:, None], fresh, assign)
                keep = ~finished
                aux = socios.Auxiliar(
                    jnp.where(keep, aux.first, 0),
                    jnp.where(keep, aux.second, 0),
                    aux.fired & keep[:, None],
                )
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
                "valid": runner.obs.alive & (runner.assign == socios.APRENDIZ),
            }
            return RunnerState(
                state,
                obs,
                carry,
                jnp.broadcast_to(finished[:, None], obs.alive.shape),
                assign,
                aux,
                pop_carry,
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


def ppo_loss(params, batch, cfg: Config, ent_coef=None):
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
    ent_coef = cfg.train.ent_coef if ent_coef is None else ent_coef
    loss = actor + cfg.train.vf_coef * critic - ent_coef * entropy
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
    def update(state, batch, ent_coef=cfg.train.ent_coef):
        """Actualiza parámetros y devuelve diagnósticos del gradiente."""
        (_, metrics), grads = jax.value_and_grad(ppo_loss, has_aux=True)(
            state.params, batch, cfg, ent_coef
        )
        return state.apply_gradients(grads=grads), {**metrics, "grad_norm": optax.tree.norm(grads)}

    return update


def take_envs(batch: dict, index: jax.Array) -> dict:
    """Selecciona entornos completos para que cada minibatch conserve sus secuencias GRU."""
    envs_last = {name: value[index] for name, value in batch.items() if name == "initial_carry"}
    rest = {
        name: jax.tree.map(lambda x: x[:, index], value)
        for name, value in batch.items()
        if name != "initial_carry"
    }
    return {**rest, **envs_last}


def run_iteration(
    cfg: Config,
    env: DarkForestEnv,
    rollout,
    update,
    state: TrainState,
    runner: RunnerState,
    key: jax.Array,
    assignment: jax.Array | None,
    progress: float = 0.0,
    pop_params=None,
) -> tuple[TrainState, RunnerState, jax.Array, dict[str, jax.Array]]:
    """Una iteración: recolecta, calcula ventajas, actualiza y devuelve las métricas.

    progress en [0, 1] indica el avance del entrenamiento y solo sirve para decaer la entropía.
    pop_params son los pesos del snapshot que juega de socio (solo con self_play.rho > 0) y
    assignment None conserva las asignaciones que ya lleva el runner.
    """
    key, rollout_key = jax.random.split(key)
    assignment = runner.assign if assignment is None else assignment
    runner, batch = rollout(rollout_key, state.params, pop_params, assignment, runner)
    advantages = compute_gae(
        batch["rewards"] * cfg.train.reward_scale,
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
    batch["advantages"] = jax.lax.stop_gradient((advantages - mean) / jnp.sqrt(variance + 1e-8))
    final = cfg.train.ent_coef_final
    ent_coef = cfg.train.ent_coef
    if final is not None:
        ent_coef = cfg.train.ent_coef + (final - cfg.train.ent_coef) * min(max(progress, 0.0), 1.0)
    parts = cfg.train.num_minibatches
    for _ in range(cfg.train.update_epochs):
        if parts == 1:
            state, metrics = update(state, batch, ent_coef)
            continue
        key, shuffle_key = jax.random.split(key)
        order = jax.random.permutation(shuffle_key, cfg.train.n_envs).reshape(parts, -1)
        for index in order:
            state, metrics = update(state, take_envs(batch, index), ent_coef)
    return state, runner, key, {**metrics, **rollout_metrics(batch, env)}


def save_checkpoint(
    directory: Path, state: TrainState, runner: RunnerState, key: jax.Array, iteration: int
) -> None:
    """Escribe ultimo.msgpack de forma atómica para que una interrupción no lo deje a medias."""
    folder = directory / "checkpoints"
    folder.mkdir(exist_ok=True)
    payload = {"train_state": state, "runner": runner, "key": key, "iteration": iteration}
    temporary = folder / "ultimo.msgpack.tmp"
    temporary.write_bytes(serialization.to_bytes(payload))
    os.replace(temporary, folder / "ultimo.msgpack")


def save_snapshot(directory: Path, params, iteration: int) -> None:
    """Guarda solo los pesos de una iteración con el formato que lee checkpoint_policy."""
    folder = directory / "snapshots"
    folder.mkdir(exist_ok=True)
    payload = {"train_state": {"params": params}, "iteration": iteration}
    (folder / f"iter_{iteration:05d}.msgpack").write_bytes(serialization.to_bytes(payload))


def train(cfg: Config, name: str, *, resume: Path | None = None) -> TrainState:
    """Entrena IPPO recurrente y guarda un checkpoint y episodios de comparación.

    Con resume continúa desde checkpoints/ultimo.msgpack de esa corrida, con el mismo estado,
    memoria y llave aleatoria, de modo que el resultado coincide con el de una corrida seguida.
    """
    env = DarkForestEnv(cfg.env)
    rollout = jax.jit(make_rollout(env, cfg))
    chunk = cfg.train.n_envs * cfg.train.rollout_len
    if cfg.train.total_steps % chunk:
        raise ValueError("train.total_steps: debe ser múltiplo de n_envs * rollout_len")
    key, init_key, evaluation_key = jax.random.split(jax.random.PRNGKey(cfg.train.seed), 3)
    state, runner = initialize(cfg, init_key)
    initial_params = state.params
    model = ActorCriticGRU(cfg.env.n_discrete, cfg.train.gru_hidden)
    if cfg.train.init_from is not None:
        loaded = serialization.msgpack_restore(Path(cfg.train.init_from).read_bytes())
        state = state.replace(
            params=serialization.from_state_dict(state.params, loaded["train_state"]["params"])
        )
        initial_params = state.params
    population = Population(cfg.train.self_play.max_snapshots)
    if cfg.train.partner_checkpoint is not None:
        frozen = serialization.msgpack_restore(Path(cfg.train.partner_checkpoint).read_bytes())
        population.add(serialization.from_state_dict(state.params, frozen["train_state"]["params"]))
    else:
        population.add(state.params)
    update = make_update(cfg)
    total = cfg.train.total_steps // chunk
    start = 0
    if resume is not None:
        saved = resume / "checkpoints" / "ultimo.msgpack"
        target = {"train_state": state, "runner": runner, "key": key, "iteration": 0}
        restored = serialization.from_bytes(target, saved.read_bytes())
        state, runner, key = restored["train_state"], restored["runner"], restored["key"]
        start = int(restored["iteration"])
        if start >= total:
            raise ValueError(
                f"train.total_steps: la corrida ya hizo {start} de {total} iteraciones"
            )
        record_resume(resume, start, cfg.train.total_steps)
    logger = Logger(cfg, name, resume)
    print(f"Corrida: {logger.dir}", flush=True)
    try:
        before = collect_episode(env, NetworkPolicy(model, initial_params), evaluation_key)
        if resume is None:
            save_episode(logger.dir, before, cfg=cfg, policy_name="red_inicial", episode_id=1)
        for index in range(start, total):
            partner = (
                population.sample(jax.random.fold_in(key, index))
                if cfg.train.self_play.rho
                else None
            )
            state, runner, key, metrics = run_iteration(
                cfg, env, rollout, update, state, runner, key, None, index / total, partner
            )
            if (
                cfg.train.partner_checkpoint is None
                and (index + 1) % cfg.train.self_play.snapshot_every == 0
            ):
                population.add(state.params)
            metrics = {name: float(value) for name, value in metrics.items()}
            if not all(np.isfinite(value) for value in metrics.values()):
                raise FloatingPointError("El entrenamiento produjo métricas no finitas")
            logger.log((index + 1) * chunk, metrics)
            if cfg.train.save_snapshots and (index + 1) % cfg.train.self_play.snapshot_every == 0:
                save_snapshot(logger.dir, state.params, index + 1)
            every = cfg.train.checkpoint_every
            if every and (index + 1) % every == 0 and index + 1 < total:
                save_checkpoint(logger.dir, state, runner, key, index + 1)
        final = collect_episode(env, NetworkPolicy(model, state.params), evaluation_key)
        episode_id = len(list((logger.dir / "trajectories").glob("episode_*.npz"))) + 1
        save_episode(logger.dir, final, cfg=cfg, policy_name="red_entrenada", episode_id=episode_id)
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
        checkpoints.mkdir(exist_ok=True)
        (checkpoints / "final.msgpack").write_bytes(
            serialization.to_bytes({"train_state": state, "runner": runner, "key": key})
        )
        save_checkpoint(logger.dir, state, runner, key, total)
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
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--config")
    source.add_argument("--resume", help="directorio de una corrida con checkpoints/ultimo.msgpack")
    p.add_argument("--name")
    p.add_argument("overrides", nargs="*", help="clave.punteada=valor")
    args = p.parse_args()
    overrides = {k: _parse(v) for k, v in (o.split("=", 1) for o in args.overrides)}
    if args.resume:
        allowed = {"train.total_steps", "train.checkpoint_every"}
        if not set(overrides) <= allowed:
            p.error(f"al reanudar solo se pueden cambiar {sorted(allowed)}")
        directory = Path(args.resume)
        train(
            load_config(directory / "config.yaml", overrides),
            directory.parent.name,
            resume=directory,
        )
        return
    cfg = load_config(args.config, overrides)
    train(cfg, name=args.name or Path(args.config).stem)


def _parse(v: str):
    """Interpreta el valor de un override con los tipos de YAML."""
    import yaml

    return yaml.safe_load(v)


if __name__ == "__main__":
    main()
