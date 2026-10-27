"""Ejecutamos políticas y guardamos episodios reproducibles."""

import argparse
import hashlib
import json
from dataclasses import fields, replace
from pathlib import Path

import jax
import numpy as np
import yaml
from flax import serialization

from darkforest.archetypes import ARCHETYPES
from darkforest.config import Config, load_config
from darkforest.env import DarkForestEnv
from darkforest.networks import ActorCriticGRU, NetworkPolicy
from darkforest.policies import Policy, RandomPolicy
from train.logging_utils import create_run


def collect_episode(env: DarkForestEnv, policy: Policy, key: jax.Array) -> dict[str, np.ndarray]:
    """Recoge T transiciones y T+1 estados sin mezclar episodios."""
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)
    carry = policy.init_carry(env.n_agents)
    states, observations, transitions = [state], [obs], []
    step, act = jax.jit(env.step), jax.jit(policy.act)
    for _ in range(env.cfg.max_steps):
        key, act_key, step_key = jax.random.split(key, 3)
        carry, actions = act(act_key, carry, obs)
        out = step(step_key, state, actions)
        transitions.append((actions, out.reward, out.done, out.info))
        obs, state = out.obs, out.state
        observations.append(obs)
        states.append(state)
        if bool(out.done.all()):
            break
    trajectory = {}
    for values, prefix in (
        (states, ""),
        (observations, "obs_"),
        ([transition[3] for transition in transitions], "info_"),
    ):
        for field in fields(values[0]):
            if getattr(values[0], field.name) is not None:
                trajectory[prefix + field.name] = np.stack(
                    [np.asarray(getattr(value, field.name)) for value in values]
                )
    for index, name in enumerate(("actions", "rewards", "done")):
        trajectory[name] = np.stack([np.asarray(value[index]) for value in transitions])
    return trajectory


def save_episode(
    directory: Path,
    trajectory: dict[str, np.ndarray],
    *,
    cfg: Config,
    policy_name: str,
    episode_id: int = 1,
) -> Path:
    """Guarda arrays sin pickle y un manifiesto con formas y procedencia."""
    folder = directory / "trajectories"
    folder.mkdir(exist_ok=True)
    path = folder / f"episode_{episode_id:06d}.npz"
    metadata = {
        "schema_version": 1,
        "config_id": directory.parent.name,
        "seed": cfg.train.seed,
        "episode_id": episode_id,
        "policy": policy_name,
        "synthetic": False,
        "n_symbols": cfg.env.n_symbols,
        "signal_speed": cfg.env.signal_speed,
        "proj_speed": cfg.env.proj_speed,
        "arrays": {
            name: {"shape": list(array.shape), "dtype": str(array.dtype)}
            for name, array in trajectory.items()
        },
    }
    with path.open("xb") as stream:
        np.savez_compressed(stream, **trajectory)
    with path.with_suffix(".json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    return path


def checkpoint_policy(path: str | Path) -> tuple[Config, NetworkPolicy]:
    """Carga los pesos para inferencia; conserva la configuración del entrenamiento."""
    path = Path(path)
    cfg = load_config(path.parent.parent / "config.yaml")
    checkpoint = serialization.msgpack_restore(path.read_bytes())
    model = ActorCriticGRU(cfg.env.n_discrete, cfg.train.gru_hidden)
    return cfg, NetworkPolicy(model, checkpoint["train_state"]["params"])


def main() -> None:
    """Genera un episodio con heurísticas o con un checkpoint ya entrenado."""
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config")
    source.add_argument("--checkpoint")
    parser.add_argument("--policy", choices=["random", *ARCHETYPES], default="random")
    parser.add_argument("--name", default="episodio")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out-dir")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    checkpoint_metadata = {}
    if args.checkpoint:
        if args.overrides:
            parser.error("Un checkpoint conserva su configuración; usa --name para otra evaluación")
        cfg, policy = checkpoint_policy(args.checkpoint)
        checkpoint_metadata = {
            "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
            "training_seed": cfg.train.seed,
        }
        cfg = replace(cfg, logging=replace(cfg.logging, out_dir=args.out_dir or "runs"))
        policy_name = "checkpoint"
    else:
        overrides = {
            key: yaml.safe_load(value)
            for key, value in (item.split("=", 1) for item in args.overrides)
        }
        cfg = load_config(args.config, overrides)
        if args.out_dir is not None:
            cfg = replace(cfg, logging=replace(cfg.logging, out_dir=args.out_dir))
        policy = (
            RandomPolicy(cfg.env.n_discrete)
            if args.policy == "random"
            else ARCHETYPES[args.policy](cfg.env)
        )
        policy_name = args.policy
    if args.seed is not None:
        cfg = replace(cfg, train=replace(cfg.train, seed=args.seed))
    directory = create_run(cfg, args.name, "evaluacion_exploratoria")
    if checkpoint_metadata:
        metadata_path = directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata_path.write_text(
            json.dumps({**metadata, **checkpoint_metadata}, indent=2), encoding="utf-8"
        )
    trajectory = collect_episode(DarkForestEnv(cfg.env), policy, jax.random.PRNGKey(cfg.train.seed))
    path = save_episode(directory, trajectory, cfg=cfg, policy_name=policy_name)
    print(f"Episodio guardado: {path}")


if __name__ == "__main__":
    main()
