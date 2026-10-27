"""Ventajas, secuencias recurrentes y persistencia de una actualización real."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from darkforest.config import Config, EnvConfig, LoggingConfig, SelfPlayConfig, TrainConfig
from darkforest.env import DarkForestEnv
from darkforest.trajectories import load_episode
from train.episodes import checkpoint_policy, collect_episode
from train.ippo import compute_gae, initialize, make_rollout, train
from train.logging_utils import run_dir


def tiny_config(out_dir="runs"):
    """Limita el presupuesto de la prueba a ocho pasos de entorno."""
    return Config(
        env=EnvConfig(n_civs=2, max_steps=3),
        train=TrainConfig(
            n_envs=2,
            rollout_len=2,
            total_steps=8,
            gru_hidden=8,
            update_epochs=1,
            self_play=SelfPlayConfig(rho=0),
        ),
        logging=LoggingConfig(out_dir=str(out_dir)),
    )


def test_gae_matches_hand_calculation_and_time_limit_bootstrap():
    """Distingue muerte, fin de episodio y corte de recolección."""
    rewards = jnp.array([[1.0], [2.0]])
    values = jnp.array([[0.5], [1.0]])
    dones = jnp.array([[False], [True]])
    actual = compute_gae(rewards, values, dones, jnp.array([10.0]), 0.9, 0.8)
    np.testing.assert_allclose(actual, [[2.12], [1.0]], atol=1e-6)
    truncated = compute_gae(
        rewards, values, dones, jnp.array([10.0]), 0.9, 0.8, terminated=jnp.zeros_like(dones)
    )
    np.testing.assert_allclose(truncated, [[8.6], [10.0]], atol=1e-6)


def test_rollout_preserves_memory_and_resets_at_episode_boundaries():
    """Un fragmento intermedio conserva carry y un episodio nuevo lo reinicia."""
    cfg = tiny_config()
    key = jax.random.PRNGKey(0)
    state, runner = initialize(cfg, key)
    rollout = jax.jit(make_rollout(DarkForestEnv(cfg.env), cfg))
    assignments = jnp.full((2, 2), -1)
    keys = jax.random.split(key, 2)
    runner, first = rollout(keys[0], state.params, None, assignments, runner)
    assert np.all(np.asarray(runner.state.t) == 2)
    assert np.any(np.asarray(runner.carry) != 0)
    assert not runner.reset.any()
    carry = runner.carry
    runner, second = rollout(keys[1], state.params, None, assignments, runner)
    np.testing.assert_array_equal(second["initial_carry"], carry)
    assert second["resets"][1].all()
    assert np.all(np.asarray(runner.state.t) == 1)
    assert first["rewards"].shape == (2, 2, 2)


def test_training_updates_and_checkpoint_replays(tmp_path):
    """Entrena, recarga los pesos y reproduce exactamente el episodio evaluado."""
    cfg = tiny_config(tmp_path)
    state = train(cfg, "prueba")
    directory = run_dir(cfg, "prueba")
    import json

    report = json.loads((directory / "evaluation.json").read_text())
    assert report["parameter_delta_norm"] > 0
    assert int(state.step) == 2
    restored_cfg, policy = checkpoint_policy(directory / "checkpoints/final.msgpack")
    assert restored_cfg == cfg
    evaluation_key = jax.random.split(jax.random.PRNGKey(cfg.train.seed), 3)[2]
    replay = collect_episode(DarkForestEnv(cfg.env), policy, evaluation_key)
    saved, _ = load_episode(directory / "trajectories/episode_000002.npz")
    np.testing.assert_array_equal(replay["actions"], saved["actions"])
    np.testing.assert_array_equal(replay["rewards"], saved["rewards"])
    with pytest.raises(FileExistsError):
        train(cfg, "prueba")


def test_nonzero_selfplay_is_not_silently_ignored():
    """El entrenador inicial informa que todavía no puede usar snapshots."""
    cfg = tiny_config()
    cfg = replace(cfg, train=replace(cfg.train, self_play=SelfPlayConfig(rho=0.3)))
    with pytest.raises(NotImplementedError, match="rho=0"):
        make_rollout(DarkForestEnv(cfg.env), cfg)
