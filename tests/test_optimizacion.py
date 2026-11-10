"""Opciones del optimizador de IPPO: minibatches por entorno, escala de reward y entropía."""

import dataclasses as dc

import jax
import jax.numpy as jnp
import pytest

from darkforest.config import Config, TrainConfig, load_config
from darkforest.env import DarkForestEnv
from train.ippo import initialize, make_rollout, make_update, run_iteration, take_envs


def _config(**train) -> Config:
    base = load_config("configs/aprendizaje_dos.yaml")
    values = dict(n_envs=8, rollout_len=8, update_epochs=2, **train)
    values["self_play"] = dc.replace(base.train.self_play, rho=0.0)
    return dc.replace(base, train=dc.replace(base.train, **values))


def test_minibatches_must_divide_envs():
    with pytest.raises(ValueError, match="num_minibatches"):
        TrainConfig(n_envs=10, num_minibatches=4)


def test_take_envs_keeps_time_axis_and_initial_carry():
    batch = {"x": jnp.zeros((8, 6, 2)), "initial_carry": jnp.zeros((6, 2, 4))}
    sub = take_envs(batch, jnp.array([0, 3]))
    assert sub["x"].shape == (8, 2, 2)
    assert sub["initial_carry"].shape == (2, 2, 4)


@pytest.mark.parametrize("parts", [1, 4])
def test_iteration_with_options_is_finite_and_changes_weights(parts):
    cfg = _config(num_minibatches=parts, reward_scale=0.2, ent_coef=0.05, ent_coef_final=0.005)
    env = DarkForestEnv(cfg.env)
    state, runner = initialize(cfg, jax.random.PRNGKey(0))
    before = state.params
    assignment = jnp.full((cfg.train.n_envs, cfg.env.n_civs), -1)
    state, runner, _, metrics = run_iteration(
        cfg,
        env,
        jax.jit(make_rollout(env, cfg)),
        make_update(cfg),
        state,
        runner,
        jax.random.PRNGKey(1),
        assignment,
        progress=0.5,
    )
    assert all(bool(jnp.isfinite(v)) for v in metrics.values())
    assert int(state.step) == cfg.train.update_epochs * parts
    changed = jax.tree.map(lambda a, b: bool((a != b).any()), before, state.params)
    assert any(jax.tree.leaves(changed))
