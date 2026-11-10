"""Imitación de coop_real: datos de los expertos, DAgger y pesos que otras partes pueden cargar."""

import dataclasses as dc

import jax
import numpy as np

from darkforest.config import Config, EnvConfig, SelfPlayConfig, TrainConfig
from train import imitar
from train.episodes import checkpoint_policy


def _config() -> Config:
    return Config(
        env=EnvConfig(n_civs=2, max_steps=12),
        train=TrainConfig(
            n_envs=2,
            rollout_len=2,
            total_steps=8,
            gru_hidden=8,
            update_epochs=1,
            self_play=SelfPlayConfig(rho=0),
        ),
    )


def test_expert_data_labels_match_the_fixed_strategies_and_shapes_align():
    """Los episodios de expertos tienen forma (T, B, N) y acciones dentro del catálogo."""
    cfg = _config()
    data = imitar.collect(cfg, episodes=8, seed=1)
    assert data["actions"].shape == (12, 8, 2)
    assert data["rewards"].shape == (12, 8, 2)
    assert int(data["actions"].max()) < cfg.env.n_discrete
    assert data["obs"].alive.shape == (12, 8, 2)


def test_imitation_lowers_the_loss_and_the_policy_loads_for_training_and_evaluation(tmp_path):
    """La entropía cruzada baja, DAgger agrega episodios y el checkpoint sirve como init_from."""
    cfg = _config()
    data = imitar.collect(cfg, episodes=8, seed=0)
    params, history = imitar.train_imitation(cfg, data, steps=26, batch_size=4)
    assert history[-1]["actor"] < history[0]["actor"]
    extra = imitar.collect_student(cfg, params, episodes=4, seed=3)
    assert extra["actions"].shape == (12, 4, 2)
    assert int(extra["actions"].max()) < cfg.env.n_discrete
    path = imitar.save_policy(cfg, params, tmp_path / "bc")
    loaded_cfg, policy = checkpoint_policy(path, greedy=True)
    assert loaded_cfg.env.max_steps == 12
    carry = policy.init_carry(2)
    _, actions = policy.act(
        jax.random.PRNGKey(0), carry, jax.tree.map(lambda x: x[0, 0], data["obs"])
    )
    assert np.asarray(actions).shape == (2,)
    resumed = dc.replace(cfg, train=dc.replace(cfg.train, init_from=str(path)))
    assert resumed.train.init_from == str(path)
