"""Integración de episodio, archivos, análisis y reproducción."""

from dataclasses import replace

import jax
import numpy as np
import pytest
from PIL import Image

from analysis.metrics import summarize
from darkforest.archetypes import AlwaysTransmit
from darkforest.config import Config, EnvConfig, LoggingConfig
from darkforest.env import DarkForestEnv
from darkforest.trajectories import load_episode
from train.episodes import collect_episode, save_episode
from train.logging_utils import create_run, run_dir
from viz.galaxy import render_episode


@pytest.mark.parametrize("n", [2, 4])
def test_simulate_save_analyze_and_replay(tmp_path, n):
    """Reproduce archivos sin permitir que el lector vuelva a ejecutar el simulador."""
    cfg = Config(env=EnvConfig(n_civs=n, max_steps=4), logging=LoggingConfig(out_dir=str(tmp_path)))
    env = DarkForestEnv(cfg.env)
    trajectory = collect_episode(env, AlwaysTransmit(cfg.env), jax.random.PRNGKey(0))
    directory = create_run(cfg, "integracion", "evaluacion_exploratoria")
    path = save_episode(directory, trajectory, cfg=cfg, policy_name="transmit")
    restored, metadata = load_episode(path)
    for name in trajectory:
        np.testing.assert_array_equal(restored[name], trajectory[name])
    assert summarize(restored)["steps"] == 4
    output = directory / "videos" / "episodio.gif"
    render_episode(restored, output, fps=4, metadata=metadata)
    with Image.open(output) as image:
        assert image.n_frames == 5
    with pytest.raises(FileExistsError):
        save_episode(directory, trajectory, cfg=cfg, policy_name="transmit")
    assert run_dir(cfg, "integracion") != run_dir(
        replace(cfg, env=replace(cfg.env, growth=0.1)), "integracion"
    )
