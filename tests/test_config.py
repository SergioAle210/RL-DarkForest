from pathlib import Path

import pytest

from darkforest.config import load_config
from train.sweep import expand_grid

CONFIGS = Path(__file__).parents[1] / "configs"


def test_base_loads():
    cfg = load_config(CONFIGS / "base.yaml")
    assert cfg.env.n_civs == 4
    assert cfg.env.proj_speed <= cfg.env.signal_speed


def test_extends_and_overrides():
    cfg = load_config(CONFIGS / "h1_sanity.yaml", {"env.signal_speed": 0.5})
    assert cfg.env.anomaly.lethality == 0.0
    assert cfg.env.n_civs == 2
    assert cfg.env.signal_speed == 0.5


def test_unknown_key_rejected():
    with pytest.raises(KeyError):
        load_config(CONFIGS / "base.yaml", {"env.nope": 1})


def test_grid_expansion():
    cells = expand_grid({"a": [1, 2], "b": [3, 4, 5]})
    assert len(cells) == 6
