"""Diagnósticos con políticas fijas: valores calculables y reproducibilidad."""

import jax.numpy as jnp
import numpy as np
import pytest

from train.matriz_pagos import run_pair
from train.valor_cooperacion import PERIODO_ALEATORIO, PERIODO_FIJO, run, update_bursts

SMALL = {"episodes": 4, "steps": 30}


def test_pure_growth_and_silence_match_the_closed_form():
    """Sin anomalía ni ataques el retorno de crecer telescopa a log(1 + 1.05^T) - log 2."""
    growth, survival = run_pair(0.0, "crecer", "crecer", 0, **SMALL)
    assert growth == pytest.approx(np.log1p(1.05**30) - np.log(2), abs=1e-4)
    assert survival == 1
    silence, survival = run_pair(0.0, "silencio", "silencio", 0, **SMALL)
    assert silence == 0
    assert survival == 1


def test_certain_single_shot_kills_a_grower_and_the_victim_loses_log_two():
    """Con p_kill 1 el crecedor muere, y su retorno total es -log 2 aunque haya crecido."""
    over = {"env.p_kill": 1.0}
    victim, survival = run_pair(
        0.0, "crecer", "ataque_unico", 0, episodes=8, steps=100, env_over=over
    )
    assert victim == pytest.approx(-np.log(2), abs=1e-4)
    assert survival == 0


def test_oracle_shelters_survive_where_ignoring_the_anomaly_does_not():
    """Con λ = 1 el oráculo supera a ignorar y la corrida se repite con la misma seed."""
    kwargs = {"episodes": 32, "steps": 100}
    ignore, ignore_survival, _ = run(1.0, PERIODO_ALEATORIO, "ignorar", seed=0, **kwargs)
    oracle, oracle_survival, shelters = run(1.0, PERIODO_ALEATORIO, "oraculo", seed=0, **kwargs)
    assert oracle > ignore
    assert oracle_survival == 1 > ignore_survival
    assert shelters > 0
    assert run(1.0, PERIODO_ALEATORIO, "oraculo", seed=0, **kwargs)[0] == oracle


@pytest.mark.parametrize("world", [PERIODO_FIJO, PERIODO_ALEATORIO])
def test_cooperating_policies_run_and_stay_finite(world):
    """Las políticas solo y cooperativa se ejecutan y devuelven valores finitos."""
    for mode in ("solo", "coop_ideal", "coop_real"):
        values = run(0.5, world, mode, 0.1, seed=0, episodes=8, steps=40)
        assert np.all(np.isfinite(values))


def test_update_bursts_records_the_first_two_changes_only():
    """Registra el primer y el segundo estallido y conserva los siguientes sin cambios."""
    first = second = jnp.zeros(2, dtype=jnp.int32)
    last = jnp.zeros(2, dtype=jnp.int32)
    for burst in (jnp.array([5, 0]), jnp.array([5, 7]), jnp.array([9, 7])):
        first, second = update_bursts(burst, last, first, second)
        last = burst
    np.testing.assert_array_equal(first, [5, 7])
    np.testing.assert_array_equal(second, [9, 0])
