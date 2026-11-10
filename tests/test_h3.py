"""H3: señal verificable, engaño y señales costosas."""

import dataclasses as dc

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from darkforest.config import EnvConfig
from darkforest.env import DarkForestEnv


def _setup(**options):
    cfg = dc.replace(EnvConfig(n_civs=2, max_steps=8), **options)
    env = DarkForestEnv(cfg)
    _, state = env.reset(jax.random.PRNGKey(0))
    state = env.step(jax.random.PRNGKey(1), state, jnp.full((2,), 2)).state  # ambas miden
    return cfg, env, state


def _send(env, state, actions):
    """Un paso en que solo la civilización 0 actúa y la 1 crece."""
    return env.step(jax.random.PRNGKey(2), state, jnp.array([actions, 0]))


def test_share_is_executed_only_when_the_verifiable_signal_exists():
    """Con la señal verificable se ejecuta y cuesta; sin ella queda anulada y no cuesta."""
    cfg, env, before = _setup()
    share = 3 + cfg.n_symbols
    out = _send(env, before, share)
    assert int(out.info.executed_kind[0]) == 4
    assert float(out.state.resources[0]) < float(before.resources[0])
    cfg, env, before = _setup(share_enabled=False)
    out = _send(env, before, share)
    assert int(out.info.executed_kind[0]) == -1
    assert float(out.state.resources[0]) == float(before.resources[0])


def test_a_lie_looks_like_an_unverified_share_with_a_false_measurement():
    """La mentira lleva la misma etiqueta que un compartir sin verificar y otra medición."""
    cfg, env, before = _setup(deception=True)
    lie = 3 + cfg.n_symbols - 1
    share = 3 + cfg.n_symbols
    truthful = _send(env, before, share)
    liar = _send(env, before, lie)
    slot = int(before.t) % (cfg.signal_slots // cfg.n_civs) * cfg.n_civs  # emisión de la civ 0
    true_content = np.asarray(truthful.state.sig_content[slot])
    fake_content = np.asarray(liar.state.sig_content[slot])
    assert fake_content[0] == true_content[0] == cfg.n_symbols  # misma etiqueta sin verificar
    assert not np.allclose(fake_content[1:], true_content[1:])  # otra medición


def test_costly_signals_make_the_true_share_verifiable_and_dearer():
    """Con costly_signals el compartir verdadero cuesta más y lleva una etiqueta propia."""
    cfg, env, before = _setup(deception=True, costly_signals=True)
    lie, share = 3 + cfg.n_symbols - 1, 3 + cfg.n_symbols
    slot = int(before.t) % (cfg.signal_slots // cfg.n_civs) * cfg.n_civs
    truthful, liar = _send(env, before, share), _send(env, before, lie)
    assert float(truthful.state.sig_content[slot, 0]) == cfg.n_symbols + 1  # verificado
    assert float(liar.state.sig_content[slot, 0]) == cfg.n_symbols  # la mentira no puede
    paid_true = float(before.resources[0] - truthful.state.resources[0])
    paid_lie = float(before.resources[0] - liar.state.resources[0])
    assert paid_true == pytest.approx(cfg.cost_verify) and paid_lie == pytest.approx(cfg.cost_tx)


def test_default_world_is_unchanged_by_the_new_options():
    """Sin deception ni costly_signals el último símbolo es un símbolo normal."""
    cfg, env, before = _setup()
    last_symbol = 3 + cfg.n_symbols - 1
    out = _send(env, before, last_symbol)
    slot = int(before.t) % (cfg.signal_slots // cfg.n_civs) * cfg.n_civs
    assert float(out.state.sig_content[slot, 0]) == cfg.n_symbols - 1
    assert float(jnp.abs(out.state.sig_content[slot, 1:]).sum()) == 0.0
