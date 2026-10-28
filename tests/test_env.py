"""Invariantes físicos y causales del simulador."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from darkforest.config import AnomalyConfig, EnvConfig
from darkforest.env import DarkForestEnv, reward_fn


def environment(**overrides) -> DarkForestEnv:
    """Construye un control sin anomalía letal y con detección determinista."""
    cfg = EnvConfig(
        n_civs=2,
        detection_low=1,
        detection_high=1,
        detection_decay=0,
        anomaly=AnomalyConfig(lethality=0),
    )
    return DarkForestEnv(replace(cfg, **overrides))


@pytest.mark.parametrize("n", [2, 4, 6, 8])
def test_reset_shapes_reproducibility_and_batching(n):
    """Verifica formas estáticas y repetición con la misma clave."""
    env = environment(n_civs=n)
    key = jax.random.PRNGKey(7)
    obs, state = jax.jit(env.reset)(key)
    assert obs.detections.shape == (n, 4, 10)
    assert not obs.det_mask.any()
    np.testing.assert_array_equal(state.pos, env.reset(key)[1].pos)
    batch = jax.jit(jax.vmap(env.reset))(jax.random.split(key, 2))[1]
    assert batch.pos.shape == (2, n, 2)
    assert np.all((state.pos >= 0) & (state.pos < 1))


def test_signals_travel_at_c_and_cross_the_torus():
    """Una transmisión distante aparece al cruzar su frente, nunca antes."""
    env = environment()
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    state = state.replace(pos=jnp.array([[0.9, 0.0], [0.15, 0.0]]))
    step = jax.jit(env.step)
    out = step(key, state, jnp.array([3, 1]))
    assert not out.obs.det_mask.any()
    out = step(key, out.state, jnp.array([1, 1]))
    assert not out.obs.det_mask.any()
    out = step(key, out.state, jnp.array([1, 1]))
    assert bool(out.obs.det_mask[1, 0])
    assert float(out.obs.detections[1, 0, 2]) == pytest.approx(0.25)
    assert float(out.obs.detections[1, 0, 3]) == 0
    assert bool(out.info.receptions[1, 0])


def test_projectile_delay_death_and_dead_agents():
    """El ataque tarda en llegar y una civilización eliminada deja de actuar."""
    env = environment(p_kill=1)
    key = jax.random.PRNGKey(2)
    _, state = env.reset(key)
    state = state.replace(
        pos=jnp.array([[0.0, 0.0], [0.2, 0.0]]), contact_t=jnp.array([[-1, 0], [-1, -1]])
    )
    step = jax.jit(env.step)
    out = step(key, state, jnp.array([8, 1]))
    assert bool(out.state.alive[1])
    for _ in range(3):
        out = step(key, out.state, jnp.array([1, 1]))
    assert not bool(out.state.alive[1])
    assert int(out.info.death_cause[1]) == 2
    assert float(out.state.resources[1]) == 0
    assert float(out.reward[1]) == pytest.approx(-np.log(2))
    later = step(key, out.state, jnp.array([1, 0]))
    assert int(later.info.executed_kind[1]) == -1
    assert float(later.reward[1]) == 0
    assert float(later.state.resources[1]) == 0


@pytest.mark.parametrize("cancel", [False, True])
def test_first_strike_cancels_projectiles_of_the_dead_only_when_enabled(cancel):
    """Con la opción activa, el eliminado no llega a impactar con su proyectil en vuelo."""
    env = environment(p_kill=1, projectiles_cancel_on_death=cancel)
    key = jax.random.PRNGKey(2)
    _, state = env.reset(key)
    state = state.replace(
        pos=jnp.array([[0.0, 0.0], [0.2, 0.0]]), contact_t=jnp.array([[-1, 0], [0, -1]])
    )
    step = jax.jit(env.step)
    out = step(key, state, jnp.array([8, 1]))
    out = step(key, out.state, jnp.array([1, 1]))
    out = step(key, out.state, jnp.array([1, 8]))
    out = step(key, out.state, jnp.array([1, 1]))
    np.testing.assert_array_equal(out.state.alive, [True, False])
    for _ in range(2):
        out = step(key, out.state, jnp.array([1, 1]))
    assert bool(out.state.alive[0]) is cancel


def test_insufficient_resources_unknown_targets_and_invalid_actions():
    """Las acciones imposibles no consumen recursos ni producen efectos."""
    env = environment(initial_resources=0.001)
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    step = jax.jit(env.step)
    for actions in ([2, 3], [8, 12], [-1, 100]):
        out = step(key, state, jnp.array(actions))
        np.testing.assert_array_equal(out.state.resources, state.resources)
        assert np.all(out.info.executed_kind == -1)
        assert not out.info.launches.any()


def test_growth_reward_and_terminal_absorption():
    """El límite temporal congela el estado sin recompensas posteriores."""
    env = environment(max_steps=1)
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    step = jax.jit(env.step)
    out = step(key, state, jnp.array([0, 1]))
    np.testing.assert_allclose(out.state.resources, [1.05, 1])
    np.testing.assert_allclose(out.reward, np.log1p([1.05, 1]) - np.log1p([1, 1]), atol=1e-7)
    later = step(key, out.state, jnp.array([0, 0]))
    assert later.done.all()
    assert int(later.state.t) == 1
    np.testing.assert_array_equal(later.reward, [0, 0])


def test_refuge_and_anomaly():
    """El refugio pagado protege este paso y se consume tras el estallido."""
    env = environment(anomaly=AnomalyConfig(lethality=1, period=1, radius=0.5))
    key = jax.random.PRNGKey(1)
    _, state = env.reset(key)
    state = state.replace(
        pos=jnp.array([[0.4, 0.4], [0.6, 0.6]]), theta=jnp.array([0.0, 1.0, 0.5, 0.5, 0.5])
    )
    out = jax.jit(env.step)(key, state, jnp.array([12, 1]))
    np.testing.assert_array_equal(out.state.alive, [True, False])
    assert int(out.info.shelters_correct) == 1
    assert not out.state.refuge.any()
    assert float(out.state.resources[0]) == pytest.approx(0.98)


@pytest.mark.parametrize("n", [2, 4])
def test_full_episode_invariants_and_vmap(n):
    """Audita recursos, recompensa y reproducibilidad durante un episodio completo."""
    env = environment(n_civs=n, max_steps=24)
    key = jax.random.PRNGKey(9)
    _, states = jax.jit(jax.vmap(env.reset))(jax.random.split(key, 2))
    step = jax.jit(jax.vmap(env.step))
    for _ in range(env.cfg.max_steps):
        key, action_key, step_key = jax.random.split(key, 3)
        actions = jax.random.randint(action_key, (2, n), 0, env.n_discrete)
        out = step(jax.random.split(step_key, 2), states, actions)
        assert np.all(out.state.resources >= 0)
        np.testing.assert_allclose(
            out.reward, reward_fn(states.resources, out.state.resources, states.alive), atol=1e-6
        )
        assert np.all(np.asarray(out.reward)[~np.asarray(states.alive)] == 0)
        states = out.state
    assert out.done.all()


def test_hidden_state_cannot_change_an_unmeasured_observation():
    """Mover emisores desconocidos o cambiar θ no filtra datos a la política."""
    env = environment()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    changed = state.replace(pos=1 - state.pos, theta=state.theta + 0.1)
    other = env.get_obs(key, changed)
    for left, right in zip(jax.tree.leaves(obs), jax.tree.leaves(other), strict=True):
        np.testing.assert_array_equal(left, right)


def test_reward_is_only_log_growth():
    """Audita la fórmula exacta y la máscara de agentes ya eliminados."""
    before, after = jnp.array([1.0, 2.0, 0.0]), jnp.array([0.0, 3.0, 0.0])
    out = reward_fn(before, after, jnp.array([True, True, False]))
    np.testing.assert_allclose(out, np.log1p(after) - np.log1p(before), atol=1e-7)


def test_unimplemented_extensions_are_explicit():
    """Evita ejecutar ablaciones con opciones sin efecto."""
    with pytest.raises(NotImplementedError, match="pendientes"):
        environment(costly_signals=True).reset(jax.random.PRNGKey(0))


@pytest.mark.parametrize("channel", [True, False])
def test_measurement_sharing_and_channel_ablation(channel):
    """El canal entrega mediciones completas; su ablación conserva solo detección física."""
    env = environment(channel_enabled=channel)
    key = jax.random.PRNGKey(11)
    _, state = env.reset(key)
    state = state.replace(pos=jnp.array([[0.0, 0.0], [0.05, 0.0]]))
    step = jax.jit(env.step)
    measured = step(key, state, jnp.array([2, 1]))
    assert float(measured.state.last_meas[0, 0]) == 1
    shared = step(key, measured.state, jnp.array([7, 1]))
    assert bool(shared.obs.det_mask[1, 0])
    if channel:
        assert float(shared.obs.detections[1, 0, 3]) == env.cfg.n_symbols
        np.testing.assert_allclose(shared.obs.detections[1, 0, 4:9], measured.state.last_meas[0])
    else:
        assert float(shared.obs.detections[1, 0, 3]) == -1
        assert np.all(np.asarray(shared.obs.detections[1, 0, 4:9]) == 0)


def test_expired_contacts_are_masked_and_cannot_be_attacked():
    """La memoria de contactos no permite apuntar a información caducada."""
    env = environment(detection_ttl=2)
    key = jax.random.PRNGKey(3)
    _, state = env.reset(key)
    state = state.replace(t=jnp.array(3), contact_t=jnp.array([[-1, 0], [-1, -1]]))
    assert not env.get_obs(key, state).det_mask.any()
    out = jax.jit(env.step)(key, state, jnp.array([8, 1]))
    assert not out.info.launches.any()
