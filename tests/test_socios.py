"""Socios de entrenamiento: estrategias fijas, asignación y rollout con población."""

import dataclasses as dc

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from darkforest.config import Config, SelfPlayConfig, load_config
from darkforest.env import DarkForestEnv
from train import socios
from train.ippo import initialize, make_rollout, make_update, run_iteration
from train.selfplay import Population
from train.valor_cooperacion import PERIODO_ALEATORIO, build_env, run


@pytest.mark.parametrize("code,mode", [(0, "coop_real"), (1, "solo"), (3, "ignorar")])
def test_fixed_strategies_reproduce_valor_cooperacion(code, mode):
    """Las estrategias vectorizadas dan el mismo retorno que train.valor_cooperacion.run."""
    episodes, steps, seed = 6, 40, 3
    env = build_env(0.5, PERIODO_ALEATORIO, steps)
    cfg = Config(env=env.cfg)
    reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
    key = jax.random.PRNGKey(seed)
    obs, state = reset(jax.random.split(key, episodes))
    aux = socios.initial_auxiliar(episodes, 2)
    total = jnp.zeros((episodes, 2))
    for _ in range(steps):
        key, step_key = jax.random.split(key)
        actions = socios.fixed_actions(cfg, state, obs, aux)[code]
        out = step(jax.random.split(step_key, episodes), state, actions)
        aux = socios.update_auxiliar(cfg, aux, state, out.state, actions)
        total += out.reward
        state, obs = out.state, out.obs
    expected = run(
        0.5, PERIODO_ALEATORIO, mode, socios.MARGEN, seed, episodes=episodes, steps=steps
    )
    assert float(total.mean()) == pytest.approx(expected[0], abs=1e-5)


def test_assignment_draws_learner_snapshot_and_fixed_partners():
    """Con rho=1 cada entorno tiene exactamente un socio y los códigos respetan los pesos."""
    base = load_config("configs/aprendizaje_dos.yaml")
    play = SelfPlayConfig(rho=1.0, w_snapshot=1.0, w_coop_real=1.0, w_solo=0.0, w_ataque_solo=1.0)
    cfg = dc.replace(base, train=dc.replace(base.train, self_play=play))
    codes = np.asarray(socios.draw_assignment(jax.random.PRNGKey(0), cfg, 3000))
    assert ((codes != socios.APRENDIZ).sum(axis=1) == 1).all()
    partners = codes[codes != socios.APRENDIZ]
    assert set(np.unique(partners)) == {socios.SNAPSHOT, 0, 2}
    assert np.mean(partners == 0) == pytest.approx(1 / 3, abs=0.04)
    none = dc.replace(base, train=dc.replace(base.train, self_play=SelfPlayConfig(rho=0.0)))
    assert (np.asarray(socios.draw_assignment(jax.random.PRNGKey(0), none, 5)) == -1).all()


def test_population_keeps_the_newest_snapshots():
    """Al llenarse descarta el más antiguo y sample devuelve uno de los guardados."""
    pop = Population(max_size=2)
    for value in (1.0, 2.0, 3.0):
        pop.add({"w": jnp.array(value)})
    assert [float(p["w"]) for p in pop.snapshots] == [2.0, 3.0]
    assert float(pop.sample(jax.random.PRNGKey(0))["w"]) in (2.0, 3.0)


def _partner_config(**weights) -> Config:
    base = load_config("configs/aprendizaje_dos.yaml")
    play = SelfPlayConfig(rho=1.0, snapshot_every=1, **weights)
    train = dc.replace(base.train, n_envs=8, rollout_len=8, update_epochs=1, self_play=play)
    return dc.replace(base, train=train)


def test_training_with_partners_trains_only_on_the_learner_and_stays_finite():
    """Los socios no aportan datos de entrenamiento y la iteración deja pesos finitos."""
    cfg = _partner_config(w_snapshot=1.0, w_coop_real=1.0, w_ataque_solo=1.0, w_ignorar=1.0)
    env = DarkForestEnv(cfg.env)
    state, runner = initialize(cfg, jax.random.PRNGKey(0))
    assert ((np.asarray(runner.assign) != socios.APRENDIZ).sum(axis=1) == 1).all()
    rollout = jax.jit(make_rollout(env, cfg))
    runner2, batch = rollout(
        jax.random.PRNGKey(1), state.params, state.params, runner.assign, runner
    )
    expected = np.asarray(runner.obs.alive) & (np.asarray(runner.assign) == socios.APRENDIZ)
    np.testing.assert_array_equal(np.asarray(batch["valid"][0]), expected)
    state, runner, _, metrics = run_iteration(
        cfg,
        env,
        rollout,
        make_update(cfg),
        state,
        runner,
        jax.random.PRNGKey(2),
        None,
        0.0,
        state.params,
    )
    assert all(bool(jnp.isfinite(v)) for v in metrics.values())
    leaves = jax.tree.leaves(state.params)
    assert all(bool(jnp.isfinite(x).all()) for x in leaves)


def test_snapshot_partner_needs_weights():
    """Con peso de snapshot y sin pesos congelados el rollout avisa en vez de ignorarlo."""
    cfg = _partner_config(w_snapshot=1.0)
    env = DarkForestEnv(cfg.env)
    state, runner = initialize(cfg, jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="snapshot"):
        make_rollout(env, cfg)(jax.random.PRNGKey(1), state.params, None, runner.assign, runner)


def test_cross_play_between_fixed_strategies_matches_known_ordering():
    """Dos coop_real ganan más que dos solo, y quien ataca a un solo no gana más que cooperar."""
    from train.cruce import cross_play

    env = build_env(0.5, PERIODO_ALEATORIO, 200)
    cfg = Config(env=env.cfg)
    result = cross_play(cfg, ["fijo:coop_real", "fijo:solo", "fijo:recip"], episodes=24)
    ret = np.asarray(result["retorno"])
    assert ret[0, 0] > ret[1, 1]
    assert ret[2, 2] > ret[1, 1]
    assert np.allclose(np.asarray(result["disparos"]), 0)


def test_training_with_population_then_init_from_and_cross_play(tmp_path):
    """Entrena con socios, reanuda los pesos en otra corrida y cruza la red con estrategias."""
    from darkforest.config import EnvConfig, LoggingConfig, TrainConfig
    from train.cruce import cross_play
    from train.ippo import train
    from train.logging_utils import run_dir

    play = SelfPlayConfig(
        rho=1.0, snapshot_every=1, w_snapshot=1.0, w_recip=1.0, w_coop_real=1.0, w_ataque_solo=1.0
    )
    base = Config(
        env=EnvConfig(n_civs=2, max_steps=6),
        train=TrainConfig(
            n_envs=4, rollout_len=2, total_steps=16, gru_hidden=8, update_epochs=1, self_play=play
        ),
        logging=LoggingConfig(out_dir=str(tmp_path)),
    )
    train(base, "pob")
    final = run_dir(base, "pob") / "checkpoints" / "final.msgpack"
    assert final.exists()
    second = dc.replace(base, train=dc.replace(base.train, seed=1, init_from=str(final)))
    train(second, "pob2")
    result = cross_play(base, ["fijo:solo", "fijo:recip", f"red:{final}:greedy"], episodes=4)
    assert np.isfinite(np.asarray(result["retorno"])).all()
    assert np.asarray(result["retorno"]).shape == (3, 3)


def test_population_training_runs_with_more_than_two_civilizations(tmp_path):
    """Con 4 civilizaciones los socios son solo snapshots y el entrenamiento termina."""
    from darkforest.config import EnvConfig, LoggingConfig, TrainConfig
    from train.ippo import train

    play = SelfPlayConfig(rho=0.5, snapshot_every=1, w_snapshot=1.0)
    cfg = Config(
        env=EnvConfig(n_civs=4, max_steps=6),
        train=TrainConfig(
            n_envs=4, rollout_len=2, total_steps=16, gru_hidden=8, update_epochs=1, self_play=play
        ),
        logging=LoggingConfig(out_dir=str(tmp_path)),
    )
    train(cfg, "n4")


def test_fixed_strategies_work_with_more_than_two_civilizations():
    """Con cuatro civilizaciones cooperar rinde más que jugar sola y comparte mensajes."""
    episodes, steps = 24, 200
    env = build_env(0.5, PERIODO_ALEATORIO, steps, {"env.n_civs": 4, "env.cost_ref": 0.25})
    cfg = Config(env=env.cfg)
    reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))

    def jugar(modo: str) -> tuple[float, int]:
        code = socios.FIJOS.index(modo)
        key = jax.random.PRNGKey(2)
        obs, state = reset(jax.random.split(key, episodes))
        aux = socios.initial_auxiliar(episodes, 4)
        total, shares = jnp.zeros((episodes, 4)), 0
        for _ in range(steps):
            key, step_key = jax.random.split(key)
            actions = socios.fixed_actions(cfg, state, obs, aux)[code]
            out = step(jax.random.split(step_key, episodes), state, actions)
            aux = socios.update_auxiliar(cfg, aux, state, out.state, actions)
            shares += int((out.info.executed_kind == 4).sum())
            total += out.reward
            state, obs = out.state, out.obs
        return float(total.mean()), shares

    coop, shares = jugar("coop_real")
    solo, _ = jugar("solo")
    assert shares > 0
    assert coop > solo + 1.0
