"""Mide cuánto aporta cooperar frente a un agente solo, con políticas fijas y sin entrenar.

Es un diagnóstico previo al entrenamiento: dice si las reglas dejan algo que aprender, no
reemplaza a IPPO. Dos civilizaciones, sin ataques. Todas las políticas con medición gastan el
paso 0 en observar. Resultados y lectura en docs/DIAGNOSTICO_DISENO.md.

solo: usa su propia medición y los estallidos que ve por since_burst.
coop_ideal: combina las dos mediciones al instante y sin costo (cota superior).
coop_real: comparte con la acción compartir del entorno; paga cost_tx, la señal tarda según
    signal_speed y se entrega con probabilidad. Mientras no recibe, actúa como solo.
oraculo: conoce θ exacto (techo). oraculo_cuando solo conoce los instantes de estallido.
oculta: no crece y se refugia en cada estallido (techo de quien se esconde).

Uso: python -m train.valor_cooperacion --world periodo_aleatorio
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from darkforest import anomaly
from darkforest.config import load_config
from darkforest.env import DarkForestEnv
from darkforest.geometry import distance
from darkforest.types import EnvState

CONFIG = Path(__file__).parents[1] / "configs" / "aprendizaje_dos.yaml"
PERIODO_FIJO, PERIODO_ALEATORIO = "periodo_fijo", "periodo_aleatorio"
OBSERVAR = 2
# anomaly.period_spread que define el mundo de periodo aleatorio (periodo entre 12 y 28)
SPREAD = 8


def mean_se(values: Any) -> tuple[float, float]:
    """Media y error estándar entre seeds."""
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def build_env(
    lethality: float, world: str, steps: int, env_over: dict[str, Any] | None = None
) -> DarkForestEnv:
    """Crea el entorno de dos civilizaciones del diagnóstico."""
    overrides: dict[str, Any] = {
        "env.n_civs": 2,
        "env.max_steps": steps,
        "env.anomaly.lethality": lethality,
        "env.anomaly.period_spread": SPREAD if world == PERIODO_ALEATORIO else 0,
        **(env_over or {}),
    }
    return DarkForestEnv(load_config(CONFIG, overrides).env)


def pooled(
    own: tuple,
    other: tuple,
    got: jax.Array,
    fixed: bool,
    t_next: jax.Array,
    last: jax.Array,
    bursts: tuple,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Estimación de cada civilización al combinar su medición con la de la otra, forma (B, 2)."""
    ca, sa, z1, z2 = own
    cb, sb, w1, w2 = other
    first_burst, second_burst = bursts
    det = ca * sb - sa * cb
    ok = got & (jnp.abs(det) > 0.05)
    safe = jnp.where(ok, det, 1.0)
    phase, period = (z1 * sb - sa * w1) / safe, (ca * w1 - cb * z1) / safe
    cx, cy = (z2 * sb - sa * w2) / safe, (ca * w2 - cb * z2) / safe
    if fixed:
        mean = 0.5 * ((z1 - sa * 20.0) / ca + (w1 - sb * 20.0) / cb)
        ref = jnp.where(last > 0, last, jnp.round(mean))
        due = ((t_next - ref) % 20) == 0
    else:
        gap = jnp.where(
            second_burst[:, None] > 0,
            (second_burst - first_burst)[:, None],
            jnp.round(period),
        )
        first = jnp.where(jnp.round(phase) > 0, jnp.round(phase), gap)
        after = ((t_next - last) == gap) & (ok | (second_burst[:, None] > 0))
        due = jnp.where(first_burst[:, None] > 0, after, (t_next == first) & ok)
    return due, jnp.stack([cx, cy], axis=-1), ok


@partial(jax.jit, static_argnames=("mode", "world", "shelter", "share"))
def decide(
    state: EnvState,
    first_burst: jax.Array,
    second_burst: jax.Array,
    margin: float,
    mode: str,
    world: str,
    shelter: int,
    share: int,
    extra_shares: int = 4,
    give_up: int = 6,
) -> jax.Array:
    """Acción de cada civilización según el nivel de información de la política."""
    fixed = world == PERIODO_FIJO
    batch = state.t.shape[0]
    t_next = (state.t + 1).astype(jnp.float32)[:, None]
    pos, theta = state.pos, state.theta
    angle = jnp.arctan2(pos[..., 1] - 0.5, pos[..., 0] - 0.5)
    ca, sa = jnp.cos(angle), jnp.sin(angle)
    z1, z2 = state.last_meas[..., 3], state.last_meas[..., 4]
    radius = theta[:, 4][:, None]
    last = state.last_burst.astype(jnp.float32)[:, None]
    bursts = first_burst.astype(jnp.float32), second_burst.astype(jnp.float32)
    line = jnp.abs(ca * pos[..., 0] + sa * pos[..., 1] - z2) <= radius + margin

    if mode == "oculta":
        due = jax.vmap(anomaly.burst_due)(state.t + 1, theta)[:, None]
        return jnp.where(jnp.broadcast_to(due, (batch, 2)), shelter, 1).astype(jnp.int32)

    if mode in ("oraculo_cuando", "oraculo"):
        due = jax.vmap(anomaly.burst_due)(state.t + 1, theta)[:, None]
        in_zone = distance(pos, theta[:, None, 2:4]) <= radius
        go = due & (in_zone if mode == "oraculo" else True)
    else:
        if fixed:
            ref = jnp.where(last > 0, last, jnp.round((z1 - sa * 20.0) / ca))
            due = ((t_next - ref) % 20) == 0
        else:
            gap = (bursts[1] - bursts[0])[:, None]
            due = (bursts[1][:, None] > 0) & ((t_next - last) == gap)
        go = due & line
        if mode != "solo":
            own = (ca, sa, z1, z2)
            if mode == "coop_ideal":
                other = tuple(x[:, ::-1] for x in own)
                got = jnp.ones_like(line)
            else:
                heard = jnp.stack(
                    [state.contact_content[:, 0, 1], state.contact_content[:, 1, 0]], axis=1
                )
                other = (heard[..., 2], heard[..., 3], heard[..., 4], heard[..., 5])
                got = jnp.stack([state.message_t[:, 0, 1] >= 0, state.message_t[:, 1, 0] >= 0], 1)
            p_due, center, ok = pooled(own, other, got, fixed, t_next, last, bursts)
            inside = distance(pos, center) <= radius + margin
            go = jnp.where(ok, p_due & inside, go)

    actions = jnp.where(jnp.broadcast_to(go, (batch, 2)), shelter, 0)
    if mode in ("solo", "coop_ideal", "coop_real"):
        actions = jnp.where((state.t == 0)[:, None], OBSERVAR, actions)
    if mode == "coop_real":
        got = jnp.stack([state.message_t[:, 0, 1] >= 0, state.message_t[:, 1, 0] >= 0], axis=1)
        age = state.t[:, None]
        waiting = ~got & (age <= give_up)
        sharing = (~go) & (age > 0) & (waiting | (age <= extra_shares))
        actions = jnp.where(sharing, share, actions)
    return actions.astype(jnp.int32)


def update_bursts(
    new_last: jax.Array, old_last: jax.Array, first: jax.Array, second: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Registra el primer y el segundo estallido observados por since_burst."""
    new = new_last > old_last
    second = jnp.where(new & (first > 0) & (second == 0), new_last, second)
    first = jnp.where(new & (first == 0), new_last, first)
    return first, second


def run(
    lethality: float,
    world: str,
    mode: str,
    margin: float = 0.0,
    seed: int = 0,
    *,
    episodes: int = 400,
    steps: int = 200,
    env_over: dict[str, Any] | None = None,
) -> tuple[float, float, float]:
    """Devuelve retorno medio por civilización, supervivencia final y refugios por episodio."""
    env = build_env(lethality, world, steps, env_over)
    reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
    shelter, share = env.n_discrete - 1, 3 + env.cfg.n_symbols
    key = jax.random.PRNGKey(seed)
    _, state = reset(jax.random.split(key, episodes))
    first = second = jnp.zeros(episodes, dtype=jnp.int32)
    total, shelters = jnp.zeros((episodes, 2)), 0
    for _ in range(steps):
        key, step_key = jax.random.split(key)
        if mode == "ignorar":
            actions = jnp.zeros((episodes, 2), dtype=jnp.int32)
        else:
            actions = decide(state, first, second, margin, mode, world, shelter, share)
        shelters += int((actions == shelter).sum())
        out = step(jax.random.split(step_key, episodes), state, actions)
        first, second = update_bursts(out.state.last_burst, state.last_burst, first, second)
        total += out.reward
        state = out.state
    return float(total.mean()), float(state.alive.mean()), shelters / episodes


def tune_margin(lethality: float, world: str, mode: str, args: argparse.Namespace) -> float:
    """Elige el margen de la zona en una seed de ajuste, aparte de las de evaluación."""
    scores = [
        run(lethality, world, mode, m, args.tune_seed, episodes=args.episodes, steps=args.steps)[0]
        for m in args.margins
    ]
    return args.margins[int(np.argmax(scores))]


def report_world(world: str, args: argparse.Namespace) -> None:
    """Imprime retorno por política y aporte de cooperar sobre el agente solo."""
    seeds = range(args.seeds)
    print(
        f"Mundo {world}, {args.episodes} episodios de {args.steps} pasos por seed, "
        f"{args.seeds} seeds"
    )
    print()
    for lethality in args.lethality:
        chosen = {
            m: tune_margin(lethality, world, m, args) for m in ("solo", "coop_ideal", "coop_real")
        }
        returns = {
            mode: [
                run(
                    lethality,
                    world,
                    mode,
                    chosen.get(mode, 0.0),
                    s,
                    episodes=args.episodes,
                    steps=args.steps,
                )[0]
                for s in seeds
            ]
            for mode in ("ignorar", "solo", "coop_ideal", "coop_real", "oraculo")
        }
        print(f"lambda={lethality}, margenes elegidos {chosen}")
        for mode, values in returns.items():
            mean, se = mean_se(values)
            print(f"  {mode:11s} retorno={mean:+.3f} +- {se:.3f}")
        for mode in ("coop_ideal", "coop_real"):
            mean, se = mean_se(np.asarray(returns[mode]) - np.asarray(returns["solo"]))
            print(f"  aporte de {mode} sobre solo: {mean:+.3f} +- {se:.3f}")
        print()


def report_sweep(world: str, args: argparse.Namespace) -> None:
    """Imprime el aporte de coop_real sobre solo al variar signal_speed y cost_tx."""
    print(f"Mundo {world}, aporte de coop_real sobre solo, margen de coop_real {args.sweep_margin}")
    print()
    kwargs = {"episodes": args.episodes, "steps": args.steps}
    # solo no usa el canal: su retorno no depende de signal_speed ni de cost_tx
    solo = {}
    for lethality in args.lethality:
        margin = tune_margin(lethality, world, "solo", args)
        solo[lethality] = np.mean(
            [run(lethality, world, "solo", margin, s, **kwargs)[0] for s in range(args.seeds)]
        )
    for speed in args.signal_speed:
        for cost in args.cost_tx:
            over = {"env.signal_speed": speed, "env.cost_tx": cost}
            row = []
            for lethality in args.lethality:
                real = [
                    run(
                        lethality, world, "coop_real", args.sweep_margin, s, env_over=over, **kwargs
                    )[0]
                    for s in range(args.seeds)
                ]
                row.append(f"{np.mean(real) - solo[lethality]:+.3f}")
            print(f"signal_speed={speed} cost_tx={cost}: {row} para lambda {args.lethality}")
    print()


def main() -> None:
    """Imprime la tabla del mundo elegido o el barrido de signal_speed y cost_tx."""
    p = argparse.ArgumentParser()
    p.add_argument("--world", choices=(PERIODO_FIJO, PERIODO_ALEATORIO), default=PERIODO_ALEATORIO)
    p.add_argument("--lethality", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--tune-seed", type=int, default=100)
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--margins", type=float, nargs="+", default=[0.0, 0.1, 0.2])
    p.add_argument("--signal-speed", type=float, nargs="+")
    p.add_argument("--cost-tx", type=float, nargs="+", default=[0.02])
    p.add_argument("--sweep-margin", type=float, default=0.2)
    args = p.parse_args()
    if args.signal_speed:
        report_sweep(args.world, args)
    else:
        report_world(args.world, args)


if __name__ == "__main__":
    main()
