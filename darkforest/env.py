"""POSG Dark Forest en JAX. API funcional al estilo JaxMARL: reset/step puros.

Todas las funciones son jit/vmap-compatibles: sin estado global, formas estáticas.
"""

from __future__ import annotations

from numbers import Integral
from typing import NamedTuple

import jax
import jax.numpy as jnp

from darkforest import anomaly
from darkforest.config import ACTIONS, EnvConfig
from darkforest.geometry import displacement, distance
from darkforest.types import EnvState, Obs, StepInfo


class StepOut(NamedTuple):
    obs: Obs
    state: EnvState
    reward: jax.Array  # (N,)
    done: jax.Array  # (N,) eliminada o fin del episodio
    info: StepInfo


class DarkForestEnv:
    """Entorno multiagente. Los agentes comparten espacio de acciones y de observación."""

    def __init__(self, cfg: EnvConfig) -> None:
        """Conserva la configuración y las dimensiones del espacio de acciones."""
        self.cfg = cfg
        self.n_agents = cfg.n_civs
        self.n_actions = len(ACTIONS)
        self.n_discrete = cfg.n_discrete

    def reset(self, key: jax.Array) -> tuple[Obs, EnvState]:
        """Posiciones aleatorias en el toro, θ aleatorio, colas vacías."""
        self._check_features()
        position_key, theta_key, obs_key = jax.random.split(key, 3)
        n, qs, qp = self.n_agents, self.cfg.signal_slots, self.cfg.projectile_slots
        state = EnvState(
            t=jnp.array(0),
            pos=jax.random.uniform(position_key, (n, 2)),
            resources=jnp.full(n, self.cfg.initial_resources),
            alive=jnp.ones(n, dtype=bool),
            emission=jnp.zeros(n, dtype=jnp.int32),
            refuge=jnp.zeros(n, dtype=bool),
            theta=anomaly.sample_theta(theta_key, self.cfg.anomaly),
            last_burst=jnp.array(0),
            last_meas=jnp.zeros((n, 5)),
            sig_origin=jnp.zeros(qs, dtype=jnp.int32),
            sig_t=jnp.zeros(qs, dtype=jnp.int32),
            sig_content=jnp.zeros((qs, 6)),
            sig_valid=jnp.zeros(qs, dtype=bool),
            prj_origin=jnp.zeros(qp, dtype=jnp.int32),
            prj_target=jnp.zeros((qp, 2)),
            prj_t=jnp.zeros(qp, dtype=jnp.int32),
            prj_valid=jnp.zeros(qp, dtype=bool),
            sig_emission=jnp.zeros(qs, dtype=jnp.int32),
            contact_t=jnp.full((n, n), -1, dtype=jnp.int32),
            contact_content=jnp.zeros((n, n, 6)).at[:, :, 0].set(-1),
            message_t=jnp.full((n, n), -1, dtype=jnp.int32),
            refuge_until=jnp.zeros(n, dtype=jnp.int32),
            has_attacked=jnp.zeros(n, dtype=bool),
        )
        return self.get_obs(obs_key, state), state

    def step(self, key: jax.Array, state: EnvState, actions: jax.Array) -> StepOut:
        """Avanza acciones, propagación, detección, anomalía y recompensa."""
        self._check_features()
        if jnp.shape(actions) != (self.n_agents,):
            raise ValueError("actions: se requiere un identificador por civilización")
        return jax.lax.cond(
            (state.t >= self.cfg.max_steps) | ~jnp.any(state.alive),
            lambda: self._absorbing(key, state),
            lambda: self._transition(key, state, actions),
        )

    def get_obs(self, key: jax.Array, state: EnvState) -> Obs:
        """Expone únicamente contactos recibidos y mediciones disponibles."""
        sources, mask = self._contacts(state)
        agents = jnp.arange(self.n_agents)[:, None]
        delta = displacement(state.pos[:, None, :], state.pos[sources])
        content = state.contact_content[agents, sources]
        message_t = state.message_t[agents, sources]
        stamp = jnp.where(message_t >= 0, message_t, state.contact_t[agents, sources])
        age = jnp.maximum(state.t - stamp, 0)
        detections = jnp.concatenate(
            [delta, jnp.linalg.norm(delta, axis=-1, keepdims=True), content, age[..., None]],
            axis=-1,
        )
        return Obs(
            own=jnp.stack([state.resources, state.emission], axis=-1),
            detections=jnp.where(mask[..., None], detections, 0),
            det_mask=mask,
            own_meas=state.last_meas,
            since_burst=jnp.full(self.n_agents, state.t - state.last_burst),
            alive=state.alive,
            agent_id=jnp.arange(self.n_agents),
        )

    def _check_features(self) -> None:
        """Rechaza extensiones cuyo mecanismo aún no está implementado."""
        if self.cfg.costly_signals or self.cfg.anomaly.interfere or self.cfg.shuffle_messages:
            raise NotImplementedError(
                "Señales costosas, interferencia y barajado siguen pendientes"
            )

    def _contacts(self, state: EnvState) -> tuple[jax.Array, jax.Array]:
        """Ordena contactos recientes sin revelar el número total de emisores."""
        sources = jnp.argsort(-state.contact_t, axis=1, stable=True)
        padding = max(self.cfg.max_detections - self.n_agents, 0)
        sources = jnp.pad(sources, ((0, 0), (0, padding)))[:, : self.cfg.max_detections]
        times = jnp.take_along_axis(state.contact_t, sources, axis=1)
        mask = (times >= 0) & ((state.t - times) <= self.cfg.detection_ttl)
        mask &= jnp.arange(self.cfg.max_detections)[None, :] < self.n_agents
        return sources, mask & state.alive[:, None]

    def _empty_info(self) -> StepInfo:
        """Construye diagnósticos vacíos con las mismas formas del paso activo."""
        n = self.n_agents
        return StepInfo(
            jnp.zeros(n, dtype=jnp.int32),
            jnp.full(n, -1, dtype=jnp.int32),
            jnp.array(0),
            jnp.array(0),
            jnp.array(0),
            jnp.full(n, -1, dtype=jnp.int32),
            jnp.zeros((n, n), dtype=bool),
            jnp.zeros(n, dtype=bool),
        )

    def _absorbing(self, key: jax.Array, state: EnvState) -> StepOut:
        """Mantiene inmóvil un episodio finalizado y devuelve recompensa cero."""
        return StepOut(
            self.get_obs(key, state),
            state,
            jnp.zeros(self.n_agents),
            jnp.ones(self.n_agents, dtype=bool),
            self._empty_info(),
        )

    def _transition(self, key: jax.Array, state: EnvState, actions: jax.Array) -> StepOut:
        """Ejecuta acciones simultáneas y resuelve sus efectos causales."""
        measure_key, detect_key, hit_key, burst_key, obs_key = jax.random.split(key, 5)
        cfg, n, now = self.cfg, self.n_agents, state.t + 1
        agents = jnp.arange(n)
        kind, argument = self.decode_action(actions)
        sources, masks = self._contacts(state)
        slot = jnp.clip(argument, 0, cfg.max_detections - 1)
        targets = sources[agents, slot]
        target_known = masks[agents, slot]
        costs = jnp.array(
            [0, 0, cfg.cost_obs, cfg.cost_tx, cfg.cost_tx, cfg.cost_atk, cfg.cost_ref]
        )
        cost = costs[jnp.maximum(kind, 0)]
        allowed = state.alive & (kind >= 0) & (state.resources >= cost)
        allowed &= (kind != 5) | target_known
        allowed &= (kind != 4) | (state.last_meas[:, 0] > 0)
        kind = jnp.where(allowed, kind, -1)
        resources = state.resources - jnp.where(allowed, cost, 0)
        resources *= jnp.where(kind == 0, 1 + cfg.growth, 1)
        high = (kind == 0) | (kind == 3) | (kind == 4)
        emission = jnp.where(kind == 1, 0, jnp.where(high, 1, state.emission))
        angle = jnp.arctan2(state.pos[:, 1] - 0.5, state.pos[:, 0] - 0.5)
        measured = anomaly.measure(measure_key, state.theta, angle, cfg.anomaly.noise_std)
        measurement = jnp.concatenate(
            [jnp.ones((n, 1)), jnp.cos(angle)[:, None], jnp.sin(angle)[:, None], measured],
            axis=1,
        )
        last_meas = jnp.where((kind == 2)[:, None], measurement, state.last_meas)
        until = jnp.where(kind == 6, state.t + cfg.refuge_duration, state.refuge_until)
        refuge = (until >= now) & state.alive
        symbol = jnp.where(kind == 3, argument, jnp.where(kind == 4, cfg.n_symbols, -1))
        symbol = jnp.where(cfg.channel_enabled, symbol, -1)
        content = jnp.concatenate(
            [
                symbol[:, None],
                jnp.where(((kind == 4) & cfg.channel_enabled)[:, None], last_meas, 0),
            ],
            axis=1,
        )
        signal_slots = (state.t % (cfg.signal_slots // n)) * n + agents
        projectile_slots = (state.t % (cfg.projectile_slots // n)) * n + agents
        launched = kind == 5
        next_state = state.replace(
            resources=resources,
            emission=emission,
            last_meas=last_meas,
            refuge=refuge,
            refuge_until=until,
            sig_origin=state.sig_origin.at[signal_slots].set(agents),
            sig_t=state.sig_t.at[signal_slots].set(state.t),
            sig_content=state.sig_content.at[signal_slots].set(content),
            sig_emission=state.sig_emission.at[signal_slots].set(emission),
            sig_valid=state.sig_valid.at[signal_slots].set(state.alive),
            prj_origin=state.prj_origin.at[projectile_slots].set(agents),
            prj_t=state.prj_t.at[projectile_slots].set(state.t),
            prj_target=state.prj_target.at[projectile_slots].set(state.pos[targets]),
            prj_valid=state.prj_valid.at[projectile_slots].set(launched),
            has_attacked=state.has_attacked | launched,
        )
        next_state, receptions = self._receive(detect_key, next_state, now)
        origins = next_state.prj_origin
        travel = distance(state.pos[origins], next_state.prj_target)
        arrived = next_state.prj_valid & (travel <= cfg.proj_speed * (now - next_state.prj_t))
        target_match = distance(next_state.prj_target[:, None, :], state.pos[None, :, :])
        hits = arrived[:, None] & (target_match <= cfg.collision_tolerance)
        hits &= jax.random.bernoulli(hit_key, float(cfg.p_kill), hits.shape)
        hits &= state.alive[None, :] & (origins[:, None] != agents[None, :])
        if cfg.projectiles_cancel_on_death:
            # el primer impacto cancela los proyectiles en vuelo del eliminado
            hits &= state.alive[origins][:, None]
        attacked = jnp.any(hits, axis=0)
        attacker = jnp.where(attacked, origins[jnp.argmax(hits, axis=0)], -1)
        burst = anomaly.burst_due(now, state.theta)
        victims = burst & anomaly.burst_victims(
            burst_key, state.pos, state.theta, refuge, cfg.anomaly.lethality
        )
        victims &= state.alive & ~attacked
        alive = state.alive & ~attacked & ~victims
        protected = burst & refuge & (distance(state.pos, state.theta[2:4]) <= state.theta[4])
        resources = jnp.where(alive, resources, 0)
        info = StepInfo(
            jnp.where(attacked, 2, jnp.where(victims, 1, 0)),
            attacker,
            jnp.sum(launched & ~state.has_attacked[targets]),
            jnp.sum(protected & ~attacked),
            burst.astype(jnp.int32),
            kind,
            receptions,
            launched,
        )
        next_state = next_state.replace(
            t=now,
            alive=alive,
            resources=resources,
            refuge=refuge & alive & ~burst,
            refuge_until=jnp.where(burst | ~alive, 0, until),
            last_burst=jnp.where(burst, now, state.last_burst),
            prj_valid=next_state.prj_valid & ~arrived,
            sig_valid=next_state.sig_valid
            & (cfg.signal_speed * (now - next_state.sig_t) < jnp.sqrt(0.5)),
        )
        return StepOut(
            self.get_obs(obs_key, next_state),
            next_state,
            reward_fn(state.resources, resources, state.alive),
            ~alive | (now >= cfg.max_steps),
            info,
        )

    def _receive(self, key: jax.Array, state: EnvState, now: jax.Array):
        """Entrega cada frente de señal una vez y conserva el último mensaje."""
        cfg, n = self.cfg, self.n_agents
        origins = state.sig_origin
        distances = distance(state.pos[:, None, :], state.pos[origins][None, :, :])
        before = cfg.signal_speed * (state.t - state.sig_t)
        after = cfg.signal_speed * (now - state.sig_t)
        reached = (distances <= after) & ((distances > before) | (before == 0))
        probability = jnp.where(state.sig_emission, cfg.detection_high, cfg.detection_low)
        probability = probability[None, :] * jnp.exp(-cfg.detection_decay * distances)
        received = reached & jax.random.bernoulli(key, probability)
        received &= state.sig_valid[None, :] & state.alive[:, None]
        received &= jnp.arange(n)[:, None] != origins[None, :]
        by_source = received[:, :, None] & (origins[None, :, None] == jnp.arange(n))
        stamp = jnp.where(by_source, state.sig_t[None, :, None], -1)
        latest = jnp.max(stamp, axis=1)
        messages = by_source & (state.sig_content[None, :, 0, None] >= 0)
        message_stamp = jnp.where(messages, state.sig_t[None, :, None], -1)
        latest_message = jnp.max(message_stamp, axis=1)
        message_index = jnp.argmax(message_stamp, axis=1)
        update = latest_message > state.message_t
        return state.replace(
            contact_t=jnp.maximum(state.contact_t, latest),
            contact_content=jnp.where(
                update[..., None], state.sig_content[message_index], state.contact_content
            ),
            message_t=jnp.maximum(state.message_t, latest_message),
        ), jnp.any(by_source, axis=1)

    def decode_action(self, action: int | jax.Array) -> tuple[jax.Array, jax.Array]:
        """Decodifica un entero o array JAX; una acción inválida produce (-1, -1)."""
        if isinstance(action, Integral) and not isinstance(action, bool):
            if not 0 <= action < self.n_discrete:
                action = -1
        try:
            action = jnp.asarray(action)
        except (TypeError, ValueError) as error:
            raise TypeError("La acción debe tener un tipo entero.") from error
        if not jnp.issubdtype(action.dtype, jnp.integer):
            raise TypeError("La acción debe tener un tipo entero.")

        valid = action >= 0
        if self.n_discrete <= jnp.iinfo(action.dtype).max:
            valid = valid & (action < self.n_discrete)
        # Se valida antes de convertir para conservar enteros sin signo fuera de rango.
        integer_dtype = jnp.int64 if jax.config.x64_enabled else jnp.int32
        action = jnp.where(valid, action, 0).astype(integer_dtype)
        share = 3 + self.cfg.n_symbols
        attack = share + 1
        refuge = self.n_discrete - 1

        kind = jnp.select(
            [action < 3, action < share, action == share, action < refuge],
            [action, 3, 4, 5],
            default=6,
        )
        argument = jnp.where(
            (action >= 3) & (action < share),
            action - 3,
            jnp.where((action >= attack) & (action < refuge), action - attack, -1),
        )
        return jnp.where(valid, kind, -1), jnp.where(valid, argument, -1)


def reward_fn(resources_t: jax.Array, resources_t1: jax.Array, alive_t: jax.Array) -> jax.Array:
    """Única recompensa: log(1 + r_{t+1}) - log(1 + r_t); 0 si ya estaba eliminada."""
    return jnp.where(alive_t, jnp.log1p(resources_t1) - jnp.log1p(resources_t), 0)
