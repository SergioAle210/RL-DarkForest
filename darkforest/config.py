"""Configuración tipada. Única fuente de hiperparámetros."""

from __future__ import annotations

import dataclasses as dc
from pathlib import Path
from typing import Any

import yaml

# Acciones discretas (orden fijo; ver sección 3.3 de la propuesta).
ACTIONS = ("crecer", "ocultarse", "observar", "transmitir", "compartir", "atacar", "refugiarse")


@dc.dataclass(frozen=True)
class AnomalyConfig:
    lethality: float = 0.5  # lambda
    period: int = 20
    noise_std: float = 0.1
    interfere: bool = False  # extensión J-04


@dc.dataclass(frozen=True)
class EnvConfig:
    n_civs: int = 4
    max_steps: int = 200
    max_detections: int = 4  # M
    n_symbols: int = 4  # K
    growth: float = 0.05
    signal_speed: float = 0.1  # c
    proj_speed: float = 0.05  # v_p <= c
    p_kill: float = 0.75
    cost_obs: float = 0.01
    cost_tx: float = 0.02
    cost_atk: float = 0.10
    cost_ref: float = 0.02
    costly_signals: bool = False
    channel_enabled: bool = True  # False = ablación del canal
    shuffle_messages: bool = False  # ablación: barajar mensajes
    anomaly: AnomalyConfig = AnomalyConfig()


@dc.dataclass(frozen=True)
class SelfPlayConfig:
    rho: float = 0.3
    snapshot_every: int = 50


@dc.dataclass(frozen=True)
class TrainConfig:
    algo: str = "ippo"
    seed: int = 0
    n_envs: int = 1024
    total_steps: int = 50_000_000
    rollout_len: int = 64
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    gru_hidden: int = 128
    self_play: SelfPlayConfig = SelfPlayConfig()


@dc.dataclass(frozen=True)
class LoggingConfig:
    backend: str = "tensorboard"  # tensorboard | wandb
    out_dir: str = "runs"


@dc.dataclass(frozen=True)
class Config:
    env: EnvConfig = EnvConfig()
    train: TrainConfig = TrainConfig()
    logging: LoggingConfig = LoggingConfig()


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _from_dict(cls: type, data: dict[str, Any]):
    hints = {f.name: f for f in dc.fields(cls)}
    unknown = set(data) - set(hints)
    if unknown:
        raise KeyError(f"Claves desconocidas en {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name, v in data.items():
        default = getattr(cls(), name)
        kwargs[name] = _from_dict(type(default), v) if dc.is_dataclass(default) else v
    return cls(**kwargs)


def apply_overrides(raw: dict, overrides: dict[str, Any]) -> dict:
    """Aplica overrides con claves punteadas, p. ej. {'env.anomaly.lethality': 0.5}."""
    for dotted, value in overrides.items():
        node = raw
        *path, leaf = dotted.split(".")
        for p in path:
            node = node.setdefault(p, {})
        node[leaf] = value
    return raw


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Carga un YAML (con `extends:`) y aplica overrides con claves punteadas."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    parent = raw.pop("extends", None)
    if parent:
        raw = _merge(yaml.safe_load((path.parent / parent).read_text()) or {}, raw)
    raw = apply_overrides(raw, overrides or {})
    return _from_dict(Config, raw)


def to_dict(cfg: Config) -> dict:
    return dc.asdict(cfg)
