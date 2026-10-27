"""Configuración tipada. Única fuente de hiperparámetros."""

from __future__ import annotations

import dataclasses as dc
from copy import deepcopy
from math import isfinite
from pathlib import Path
from typing import Any

import yaml

# Acciones discretas (orden fijo; ver sección 3.3 de la propuesta).
ACTIONS = ("crecer", "ocultarse", "observar", "transmitir", "compartir", "atacar", "refugiarse")


def _require_type(name: str, value: Any, expected: type) -> None:
    """Rechaza tipos incompatibles y booleanos usados como enteros."""
    if not isinstance(value, expected) or (expected is int and isinstance(value, bool)):
        raise TypeError(f"{name}: se esperaba {expected.__name__}")


def _integer(name: str, value: int, minimum: int = 1, maximum: int | None = None) -> None:
    """Valida un entero dentro del intervalo permitido."""
    _require_type(name, value, int)
    if value < minimum or (maximum is not None and value > maximum):
        interval = f"entre {minimum} y {maximum}" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name}: debe ser {interval}")


def _number(
    name: str,
    value: float,
    maximum: float | None = None,
    *,
    positive: bool = False,
) -> None:
    """Valida un número finito no negativo o estrictamente positivo."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name}: se esperaba un número")
    if not isfinite(value):
        raise ValueError(f"{name}: debe ser finito")
    if value < 0 or (positive and value == 0) or (maximum is not None and value > maximum):
        raise ValueError(f"{name}: número fuera de rango")


def _choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    """Valida una opción textual admitida."""
    _require_type(name, value, str)
    if value not in choices:
        raise ValueError(f"{name}: opción no válida")


@dc.dataclass(frozen=True)
class AnomalyConfig:
    lethality: float = 0.5  # lambda
    period: int = 20
    noise_std: float = 0.1
    interfere: bool = False  # extensión J-04

    def __post_init__(self) -> None:
        """Valida los parámetros de la anomalía."""
        _number("anomaly.lethality", self.lethality, maximum=1)
        _integer("anomaly.period", self.period)
        _number("anomaly.noise_std", self.noise_std)
        _require_type("anomaly.interfere", self.interfere, bool)


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

    def __post_init__(self) -> None:
        """Valida conteos, costes y velocidades del entorno."""
        _integer("env.n_civs", self.n_civs)
        if self.n_civs not in (2, 4, 6, 8):
            raise ValueError("env.n_civs: debe ser 2, 4, 6 u 8")
        for name in ("max_steps", "max_detections", "n_symbols"):
            _integer(f"env.{name}", getattr(self, name))
        for name in ("growth", "cost_obs", "cost_tx", "cost_atk", "cost_ref"):
            _number(f"env.{name}", getattr(self, name))
        _number("env.signal_speed", self.signal_speed, positive=True)
        _number("env.proj_speed", self.proj_speed, positive=True)
        if self.proj_speed > self.signal_speed:
            raise ValueError("env.proj_speed: no debe superar signal_speed")
        _number("env.p_kill", self.p_kill, maximum=1)
        for name in ("costly_signals", "channel_enabled", "shuffle_messages"):
            _require_type(f"env.{name}", getattr(self, name), bool)
        _require_type("env.anomaly", self.anomaly, AnomalyConfig)

    @property
    def n_discrete(self) -> int:
        """Cuenta acciones simples y opciones parametrizadas."""
        return 5 + self.n_symbols + self.max_detections


@dc.dataclass(frozen=True)
class SelfPlayConfig:
    rho: float = 0.3
    snapshot_every: int = 50

    def __post_init__(self) -> None:
        """Valida la mezcla de políticas y la frecuencia de guardado."""
        _number("self_play.rho", self.rho, maximum=1)
        _integer("self_play.snapshot_every", self.snapshot_every)


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

    def __post_init__(self) -> None:
        """Valida el entrenamiento y el dominio de seeds de 32 bits."""
        _choice("train.algo", self.algo, ("ippo",))
        _integer("train.seed", self.seed, minimum=0, maximum=2**32 - 1)
        for name in ("n_envs", "total_steps", "rollout_len", "gru_hidden"):
            _integer(f"train.{name}", getattr(self, name))
        for name in ("lr", "clip_eps"):
            _number(f"train.{name}", getattr(self, name), positive=True)
        for name in ("gamma", "gae_lambda"):
            _number(f"train.{name}", getattr(self, name), maximum=1)
        for name in ("ent_coef", "vf_coef"):
            _number(f"train.{name}", getattr(self, name))
        _require_type("train.self_play", self.self_play, SelfPlayConfig)


@dc.dataclass(frozen=True)
class LoggingConfig:
    backend: str = "tensorboard"  # tensorboard | wandb
    out_dir: str = "runs"

    def __post_init__(self) -> None:
        """Valida el backend y la ruta de salida."""
        _choice("logging.backend", self.backend, ("tensorboard", "wandb"))
        _require_type("logging.out_dir", self.out_dir, str)
        if not self.out_dir.strip():
            raise ValueError("logging.out_dir: no puede estar vacío")
        if "\0" in self.out_dir:
            raise ValueError("logging.out_dir: no puede contener NUL")
        if Path(self.out_dir).is_file():
            raise ValueError("logging.out_dir: corresponde a un archivo")


@dc.dataclass(frozen=True)
class Config:
    env: EnvConfig = EnvConfig()
    train: TrainConfig = TrainConfig()
    logging: LoggingConfig = LoggingConfig()

    def __post_init__(self) -> None:
        """Exige configuraciones tipadas en cada sección."""
        _require_type("env", self.env, EnvConfig)
        _require_type("train", self.train, TrainConfig)
        _require_type("logging", self.logging, LoggingConfig)


def _merge(base: dict, over: dict) -> dict:
    """Combina los campos de un YAML hijo con su base."""
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _from_dict(cls: type, data: dict[str, Any]):
    """Construye las dataclasses y rechaza claves desconocidas."""
    _require_type(cls.__name__, data, dict)
    hints = {f.name: f for f in dc.fields(cls)}
    unknown = set(data) - set(hints)
    if unknown:
        raise KeyError(f"Claves desconocidas en {cls.__name__}: {sorted(unknown, key=str)}")
    kwargs = {}
    for name, v in data.items():
        default = hints[name].default
        kwargs[name] = _from_dict(type(default), v) if dc.is_dataclass(default) else v
    return cls(**kwargs)


def apply_overrides(raw: dict, overrides: dict[str, Any]) -> dict:
    """Aplica claves punteadas sin modificar los diccionarios originales."""
    _require_type("configuración", raw, dict)
    _require_type("overrides", overrides, dict)
    out = deepcopy(raw)
    for dotted, value in overrides.items():
        _require_type("clave de override", dotted, str)
        if not all(dotted.split(".")):
            raise KeyError("Clave de override vacía")
        node = out
        *path, leaf = dotted.split(".")
        for p in path:
            node = node.setdefault(p, {})
            _require_type(f"override {dotted}", node, dict)
        node[leaf] = deepcopy(value)
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    """Lee un mapa YAML; un archivo vacío usa los valores por defecto."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    _require_type(str(path), raw, dict)
    return raw


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Carga un YAML (con `extends:`) y aplica overrides con claves punteadas."""
    path = Path(path)
    raw = _read_yaml(path)
    parent = raw.pop("extends", None)
    if parent is not None:
        _require_type("extends", parent, str)
        if not parent.strip():
            raise ValueError("extends: no puede estar vacío")
        raw = _merge(_read_yaml(path.parent / parent), raw)
    raw = apply_overrides(raw, overrides if overrides is not None else {})
    return _from_dict(Config, raw)


def to_dict(cfg: Config) -> dict:
    """Devuelve una copia serializable de la configuración resuelta."""
    return dc.asdict(cfg)
