"""Registro de corridas: config resuelta, métricas y checkpoints en runs/<config>/<seed>/."""

from __future__ import annotations

from pathlib import Path

from darkforest.config import Config


def run_dir(cfg: Config, name: str) -> Path:
    return Path(cfg.logging.out_dir) / name / str(cfg.train.seed)


class Logger:
    """Fachada sobre TensorBoard o W&B; guarda `config.yaml` al iniciar."""

    def __init__(self, cfg: Config, name: str):
        self.dir = run_dir(cfg, name)
        raise NotImplementedError

    def log(self, step: int, metrics: dict[str, float]) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
