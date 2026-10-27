"""Registro de corridas: config resuelta, métricas y checkpoints en runs/<config>/<seed>/."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import jax
import yaml

from darkforest.config import Config, to_dict


def run_dir(cfg: Config, name: str) -> Path:
    """Identifica la configuración sin seed y evita colisiones entre overrides."""
    resolved = to_dict(cfg)
    resolved["train"].pop("seed")
    resolved["logging"].pop("out_dir")
    digest = hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest()[:10]
    label = re.sub(r"[^a-zA-Z0-9_-]+", "-", Path(name).stem).strip("-") or "corrida"
    return Path(cfg.logging.out_dir) / f"{label}-{digest}" / str(cfg.train.seed)


def create_run(cfg: Config, name: str, mode: str) -> Path:
    """Guarda procedencia y configuración sin sobrescribir corridas anteriores."""
    directory = run_dir(cfg, name)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.yaml").write_text(
        yaml.safe_dump(to_dict(cfg), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    metadata = {
        "schema_version": 1,
        "config_id": directory.parent.name,
        "seed": cfg.train.seed,
        "mode": mode,
        "synthetic": False,
        "git_revision": revision,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "jax": jax.__version__,
        "versions": {package: version(package) for package in ("flax", "optax", "numpy")},
        "devices": [str(device) for device in jax.devices()],
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return directory


class Logger:
    """Fachada sobre TensorBoard o W&B; guarda `config.yaml` al iniciar."""

    def __init__(self, cfg: Config, name: str) -> None:
        """Abre registros locales; W&B requiere una integración posterior explícita."""
        if cfg.logging.backend != "tensorboard":
            raise NotImplementedError("Este bloque usa TensorBoard local")
        from tensorboard.summary.writer.event_file_writer import EventFileWriter

        self.dir = create_run(cfg, name, "entrenamiento_exploratorio")
        self.writer = EventFileWriter(str(self.dir / "tensorboard"))
        self.metrics = (self.dir / "training.jsonl").open("x", encoding="utf-8")

    def log(self, step: int, metrics: dict[str, float]) -> None:
        """Escribe las métricas como JSONL y eventos de TensorBoard."""
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary

        self.metrics.write(json.dumps({"step": step, **metrics}, allow_nan=False) + "\n")
        self.metrics.flush()
        summary = Summary(value=[Summary.Value(tag=k, simple_value=v) for k, v in metrics.items()])
        self.writer.add_event(Event(wall_time=time.time(), step=step, summary=summary))

    def close(self) -> None:
        """Vacía los registros y cierra los archivos de la corrida."""
        self.writer.close()
        self.metrics.close()
