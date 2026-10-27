"""Lectura compartida del formato de episodios, sin ejecutar el entorno."""

import json
from pathlib import Path

import numpy as np


def load_episode(path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    """Carga arrays numéricos y comprueba la alineación temporal del formato."""
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise ValueError("Versión de trayectoria no reconocida")
    with np.load(path, allow_pickle=False) as archive:
        trajectory = {name: archive[name] for name in archive.files}
    steps, agents = trajectory["actions"].shape
    if trajectory["alive"].shape != (steps + 1, agents):
        raise ValueError("alive: se requieren T+1 estados para T acciones")
    for name in ("rewards", "done", "info_executed_kind"):
        if trajectory[name].shape != (steps, agents):
            raise ValueError(f"{name}: forma incompatible con actions")
    for name, spec in metadata["arrays"].items():
        array = trajectory[name]
        if list(array.shape) != spec["shape"] or str(array.dtype) != spec["dtype"]:
            raise ValueError(f"{name}: el archivo no coincide con sus metadatos")
    return trajectory, metadata
