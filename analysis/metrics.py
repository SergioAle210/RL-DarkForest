"""Métricas puras sobre trayectorias y clasificación de régimen (sección 5)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

Trajectory = dict[str, np.ndarray]  # claves: obs, actions, rewards, alive, info...

REGIMES = ("cooperacion", "silencio_mutuo", "ataque_preventivo", "coalicion", "colapso")


def survival_rate(traj: Trajectory) -> dict[str, float]:
    """Total y por causa (anomalía / ataque)."""
    raise NotImplementedError


def low_emission_fraction(traj: Trajectory) -> float:
    raise NotImplementedError


def transmissions_per_episode(traj: Trajectory) -> dict[str, float]:
    """Transmisiones y observaciones compartidas."""
    raise NotImplementedError


def preventive_attacks(traj: Trajectory) -> float:
    raise NotImplementedError


def anomaly_prediction_accuracy(traj: Trajectory) -> float:
    """Refugios correctos / total de estallidos."""
    raise NotImplementedError


def message_theta_mutual_info(traj: Trajectory) -> float:
    """I(M; θ) estimada (p. ej. discretizando θ)."""
    raise NotImplementedError


def classify_regime(traj: Trajectory) -> str:
    """Asigna uno de REGIMES al episodio."""
    raise NotImplementedError


def channel_ablation_drop(baseline: Trajectory, ablated: Trajectory) -> float:
    """Caída de supervivencia al anular el canal o barajar mensajes."""
    raise NotImplementedError


METRICS: dict[str, Callable[[Trajectory], Any]] = {
    "survival": survival_rate,
    "low_emission": low_emission_fraction,
    "tx": transmissions_per_episode,
    "preventive_attacks": preventive_attacks,
    "anomaly_accuracy": anomaly_prediction_accuracy,
    "mutual_info": message_theta_mutual_info,
}
