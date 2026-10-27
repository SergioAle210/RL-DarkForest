"""Métricas puras sobre trayectorias y clasificación de régimen (sección 5)."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from darkforest.trajectories import load_episode

Trajectory = dict[str, np.ndarray]  # claves: obs, actions, rewards, alive, info...

REGIMES = ("cooperacion", "silencio_mutuo", "ataque_preventivo", "coalicion", "colapso")


def survival_rate(traj: Trajectory) -> dict[str, float]:
    """Total y por causa (anomalía / ataque)."""
    causes = traj["info_death_cause"]
    agents = traj["alive"].shape[1]
    return {
        "survival": float(traj["alive"][-1].mean()),
        "death_anomaly": float((causes == 1).sum() / agents),
        "death_attack": float((causes == 2).sum() / agents),
    }


def low_emission_fraction(traj: Trajectory) -> float:
    """Cuenta emisión baja solo en estados de civilizaciones vivas."""
    alive = traj["alive"][1:]
    return float(((traj["emission"][1:] == 0) & alive).sum() / max(int(alive.sum()), 1))


def transmissions_per_episode(traj: Trajectory) -> dict[str, float]:
    """Transmisiones y observaciones compartidas."""
    kinds = traj["info_executed_kind"]
    return {"symbols": int((kinds == 3).sum()), "shared_measurements": int((kinds == 4).sum())}


def preventive_attacks(traj: Trajectory) -> float:
    """Cuenta lanzamientos contra civilizaciones que todavía no habían atacado."""
    return float(traj["info_preventive_attacks"].sum())


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


def summarize(traj: Trajectory) -> dict:
    """Resume un episodio sin asignarle un régimen científico no validado."""
    return {
        **survival_rate(traj),
        "low_emission": low_emission_fraction(traj),
        **transmissions_per_episode(traj),
        "preventive_attacks": preventive_attacks(traj),
        "mean_return": float(traj["rewards"].sum(axis=0).mean()),
        "steps": int(traj["actions"].shape[0]),
    }


def main() -> None:
    """Calcula resultados desde un episodio guardado, sin ejecutar simulaciones."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    trajectory, metadata = load_episode(args.episode)
    report = {
        "schema_version": 1,
        "source": str(args.episode),
        "config_id": metadata["config_id"],
        "seed": metadata["seed"],
        "exploratory": True,
        "metrics": summarize(trajectory),
    }
    output = args.out or args.episode.parent.parent / "results.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


METRICS: dict[str, Callable[[Trajectory], Any]] = {
    "survival": survival_rate,
    "low_emission": low_emission_fraction,
    "tx": transmissions_per_episode,
    "preventive_attacks": preventive_attacks,
    "anomaly_accuracy": anomaly_prediction_accuracy,
    "mutual_info": message_theta_mutual_info,
}


if __name__ == "__main__":
    main()
