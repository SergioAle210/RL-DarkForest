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
    """Refugios correctos por estallido: suma de aciertos sobre el número de estallidos."""
    bursts = int(traj["info_bursts"].sum())
    return float(traj["info_shelters_correct"].sum() / max(bursts, 1))


def mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """Información mutua en bits entre dos variables discretas, con corrección de Miller-Madow."""
    x, y = np.asarray(x), np.asarray(y)
    if x.size == 0:
        return 0.0
    _, xi = np.unique(x, return_inverse=True)
    _, yi = np.unique(y, return_inverse=True)
    joint = np.zeros((xi.max() + 1, yi.max() + 1))
    np.add.at(joint, (xi, yi), 1)
    n = joint.sum()
    px, py, pxy = joint.sum(1) / n, joint.sum(0) / n, joint / n
    nonzero = pxy > 0
    plugin = float((pxy[nonzero] * np.log2(pxy[nonzero] / np.outer(px, py)[nonzero])).sum())
    bias = (joint.shape[0] - 1) * (joint.shape[1] - 1) / (2 * n * np.log(2))
    return max(plugin - bias, 0.0)


def message_theta_mutual_info(trajs: list[Trajectory], n_symbols: int = 4, bins: int = 4) -> float:
    """I(M; θ) en bits sobre varios episodios: símbolo transmitido contra el periodo de θ.

    Cada transmisión aporta un par (símbolo, cuartil del periodo del episodio). Los compartir no
    entran porque llevan una medición continua, no un símbolo. Con pocos mensajes la corrección
    de sesgo lo deja en 0.
    """
    symbols, periods = [], []
    for traj in trajs:
        actions, alive = traj["actions"], traj["alive"][:-1]
        sent = (actions >= 3) & (actions < 3 + n_symbols) & alive
        count = int(sent.sum())
        symbols += list(actions[sent] - 3)
        periods += [float(traj["theta"][0][1])] * count
    if not symbols:
        return 0.0
    edges = np.quantile(periods, np.linspace(0, 1, bins + 1)[1:-1])
    return mutual_information(np.array(symbols), np.digitize(periods, edges))


def classify_regime(traj: Trajectory) -> str:
    """Etiqueta descriptiva por reglas: ataque, colapso, cooperación o silencio.

    No es un régimen validado: son umbrales fijos (ver REGIMES) sobre tres cantidades. Sin
    detección de comunidades, "coalicion" no se asigna aquí.
    """
    survival = float(traj["alive"][-1].mean())
    if survival == 0:
        return "colapso"
    if preventive_attacks(traj) > 0:
        return "ataque_preventivo"
    talks = sum(transmissions_per_episode(traj).values())
    if talks > 0 and survival >= 0.8:
        return "cooperacion"
    return "silencio_mutuo" if talks == 0 else "colapso"


def channel_ablation_drop(baseline: Trajectory, ablated: Trajectory) -> float:
    """Caída de supervivencia al anular el canal o barajar mensajes."""
    return float(baseline["alive"][-1].mean() - ablated["alive"][-1].mean())


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
