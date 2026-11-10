"""Convierte un episodio guardado (.npz y .json) a un JSON compacto para el visor en Bevy.

Solo lee la trayectoria: no ejecuta el entorno. Las civilizaciones no se mueven, así que la
posición se guarda una vez; los eventos (señales y proyectiles) se deduplican por (origen, turno).

Uso: python -m viz.exportar_json --episode runs/<corrida>/0/trajectories/episode_000002.npz \
         --out visor/episodios/ejemplo.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from darkforest.trajectories import load_episode


def extraer_muertes(causa: np.ndarray, atacante: np.ndarray) -> list[dict]:
    """Primera muerte de cada civilización: turno (estado ya muerta), causa y atacante."""
    muertes = []
    for civ in range(causa.shape[1]):
        turnos = np.flatnonzero(causa[:, civ] != 0)
        if turnos.size:
            t = int(turnos[0])
            muertes.append(
                {
                    "t": t + 1,
                    "civ": civ,
                    "causa": "anomalia" if int(causa[t, civ]) == 1 else "ataque",
                    "atacante": int(atacante[t, civ]),
                }
            )
    return sorted(muertes, key=lambda m: (m["t"], m["civ"]))


def export_episode(
    traj: dict[str, np.ndarray],
    metadata: dict,
    deception: bool = False,
    politicas: list[str] | None = None,
) -> dict:
    """Estructura serializable: estados por turno y eventos de señales, proyectiles y estallidos."""
    steps = traj["actions"].shape[0]
    n = traj["alive"].shape[1]
    symbols = int(metadata["n_symbols"])
    lies: set[tuple[int, int]] = set()
    if deception:
        mask = (traj["info_executed_kind"] == 3) & (traj["actions"] == 2 + symbols)
        for t, o in zip(*np.nonzero(mask), strict=True):
            lies.add((int(o), int(t)))
    signals: dict[tuple[int, int], int] = {}
    projectiles: dict[tuple[int, int], tuple[float, float]] = {}
    for frame in range(steps + 1):
        for index in np.flatnonzero(traj["sig_valid"][frame]):
            key = (int(traj["sig_origin"][frame, index]), int(traj["sig_t"][frame, index]))
            first = int(traj["sig_content"][frame, index, 0])
            message = 0 if first < 0 else (2 if first == symbols else 1)
            signals[key] = 3 if key in lies else max(signals.get(key, 0), message)
        for index in np.flatnonzero(traj["prj_valid"][frame]):
            key = (int(traj["prj_origin"][frame, index]), int(traj["prj_t"][frame, index]))
            target = traj["prj_target"][frame, index]
            projectiles[key] = (float(target[0]), float(target[1]))
    sello = traj["message_t"]
    turnos, receptores, emisores = np.nonzero(sello[1:] > sello[:-1])
    mensajes = [
        [int(t) + 1, int(r), int(e)] for t, r, e in zip(turnos, receptores, emisores, strict=True)
    ]
    theta = traj["theta"][0]
    return {
        "n": n,
        "steps": steps,
        "c": float(metadata["signal_speed"]),
        "v": float(metadata["proj_speed"]),
        "policy": str(metadata.get("policy", "")),
        "seed": int(metadata.get("seed", 0)),
        "pos": np.round(traj["pos"][0], 5).tolist(),
        "alive": traj["alive"].astype(int).tolist(),
        "res": np.round(traj["resources"], 4).tolist(),
        "refuge": traj["refuge"].astype(int).tolist(),
        "kind": traj["info_executed_kind"].astype(int).tolist(),
        "act": traj["actions"].astype(int).tolist(),
        "mensajes": mensajes,  # [turno, receptor, emisor] de cada mensaje explícito recibido
        # luz con que cada civilización es detectable (EnvState.emission): 0 baja, 1 alta;
        # el entorno no produce 2 (ese nivel solo existe en sig_emission, para mensajes)
        "emision": traj["emission"].astype(int).tolist(),
        "muertes": extraer_muertes(traj["info_death_cause"], traj["info_attacker"]),
        "politicas": list(politicas) if politicas is not None else [],
        "deception": bool(deception),
        "n_symbols": symbols,
        "signals": [
            {"o": o, "t": t, "msg": m}
            for (o, t), m in sorted(signals.items(), key=lambda x: x[0][1])
        ],
        "projectiles": [
            {"o": o, "t": t, "tx": x, "ty": y}
            for (o, t), (x, y) in sorted(projectiles.items(), key=lambda x: x[0][1])
        ],
        "bursts": [int(f) for f in range(1, steps + 1) if traj["info_bursts"][f - 1]],
        "zone": {"x": float(theta[2]), "y": float(theta[3]), "r": float(theta[4])},
    }


def main() -> None:
    """Exporta un episodio guardado a JSON."""
    p = argparse.ArgumentParser()
    p.add_argument("--episode", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    traj, metadata = load_episode(args.episode)
    data = export_episode(traj, metadata)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(
        f"{args.out}: {data['n']} civilizaciones, {data['steps']} pasos, "
        f"{len(data['signals'])} señales, {len(data['projectiles'])} proyectiles"
    )


if __name__ == "__main__":
    main()
