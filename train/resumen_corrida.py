"""Resumen en JSON de un checkpoint: retorno, ataques y uso del canal, con y sin canal.

Evalúa la red muestreada y en greedy, mide la secuencia contacto, mensaje y ataque, y repite la
evaluación con el canal apagado (env.channel_enabled) y con los mensajes barajados
(env.shuffle_messages). Si apagar o barajar el canal no cambia el retorno, la red no lo usa.
También sirve para los snapshots de una corrida (snapshots/iter_*.msgpack).

Uso: python -m train.resumen_corrida --checkpoint <ruta> --out resumen.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from train.diagnostico_ataque import diagnose
from train.episodes import checkpoint_policy
from train.evaluar_checkpoint import evaluate

CANAL = {
    "canal_apagado": {"channel_enabled": False},
    "mensajes_barajados": {"shuffle_messages": True},
    "sin_compartir": {"share_enabled": False},  # H3: sin la señal verificable, solo charla barata
}


def summarize(path: str | Path, episodes: int = 400, seed: int = 100) -> dict:
    """Resumen de un checkpoint con las mismas semillas de evaluación para todas las variantes."""
    result: dict = {"checkpoint": str(path), "episodes": episodes}
    for greedy in (False, True):
        cfg, policy = checkpoint_policy(path, greedy=greedy)
        label = "greedy" if greedy else "muestreo"
        result[label] = {
            "evaluacion": evaluate(cfg, policy, seed, episodes),
            "diagnostico": diagnose(cfg, policy, seed, episodes),
        }
        for name, changes in CANAL.items():
            variant = replace(cfg, env=replace(cfg.env, **changes))
            result[label][name] = evaluate(variant, policy, seed, episodes)
    return result


def main() -> None:
    """Calcula el resumen y lo guarda o lo imprime."""
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out")
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--seed", type=int, default=100)
    args = p.parse_args()
    text = json.dumps(summarize(args.checkpoint, args.episodes, args.seed), indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
