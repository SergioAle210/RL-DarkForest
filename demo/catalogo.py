"""Genera un catálogo de escenarios para la demo, sin servidor, a partir de redes ya entrenadas.

Cada escenario es un episodio completo (mundo, política de cada civilización y semilla) guardado
como JSON en demo/web/catalogo/ junto con un índice. La página los carga sola cuando no encuentra
el servidor, así que sirve para presentar sin instalar nada.

Uso: python -m demo.catalogo [--redes ../kout8 ../kout3]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from demo.servidor import RAIZ, descubrir_redes, simular

SALIDA = RAIZ / "web" / "catalogo"
MUNDO = {"lethality": 0.5, "signal_speed": 0.22, "noise_std": 0.3, "cost_ref": 0.25}


def red(redes: list[dict], texto: str) -> str | None:
    """Identificador de la primera red cuyo nombre contiene el texto."""
    return next((r["id"] for r in redes if texto in r["nombre"]), None)


def escenarios(redes: list[dict]) -> list[dict]:
    """Escenarios de la presentación; los que necesitan una red ausente se omiten."""
    cooperar = {"seed": 1}
    lista = [
        ("Dos estrategias cooperan", MUNDO, ["fijo:coop_real", "fijo:coop_real"], cooperar),
        ("Dos estrategias solas", MUNDO, ["fijo:solo", "fijo:solo"], cooperar),
        ("Cooperar contra quien dispara", MUNDO, ["fijo:coop_real", "fijo:ataque_solo"], cooperar),
        ("Anomalía letal (λ = 1)", {**MUNDO, "lethality": 1.0}, ["fijo:solo", "fijo:solo"], {}),
        (
            "Mentiroso contra cooperador ingenuo",
            {**MUNDO, "deception": True},
            ["fijo:mentiroso", "fijo:coop_real"],
            {"seed": 3},
        ),
        (
            "Mentiroso contra prudente (señales costosas)",
            {**MUNDO, "deception": True, "costly_signals": True},
            ["fijo:mentiroso", "fijo:prudente"],
            {"seed": 3},
        ),
    ]
    entrenadas = [
        ("Red entrenada contra cooperador", "ft_l0p5_c0p22", "fijo:coop_real", MUNDO),
        ("Red entrenada contra quien dispara", "ft_l0p5_c0p22", "fijo:ataque_solo", MUNDO),
        ("Red sin imitación contra cooperador", "cero_l0p5_c0p22", "fijo:coop_real", MUNDO),
        (
            "Red con señales rápidas (c = 1, λ = 0.25)",
            "ft_l0p25_c1p0",
            "fijo:coop_real",
            {**MUNDO, "lethality": 0.25, "signal_speed": 1.0},
        ),
        (
            "Red con λ = 1: casi no usa el canal",
            "ft_l1p0_c1p0",
            "fijo:coop_real",
            {**MUNDO, "lethality": 1.0, "signal_speed": 1.0},
        ),
    ]
    for nombre, texto, otra, mundo in entrenadas:
        propia = red(redes, texto)
        if propia:
            lista.append((nombre, mundo, [propia, otra], {}))
    seis = red(redes, "n6")
    if seis:
        lista.append(("Seis civilizaciones con la misma red", {**MUNDO, "n_civs": 6}, [seis], {}))
    return [{"nombre": n, "mundo": m, "politicas": p, **o} for n, m, p, o in lista]


def main() -> None:
    """Simula cada escenario y escribe el índice del catálogo."""
    p = argparse.ArgumentParser()
    p.add_argument("--redes", nargs="*", type=Path, default=[])
    args = p.parse_args()
    carpetas = args.redes or sorted(RAIZ.parent.parent.glob("kout*"))
    redes = descubrir_redes(carpetas)
    SALIDA.mkdir(parents=True, exist_ok=True)
    indice = []
    for numero, esc in enumerate(escenarios(redes), start=1):
        datos = simular(esc["mundo"], esc["politicas"], esc.get("seed", 0), True, 40)
        archivo = f"escenario_{numero:02d}.json"
        (SALIDA / archivo).write_text(json.dumps(datos, separators=(",", ":")), encoding="utf-8")
        indice.append({"nombre": esc["nombre"], "archivo": archivo})
        promedio = datos.get("promedio", {}).get("retorno")
        print(
            f"{archivo}: {esc['nombre']} episodio {datos['resumen']['retorno'][:2]}"
            f" promedio {promedio}",
            flush=True,
        )
    (SALIDA / "index.json").write_text(json.dumps(indice, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
